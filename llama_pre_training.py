# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import pathlib
import time

import fire

# Unused imports removed
from utils import fsdp_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
    BitsAndBytesConfig
)
import torch.distributed as dist
# Unused imports removed
from utils.pre_train_utils import (
    save_train_params,
    train,
    save_model,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    print_model_size,
    get_policies  
)

from transformers.models.llama import modeling_llama
from utils.dataset_utils import get_preprocessed_dataset

from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from peft import (
    PeftModel, PeftConfig,
    get_peft_model, TaskType, prepare_model_for_int8_training
)

import configs
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.utils.data import DistributedSampler
import policies
from policies import AnyPrecisionAdamW
from configs import fsdp_config, train_config
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from model_checkpointing import load_optimizer_checkpoint


def main(**kwargs):

    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1.

    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
        setup_environ_flags(rank)
    
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = train_config.batch_size_training // train_config.micro_batch_size

    model_name = train_config.model_name
    # gpu3 cpu memory is not enough, lazy loading with 20s after
    if train_config.pre_train_model_path and pathlib.Path(train_config.pre_train_model_path).exists():
        if rank == 0 or rank == 1:
            time.sleep(20)
        model_name = train_config.pre_train_model_path
    print(f'{"x" * 20}    {model_name}    {"x" * 20}')

    # Load the pre-trained model and setup its configuration
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True if train_config.quantization else None,
        device_map="auto" if train_config.quantization else None,
    )
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)
    
    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)
        
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )

    if train_config.peft_model:
        model = PeftModel.from_pretrained(model, train_config.peft_model)
        for name, param in model.named_parameters():
            if 'lora' in name or 'Lora' in name:
                param.requires_grad = True
        model.print_trainable_parameters()

    elif train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
   
        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)
    if train_config.dataset_sub_dir_prefix != '':
        dataset_config.sub_dir_prefix = train_config.dataset_sub_dir_prefix
    print(dataset_config.dataset, dataset_config.sub_dir_prefix)
    
    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

    if train_config.optimizer_checkpoint_path:
        from pathlib import Path
        path = Path(train_config.optimizer_checkpoint_path)
        if path.exists():
            sharded_osd = load_optimizer_checkpoint(model, Path(train_config.optimizer_checkpoint_path), rank)
            optimizer.load_state_dict(sharded_osd)
            del sharded_osd
            torch.cuda.empty_cache()

    from transformers import get_scheduler, SchedulerType


    # pre-train 的学习率不变
    scheduler = get_scheduler(
        SchedulerType.CONSTANT,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=-1,
    )
    # scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    def build_dataset(split, shuffle=False):
        dataset = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split=split,
        )

        if not train_config.enable_fsdp or rank == 0:
            print(f"--> split Set Length = {len(dataset)}")

        sampler = DistributedSampler(
            dataset,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=shuffle,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_config.micro_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=sampler,
            drop_last=True,
            collate_fn=default_data_collator,
        )

        return sampler, dataloader

    accu_step = 0
    dataset_config.input_file = 'valid.bin'
    valid_sampler, valid_dataloader = build_dataset('test', False)

    os.makedirs(train_config.output_dir, exist_ok=True)
    if train_config.enable_fsdp and not train_config.use_peft:
        save_dir = train_config.output_dir
        save_train_params(save_dir, train_config, fsdp_config, rank)

    for epoch in range(train_config.num_epochs):
        filenames = [f for f in sorted(os.listdir(dataset_config.root + '/' + dataset_config.sub_dir_prefix)) if 'train' in f]
        for i, filename in enumerate(filenames):
            # 指定训练文件
            dataset_config.input_file = filename
            # 加载分片训练数据
            train_sampler, train_dataloader = build_dataset('train', True)

            num_training_steps = len(train_dataloader)
            if rank == 0:
                print(f'[{i}], input = {filename}')
                print(f'[{i}], num training steps = {num_training_steps}')
                print(f'[{i}], num eval steps = {len(valid_dataloader)}')
                print(f'[{i}], num data batches = {len(train_dataloader)}')

            # Start the training process
            accu_step = train(
                model,
                train_dataloader,
                train_sampler,
                valid_dataloader,
                tokenizer,
                optimizer,
                scheduler,
                gradient_accumulation_steps,
                train_config,
                fsdp_config if train_config.enable_fsdp else None,
                local_rank if train_config.enable_fsdp else None,
                rank if train_config.enable_fsdp else None,
                first_step=accu_step
            )

            if (i+1) % 5 == 0:
                save_dir = save_model(model, train_config, fsdp_config, rank, optimizer, accu_step=accu_step)
            else:
                save_dir = save_model(model, train_config, fsdp_config, rank, None, accu_step=accu_step)
            print(f'{accu_step} {filename}', file=open(f'{save_dir}/file.txt', 'w'))

        save_model(model, train_config, fsdp_config, rank, optimizer, epoch=epoch)

if __name__ == "__main__":
    fire.Fire(main)
