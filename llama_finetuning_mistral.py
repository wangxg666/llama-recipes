# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import pathlib
import time

import fire

# Unused imports removed
from utils import fsdp_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    BitsAndBytesConfig
)
import torch.distributed as dist
# Unused imports removed
from utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    print_model_size,
    get_policies  
)

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
import torch
import torch.distributed as dist
from model_checkpointing import load_optimizer_checkpoint

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch

import functools

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
    gradient_accumulation_steps = train_config.train_batch_size // train_config.micro_batch_size

    model_name = train_config.model_name
    # gpu3 cpu memory is not enough, lazy loading with 20s after
    if train_config.pre_train_model_path and pathlib.Path(train_config.pre_train_model_path).exists():
        model_name = train_config.pre_train_model_path

    print(f'{"x" * 20}    {model_name}    {"x" * 20}')

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=2048,
    )
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )

    # Load the pre-trained model and setup its configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True if train_config.quantization else None,
        device_map="auto" if train_config.quantization else None,
    )
    model.to(torch.bfloat16)
    print(model)

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            MistralDecoderLayer,
        },
    )

    local_fsdp_config = dict(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=None,
        param_init_fn=None,
        cpu_offload=None,
    )
    model = FSDP(model, **local_fsdp_config)
    policies.apply_fsdp_checkpointing(model)

    dataset_config = generate_dataset_config(train_config, kwargs)
    if train_config.dataset_dir != '':
        dataset_config.dataset_dir = train_config.dataset_dir
    print(dataset_config.dataset, dataset_config.dataset_dir)
    
     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
    
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = DistributedSampler(
        dataset_train,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        dataset_val,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
    )
        
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.micro_batch_size,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=train_config.valid_batch_size,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=val_sampler if val_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )
        
    # Initialize the optimizer and learning rate scheduler
    optimizer = AnyPrecisionAdamW(
        model.parameters(),
        lr=train_config.lr,
        momentum_dtype=torch.bfloat16,
        variance_dtype=torch.bfloat16,
        use_kahan_summation=False,
    )

    from transformers import get_scheduler
    num_training_steps = train_config.num_epochs * len(train_dataloader)
    if rank == 0:
        print(f'num training steps = {num_training_steps}')
        print(f'num eval steps = {len(eval_dataloader)}')
        print(f'num data batches = {len(train_dataloader)}')

    from transformers import get_scheduler, SchedulerType
    total_steps = len(train_dataloader) * train_config.num_epochs
    scheduler = get_scheduler(
        SchedulerType.COSINE,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * 0.03),
        num_training_steps=total_steps,
    )

    # Start the training process
    results = train(
        model,
        train_dataloader,
        train_sampler,
        eval_dataloader, 
        tokenizer,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
