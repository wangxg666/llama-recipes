# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from trl import PPOConfig, set_seed

from ppo_trainer import PPOTrainer, print_rank_0
from modeling_value_head import AutoModelForCausalLMWithValueHead

tqdm.pandas()


@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="agent_sft_act_dataset.7b.2e-5.full.B16.E1.hf",
            query_dataset="",
            reward_model="",
            learning_rate=1e-8,
            log_with='wandb',
            mini_batch_size=4,
            batch_size=4,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            vf_coef=0.1,
            seed=0,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
            ppo_epochs=1,
        )
    )
    """whether to use seq2seq models"""
    use_peft: bool = False
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_checkpoint_dir: str = ''

    pre_train_critic: bool = False
    pre_train_critic_data_dir: str = ''
    pre_train_critic_checkpoint_dir: str = ''

args = tyro.cli(ScriptArguments)

# set seed before initializing value head for deterministic eval
set_seed(args.ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.

if not args.use_peft:
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.ppo_config.model_name,
        trust_remote_code=args.trust_remote_code
    )
    device_map = None
    peft_config = None
else:
    peft_config = args.peft_config
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    args.ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)

if args.pre_train_critic:
    ref_model = model
    # 冻结模型参数
    for i, layer in enumerate(model.pretrained_model.model.layers):
        for param in layer.parameters():
            param.requires_grad = False
    for param in model.pretrained_model.model.embed_tokens.parameters():
        param.requires_grad = False


if os.path.exists(f'{args.pre_train_critic_checkpoint_dir}'):
    state_dict = {}
    for filename in os.listdir(f'{args.pre_train_critic_checkpoint_dir}'):
        if filename.startswith('pytorch_model') and filename.endswith('.bin'):
            state_dict.update(torch.load(f'{args.pre_train_critic_checkpoint_dir}/{filename}'))
    state_dict = {k: v for k, v in state_dict.items() if 'v_head' in k}
    model.load_state_dict(state_dict, strict=False)
    print_rank_0(f'load {json.dumps(list(state_dict.keys()), indent=2)} from pre-trained critic model')

tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=None, data_collator=None)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        pass

# from agent.generate_two_stage_origin import get_batch, parse_dialog
from agent.generate_two_stage_replace import get_batch, parse_dialog
import torch.distributed as dist

def safty_get_batch(batch_size, policy_model, policy_tokenizer, device):
    while True:
        try:
            batch_input = get_batch(batch_size, policy_model, policy_tokenizer, device)
            return batch_input
        except Exception as e:
            logging.error(f'error at generation batch, {e}')
            continue


if args.pre_train_critic \
        and os.path.exists(args.pre_train_critic_data_dir) \
        and dist.get_world_size() == 1:

    # critic 只在单卡状态下训练
    datas = []
    for filename in os.listdir(args.pre_train_critic_data_dir):
        try:
            datas.extend([json.loads(line) for line in open(f'{args.pre_train_critic_data_dir}/{filename}')])
        except:
            continue
    print_rank_0(f'load {len(datas)} datas from {args.pre_train_critic_data_dir}')

    train_datas = {
        'query_tensors': [],
        'response_tensors': [],
        'reward_tensors': []
    }

    for data in tqdm(datas):
        batch_input = parse_dialog(data['dialog'], data['reward'], batch_size=-1, policy_tokenizer=tokenizer)
        for key, val in batch_input.items():
            train_datas[key].extend(val)
    print_rank_0(f'load {len(train_datas["query_tensors"])} training datas')

    idxs = [i for i in range(len(train_datas['query_tensors']))]
    random.shuffle(idxs)

    eos = (len(idxs) // args.ppo_config.batch_size) * args.ppo_config.batch_size
    for bos in range(0, eos, args.ppo_config.batch_size):
        batch_input = {
            k: [v[idx] for idx in idxs[bos: bos+args.ppo_config.batch_size]]
            for k, v in train_datas.items()
        }
        query_tensors = batch_input['query_tensors']
        response_tensors = batch_input['response_tensors']
        reward_tensors = batch_input['reward_tensors']
        print(bos, len(query_tensors), len(response_tensors), len(reward_tensors))

        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors, train_generation=False)
        ppo_trainer.log_stats(stats, {}, reward_tensors,
                              columns_to_log=["query", "response", "ref_response", "ref_rewards"])
    ppo_trainer.model.save_pretrained(args.pre_train_critic_checkpoint_dir)

else:
    for step in tqdm(range(5000)):
        print('+' * 20, f'step = {step}', '+' * 20)
        model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)

        batch_input = safty_get_batch(args.ppo_config.batch_size,
                                      policy_model=model,
                                      policy_tokenizer=tokenizer,
                                      device=ppo_trainer.current_device)

        query_tensors = batch_input['query_tensors']
        response_tensors = batch_input['response_tensors']
        reward_tensors = batch_input['reward_tensors']

        dist.barrier()
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors, train_generation=True)
        ppo_trainer.log_stats(stats, {}, reward_tensors, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

        if (step + 1) % 200 == 0:
            sub_dir = f'step_{str(10000 + step + 1)[1:]}'
            ppo_trainer.model.save_pretrained(args.output_checkpoint_dir + '/' + sub_dir)
