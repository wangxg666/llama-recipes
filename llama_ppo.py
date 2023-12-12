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
import asyncio
import collections
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Optional

import torch
from accelerate import Accelerator
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, set_seed
from ppo_trainer import PPOTrainer

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# from torch import distributed as dist
# dist.init_process_group("nccl")
# torch.cuda.set_device(rank)
# print('+' * 10, rank, dist.get_rank(), '*' * 10, flush=True)

from agent.generate import *


tqdm.pandas()

from torch.utils.data import Dataset


class PPODataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.samples['rewards'] = [(x.item()+10.) / 5. for x in self.samples['rewards']]
        self.samples['query_tensors'] = [x.tolist() for x in self.samples['query_tensors']]
        self.samples['response_tensors'] = [x.tolist() for x in self.samples['response_tensors']]
    def __len__(self):
        return len(self.samples['rewards'])
    def __getitem__(self, index):
        return {k: v[index] for k, v in self.samples.items()}


def collator(data):
    return dict((key, [torch.tensor(d[key]) for d in data]) for key in data[0])


@dataclass
class ScriptArguments:
    model_name: Optional[str] = "meta-llama/Llama-2-13b"

    target_kl: Optional[float] = 6.0
    kl_penalty: Optional[str] = 'kl'

    learning_rate: Optional[float] = 1.41e-5
    batch_size: Optional[int] = 4
    mini_batch_size: Optional[int] = 4
    gradient_accumulation_steps: Optional[int] = 1

    use_peft: Optional[bool] = False

    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 16

    data_input_file: Optional[str] = '/home/paperspace/xingguang/datasets/ppo_test/data.bin'
    model_output_dir: Optional[str] = '/home/paperspace/xingguang/models/ppo_test/'


if __name__ == '__main__':
    from transformers import HfArgumentParser
    script_args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]

    ppo_config = PPOConfig(
        model_name=script_args.model_name,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        target_kl=script_args.target_kl,
        kl_penalty=script_args.kl_penalty,
        learning_rate=script_args.learning_rate,
        seed=0,
        use_score_scaling=False,
        use_score_norm=False,
        early_stopping=False,
        score_clip=None,
        log_with=None,
    )

    os.makedirs(script_args.model_output_dir, exist_ok=True)

    peft_config = None
    if script_args.use_peft:
        peft_config = LoraConfig(r=script_args.lora_rank, lora_alpha=script_args.lora_alpha)

    ppo_config.model_name = '/home/paperspace/xingguang/models/my_agent_sft_dataset.13b.2e-5.full.B4.E1.v07.all.hf'

    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

    # set seed before initializing value head for deterministic eval
    set_seed(ppo_config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    if not script_args.use_peft:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            ppo_config.model_name,
            # load_in_8bit=True,
            trust_remote_code=False
        )
        device_map = None
        peft_config = None
    else:
        peft_config = script_args.peft_config
        ref_model = None
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        trust_remote_code=False,
        device_map=device_map,
        peft_config=peft_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    tokenizer.pad_token_id = tokenizer.eos_token_id

    samples = pickle.load(open(script_args.data_input_file, 'rb'))
    datasets = PPODataset(samples)

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=datasets, data_collator=collator)
    device = ppo_trainer.accelerator.device
    print('*' * 10, device, '*' * 10, flush=True)

    random.seed(rank)

    for batch in ppo_trainer.dataloader:
        print(f'{rank}, {batch.keys()}', flush=True)

        query_tensors = batch['query_tensors']
        response_tensors = batch['response_tensors']
        rewards = batch['rewards']

        print(f'{rank}, {rewards}', flush=True)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    import os
    ppo_trainer.model.save_pretrained(script_args.model_output_dir)
