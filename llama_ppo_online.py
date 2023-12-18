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
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, set_seed

from ppo_trainer import PPOTrainer, print_rank_0


tqdm.pandas()


@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="agent_sft_act_dataset.7b.2e-5.full.B16.E1.hf",
            query_dataset="",
            reward_model="",
            learning_rate=1.41e-6,
            log_with=None,
            mini_batch_size=4,
            batch_size=4,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
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

args = tyro.cli(ScriptArguments)
print(args)


from torch.utils.data import Dataset
class PPODataset(Dataset):
    def __init__(self, size=10000):
        self.size = size
        self.data = list(range(size))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {'index': index}


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = PPODataset(size=1000)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(args.ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.ppo_config.model_name,
        load_in_8bit=True,
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

tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

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

from agent.generate_two_stage import get_batch
import torch.distributed as dist

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
    batch_input = get_batch(len(batch),
                            policy_model=model,
                            policy_tokenizer=tokenizer,
                            device=ppo_trainer.current_device)

    query_tensors = batch['query_tensors']
    response_tensors = batch['response_tensors']
    reward_tensors = batch['reward_tensors']

    for i in range(len(query_tensors)):
        print_rank_0(f'query: {tokenizer.decode(query_tensors[i])}')
        print_rank_0(f'response: {tokenizer.decode(response_tensors[i])}')
        print_rank_0(f'reward: {reward_tensors[i]}')
        break

    dist.barrier()
    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
    # ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

ppo_trainer.model.save_pretrained(args.output_checkpoint_dir)
