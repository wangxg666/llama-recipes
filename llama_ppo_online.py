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
            model_name="/home/paperspace/xingguang/models/my_agent_sft_dataset.13b.2e-5.full.B4.E1.v07.all.hf",
            query_dataset="/home/paperspace/xingguang/datasets/agent_raft.v07/ppo.train.jsonl",
            reward_model="",
            learning_rate=1.41e-6,
            log_with=None,
            mini_batch_size=1,
            batch_size=1,
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
    def __init__(self, input_file):
        self.samples = {
            'rewards': [],
            'query_tensors': [],
            'response_tensors': [],
        }
        for data in open(input_file):
            data = json.loads(data)
            keys = [key[:-1] for key in self.samples if key[:-1] in data]
            if data['key'] == 'rag':
                continue
            if len(keys) != len(self.samples):
                continue
            if len(data['query_tensor']) + len(data['response_tensor']) > 1536:
                continue
            if len(data['response_tensor']) == 0:
                continue
            for key in keys:
                self.samples[key + 's'].append(torch.tensor(data[key]))
        print(f'load {len(self.samples["rewards"])} from {input_file}')
        self.norm_rewards()

    def norm_rewards(self):
        self.samples['rewards'] = [r / 10. for r in self.samples['rewards']]

    def __len__(self):
        return len(self.samples['rewards'])

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.samples.items()}


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = PPODataset(args.ppo_config.query_dataset)

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
model = model.half()

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


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}


from agent.generate import generate_dialog
from agent.convert_ppo_offline import convert_sft_types, tokenize_samples


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    out_services, out_turns, out_rewards = generate_dialog(ppo_trainer)
    key2turns = convert_sft_types(out_turns)
    reward = out_rewards['reward']['avg_score']
    if reward >= 4:
        for key, turns in key2turns.items():
            key2sout[key].write('\n'.join([json.dumps(turn) for turn in turns]) + '\n')
            key2sout[key].flush()
    samples = tokenize_samples(reward, key2turns, tokenizer)


    # Get response from gpt2
    # response_tensors, ref_response_tensors = ppo_trainer.generate(
    #     query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    # )

    # batch["response"] = tokenizer.batch_decode(response_tensors)
    # batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)
    # batch["ref_rewards"] = rewards

    print_rank_0(tokenizer.batch_decode(batch["response_tensors"]))

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

ppo_trainer.model.save_pretrained(args.output_checkpoint_dir)
