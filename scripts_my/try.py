
from dataclasses import dataclass, field
from typing import Optional

import tyro
from peft import LoraConfig
from tqdm import tqdm

from trl import PPOConfig


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

args = tyro.cli(ScriptArguments)

print(args.ppo_config.model_name)