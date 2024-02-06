# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field
from typing import ClassVar, List, Tuple


@dataclass
class lora_config:
    r: int = 32
    lora_alpha: int = 32
    target_modules: Tuple[str] = ("q_proj", "v_proj", "k_proj", "o_proj")
    # target_modules: Tuple[str] = ("gate_proj", "down_proj", 'up_proj')
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False


@dataclass
class llama_adapter_config:
    adapter_len: int = 10
    adapter_layers: int = 30
    task_type: str = "CAUSAL_LM"


@dataclass
class prefix_config:
    num_virtual_tokens: int = 30
    task_type: str = "CAUSAL_LM"
