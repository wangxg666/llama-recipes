# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import pandas as pd
import torch
import os
import sys
import time
import json
from typing import List

from transformers import LlamaTokenizer
from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model, load_llama_from_config

def main(
    model_name,
    peft_model: str=None,
    dataset: str=None,
    quantization: bool=False,
    max_new_tokens = 100, #The maximum numbers of tokens to generate
    input_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    **kwargs
):

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )

    if peft_model:
        model = load_peft_model(model, peft_model)

    from ft_datasets.my_clickbait_dataset import PROMOT_DICT
    if dataset not in PROMOT_DICT:
        exit(0)
    prompt_template = PROMOT_DICT[dataset]

    model.eval()

    datas = [data.strip() for data in open(input_file)]
    for iid, data in enumerate(datas):
        obj = json.loads(data)
        prompt = prompt_template.format_map(obj)

        batch = tokenizer(prompt, return_tensors="pt")

        batch = {k: v.to("cuda") for k, v in batch.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
        # e2e_inference_time = (time.perf_counter()-start)*1000
        # print(f"the inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(iid)
        print(output_text.replace(prompt, ''))
        print(obj['description'])
        print('*' * 35)


if __name__ == "__main__":
    fire.Fire(main)
