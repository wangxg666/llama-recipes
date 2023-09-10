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
from inference.model_utils import load_model, load_peft_model, load_llama_from_config


class PredictionWriter:
    def __init__(self, output_file):
        self.output_file = output_file
        self.type2sout = {}

    def write(self, type, real, pred):
        if type not in self.type2sout:
            self.type2sout[type] = open(self.output_file + '.' + type, 'w')
        obj = {
            'type': type,
            'real': real,
            'pred': pred,
        }
        self.type2sout[type].write(f'{json.dumps(obj)}\n')
        self.type2sout[type].flush()

    def close(self):
        for type, sout in self.type2sout.items():
            sout.close()


def get_input_file_abs_path(input_file):
    if os.path.exists(f'/home/cpp/xingguang/datasets/{input_file}'):
        return f'/home/cpp/xingguang/datasets/{input_file}'
    if os.path.exists(f'/home/paperspace/datasets/{input_file}'):
        return f'/home/paperspace/datasets/{input_file}'
    if os.path.exists(f'/home/paperspace/xingguang/datasets/{input_file}'):
        return f'/home/paperspace/xingguang/datasets/{input_file}'
    if os.path.exists(f'/mnt/nlp/xingguang/llama/datasets/nb_training/{input_file}'):
        return f'/mnt/nlp/xingguang/llama/datasets/nb_training/{input_file}'
    print(f'{input_file} is not valid, exit(0)')
    exit(0)


def main(
    model_name,
    peft_model: str=None,
    dataset: str=None,
    quantization: bool=False,
    max_new_tokens = 100, #The maximum numbers of tokens to generate
    input_file: str=None,
    num_beams: int=10,
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

    print(model)

    if peft_model:
        model = load_peft_model(model, peft_model)
        print(model)

    from ft_datasets import (
        get_my_allin_one_dataset
    )
    DATASET_PREPROC = {
        "my_allin_one_dataset": get_my_allin_one_dataset,
    }
    input_file_abs_path = get_input_file_abs_path(input_file=input_file)

    model.eval()

    datas = [json.loads(data) for data in open(input_file_abs_path)]
    datas = sorted(datas, key=lambda x:x['label'])

    writer = PredictionWriter(input_file_abs_path + '.pred')

    for iid, obj in enumerate(datas):
        prompt = DATASET_PREPROC[dataset].prompting(obj)
        batch = tokenizer(prompt, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
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

        type = obj['type']
        pred = output_text.replace(prompt, '').strip()
        real = obj['label']

        writer.write(type, real, pred)

        print(iid)
        print(f'pred = {pred}')
        print(f'real = {real}')
        print('*' * 35)

    writer.close()


if __name__ == "__main__":
    fire.Fire(main)
