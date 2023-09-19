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
from ft_datasets.my_allin_one_dataset import *
from transformers import LlamaTokenizer
from inference.model_utils import load_model, load_peft_model, load_llama_from_config


def is_default_ans(ans):
    if 'not be ans' in ans:
        return True
    for mark in [
        'the knowledge provided does not',
        'Therefore, based on this information',
        'information provided is limited',
        'not mentioned in the given',
        'not specifically mentioned in the given',
        'not mentioned in the provided'
        'the provided knowledge does not',
        'is not provided in the given knowledge',
        'is not directly provided in the given knowledge',
        'is not specified in the provided',
        'not be determined from the given',
        'based on the given knowledge'
    ]:
        if mark in ans:
            return True
    return False

def main(
    model_name,
    port: int=1201,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 100, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=False, # Whether or not to use sampling ; use greedy decoding otherwise.
    num_beams: int=4,
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    **kwargs
):

    print(do_sample, num_beams)
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

    model.eval()
    model.half()

    async def encode(request):
        obj = await request.json()
        input = MyAllInOneDataset.prompting(obj)
        batch = tokenizer(input, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
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
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(output_text)
        pred = output_text.replace(input, '').strip()

        if is_default_ans(pred):
            # pred = json.dumps({'answer': 'Sorry, the query can not be answered'})
            pred = 'Sorry, the query can not be answered'

        return web.json_response(data={'pred': pred})

    from aiohttp import web
    app = web.Application()
    app.add_routes([web.post('/do', encode)])
    web.run_app(app, port=port, access_log=None)


if __name__ == "__main__":
    fire.Fire(main)
