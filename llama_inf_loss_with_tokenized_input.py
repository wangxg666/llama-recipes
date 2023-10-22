# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import collections

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import pandas as pd
import torch
import os
import sys
import time
import json
import pickle
import numpy as np
from typing import List

import tqdm
from transformers import LlamaTokenizer
from inference.model_utils import load_model, load_peft_model, load_llama_from_config


def get_input_datas(root, sub_dir_prefix):
    input_files = []
    for input_file in os.listdir(root + '/' + sub_dir_prefix):
        input_files.append(f'{root}/{sub_dir_prefix}/{input_file}')
    print(f'load input from {len(input_files)} files', flush=True)

    token_ids_list = []
    for input_file in input_files:
        datas = pickle.load(open(input_file, 'rb'))[0:100]
        token_ids_list.extend(datas)
        # print(f'load total {len(input_ids)} after load {input_file}', flush=True)
    print(f'load, num articles = {len(token_ids_list)}', flush=True)
    all_tokens_ids = np.concatenate(token_ids_list)
    print(f'load {all_tokens_ids.shape[0]} tokens, {round(all_tokens_ids.shape[0] / 1.e9, 3)}B token')
    pickle.dump(token_ids_list, open(f'{root}/{sub_dir_prefix}.valid.bin', 'wb'))
    return token_ids_list


def main(
        model_name: str = '',
        peft_model: str = None,
        quantization: bool = False,
        input_tokenized_file: str = '',
        tokenized_data_root: str = '',
        tokenized_data_sub_dir_prefix: str = '',
        seed: int = 42,  # seed value for reproducibility
):
    if input_tokenized_file:
       token_ids_list = pickle.load(open(input_tokenized_file, 'rb'))
    elif tokenized_data_root and tokenized_data_sub_dir_prefix:
        token_ids_list = get_input_datas(tokenized_data_root, tokenized_data_sub_dir_prefix)
    else:
        print(f'input file is not specified, please take a look.')
        return

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
    # model.to(torch.bfloat16)
    model.half()

    output = collections.defaultdict(list)
    for token_ids in tqdm.tqdm(token_ids_list):
        input = torch.from_numpy(np.asarray([token_ids[0:2048]]))
        batch = {
            'input_ids': input,
            'labels': input,
            'attention_mask': torch.ones_like(input, dtype=torch.float32),
        }
        batch = {k: v.to("cuda") for k, v in batch.items()}
        text = tokenizer.decode(token_ids=token_ids)
        doc = text.split('\n')[0].replace('<s>', '').strip()
        output['doc'].append(doc)
        with torch.no_grad():
            loss = model(**batch).loss
        output['loss'].append(loss.detach().cpu().numpy()[0])
        pd.DataFrame(output).to_csv(f'./loss.{model_name.split("/")[-1]}.csv', index=False)

if __name__ == "__main__":
    fire.Fire(main)
