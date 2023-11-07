# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import gzip
import pickle
import numpy as np
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
from typing import List

IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

YOU = 'You'
PROMPT = (
    "<<SYS>>\n"
    "You are a senior news commentator, that is good at understanding the key points of news and making accurate comments.\n"
    "Here is a news data with `title`, `content` and some comment history.\n"
    "Please make a new comment or reply to the specified user based on the command.\n"
    "### title:\n{title}\n"
    "### content:\n{content}\n"
    "<</SYS>>"
    # "You:\n{comment} </s>\n"
    # "[INST] User A reply to You: {other_coments} [/INST]\n" # 如果有多轮，重复多行
    # "[INST] User A reply to User B: {other_coments} [/INST]\n" # 如果有多轮，重复多行
    # "Your reply to User A: {reply} </s>"
)


def get_input_files(dataset_config):
    input_files = []
    dataset_dir = f'{dataset_config.root}/{dataset_config.dataset_dir}'
    for input_file in os.listdir(dataset_dir):
        if dataset_config.input_file and input_file != dataset_config.input_file:
            continue
        input_files.append(f'{dataset_dir}/{input_file}')
    print(f'load input from {len(input_files)} files', flush=True)
    return sorted(input_files)


class NewsCommentDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=2048):
        self.items = []
        self.max_words = max_words
        self.tokenizer = tokenizer

        for input_file in get_input_files(dataset_config):
            self.items.extend(pickle.load(open(input_file, 'rb')))
        self.items = [x for x in self.items if len(x['input_ids']) <= max_words]
        print(f'load [{partition}], from files = {get_input_files(dataset_config)}, data size = {len(self.items)}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        input_ids = item['input_ids'].astype(np.int64)
        labels_pos = item['labels_pos']
        labels = np.zeros_like(input_ids, dtype=np.int64) + IGNORE_INDEX
        labels[labels_pos] = input_ids[labels_pos]

        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        padding = self.max_words - input_ids.shape[0]
        if padding > 0:
            input_ids = torch.cat((input_ids, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat((labels, torch.zeros(padding, dtype=torch.int64) + IGNORE_INDEX))
        input_mask = input_ids.ge(0)
        input_ids[~input_mask] = 0

        return {
            "input_ids": input_ids[: self.max_words],
            "labels": labels[: self.max_words],
            "attention_mask": input_mask[: self.max_words],
        }


if __name__ == '__main__':
    input_file = '/home/paperspace/xingguang/datasets/comment.v01/train.7b.bin'
    items = pickle.load(open(input_file, 'rb'))

    item = items[0]
    input_ids = item['input_ids'].astype(np.int64)
    labels_pos = item['labels_pos']
    labels = np.zeros_like(input_ids, dtype=np.int64) + IGNORE_INDEX
    labels[labels_pos] = input_ids[labels_pos]

    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    labels = torch.tensor(labels, dtype=torch.int64)

    padding = 4096 - input_ids.shape[0]
    if padding > 0:
        input_ids = torch.cat((input_ids, torch.zeros(padding, dtype=torch.int64) - 1))
        labels = torch.cat((labels, torch.zeros(padding, dtype=torch.int64) + IGNORE_INDEX))

    input_mask = input_ids.ge(0)
    input_ids[~input_mask] = 0

    print(json.dumps({
        "input_ids": input_ids.numpy().tolist(),
        "labels": labels.numpy().tolist(),
        "attention_mask": input_mask.numpy().tolist(),
    }))