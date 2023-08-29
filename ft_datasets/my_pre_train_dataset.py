# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import copy
# For dataset details visit: https://huggingface.co/datasets/samsum

import os
import random

import torch
from torch.utils.data import Dataset
from ft_datasets.utils import ConcatDataset


class _MyPreTrainDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split, max_words=-1, padding=False, debug=False):
        input_files = [
            f'{dataset_config.root}/{sub_dir}/{split}.txt'
            for sub_dir in os.listdir(dataset_config.root)
            if os.path.exists(f'{dataset_config.root}/{sub_dir}/{split}.txt')
               and dataset_config.target_sub_dir in sub_dir
        ]

        self.raw_datas = []
        for input_file in input_files:
            self.raw_datas.extend([x.strip() for x in open(input_file)])
        self.raw_datas = [x for x in self.raw_datas if len(x.split(' ')) >= 3]
        # self.raw_datas = self.raw_datas[0:1000]
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.padding = padding

    def __len__(self):
        return len(self.raw_datas)

    def __getitem__(self, item):
        example = self.tokenizer.encode(self.raw_datas[item]) + [self.tokenizer.eos_token_id]

        if self.padding and self.max_words > 0:
            example = (example + [0 for _ in range(self.max_words)])[0:self.max_words]
        labels = copy.deepcopy(example)
        example_mask = [1. if x > 0 else 0. for x in example]

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


def get_my_pre_train_dataset(dataset_config, tokenizer, split):
    dataset = _MyPreTrainDataset(dataset_config, tokenizer, split, max_words=-1, padding=False)
    dataset = ConcatDataset(dataset, chunk_size=2048)
    return dataset


def get_my_pre_train_pad_dataset(dataset_config, tokenizer, split):
    dataset = _MyPreTrainDataset(dataset_config, tokenizer, split, max_words=256, padding=True)
    return dataset


if __name__ == '__main__':
    from configs.datasets import my_pre_train_pad_dataset

    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    dataset = get_my_pre_train_pad_dataset(my_pre_train_pad_dataset, tokenizer, 'valid')
    for i in range(10):
        for k, v in dataset[i].items():
            print(k)
            print(len(v), v)
