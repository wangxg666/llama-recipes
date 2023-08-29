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
        # self.raw_datas = self.raw_datas[0:1000]
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.padding = padding

    def __len__(self):
        return len(self.raw_datas)

    def __getitem__(self, item):
        keys = [
            'Name: ',
            'Cuisine: ',
            'Price range: ',
            'Address: ',
            'Rating: ',
            'Telephone: ',
            'Website: ',
            'Hours: ',
            'Categories: ',
            'Popular dishes: ',
        ]
        text = self.raw_datas[item]
        poses = [text.find(key) for key in keys]

        key_pos_list = [(key, pos) for key, pos in zip(keys, poses) if pos != -1]
        key_pos_list.append(('', len(text)))

        example, labels, example_mask = [], [], []
        for i in range(len(key_pos_list)-1):
            key, pos = key_pos_list[i]
            end = key_pos_list[i+1][1]
            val = text[pos: end].replace(key, '')
            if not val:
                continue

            # print(f'[{item}]: key = {key}, val = {val}')

            key_ids = self.tokenizer.encode(key)
            val_ids = self.tokenizer.encode(val)

            # 跳过 bos token id
            if example:
                key_ids = key_ids[1:]
            val_ids = val_ids[1:]

            example.extend(key_ids)
            labels.extend([-100 for _ in key_ids])  # no loss for key
            example_mask.extend([1. for _ in key_ids])

            example.extend(val_ids)
            labels.extend(val_ids)
            example_mask.extend([1. for _ in val_ids])

        example.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        example_mask.append(1.)

        if self.padding and self.max_words > 0:
            example = (example + [0 for _ in range(self.max_words)])[0:self.max_words]
            labels = (labels + [0 for _ in range(self.max_words)])[0:self.max_words]
            example_mask = (example_mask + [0. for _ in range(self.max_words)])[0:self.max_words]

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


def get_my_pre_train_dataset(dataset_config, tokenizer, split):
    dataset = _MyPreTrainDataset(dataset_config, tokenizer, split, max_words=-1, padding=False)
    dataset = ConcatDataset(dataset, chunk_size=1024)
    return dataset


def get_my_pre_train_pad_dataset(dataset_config, tokenizer, split):
    dataset = _MyPreTrainDataset(dataset_config, tokenizer, split, max_words=1024, padding=True)
    return dataset


if __name__ == '__main__':
    from configs.datasets import my_pre_train_dataset

    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    dataset = _MyPreTrainDataset(my_pre_train_dataset, tokenizer, 'valid', max_words=-1, padding=False)
    for i in range(10):
        for k, v in dataset[i].items():
            print(k)
            print(len(v), v)
