# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import copy
# For dataset details visit: https://huggingface.co/datasets/samsum

import os
import torch
from torch.utils.data import Dataset
from ft_datasets.utils import ConcatDataset


class _MyPreTrainDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split, debug=False):
        input_files = [
            f'{dataset_config.root}/{sub_dir}/{split}.txt'
            for sub_dir in os.listdir(dataset_config.root)
            if os.path.exists(f'{dataset_config.root}/{sub_dir}/{split}.txt')
               and sub_dir == 'pre_train_yelp_ny'
        ]

        self.raw_datas = []
        for input_file in input_files:
            self.raw_datas.extend([x.strip() for x in open(input_file)])

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.raw_datas)

    def __getitem__(self, item):
        example = self.tokenizer.encode(self.raw_datas[item]) + [self.tokenizer.eos_token_id]
        example = example

        labels = copy.deepcopy(example)
        example_mask = [1.0 for _ in range(len(labels))]

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


def get_my_pre_train_dataset(dataset_config, tokenizer, split):
    dataset = _MyPreTrainDataset(dataset_config, tokenizer, split)
    dataset = ConcatDataset(dataset, chunk_size=1024)
    return dataset


if __name__ == '__main__':
    from configs.datasets import my_pre_train_dataset

    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    dataset = get_my_pre_train_dataset(my_pre_train_dataset, tokenizer, 'train')
    for i in range(10):
        for k, v in dataset[i].items():
            print(k)
            print(len(v), v)
