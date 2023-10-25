# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import copy
import json
import multiprocessing
# For dataset details visit: https://huggingface.co/datasets/samsum

import os
import pickle
import random
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
from ft_datasets.utils import ConcatDatasetNumpy


def get_input_files(dataset_config):
    input_files = []
    for input_file in os.listdir(dataset_config.root + '/' + dataset_config.sub_dir_prefix):
        if dataset_config.input_file and input_file != dataset_config.input_file:
            continue
        input_files.append(f'{dataset_config.root}/{dataset_config.sub_dir_prefix}/{input_file}')
    print(f'load input from {len(input_files)} files', flush=True)
    return sorted(input_files)


class _MyPreTrainDataset(Dataset):

    def __init__(self, dataset_config, tokenizer, split, chunk_size=3998):
        """
        Pre Training 的dataset不需要 shuffle
        :param padding: 是否padding，如果使用 concate dataset，padding 是 False
        :param max_tokens: 最大 token 长度，如果使用 concate dataset，max token 失效，配合padding使用
        """
        self.input_files = []
        self.input_token_ids = None

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

        self.get_input_datas(dataset_config, split)

    def get_input_datas(self, dataset_config, split):
        input_ids = []
        self.input_files = get_input_files(dataset_config)

        if len(self.input_files) < 10:
            print(f'load {split} from {json.dumps(self.input_files, indent=2)}')
        else:
            print(f'load {split} from {len(self.input_files)} input files')

        for input_file in self.input_files:
            input_ids.extend(pickle.load(open(input_file, 'rb')))
        self.input_token_ids = np.concatenate(input_ids)

        print(f'load [{split}], num articles = {len(input_ids)}, '
              f'{self.input_token_ids.shape[0]} tokens, '
              f'{round(self.input_token_ids.shape[0] / (2.**30), 3)}B token', flush=True)

    def __len__(self):
        return self.input_token_ids.shape[0] // self.chunk_size

    def __getitem__(self, idx):
        bos = idx * self.chunk_size
        input_ids = self.input_token_ids[bos: bos + self.chunk_size]
        labels = copy.deepcopy(input_ids)
        data = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': np.ones_like(input_ids, dtype=np.float32),
        }
        return data


def get_my_pre_train_dataset(dataset_config, tokenizer, split):
    dataset = _MyPreTrainDataset(dataset_config, tokenizer, split, chunk_size=4096)
    return dataset


def get_my_pre_train_pad_dataset(dataset_config, tokenizer, split):
    dataset = _MyPreTrainDataset(dataset_config, tokenizer, split, chunk_size=512)
    return dataset


