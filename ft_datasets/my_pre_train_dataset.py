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

    def get_input_files(self, dataset_config):
        input_files = []
        for input_file in os.listdir(dataset_config.root + '/' + dataset_config.sub_dir_prefix):
            if dataset_config.input_file and input_file != dataset_config.input_file:
                continue
            input_files.append(f'{dataset_config.root}/{dataset_config.sub_dir_prefix}/{input_file}')
        print(f'load input from {len(input_files)} files', flush=True)
        return sorted(input_files)

    def get_input_datas(self, dataset_config, split):
        input_ids = []
        self.input_files = self.get_input_files(dataset_config)

        if len(self.input_files) < 10:
            print(f'load {split} from {json.dumps(self.input_files, indent=2)}')
        else:
            print(f'load {split} from {len(self.input_files)} input files')

        for input_file in self.input_files:
            datas = pickle.load(open(input_file, 'rb'))
            if split == 'train':
                datas = datas[500:]
            else:
                datas = datas[0:500]
            input_ids.extend(datas)
            # print(f'load total {len(input_ids)} after load {input_file}', flush=True)
        print(f'load [{split}], num articles = {len(input_ids)}', flush=True)
        self.input_token_ids = np.concatenate(input_ids)
        print(f'load [{split}], {self.input_token_ids.shape[0]} tokens, {round(self.input_token_ids.shape[0] / 1.e9, 3)}B token')

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


def pre_tokenize(input_file, output_file):
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')

    datas = []
    for data in open(input_file):
        obj = json.loads(data)
        text = ''
        if 'title' in obj:
            text += obj['title'] + ' '
        if 'content' in obj:
            text += obj['content'] + ' '
        datas.append(text)
    tokenized_datas = []
    for data in tqdm.tqdm(datas):
        tokenized_datas.append(np.asarray(tokenizer.encode(data) + [tokenizer.eos_token_id], np.int64))
    pickle.dump(tokenized_datas, open(output_file, 'wb'))


if __name__ == '__main__':
    pool = multiprocessing.Pool(32)
    work_dir = '/home/paperspace/xingguang/datasets/pre-training-ariticle'
    for input_file in os.listdir(f'{work_dir}/text/'):
        pool.apply_async(
            func=pre_tokenize,
            args=(
                f'{work_dir}/text/{input_file}',
                f'{work_dir}/tokenized.13b/{input_file.replace(".txt", ".bin")}',
            )
        )
    pool.close()
    pool.join()
