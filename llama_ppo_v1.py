# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pickle
import torch
from torch.utils.data import Dataset


class PPODataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.samples['rewards'] = [torch.tensor([(x.item()+10.) / 5.]) for x in self.samples['rewards']]
        self.samples['query_tensors'] = [x.cpu() for x in self.samples['query_tensors']]
        self.samples['response_tensors'] = [x.cpu() for x in self.samples['response_tensors']]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return {
            k: v[index] for k, v in self.samples.items()
        }


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])



data_input_file = '/home/paperspace/xingguang/datasets/ppo_test/data.bin'

samples = pickle.load(open(data_input_file, 'rb'))
datasets = PPODataset(samples)

batch = collator([datasets[i] for i in range(4)])

from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b")

for i in range(4):
    print(tokenizer.decode(batch['query_tensors'][i].tolist()))
    print(batch['response_tensors'][i].tolist())
    print(tokenizer.encode(tokenizer.decode(batch['response_tensors'][i].tolist()).strip())[1:])
    print(batch['rewards'][i].tolist())