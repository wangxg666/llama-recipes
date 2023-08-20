import copy
import itertools
import json
import os
import random

import torch

from torch.utils.data import Dataset

PROMPT_DICT = {
    "GRAMMAR_SINGLE": """Below is an instruction that describes a task. 
The following **text** might have some grammatical errors.
Please read it and give your audit result whether it has grammatical errors.
Your answer should be one of the following
- No, no grammatical error
- One, has only one grammatical error
- Multi, has more than one grammatical error
### text: {source_sent}
### response:""",

    "GRAMMAR_SEQ2SEQ": """Below is an instruction that describes a task. 
The following **text** have some grammatical errors.
Please fix these errors and output the new text.
### text: {source_sent}
### response:""",

    "CLICKBAIT_SINGLE": """Below is an instruction that describes a task. 
The following **text** might be writen informally, which is clickbait, over-describing, exaggerating, intimidating, curious, frightened or even angry.
Please read it and give your audit result whether it's writen informally. 
### text: {source_sent}
### response:"""
}


class MyAllInOneDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=256, debug=False):
        input_files = [
            f'{dataset_config.root}/{sub_dir}/{partition}.txt'
            for sub_dir in os.listdir(dataset_config.root)
            if os.path.exists(f'{dataset_config.root}/{sub_dir}/{partition}.txt')
        ]
        self.raw_data = [[json.loads(data) for data in open(input_file)] for input_file in input_files]
        if debug:
            from collections import defaultdict
            type2data = defaultdict(list)
            for datas in self.raw_data:
                for data in datas:
                    type = data['type']
                    if len(type2data[type]) < 10:
                        type2data[type].append(data)
            self.raw_data = list(type2data.values())

        self.raw_data = list(itertools.chain(*self.raw_data))

        print(f'load {len(self.raw_data)} {partition} datas')
        if partition == 'train':
            random.shuffle(self.raw_data)

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.debug = debug

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        item = self.raw_data[index]
        prompt = MyAllInOneDataset.prompting(item)
        example = prompt + ' ' + item['label']

        prompt = self.tokenizer.encode(prompt)
        example = self.tokenizer.encode(example) + [self.tokenizer.eos_token_id]

        if self.debug:
            n = len(prompt) - 3
            print(item['type'])
            print(len(prompt), prompt[n:])
            print(MyAllInOneDataset.prompting(item))
            print(len(example), example[n:])
            print(MyAllInOneDataset.prompting(item) + ' ' + item['label'])
            print('**' * 20)

        prompt = torch.tensor(prompt, dtype=torch.int64)
        example = torch.tensor(example, dtype=torch.int64)

        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = -100
        example_mask = example_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

    @staticmethod
    def prompting(item):
        return PROMPT_DICT[item['type']].format_map(item)


if __name__ == '__main__':
    from configs.datasets import my_allin_one_dataset

    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    dataset = MyAllInOneDataset(my_allin_one_dataset, tokenizer, partition='valid', debug=True)
    print(len(dataset))
    for i in range(len(dataset)):
        dataset[i]
