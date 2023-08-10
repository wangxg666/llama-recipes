import copy
import json
import random

import torch

from torch.utils.data import Dataset


PROMOT_DICT = {
    "my_grammar_dataset": """
Here is a short text and I need your help to review whether the words are used correctly,
text:
    {text}
your review result
    {description}
""",

    "my_clickbait_dataset": """
Here is a news title and I need your help to review whether it's wrote with clickbait, exaggerate, curious, firghtened or even angry ways,
text:
    {text}
your review result
    {description}
"""
}

class MyCommonDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=256):
        if partition == 'train':
            self.raw_data = [json.loads(data) for data in open(dataset_config.train_data_path)]
        else:
            self.raw_data = [json.loads(data) for data in open(dataset_config.valid_data_path)]
        random.shuffle(self.raw_data)

        self.max_words = max_words
        self.tokenizer = tokenizer

        self.PROMPT = PROMOT_DICT[dataset_config.dataset]

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        item = self.raw_data[index]
        prompt = self.PROMPT.format_map(item)
        example = prompt

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
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
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
