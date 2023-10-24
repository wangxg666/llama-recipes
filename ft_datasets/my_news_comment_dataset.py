# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

PROMPT_DICT = {
    'dialog': (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a comment that appropriately reflect the news's `title` and `content`\n\n"
        "### Title:\n{title}\n\n### Content:\n{content}\n\n### Comment:\n"
    )
}


class NewsCommentDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann[200:]
        else:
            self.ann = self.ann[:200]

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        prompt = PROMPT_DICT["dialog"].format_map(ann)

        owner = ann['dialog'][0]['user']
        other_user2idx = {}
        for cid, user, comment in ann['dialog']:
            if user == owner:
                continue
            if user not in other_user2idx:
                other_user2idx[user] = len(other_user2idx) + 1

        labels = []
        example = []

        for cid, user, comment in ann['dialog']:
            if user == owner:
                prompt += '\nYou: '
                example = self.tokenizer.encode(prompt)
                ex_size = len(example) - len(labels)
                labels.extend([IGNORE_INDEX for _ in range(ex_size)])
                prompt += comment
                example = self.tokenizer.encode(prompt)
                labels.extend(example[len(labels):])
            else:
                prompt += f'\nUser {other_user2idx[user]}: {comment}'
                example = self.tokenizer.encode(prompt)
                ex_size = len(example) - len(labels)
                labels.extend([IGNORE_INDEX for _ in range(ex_size)])

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
        example_mask = example.ge(0)
        example_mask = example_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
