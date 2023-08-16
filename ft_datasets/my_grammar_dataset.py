import copy
import json
import random

import torch

from torch.utils.data import Dataset


PROMOT_DICT = {
    "SINGLE": """Below is an instruction that describes a task. 
The following **text** might have some grammatical errors.
Please  read it and give your audit result whether it has grammatical errors
### text: {source_sent}
### response: """,

    "SEQ2SEQ": """Below is an instruction that describes a task. 
The following **text** have some grammatical errors.
Please fix these errors and output the new text.
### text: {source_sent}
### new text: """,
}


class MyGrammarDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=256):
        if partition == 'train':
            self.raw_data = [json.loads(data) for data in open(dataset_config.train_data_path)]
            random.shuffle(self.raw_data)
        else:
            self.raw_data = [json.loads(data) for data in open(dataset_config.valid_data_path)]

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        item = self.raw_data[index]
        prompt = MyGrammarDataset.prompting(item)
        example = prompt + item['label']

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
        labels[~label_mask] = -100
        example_mask = example_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }

    @staticmethod
    def prompting(item):
        PROMOT = PROMOT_DICT[item['type']]
        prompt = PROMOT.format_map(item)
        return prompt


if __name__ == '__main__':
    from configs.datasets import my_grammar_dataset

    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    dataset = MyGrammarDataset(my_grammar_dataset, tokenizer, partition='val')
    for k, v in dataset[0].items():
        print(k)
        print(v)
