import copy
import json
import torch

from torch.utils.data import Dataset


PROMOT = """
There are tow articles blew, and Article B is rewrote from Article A.
You need to judge whether the main content of these two articles is the same.
Article A: 
{article_a}
Article B:
{article_b}
Your judgement:
"""

class HallucinationDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=1024):
        self.raw_data = json.load(open(f'{dataset_config.data_path}'))

        if partition == "train":
            self.raw_data = self.raw_data[0:16]
        else:
            self.raw_data = self.raw_data[16:]

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        item = self.raw_data[index]
        prompt = PROMOT.format_map(item)
        example = prompt + item["output"]

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


if __name__ == '__main__':
    from configs.datasets import hallucination_dataset

    dataset = HallucinationDataset(hallucination_dataset, None, None, None)
