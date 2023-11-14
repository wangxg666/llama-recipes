import copy
import itertools
import json
import os
import random

import torch

from torch.utils.data import Dataset

PROMPT_DICT = {
    "PLATYPUS": "{instruction}",

    "NORM_PROMPT": "{prompt}",

    "FAQ_ANSWER_EXTRACT_NO_JSON_V5_FAQ": (
        "You are an excellent linguist, and I need your help to complete the following task.\n"
        "Here is a user query and some retrieved information, the retrieved information might not be the proper for the query.\n"
        "Please read the user query and the retrieved information carefully, and then try using this information to answer the query.\n"
        "The retrieved information may have duplicate content, please summarize the retrieved information with deduplication before you taking use of them.\n"
        "Your response should be accurate, grammatically fluent.\n"
        "But your answer should answer the query directly, and do not output any additional or furthermore guidelines in your response.\n"
        "Don't output any additional information that is not included in the retrieved information.\n"
        "If you think the information is not relevant to the query, please give a default response \"Sorry, the query can not be answered.\"\n"
        "If there is enumeration of entities in your response, please convert it to list of entities with markdown syntax.\n"
        "Do not provide any additional response that is not relevant with the user query.\n"
        "Please answer the query directly, do not output your explanation.\n"
        "### Query: {query}\n"
        "### Retrieved Information: {prompt_answer}\n"
        "### response:\n"
    ),

    "FAQ_ANSWER_EXTRACT_NO_JSON_V5_AIGC": (
        "You are an excellent linguist, and I need your help to complete the following task.\n"
        "Here is a user query and some retrieved ariticles, the retrieved ariticles might not be the proper for the query.\n"
        "Please read the user query and the retrieved ariticles carefully, and then try using this ariticles to answer the query.\n"
        "The retrieved ariticles may have duplicate content, please summarize the retrieved ariticles with deduplication before you taking use of them.\n"
        "Your response should be accurate, grammatically fluent.\n"
        "But your answer should answer the query directly, and do not output any additional or furthermore guidelines in your response.\n"
        "Don't output any additional ariticles that is not included in the retrieved ariticles.\n"
        "If you think the ariticles is not relevant to the query, please give a default response \"Sorry, the query can not be answered.\"\n"
        "If there is enumeration of entities in your response, please convert it to list of entities with markdown syntax.\n"
        "Do not provide any additional response that is not relevant with the user query.\n"
        "Please answer the query directly, do not output your explanation.\n"
        "### Query: {query}\n"
        "### Retrieved Information: {prompt_answer}\n"
        "### response:\n"
    ),

    "DOC_ID_QUERY": (
        "You are an excellent knowledger, and I need you to help to complate the following task.\n"
        "Given you an `8 character` string as a News ID, I need you to find out it's title from you memory.\n"
        "- The news IDs is a 8 characters string, which is start with `0`\n"
        "### Doc ID: \n{docid}\n"
    ),

    "DOC_ID_GENERATION": (
        "You are an excellent knowledger, and I need you to help to complate the following task.\n"
        "Here is a user query, I wnat you to help to find some news that you have read, and return their IDs.\n"
        "The news IDs is a 8 characters string, start with `0`\n"
        "### Query: \n{query}\n"
        "### Response: \n"
    ),

    "NEWS_COMMENT_WITH_INSTRUCTION": (
        "{prompt}\n"
        "{instruction}\n"
        "Your Comment:\n"
    ),

    "NEWS_COMMENT_WITH_INSTRUCTION_V2": "{prompt}"
}


class MyAllInOneDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=2048, debug=False):
        input_file = f'{dataset_config.root}/{dataset_config.dataset_dir}/{partition}.txt'
        self.raw_data = [json.loads(data) for data in open(input_file)]
        print(f'load {len(self.raw_data)} {partition} datas')

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.debug = debug

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        item = self.raw_data[index]
        prompt = MyAllInOneDataset.prompting(item)
        example = prompt + ' ' + str(item['label'])

        prompt = self.tokenizer.encode(prompt)
        example = self.tokenizer.encode(example) + [self.tokenizer.eos_token_id]

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

    # sent = "On 7th March, Dr Fontanelli delivered a presentation for the first regional launch of an online course on the right to property prepared in the framework of the HELP project of the Council of Europe (European Programme for Human Rights Education for Legal Professionals)"
    # print(tokenizer.tokenize(sent))
    #
    # sent = "On 7th March, Dr Fontanelli delivered a presentation for the first regional launch of an online course on the right to propert prepared in the framework of the HELP project of the Council of Europe (European Programme for Human Rights Education for Legal Professionals)"
    # print(tokenizer.tokenize(sent))

