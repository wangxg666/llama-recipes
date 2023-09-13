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
Kindly review it and provide your assessment of whether there are any grammatical errors. 
Your response should be "Good." if there are no grammatical errors and "Poor." if there are.
### text: {source_sent}
### response:""",

    "GRAMMAR_SEQ2SEQ": """Below is an instruction that describes a task. 
The following **text** might have some grammatical errors.
Please correct these errors and provide the revised text if it contains any mistakes, 
or simply provide the text as is if it is correct.
### text: {source_sent}
### response:""",

    "CLICKBAIT_SINGLE": """Below is an instruction that describes a task. 
The following **text** might be writen in an informal style, 
characterized by clickbait tendencies, excessive elaboration, hyperbole, a confrontational tone, curiosity, fear, or even anger. 
Please review it and provide your assessment of whether it is composed informally.
Your response should be "Good." if it is writen formally and "Poor." if it is not.
### text: {source_sent}
### response:""",

    "HALLUCINATION": """Below is an instruction that describes a task. 
The following text B is rewritten from text A, but text B might be inconsistent with text A in content.
Please help to check does text B is rewritten properly with text A.
### {source_sent}
### response:""",

    "FAQ_ANSWER_EXTRACT": """
You are an excellent linguist, and I need your help to complete the following task.
Give you a user query and some background knowledge, 
the knowledge is not the directly answer for the query, but it might be used to answer the query.
Please read the query and knowledge carefully, 
and then try extracting and summarizing the useful information from these knowledge to answer the query.
If you think the knowledge is not relevant with the query, please give a default answer "Sorry, the query can not be answered.".
Your answer should be in json format like {{"answer": xxx}}
### query: {query}
### knowledge: {knowledge}
### response:""",

    "PLATYPUS": """"
{instruction}
    """,
    
    "QUERY_TITLE_RELEVANCE": """
Here is a query and news title, please help to double check whether the query and the news title is highly correlated.
You need tell me the reason why they are correlated or not correlated. 
Please answer in JSON format {{"reason": xxx, "decision": yyy}}
### query: {query}
### title: {title}
### response:"""
}


# PROMPT_DICT = {
#     "GRAMMAR_SINGLE": """
# The following **text** might have some grammatical errors, give your judgement.
# ### text: {source_sent}
# ### response:""",
#
#     "GRAMMAR_SEQ2SEQ": """Below is an instruction that describes a task.
# The following **text** have some grammatical errors, please correct these errors.
# ### text: {source_sent}
# ### response:""",
#
#     "CLICKBAIT_SINGLE": """Below is an instruction that describes a task.
# The following **text** might be writen informally, give your judgement.
# ### text: {source_sent}
# ### response:"""
# }


class MyAllInOneDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=2048, debug=False):
        input_files = [
            f'{dataset_config.root}/{sub_dir}/{partition}.txt'
            for sub_dir in os.listdir(dataset_config.root)
            if os.path.exists(f'{dataset_config.root}/{sub_dir}/{partition}.txt')
            and sub_dir.startswith(dataset_config.sub_dir_prefix)
        ]
        print(json.dumps(input_files, indent=4))
        self.raw_data = []
        for input_file in input_files:
            for data in open(input_file):
                try:
                    obj = json.loads(data)
                    self.raw_data.append(obj)
                except:
                    pass
        if debug:
            from collections import defaultdict
            type2datas = defaultdict(list)
            for data in self.raw_data:
                type = data['type']
                type2datas[type].append(data)
            for type, datas in type2datas.items():
                self.raw_data.extend(datas)

        # self.raw_data = list(itertools.chain(*self.raw_data))
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
        example = prompt + ' ' + str(item['label'])

        prompt = self.tokenizer.encode(prompt)
        example = self.tokenizer.encode(example) + [self.tokenizer.eos_token_id]

        if self.debug:
            n = len(prompt) - 3
            print(item['type'])
            print(len(prompt), prompt[n:])
            print(len(example), example[n:])
            print(MyAllInOneDataset.prompting(item) + ' ' + item['label'])
            print('**' * 20)
            print('\n')

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

