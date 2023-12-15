import os

import torch
import tqdm
from torch.utils.data import Dataset

import copy
import json
from typing import Dict

from ft_datasets.agent_sft_common import PERSONA_PROMPT_DICT, agent_tokenize


ANSWER_TYPE_PROMPT = {
    'casual_generation': (
        '{persona}\n'
        'Given the conversion history, your task is to generate the next response.\n'
        'Generate an appropriate response; this response can be in one of the following two styles:\n'
        '1. Interrogative, If you think that the user\'s needs have not been met, please ask for the necessary information to provide a more accurate understanding.\n'
        '2. Direct response: If you believe the conversation is concluded, politely say goodbye; or other direct response based on the conversion history.\n'
        'Here is the conversion history:\n{history}\n'
        'and the user lastest utterence: \n{user_utterence}\n'
        'and here is the slots you\'d better to ask rhetorically when outputting your response: \n{slots}\n'
        'Please give your response:\n'
    ),
    'casual_generation_no_slots': (
        '{persona}\n'
        'Given the conversion history, your task is to generate the next response.\n'
        'Generate an appropriate response; this response can be in one of the following two styles:\n'
        '1. Interrogative, If you think that the user\'s needs have not been met, please ask for the necessary information to provide a more accurate understanding.\n'
        '2. Direct response: If you believe the conversation is concluded, politely say goodbye; or other direct response based on the conversion history.\n'
        'Here is the conversion history:\n{history}\n'
        'and the user lastest utterence: \n{user_utterence}\n'
        'Please generate a proper response based on the context.\n'
        'Please give your response:\n'
    ),
    'rag_generation': (
        '{persona}\n'
        'Given the conversion history, user utterance, and query result from database, '
        'please generate a appropriate answer based on the give conversion status.\n'
        'Here is the conversion history:\n{history}\n'
        'query result:\n{search_results}\n'
        'the user lastest utterence: \n{user_utterence}\n'
        'Please give your generation:\n'
    ),
    'api_generation': (
        '{persona}\n'
        'Given the conversion history, your task is the generate the formatted API for searching.\n'
        'Here is the conversion history:\n{history}\n'
        'the user lastest utterence: \n{user_utterence}\n'
        'and here is the slots you\'d better to refer when generating the formatted API: \n{slots}\n'
        'Please give your API:\n'
    )
}
ANSWER_TYPE_PROMPT['default'] = ANSWER_TYPE_PROMPT['casual_generation']


class AgentSFTDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=2048, do_padding=True, debug=False):
        type = 'train' if partition == 'train' else 'dev'
        input_files = [
            f'{dataset_config.root}/{dataset_config.dataset_dir}/{type}.{task}.json'
            for task in ['api', 'casual', 'rag']
        ]
        print(json.dumps(input_files, indent=2), flush=True)
        self.datas = []
        for input_file in input_files:
            datas = [json.loads(data) for data in open(input_file) if data.strip()]
            if partition == 'train':
                self.datas.extend(datas)
            else:
                n = 100 if 'casual.json' in input_file else 200
                self.datas.extend(datas[0:n])

        self.max_words = max_words
        self.do_padding = do_padding
        self.tokenizer = tokenizer
        self.debug = debug

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        item: Dict = self.datas[index]
        return AgentSFTDataset.tokenize(item, self.tokenizer, self.max_words, self.do_padding)

    @staticmethod
    def tokenize(item: Dict, tokenizer, max_words, do_padding):
        type = item['type']
        history = [x.replace('USER', 'user').replace('SYSTEM', 'you') for x in item['history']]

        if type == 'api_generation':
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            slots = {k: list(v) for k, v in item['label'].items()}

            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(history[0:-1], indent=2),
                user_utterence=history[-1].replace('user: ', ''),
                slots=json.dumps(slots)
            )
            label = item['label_type'] + '\n' + json.dumps(item['label'], separators=(',', ':'))

        elif type == 'casual_generation':
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            slots = {k: list(v) for k, v in item.get('asked_slots', {}).items()}

            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(history[0:-1], indent=2),
                user_utterence=history[-1].replace('user: ', ''),
                slots=json.dumps(slots)
            )
            label = item['label_type'] + '\n' + item['label']

        elif type == 'casual_generation_no_slots':
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(history[0:-1], indent=2),
                user_utterence=history[-1].replace('user: ', '')
            )
            label = item['label_type'] + '\n' + item['label']

        else:
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(history[0:-1], indent=2),
                user_utterence=history[-1].replace('user: ', ''),
                search_results=json.dumps(item['search_results'], separators=(',', ':'))
            )
            label = item['label']
        return agent_tokenize(tokenizer, prompt, label, max_words, do_padding)


if __name__ == '__main__':
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')

    items = json.load(open('./datas/agent_sft_gen_data.json'))
    for item in items:
        output = AgentSFTDataset.tokenize(item, tokenizer, 1024, True)
