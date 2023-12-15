import os

import torch
import tqdm
from torch.utils.data import Dataset

import copy
import json
from typing import Dict
from ft_datasets.agent_sft_common import PERSONA_PROMPT_DICT, agent_tokenize


ANSWER_TYPE_PROMPT = {
    'act_selection': (
        '{persona}\n'
        'Given the conversion history, Your task is to help determine whether the next response can be directly replied to or not.\n'
        '1. If you think it can be replied without searching, the action would be `chat`.\n'
        '2. Otherwise the action would be `search`.\n'
        'In addition, please also output the types of information that need pay attention to for the current action.\n'
        'Here is the conversion history:\n{history}\n'
        'the user lastest utterence: \n{user_utterence}\n'
        'The output should be in JSON format like {{"action": xxx, "slots": {{"service": [xxx, ...]}}}}\n'
        'Please give your decision:\n'
    )
}
ANSWER_TYPE_PROMPT['default'] = ANSWER_TYPE_PROMPT['act_selection']


class AgentActDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=2048, do_padding=True, debug=False):
        type = 'train' if partition == 'train' else 'dev'
        input_files = [
            f'{dataset_config.root}/{dataset_config.dataset_dir}/{type}.{task}.json'
            for task in ['act']
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
        return AgentActDataset.tokenize(item, self.tokenizer, self.max_words, self.do_padding)

    @staticmethod
    def tokenize(item: Dict, tokenizer, max_words, do_padding):

        type = item['type']
        history = [x.replace('USER', 'user').replace('SYSTEM', 'you') for x in item['history']]

        persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
        prompt = ANSWER_TYPE_PROMPT[type].format(
            persona=persona,
            history=json.dumps(history[0:-1], indent=2),
            user_utterence=history[-1].replace('user: ', '')
        )
        label = json.dumps(item['label'])

        return agent_tokenize(tokenizer, prompt, label, max_words, do_padding)


if __name__ == '__main__':
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')

    items = json.load(open('./datas/agent_sft_act_data.json'))
    for item in items:
        output = AgentActDataset.tokenize(item, tokenizer, 1024, True)

