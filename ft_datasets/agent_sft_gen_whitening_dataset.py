import collections
import os
import random

import torch
import tqdm
from torch.utils.data import Dataset

import copy
import json
from typing import Dict

from ft_datasets.agent_sft_common import PERSONA_PROMPT_DICT, agent_tokenize


from ft_datasets.agent_sft_gen_dataset import ANSWER_TYPE_PROMPT


class AgentSFTWhiteningDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=2048, do_padding=True, debug=False):
        type = 'train' if partition == 'train' else 'dev'
        input_files = [
            f'{dataset_config.root}/{dataset_config.dataset_dir}/{type}.{taslot_key}.json'
            for taslot_key in ['api', 'casual', 'rag']
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

        self.service2slots_base = collections.defaultdict(set)
        if partition == 'train':
            input_file = f'{dataset_config.root}/{dataset_config.dataset_dir}/{type}.api.json'
            datas = [json.loads(data) for data in open(input_file) if data.strip()]
            for data in datas:
                for key, slot_kv in data['label'].items():
                    self.service2slots_base[key].update(slot_kv.keys())

        self.max_words = max_words
        self.do_padding = do_padding
        self.tokenizer = tokenizer
        self.debug = debug

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        item: Dict = self.datas[index]
        prompt, label = AgentSFTWhiteningDataset.prompting(item, self.service2slots_base)
        return agent_tokenize(self.tokenizer, prompt, label, self.max_words, self.do_padding)

    @staticmethod
    def prompting(item: Dict, service2slots_base: Dict):
        type = item['type']
        history = [x.replace('USER', 'user').replace('SYSTEM', 'you') for x in item['history']]

        if type == 'api_generation':
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            service2slots = {k: list(v) for k, v in item['label'].items()}

            rnd = random.random()
            if service2slots_base:
                if rnd < 0.1:
                    # remove from each action
                    for service in service2slots.keys():
                        slot_keys = service2slots[service]
                        rm_slot_key = random.choice(slot_keys)
                        service2slots[service] = [slot_key for slot_key in slot_keys if slot_key != rm_slot_key]
                elif rnd < 0.2:
                    # insert for each action
                    for service in service2slots.keys():
                        slot_keys = service2slots[service]
                        ex_slot_keys = [slot_key for slot_key in service2slots_base.get(service, set()) if slot_key not in slot_keys]
                        if ex_slot_keys:
                            slot_keys.append(random.choice(ex_slot_keys))
                        service2slots[service] = slot_keys

            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(history[0:-1], indent=2),
                user_utterence=history[-1].replace('user: ', ''),
                slots=json.dumps(service2slots)
            )
            label = json.dumps(item['label'], separators=(',', ':'))

        elif type == 'casual_generation':
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            slots = {k: list(v) for k, v in item.get('aslot_keyed_slots', {}).items()}

            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(history[0:-1], indent=2),
                user_utterence=history[-1].replace('user: ', ''),
                slots=json.dumps(slots)
            )
            label = item['label']

        elif type == 'casual_generation_no_slots':
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(history[0:-1], indent=2),
                user_utterence=history[-1].replace('user: ', '')
            )
            label = item['label']

        else:
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(history[0:-1], indent=2),
                user_utterence=history[-1].replace('user: ', ''),
                search_results=json.dumps(item['search_results'], separators=(',', ':'))
            )
            label = item['label']

        return prompt, label


if __name__ == '__main__':
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')

    items = json.load(open('./datas/agent_sft_gen_data.json'))
    for item in items:
        output = AgentSFTDataset.tokenize(item, tokenizer, 1024, True)
