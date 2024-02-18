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
        "{persona}\n"
        "Given the conversation history:\n{history}\n"
        "Generate the next response to the user's latest reply: \n{user_utterence}.\n"
        "Your response must adhere strictly to one of the following two types:\n"
        "1. Concluding Sentence: If it seems that the user is looking to end the assistance, conclude your conversation appropriately. This may include, but is not limited to, saying goodbye.\n"
        "2. Connecting Sentence: If the user has not ended the interaction or introduced any new requirements, you may opt for a connecting sentence to maintain the flow of the conversation. "
        "For example, if the user expresses gratitude for your assistance but hasn't concluded the interaction, you could respond with, \"You're welcome! Is there anything else I can assist you with?\"."
        "Should the user ask a question that significantly diverges from the previous topic, you may selectively answer and politely inquire whether they wish to return to the main topic.\n"
        "Now, take a deep breath and think step-by-step, your next response should be:\n"
    ),
    'rag_generation': (
        '{persona}\n'
        'Given the conversion history and the search result, '
        'please generate a appropriate answer based on the give conversion status.\n'
        'Here is the conversion history:\n{history}\n'
        'the search result:\n{search_results}\n'
        'Please give your response based on the conversion history and search result.\n'
        'Your response:\n'
    )
}
ANSWER_TYPE_PROMPT['default'] = ANSWER_TYPE_PROMPT['casual_generation']
ANSWER_TYPE_PROMPT['casual_generation_no_slots'] = ANSWER_TYPE_PROMPT['casual_generation']


class AgentSFTDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=2048, do_padding=True, debug=False):
        type = 'train' if partition == 'train' else 'dev'
        input_files = [
            f'{dataset_config.root}/{dataset_config.dataset_dir}/{type}.{task}.json'
            for task in ['rag', 'casual']
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
        prompt, label = AgentSFTDataset.prompting(item)
        return agent_tokenize(self.tokenizer, prompt, label, self.max_words, self.do_padding)

    @staticmethod
    def prompting(item: Dict):
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
            label = json.dumps(item['label'], separators=(',', ':'))

        elif type == 'casual_generation':
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            slots = {k: list(v) for k, v in item.get('asked_slots', {}).items()}

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
            if len(item['search_results']) > 5:
                search_results = f"search result size = {len(item['search_results'])}"
            else:
                search_results = json.dumps(item['search_results'], separators=(',', ':'))
            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(history[0:-1], indent=2),
                search_results=search_results
            )
            label = item['label']

        return prompt, label


if __name__ == '__main__':
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    dir = '/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k'

    items = [json.loads(data) for data in open(f'{dir}/train.rag.json')]
    for item in items[0:10]:
        prompt, label = AgentSFTDataset.prompting(item)
        print(prompt + label, '\n\n')
