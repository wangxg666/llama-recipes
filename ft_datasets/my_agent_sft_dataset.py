import os

import torch
import tqdm
from torch.utils.data import Dataset

import copy
import json
from typing import Dict

PERSONA_PROMPT_DICT = {
    'attraction': (
        'You are a community outreach coordinator, engaging with locals and tourists alike to promote the rich heritage and attractions of the area.'
    ),
    'hotel': (
        'You are a staff member responsible for hotel reservations at a local travel agency. You understand the unique features of each local hotel and can quickly find the hotel that meets users\' preferences based on their needs.'
    ),
    'train': (
        'You are a ticket seller at the local train ticket sales center. You work diligently, have a friendly personality, and are very skilled at assisting passengers inquiring about and purchasing train tickets.'
    ),
    'restaurant': (
        'You are a locally renowned food critic who has tried almost every restaurant in the area. Whenever someone consults you for restaurant information, you are always able to respond enthusiastically and accurately.'
    ),
    'default': (
        'You are a local 114 operator, primarily handling local services such as contacting the police, finding hospitals, calling taxis, and other convenient services. Your service is efficient and of high quality, earning widespread praise from the local community.'
    )
}

ANSWER_TYPE_PROMPT = {
    'casual_generation': (
        '{persona}\n'
        'Given the conversion history, please firstly to decide what to do next. Specifically:\n'
        '- Casual, if you feel the user question is casual, which can be answered without querying the database, please provide your response based on the conversation history.\n'
        '- Query, if a database query is needed, please provide the specific query parameters.\n'
        'The first word you provide should represent the action you are going to perform.\n'
        'Then, give your response based on your decision.\n'
        'Here is the conversion history:\n{history}\n'
        'Please give your output:\n'
    ),
    'rag_generation': (
        '{persona}\n'
        'Given the conversion history, user utterance, and query result from database, '
        'please generate a appropriate answer based on the give conversion status.\n'
        'Here is the conversion history:\n{history}\n'
        'query result:\n{search_results}\n'
        'Please give your output:\n'
    )
}
ANSWER_TYPE_PROMPT['api_generation'] = ANSWER_TYPE_PROMPT['casual_generation']


class MyAgentSFTDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=2048, debug=False):
        type = 'train' if partition == 'train' else 'dev'
        input_files = [
            f'{dataset_config.root}/{dataset_config.dataset_dir}/{type}.{task}.json'
            for task in ['api', 'casual', 'rag']
        ]
        self.datas = []
        for input_file in input_files:
            datas = [json.loads(data) for data in open(input_file)]
            if partition == 'train':
                self.datas.extend(datas)
            else:
                n = 100 if 'casual.json' in input_file else 200
                self.datas.extend(datas[0:n])

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.debug = debug

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        item: Dict = self.datas[index]
        return MyAgentSFTDataset.inner_tokenize(item, self.max_words, self.tokenizer)

    @staticmethod
    def inner_tokenize(item, max_words, tokenizer, do_padding=True):
        type = item['type']

        if type == 'api_generation':
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(item['history'], indent=2)
            )
            label = item['label_type'] + '\n' + json.dumps(item['label'], separators=(',', ':'))

        elif type == 'casual_generation':
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(item['history'], indent=2)
            )
            label = item['label_type'] + '\n' + item['label']

        else:
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(item['history'], indent=2),
                search_results=json.dumps(item['search_results'], separators=(',', ':'))
            )
            label = item['label']

        example = prompt + label
        print(example + '\n\n', flush=True)

        prompt = tokenizer.encode(prompt)
        example = tokenizer.encode(example) + [tokenizer.eos_token_id]

        prompt = torch.tensor(prompt, dtype=torch.int64)
        example = torch.tensor(example, dtype=torch.int64)

        if do_padding:
            padding = max_words - example.shape[0]
            if padding > 0:
                example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example = example[: max_words]

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


if __name__ == '__main__':
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')

    items = [
        {
            "type": "api_generation",
            "action": "hotel",
            "label_type": "Query",
            "history": [
                "USER: I want a hotel with free parking in the center of town.",
                "SYSTEM: Sure, I can help you with that. Do you have a price preference?",
                "USER: Something moderate would be preferred.",
                "SYSTEM: I am sorry but there is nothing in the moderate price range. Would you like another price range?",
                "USER: What about a moderate hotel located in the North?"
            ],
            "label": [
                {
                    "service": "hotel",
                    "active_intent": "find_hotel",
                    "slot_values": {
                        "hotel-area": [
                            "north"
                        ],
                        "hotel-parking": [
                            "yes"
                        ],
                        "hotel-pricerange": [
                            "moderate"
                        ],
                        "hotel-name": [
                            "Acorn Guest House"
                        ]
                    }
                }
            ]
        },
        {
            "type": "casual_generation",
            "action": "restaurant",
            "label_type": "Normal",
            "history": [
                "USER: I need a train that leaves on monday from cambridge",
                "SYSTEM: Ok and what is your destination?",
                "USER: Norwich. I'd like to leave after 17:30",
                "SYSTEM: Okay, the TR6675 leaves at 17:36 and arrives by 18:55. Would you like me to book tickets?",
                "USER: Yes please book that for 5 people.",
                "SYSTEM: Your train tickets are booked, please retain reference number: YVHR5SZJ and the fee for the tickets is 88GBP, which you will be required to pay at the station. Anything else today?",
                "USER: I need a restaurant that serves catalan food and it should be in the west.",
                "SYSTEM: Unfortunately, I'm not finding any Catalan restaurants in the west. Would you like me to look in other areas?",
                "USER: Is there another high-end restaurant in the west area that you can recommend?",
                "SYSTEM: Are you looking for expensive food?",
                "USER: Yes, please.",
                "SYSTEM: Cocum is an expensive restaurant on the west part of town serving Indian food. Would you like more information or reserve a table?",
                "USER: I don't really like Indian food. Are there any expensive british restaurants in the west?",
                "SYSTEM: Yes, I have 2. The Graffiti and Traveller's Rest. Do you know which you would prefer?",
                "USER: I don't really have a preference. Which one would you recommend?",
                "SYSTEM: I would recommend the Travellers rest. Would you like me to book you a table?",
                "USER: Yes, I would like a table."
            ],
            "label": "What day and time will you want a table?"
        },
        {"type": "rag_generation", "action": "hotel",
         "history": ["USER: Yes, I need a place to stay that is expensive, and is a hotel please."], "search_results": {
            "hotel": [{"address": "15-17 norman way, coldhams business park", "area": "east", "internet": "yes",
                       "parking": "yes", "id": "16", "location": [52.2, 0.17],
                       "name": "express by holiday inn cambridge", "phone": "01223866800", "postcode": "cb13lh",
                       "price": {"double": "90", "family": "90", "single": "90"}, "pricerange": "expensive",
                       "stars": "2", "takesbookings": "yes", "type": "hotel"},
                      {"address": "gonville place", "area": "centre", "internet": "yes", "parking": "yes", "id": "18",
                       "location": [52.2, 0.13], "name": "gonville hotel", "phone": "01223366611", "postcode": "cb11ly",
                       "price": {"double": "95", "family": "119", "single": "79"}, "pricerange": "expensive",
                       "stars": "3", "takesbookings": "yes", "type": "hotel"}]},
         "label": "I have 5 different hotels that meet your needs. Is there a certain area you prefer to stay in?"}
    ]

    for item in items:
        output = MyAgentSFTDataset.inner_tokenize(item, 1024, tokenizer)

    # datas = []
    # input_dir = '/home/paperspace/xingguang/datasets/agent_sft.v02'
    # for input_file in os.listdir(input_dir):
    #     input_file = f'{input_dir}/{input_file}'
    #     datas.extend([json.loads(data) for data in open(input_file)])
    #
    # max_len = -1
    # for data in tqdm.tqdm(datas):
    #     out = MyAgentSFTDataset.inner_tokenize(data, 2048, tokenizer, do_padding=False)
    #     print(len(out['input_ids']))
    #     max_len = max(max_len, len(out['input_ids']))
    # print(f'max len is {max_len}')


