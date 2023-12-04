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
        '- Normal, if the user\' request can be answered without querying the database, please response directly.\n'
        '- Query, if a database query is needed to answer the user\'s request, please output the specific query parameters.\n'
        'The first word you provide should represent the action you are going to perform.\n'
        'Then, give your response based on your action decision.\n'
        'Here is the conversion history:\n{history}\n'
        'the user lastest utterence: \n{user_utterence}\n'
        'Please give your output:\n'
    ),
    'rag_generation': (
        '{persona}\n'
        'Given the conversion history, user utterance, and query result from database, '
        'please generate a appropriate answer based on the give conversion status.\n'
        'Here is the conversion history:\n{history}\n'
        'query result:\n{search_results}\n'
        'the user lastest utterence: \n{user_utterence}\n'
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
        history = [x.replace('USER', 'user').replace('SYSTEM', 'you') for x in item['history']]

        if type == 'api_generation':
            persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(history[0:-1], indent=2),
                user_utterence=history[-1].replace('user: ', '')
            )
            label = item['label_type'] + '\n' + json.dumps(item['label'], separators=(',', ':'))

        elif type == 'casual_generation':
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
        {"dialog_id": "pmul1141", "turn_id": 3, "type": "rag_generation", "action": "attraction",
         "history": ["USER: I'm looking for a place to go in the south of town.",
                     "SYSTEM: What type of place are you looking for? In the south, we have a cinema, theater, museum, parks, and night club. If you tell me what you are looking for, we can narrow it down.",
                     "USER: south. and i need to get address and postcode"], "search_results": {"attraction": [
            {"address": "14 king's parade", "area": "south", "entrance fee": "free", "id": "6",
             "location": [52.17, 0.11], "name": "byard art",
             "openhours": "it opens from 09:30 a.m. to 5:30 p.m. from monday to saturday, and from 11:00 a.m. to 4:00 p.m. on sunday",
             "phone": "01223464646", "postcode": "cb21sj", "pricerange": "free", "type": "museum"},
            {"address": "cambridge leisure park, clifton way", "area": "south", "entrance fee": "?", "id": "21",
             "location": [52.19, 0.14], "name": "cineworld cinema", "openhours": "?", "phone": "00872208000",
             "postcode": "cb17dy", "pricerange": "?", "type": "cinema"}]},
         "label": "Tenpin is a fun place in the south, located at Cambridge Leisure Park, Clifton Way. Their postcode is cb17dy."},
        {"dialog_id": "pmul3589", "turn_id": 11, "type": "casual_generation", "action": "train", "label_type": "Normal",
         "history": ["USER: Hello. I'm hoping to find a guesthouse in the north part of Cambridge. Are there any?",
                     "SYSTEM: Indeed there are. In fact, there are 11. If you'd like an inexpensive, 4 star guesthouse with free internet and free parking, Worth House is a good choice.",
                     "USER: I'm looking for a place to stay that is cheap located in the north.",
                     "SYSTEM: I would suggest this one city centre north b and b.",
                     "USER: Sure, I need a reservation for 4 people and 2 nights starting on Monday please.",
                     "SYSTEM: I have made the booking the Reference number is E6UVD9OL . Is there anything else i can help you with?",
                     "USER: Would you mind finding some info on a train into Cambridge for me?",
                     "SYSTEM: I could do that for you. Can you give me a time ?",
                     "USER: Yes, we will be coming from bishops stortford on Monday. I would like to leave after 19:30.",
                     "SYSTEM: Their is a train that leaves Monday at 21:29. it costs 10.10 pounds and it's a 38 minute trip. Would you like me to book that for you?",
                     "USER: Yes, please. Can you book me 4 tickets?"],
         "label": "Booking was successful, the total fee is 40.4 GBP payable at the station. Reference number is : H4BIFD0U . Will there be anything else today?"},
        {"dialog_id": "mul2061", "turn_id": 7, "type": "api_generation", "action": "hotel", "label_type": "Query",
         "history": [
             "USER: I am looking for a place to stay and it doesn't need free parking , but I would like it to be cheap.",
             "SYSTEM: There are lots of accommodations in the cheaper price range, but they do all offer free parking. Also, is there a certain part of town you'd like to be in?",
             "USER: Yes, the east part of town.", "SYSTEM: How about autumn house?",
             "USER: As long as it has free parking, we are good to go!",
             "SYSTEM: Great, would you like for me to set up a booking?", "USER: Yes and can I get a postcode?"],
         "label": {"hotel": {"area": "east", "name": "autumn house", "parking": "yes", "pricerange": "cheap"}}}
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


