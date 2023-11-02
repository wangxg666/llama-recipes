import gzip
import json
import random

from text_generation import Client
from pymongo import MongoClient

client = Client("http://184.105.106.16:1308")


def clean_text_fast(text):
    reps = {
        '-LRB-': '(',
        '-RRB-': ')',
        '-LSB-': '[',
        '-RSB-': ']',
        '-LCB-': '[',
        '-RCB-': ']',
    }
    return ' '.join([reps.get(w, w) for w in text.split(' ')])


YOU = 'You'
PROMPT = (
    "<<SYS>>\n"
    "You are a senior news commentator, that is good at understanding the key points of news and making accurate comments.\n"
    "Here is a news data with `title`, `content` and some comment history.\n"
    "Please make a new comment or reply to the specified user based on the command.\n"
    "### title:\n{title}\n"
    "### content:\n{content}\n"
    "<</SYS>>"

)

mongo_rs3 = MongoClient('mongodb://mongos.content.xingguang_wang:oq73xA7M55pDakL4TSFq97kuAoFKg0a9@content-mongos.mongos.nb.com/?readPreference=secondaryPreferred&authSource=admin&tls=false&connectTimeoutMS=1000&socketTimeoutMS=300000')
col = mongo_rs3['staticFeature']['document']


def enrich_prompt(doc, history):
    feature = col.find_one({'_id': doc}, {f: 1 for f in {'stitle', 'seg_content'}})
    if not feature:
        return ''
    input_txt = PROMPT.format_map({
        'title': clean_text_fast(feature.get('stitle', '')),
        'content': clean_text_fast(feature.get('seg_content', ''))
    })

    for round in history:
        user = round['user']
        reply_to = round['reply_to']
        comment = round['comment']

        round_prompt = ''
        if user == YOU:
            if reply_to:
                round_prompt += f'r reply to {reply_to}'
            round_prompt = f'\n{user}{round_prompt}:\n{comment}'

        else:
            if reply_to:
                round_prompt += f'\'s reply to {reply_to}'
            round_prompt += ':\n'
            round_prompt += comment
            round_prompt = f'\n[INST]\n{user}{round_prompt}\n[/INST]'
        input_txt += round_prompt
    return input_txt


datas = [data for data in open('/mnt/nlp/xingguang/llama/datasets/nb_training/comment.v02/valid.dialog', 'r')]
random.shuffle(datas)

for data in datas[0:200]:
    data = json.loads(data)
    label = data['dialog'][-1]['comment']
    data['dialog'][-1]['comment'] = ''

    prompt = enrich_prompt(data['docid'], data['dialog'])
    if not prompt:
        continue

    print(prompt)
    print(f'real: {label}')
    response = client.generate(prompt=prompt, max_new_tokens=100, repetition_penalty=1.1)
    print(f'pred (greedy): {response.generated_text}')
    response = client.generate(prompt=prompt, max_new_tokens=100, repetition_penalty=1., do_sample=True)
    print(f'pred (random): {response.generated_text}')
    # for response in client.generate_stream(prompt=prompt, max_new_tokens=100, repetition_penalty=1., do_sample=True):
    #     if not response.token.special:
    #         print(response.token.text, end='', flush=True)
    print('\n', flush=True)
    print('*****' * 20)
