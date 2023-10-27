import collections
import multiprocessing
import json
import os
import pickle

import numpy as np
import tqdm


IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

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


def process_item(item, feature, tokenizer):
    input_txt = PROMPT.format_map(feature)
    input_ids = tokenizer.encode(input_txt)

    labels = [IGNORE_INDEX for _ in input_ids]

    for round in item['dialog']:
        user = round['user']
        reply_to = round['reply_to']
        comment = round['comment']

        round_prompt = ''
        round_label = ''
        if user == YOU:
            if reply_to:
                round_prompt += f'r reply to {reply_to}'
            round_prompt = f'\n\n{user}{round_prompt}:\n'
            round_label = f'{comment} </s>'

        else:
            if reply_to:
                round_prompt += f'\'s reply to {reply_to}'
            round_prompt += ':\n'
            round_prompt += comment
            round_prompt = f'\n\n[INST]\n{user}{round_prompt}\n[/INST]'

        """
            input: \n\nYou:\n
            tokenize: ['▁', '<0x0A>', '<0x0A>', 'You', ':', '<0x0A>']
            input_ids: [1, 29871, 13, 13, 3492, 29901, 13]
            需要去掉的 [1, 29871, 13, 13] -> "<s> \n\n"
            加两个回车，然后通过[3:] 操作跳过前4个操作符，能够确保结果跟一次性tokenize尽可能一致
        """
        round_prompt_ids = tokenizer.encode(round_prompt)[3:]

        if round_label:
            round_prompt_label_ids = tokenizer.encode(round_prompt + round_label)[3:]
            round_label_ids = round_prompt_label_ids[len(round_prompt_ids):]
        else:
            round_label_ids = []

        input_ids.extend(round_prompt_ids + round_label_ids)
        labels.extend([IGNORE_INDEX for _ in round_prompt_ids] + round_label_ids)

    if input_ids[-1] != tokenizer.eos_token_id:
        input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    input_ids = np.asarray(input_ids, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    labels_pos = np.argwhere(labels != -100)

    return {
        "input_ids": input_ids,
        "labels_pos": labels_pos,
    }


def process_part(partition, n_thread, i_thread):

    input_file = 'train' if partition == 'train' else 'valid'
    os.makedirs(f'{work_dir}/{partition}.{model_type}', exist_ok=True)

    doc2items = collections.defaultdict(list)
    gzin = open(f'{work_dir}/{input_file}.dialog')
    if i_thread == 0:
        gzin = tqdm.tqdm(gzin)

    for i, data in enumerate(gzin):
        if i % n_thread != i_thread:
            continue
        item = json.loads(data)
        doc2items[item['docid']].append(item)
    print(f'thread {i_thread}, load {len(doc2items)} docs for {sum([len(v) for v in doc2items.values()])} comment.', flush=True)

    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(f'meta-llama/Llama-2-{model_type}-hf')


    gzin = open(f'{work_dir}/{input_file}.feature')
    if i_thread == 0:
        gzin = tqdm.tqdm(gzin)
    doc2feature = {}

    for data in gzin:
        feature = json.loads(data)
        doc = feature['docid']
        if doc not in doc2items:
            continue
        doc2feature[doc] = feature

    outputs = []
    doc_features = doc2feature.items()
    if i_thread == 0:
        doc_features = tqdm.tqdm(doc_features)
    for doc, feature in doc_features:
        for item in doc2items[doc]:
            outputs.append(process_item(item, feature, tokenizer))
    outputs = [x for x in outputs if x['input_ids'].shape[0] <= 4096]
    pickle.dump(outputs, open(f'{work_dir}/{partition}.{model_type}/part-{str(i_thread + 1000)[1:]}.bin', 'wb'))


def process_partation(partition='train', N=32):
    pool = multiprocessing.Pool(N)
    for i in range(N):
        pool.apply_async(
            func=process_part,
            args=(partition, N, i)
        )
    pool.close()
    pool.join()

    datas = []
    for filename in os.listdir(f'{work_dir}/{partition}.{model_type}/'):
        datas.extend(pickle.load(open(f'{work_dir}/{partition}.{model_type}/{filename}', 'rb')))
    pickle.dump(datas, open(f'{work_dir}/{partition}.{model_type}.bin', 'wb'))


if __name__ == '__main__':
    model_type = '13b'
    work_dir = '/home/paperspace/xingguang/datasets/comment.v02'

    process_partation('valid', 16)
    process_partation('train', 64)
