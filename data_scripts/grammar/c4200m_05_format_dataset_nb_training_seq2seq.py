import logging
import random

from llama.data_scripts.grammar.utils_ import *


if __name__ == '__main__':
    # input_file = '/mnt/nlp/xingguang/llama/datasets/nb_raw_datas/grammar_c4200m/raw.txt'
    # candidates = []
    # for data in open(input_file):
    #     parts = data.strip().split('\t')
    #     if len(parts) != 5:
    #         continue
    #     n_ups, n_words, source_sent, target_sent, updates = parts
    #     if source_sent.strip().lower() == target_sent.strip().lower():
    #         continue
    #     if len(target_sent.strip().lower().split(' ')) <= 2:
    #         continue
    #     candidates.append(Candidate(source_sent, target_sent))
    # random.shuffle(candidates)
    # candidates = candidates[0:500000]

    candidates = []
    for data in open('/mnt/nlp/xingguang/llama/datasets/nb_raw_datas/grammar_c4200m/checked.txt'):
        parts = data.strip().split('\t')
        if len(parts) != 3:
            continue
        src, trg, gpt_res = parts
        if trg == gpt_res:
            candidates.append(Candidate(src, gpt_res))
    print(f'load {len(candidates)} candidates with double check dataset')

    train_datas = []
    valid_datas = []
    for candidate in candidates:
        if len(valid_datas) < 1000 and random.random() < 0.01:
            datas = valid_datas
        else:
            datas = train_datas

        datas.append({
            'type': 'GRAMMAR_SEQ2SEQ',
            'label': candidate.target_sent,
            'source_sent': candidate.source_sent,
        })
        datas.append({
            'type': 'GRAMMAR_SEQ2SEQ',
            'label': candidate.target_sent,
            'source_sent': candidate.target_sent,
        })

    print(type, len(train_datas), len(valid_datas), np.mean([len(x["source_sent"].split(' ')) for x in train_datas]))

    output_dir = '/mnt/nlp/xingguang/llama/datasets/nb_training/grammar_c4200m_seq2seq'
    os.makedirs(output_dir, exist_ok=True)

    save_file(f'{output_dir}/train.txt', train_datas)
    save_file(f'{output_dir}/valid.txt', valid_datas)

