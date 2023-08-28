import json
import os.path
import random

import pandas as pd

from llama.data_scripts.grammar.utils_ import *


def save_file(out_file, datas):
    sout = open(out_file, 'w')
    for data in datas:
        sout.write(json.dumps(data) + '\n')
    sout.close()


if __name__ == '__main__':
    train_datas = []
    valid_datas = []

    for data in open('/mnt/nlp/xingguang/llama/c4200m/sentence_pairs.tsv-00000-of-00010'):
        parts = data.strip().split('\t')
        if len(parts) != 3:
            continue

        # source sent has grammatical erro
        source_sent, target_sent, _ = parts

        state = random.random()
        data_type = random.random()

        if data_type <= 0.5:
            # seq2seq task
            data = {
                'source_sent': source_sent,
                'type': 'SEQ2SEQ',
                'label': target_sent,
            }
        else:
            choice = random.random()
            # Yes for grammatical error
            data = {
                'source_sent': source_sent if choice < 0.5 else target_sent,
                'type': 'SINGLE',
                'label': 'Yes' if choice < 0.5 else 'No'
            }

        if state <= 0.02:
            valid_datas.append(data)
        else:
            train_datas.append(data)

        if len(train_datas) >= 200000 and len(valid_datas) >= 200000 * 0.02:
            break

    save_file('./datas/train.c4200m.txt', train_datas)
    save_file('./datas/valid.c4200m.txt', valid_datas)

