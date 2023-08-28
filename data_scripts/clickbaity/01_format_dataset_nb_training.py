import json
import os
import random

from llama.data_scripts.grammar.utils_ import *


def generate(sday, eday, rm_docs):

    docs = set()
    datas = []
    for day in time_range(sday, eday):
        if not os.path.exists(f'./datas/gpt_res/{day}.csv'):
            continue

        df = pd.read_csv(f'./datas/gpt_res/{day}.csv')
        for row in df.itertuples():
            if row.doc_id in rm_docs:
                continue
            docs.add(row.doc_id)
            raw_title = str(row.raw_title)
            rewrote_title = str(row.rewrote_title)
            datas.append({
                'type': 'CLICKBAIT_SINGLE',
                'label': 'Formally.',
                'source_sent': raw_title,
            })
            datas.append({
                'type': 'CLICKBAIT_SINGLE',
                'label': 'Informally.',
                'source_sent': rewrote_title,
            })
    return docs, datas


def save_file(out_file, datas):
    sout = open(out_file, 'w')
    for data in datas:
        sout.write(json.dumps(data) + '\n')
    sout.close()


if __name__ == '__main__':
    # clickbait  Yes or No
    train_sday = '2023-06-01'
    train_eday = '2023-06-30'
    train_docs, train_datas = generate(train_sday, train_eday, set())

    valid_sday = '2023-08-01'
    valid_eday = '2023-08-07'
    valid_docs, valid_datas = generate(valid_sday, valid_eday, train_docs)

    ex_train_datas, ex_valid_datas = [], []
    for input_file_labeled in [
        './datas/clickbait_human.train.txt',
        './datas/clickbait_human.valid.txt',
    ]:
        for data in open(input_file_labeled):
            parts = data.strip().split('\t')
            if len(parts) != 3:
                continue
            label = 'Informally.' if (parts[0] == '1' or parts[0] == '2') else 'Formally.'

            ex_train_datas.append({
                'type': 'CLICKBAIT_SINGLE',
                'label': label,
                'source_sent': parts[2],
            })

    random.seed(0)
    random.shuffle(ex_train_datas)

    ex_valid_datas = ex_train_datas[-1000:]
    ex_train_datas = ex_train_datas[0: -1000]

    all_datas = train_datas + ex_train_datas + valid_datas + ex_valid_datas
    random.shuffle(all_datas)

    train_datas_nb = all_datas[:-500]
    valid_datas_nb = all_datas[-500:]

    output_dir = '/mnt/nlp/xingguang/llama/datasets/nb_training/clickbait'
    os.makedirs(output_dir, exist_ok=True)
    save_file(f'{output_dir}/train.txt', train_datas_nb)
    save_file(f'{output_dir}/valid.txt', valid_datas_nb)
