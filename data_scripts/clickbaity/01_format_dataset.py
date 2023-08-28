import json
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
            raw_title = str(row.raw_title).lower()
            rewrote_title = str(row.rewrote_title).lower()
            style_phrase = str(row.rewrote_phrase).lower().split('|||')

            datas.append({
                'text': raw_title,
                'description': 'No.',
                'style_phrase': []
            })

            if len(style_phrase) > 1:
                phrase = '"' + '", "'.join(style_phrase[0:-1]) + '" and "' + style_phrase[-1] + '"'
            else:
                phrase = f'"{style_phrase[0]}"'

            # datas.append({
            #     'text': rewrote_title,
            #     'description': f'Yes, the phrase {phrase} {"is" if len(style_phrase) == 1 else "are"} opinion words.'
            # })
            datas.append({
                'text': rewrote_title,
                'description': 'Yes.',
                'style_phrase': style_phrase
            })
    return docs, datas


def save_file(out_file, datas):
    sout = open(out_file, 'w')
    for data in datas:
        sout.write(json.dumps(data) + '\n')
    sout.close()


if __name__ == '__main__':
    # click bait  Yes, other No

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
            label = 'Yes.' if (parts[0] == '1.0' or parts[0] == '2.0') else 'No.'
            ex_train_datas.append({
                'text': parts[2],
                'description': label,
                'style_phrase': [f'labeled {parts[0]}']
            })

    random.seed(0)
    random.shuffle(ex_train_datas)

    ex_valid_datas = ex_train_datas[-1000:]
    ex_train_datas = ex_train_datas[0: -1000]

    save_file('./datas/train.txt', train_datas + ex_train_datas)
    save_file('./datas/valid.txt', valid_datas)
    save_file('./datas/valid.ex.txt', ex_valid_datas)