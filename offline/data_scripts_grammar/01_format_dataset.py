import json
import os.path
import random

import pandas as pd

from hallucination.data_scripts_grammar.utils_ import *


def generate(sday, eday, out_file):
    datas = []
    pos, neg = 1, 1
    for day in time_range(sday, eday):
        if not os.path.exists(f'{day}.csv'):
            continue

        df = pd.read_csv(f'{day}.csv')
        for row in df.itertuples():
            change_pos_list = [int(x) for x in str(row.change_pos_list).split(',')]
            change_word_list = [x for x in str(row.change_word_list).split(',')]

            full_words = str(row.full_text).split(' ')

            pos_word_list = []

            for change_pos, change_words in zip(change_pos_list, change_word_list):
                for change_word in change_words.split('|||'):
                    if change_word == full_words[change_pos]:
                        continue
                    else:
                        pos_word_list.append([change_pos, change_word, full_words[change_pos]])
                        break

            if not pos_word_list:
                continue

            descp = 'Yes, '
            for i, (change_pos, change_word, real_word) in enumerate(pos_word_list):
                full_words[change_pos] = change_word
                if i == 0:
                    descp += f'the word \'{change_word}\' should be \'{real_word}\''
                else:
                    descp += f', and the word \'{change_word}\' should be \'{real_word}\''
            descp += '.'

            datas.append({
                'text': ' '.join(full_words),
                'description': descp,
                'info': [
                    {
                        'real': real_word,
                        'fake': change_word,
                        'pos': change_pos
                    } for change_pos, change_word, real_word in pos_word_list
                ]
            })
            neg += 1

        doc2click = defaultdict(float)

        for doc, click in get_doc_by_day(day, is_push=True).items():
            doc2click[doc] += click

        for doc, click in get_doc_by_day(day, is_push=False).items():
            doc2click[doc] += click

        doc2click = {doc: click for doc, click in sorted(doc2click.items(), key=lambda x: x[1], reverse=True)[500:1000]}
        doc2feature: Dict[str, StaticDocument] = pull_features_async(doc2click.keys(), return_dict=True,
                                                                     return_item=True)

        items = list(doc2feature.items())
        random.shuffle(items)

        for doc, feature in items[0:50]:
            title = clean_text(feature.seg_title).lower()
            title_words = title.lower().split(' ')
            content_words = get_content_words(feature.seg_content)

            if len(title_words) > 5:
                datas.append({
                    'text': ' '.join(title_words),
                    'description': 'No, there is no problem for the writing.'
                })
                pos += 1

            if len(content_words) > 30:
                datas.append({
                    'text': ' '.join(content_words),
                    'description': 'No, there is no problem for the writing.'
                })
                pos += 1

    print(neg, pos)
    sout = open(out_file, 'w')
    for data in datas:
        sout.write(json.dumps(data) + '\n')
    sout.close()


if __name__ == '__main__':
    generate('2023-07-01', '2023-07-31', 'train.txt')
    generate('2023-08-01', '2023-08-08', 'valid.txt')