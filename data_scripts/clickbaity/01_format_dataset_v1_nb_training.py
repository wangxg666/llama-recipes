import json
import os
import random

from llama.data_scripts.grammar.utils_ import *


def save_file(out_file, datas):
    sout = open(out_file, 'w')
    for data in datas:
        sout.write(json.dumps(data) + '\n')
    sout.close()


if __name__ == '__main__':
    # click bait  Yes, other No

    input_file_labeled = './datas.txt'
    ex_valid_datas = []

    key2count = defaultdict(float)
    for data in open(input_file_labeled):
        parts = data.strip().split('\t')
        if len(parts) != 3:
            continue
        label = 'Informally.' if (parts[0] == '1' or parts[0] == '2') else 'Formally.'

        feature = pull_feature(parts[1], fields=['click_bait_v3'])
        if not feature or "click_bait_v3" not in feature:
            continue

        ex_valid_datas.append({
                'type': 'CLICKBAIT_SINGLE',
                'label': label,
                'source_sent': parts[2],
            })

        key = f'{0 if parts[0] == "0" else 1} {0 if "0" in feature["click_bait_v3"] else 1}'
        key2count[key] += 1

    print(json.dumps(key2count))

    output_dir = '/mnt/nlp/xingguang/llama/datasets/nb_training/clickbait/'
    os.makedirs(output_dir, exist_ok=True)
    save_file(f'{output_dir}/valid.ex.txt', ex_valid_datas)