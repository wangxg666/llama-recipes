import sys

import pandas as pd

from llama.data_scripts.grammar.utils_ import *

from utils.str_utils import stop_tokens


if __name__ == '__main__':
    output_dir = '/mnt/nlp/xingguang/llama/datasets/nb_raw_datas/grammar_c4200m'

    commons = []
    bes = ['is', 'are', 'was', 'were', 'be', 'been']
    for x in bes:
        for y in bes:
            if x == y:
                continue
            commons.append([x, y])

    for line in open(f'{output_dir}/update_count.txt'):
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        f, t = parts[0:2]
        if f[0].lower() == t[0].lower() and f.lower() != t.lower():
            commons.append([f, t])

    out = defaultdict(list)
    for f, t in commons:
        out['from'].append(f)
        out['to'].append(t)
    pd.DataFrame(out).to_csv('./changes.csv', index=False)
    pd.DataFrame(out).to_csv(f'{output_dir}/update_select.csv', index=False)
