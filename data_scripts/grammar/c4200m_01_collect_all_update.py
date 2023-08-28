import sys
from llama.data_scripts.grammar.utils_ import *


if __name__ == '__main__':
    candidates = []
    for data in open('/mnt/nlp/xingguang/llama/c4200m/sentence_pairs.tsv-00000-of-00010'):
        parts = data.strip().split('\t')
        if len(parts) != 3:
            continue
        # source sent has grammatical erro
        source_sent, target_sent, changes = parts
        changes = json.loads(changes)
        candidates.append(Candidate(source_sent, target_sent, changes))

    up2count = defaultdict(float)
    for candidate in candidates:
        for update in candidate.updates:
            rp_from = update['rp_from']
            rp_to = update['rp_to']
            up2count[f'{rp_from} -> {rp_to}'] += 1

    output_dir = '/mnt/nlp/xingguang/llama/datasets/nb_raw_datas/grammar_c4200m'
    sout = open(f'{output_dir}/update_count.txt', 'w')
    for up, count in sorted(up2count.items(), key=lambda x:x[1], reverse=True):
        if count <= 10:
            break
        f, t = up.split(' -> ')
        if not f or not t or f != norm_text(f) or t != norm_text(t):
            continue
        sout.write(f'{f}\t{t}\t{count}\n')
    sout.close()