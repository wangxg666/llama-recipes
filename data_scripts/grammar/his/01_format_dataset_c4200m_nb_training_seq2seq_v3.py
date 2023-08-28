import json
import random

from llama.data_scripts.grammar.utils_ import *


if __name__ == '__main__':
    input_file = '/mnt/nlp/xingguang/llama/datasets/nb_raw_datas/grammar_c4200m/raw.txt'
    candidates = []
    for data in open(input_file):
        parts = data.strip().split('\t')
        if len(parts) != 5:
            continue
        n_ups, n_words, source_sent, target_sent, updates = parts
        if source_sent.strip().lower() == target_sent.strip().lower():
            continue
        if len(target_sent.strip().lower().split(' ')) <= 2:
            continue
        candidates.append(Candidate(source_sent, target_sent, updates=json.loads(updates)))
    random.shuffle(candidates)
    candidates = candidates[0:5000]

    datas = []
    for candidate in candidates:
        target_words = candidate.target_sent.split(' ')
        target_words_new = []
        for i, update in enumerate(candidate.updates):
            begin, size = update['rp_begin'], update['rp_length']
            from_word, to_word = update['rp_from'], update['rp_to']
            if i == 0:
                target_words_new.extend(target_words[0:begin])

            target_words_new.extend(f'<soc> {to_word} -> {from_word} <eoc>'.split(' '))

            if i == len(candidate.updates) - 1:
                end = len(target_words)
            else:
                end = candidate.updates[i+1]['rp_begin']
            target_words_new.extend(target_words[begin+size: end])

        data = {
            'type': 'GRAMMAR_SEQ2SEQ',
            'label': ' '.join(target_words_new),
            'source_sent': candidate.source_sent,
        }
        datas.append(data)

    n = len(datas) - 100
    print(type, len(datas[0:n]), len(datas[n:]), np.mean([len(x["source_sent"].split(' ')) for x in datas]))

    output_dir = '/mnt/nlp/xingguang/llama/datasets/nb_training/grammar_c4200m_seq2seq_v3'
    os.makedirs(output_dir, exist_ok=True)

    save_file(f'{output_dir}/train.txt', datas[0:n])
    save_file(f'{output_dir}/valid.txt', datas[n:])

