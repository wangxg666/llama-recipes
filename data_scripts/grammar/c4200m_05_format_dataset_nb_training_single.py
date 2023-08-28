from llama.data_scripts.grammar.utils_ import *


if __name__ == '__main__':
    input_file = '/mnt/nlp/xingguang/llama/datasets/nb_raw_datas/grammar_c4200m/raw.txt'
    candidates = []
    for data in open(input_file):
        parts = data.strip().split('\t')
        if len(parts) != 5:
            continue
        n_ups, n_words, source_sent, target_sent, updates = parts
        candidates.append(Candidate(source_sent, target_sent))
    random.shuffle(candidates)
    candidates = candidates[0:1000]

    datas = []
    for candidate in candidates:
        prob = random.random()
        # Yes for grammatical error
        datas.append({
            'type': 'GRAMMAR_SINGLE',
            'label': 'Good.',
            'source_sent': candidate.target_sent,
        })
        datas.append({
            'type': 'GRAMMAR_SINGLE',
            'label': 'Poor.',
            'source_sent': candidate.source_sent,
        })

    n = len(datas) - 100
    print(type, len(datas[0:n]), len(datas[n:]), np.mean([len(x["source_sent"].split(' ')) for x in datas]))

    output_dir = '/mnt/nlp/xingguang/llama/datasets/nb_training/grammar_c4200m_single'
    os.makedirs(output_dir, exist_ok=True)

    save_file(f'{output_dir}/train.txt', datas[0:n])
    save_file(f'{output_dir}/valid.txt', datas[n:])

