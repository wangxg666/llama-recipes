import sys
from llama.data_scripts.grammar.utils_ import *

from utils.str_utils import stop_tokens


if __name__ == '__main__':
    train_datas = []
    valid_datas = []
    candidates = []

    f2t = defaultdict(set)
    for rid, row in pd.read_csv('./changes.csv').iterrows():
        f2t[row['from']].add(row['to'])
    print(f'load {len(f2t)} valid updates')

    key2count = defaultdict(float)

    for data in open('/mnt/nlp/xingguang/llama/c4200m/sentence_pairs.tsv-00000-of-00010'):
        parts = data.strip().split('\t')
        if len(parts) != 3:
            continue
        # source sent has grammatical erro
        source_sent, target_sent, changes = parts
        changes = json.loads(changes)
        candidate = Candidate(source_sent, target_sent, changes)

        if not source_sent:
            continue

        target_words = [w for w in target_sent.strip().split(' ') if w]
        if not target_words:
            continue

        if target_words[0][0] not in valid_alphas and target_words[0] not in valid_years:
            continue

        if sum([x[0] in valid_alphas for x in target_words if x]) >= len(target_words) // 2:
            continue

        # 非英语的规范句子不要
        if not is_standard_en_sent(candidate.source_sent) or not is_standard_en_sent(candidate.target_sent):
            continue

        is_need = False
        valid_changes = []
        # 只看 词性，单复数，进行时，等指定的错误
        for change in changes:
            # 句子开头的修改通通不要
            if change['rp_begin'] == 0 and change['rp_to'].lower() not in stop_tokens:
                continue

            rp_from = change['rp_from']
            rp_to = change['rp_to']

            if rp_from in f2t and rp_to in f2t[rp_from]:
                valid_changes.append(change)

        if len(valid_changes) != len(changes):
            continue

        for change in changes:
            rp_from = change['rp_from']
            rp_to = change['rp_to']
            # 只要有一个key满足要求，就保留下来
            key = f'{rp_from}_{rp_to}'
            if key2count.get(key, 0) < 10:
                is_need = True
            key2count[key] += 1
        if not is_need:
            continue

        candidate.source_sent = re.sub(' +', ' ', candidate.source_sent)
        candidate.target_sent = re.sub(' +', ' ', candidate.target_sent)

        for invalid_ch in [' / ']:
            candidate.source_sent = candidate.source_sent.replace(invalid_ch, ' ')
            candidate.target_sent = candidate.target_sent.replace(invalid_ch, ' ')

        candidate.source_sent = candidate.source_sent.strip()
        candidate.target_sent = candidate.target_sent.strip()
        if candidate.source_sent.lower().strip() == candidate.target_sent.lower().strip():
            continue

        # 太短或者太长的都不要
        if len(candidate.source_words) < 8 or len(candidate.source_words) > 32 or len(candidate.updates) >= 10:
            continue

        candidates.append(candidate)

        if len(candidates) % 10000 == 0:
            print(len(candidates))
            sys.stdout.flush()

        if len(candidates) >= 2000000:
            break

    candidates = sorted(candidates, key=lambda x: x.source_sent)

    up2count = defaultdict(float)
    for candidate in candidates:
        for update in candidate.updates:
            rp_from = update['rp_from']
            rp_to = update['rp_to']
            up2count[f'{rp_from} -> {rp_to}'] += 1

    output_dir = '/mnt/nlp/xingguang/llama/datasets/nb_raw_datas/grammar_c4200m'
    os.makedirs(output_dir, exist_ok=True)

    sout = open(f'{output_dir}/raw.txt', 'w')
    for candidate in candidates:
        sout.write(f'{len(candidate.updates)}\t{len(candidate.source_words)}\t{candidate.source_sent}\t{candidate.target_sent}\t{json.dumps(candidate.updates)}\n')
    sout.close()

    sout = open(f'{output_dir}/statistics.txt', 'w')
    for key, count in sorted(key2count.items(), key=lambda x:x[1], reverse=True):
        sout.write(f'{key}\t{count}\n')
    sout.close()