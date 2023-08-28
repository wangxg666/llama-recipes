from collections import defaultdict

def load(input_file):
    sin = open(input_file)

    tn2exact = defaultdict(float)
    tn2total = defaultdict(float)

    while True:
        line = sin.readline()
        if not line:
            break
        if line.startswith('input:'):
            input = line.replace('input:', '').replace(' ', '')
            real = sin.readline().replace('real:', '').replace(' ', '')
            pred = sin.readline().replace('pred:', '').replace(' ', '')

            if input == real:
                tn = 'directly'
            else:
                tn = 'rewrite'

            if real == pred:
                tn2exact[tn] += 1
            tn2total[tn] += 1

    for tn, exact in tn2exact.items():
        total = tn2total[tn]
        print(tn, exact, total, exact / total)


if __name__ == '__main__':
    input_file = '/home/paperspace/datasets/grammar_c4200m_seq2seq/valid.txt.pred.GRAMMAR_SEQ2SEQ'
    load(input_file)