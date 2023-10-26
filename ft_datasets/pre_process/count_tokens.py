import os
import pickle
import random

if __name__ == '__main__':
    input_files = []
    for input_sub_dir in [
        '/home/paperspace/xingguang/datasets/pre-training-faq/tokenized.13b',
        '/home/paperspace/xingguang/datasets/pre-training-ariticle/tokenized.13b',
    ]:
        input_files.extend([
            f'{input_sub_dir}/{filename}' for filename in os.listdir(input_sub_dir)
        ])

    datas = []
    for input_file in sorted(input_files):
        datas.extend(pickle.load(open(input_file, 'rb')))
        print(f'load {input_file}, data size = {len(datas)}')

    random.shuffle(datas)

    n_tokens = sum([len(x) for x in datas])
    print(f'pre load {len(datas)} datas from {len(input_files)} files, {n_tokens / 1024 / 1024 / 1024}B tokens')