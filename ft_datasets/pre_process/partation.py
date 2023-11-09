import os
import pickle
import random

if __name__ == '__main__':
    input_files = []
    for input_sub_dir in [
        '/home/paperspace/xingguang/datasets/pre-training-faq/tokenized.13b',
        '/home/paperspace/xingguang/datasets/pre-training-ariticle/tokenized.13b.v02',
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
    print(f'pre load {len(datas)} datas from {len(input_files)} files, {n_tokens / 1.e9} B tokens')

    NUM_VALID = 10000

    work_dir = '/home/paperspace/xingguang/datasets/pre-training-shuffle/tokenized.13b.v02'
    os.makedirs(work_dir, exist_ok=True)

    pickle.dump(datas[0:NUM_VALID], open(f'{work_dir}/valid.bin', 'wb'))

    datas = datas[NUM_VALID:]
    n_parts = 100
    chunk = len(datas) // n_parts
    for i in range(n_parts):
        pickle.dump(datas[i*chunk: i*chunk+chunk], open(f'{work_dir}/train.part-{str(i+1000)[1:]}.bin', 'wb'))

