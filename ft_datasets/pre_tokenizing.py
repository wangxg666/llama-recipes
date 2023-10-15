import os
import sys
import json
import tqdm
import pickle
import multiprocessing
import numpy as np


def pre_tokenize(input_file, output_file):
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')

    datas = []
    for data in open(input_file):
        obj = json.loads(data)
        text = ''
        if 'title' in obj:
            text += obj['title'] + ' '
        if 'content' in obj:
            text += obj['content'] + ' '
        datas.append(text)
    tokenized_datas = []
    for data in tqdm.tqdm(datas):
        tokenized_datas.append(np.asarray(tokenizer.encode(data) + [tokenizer.eos_token_id], np.int64))
    pickle.dump(tokenized_datas, open(output_file, 'wb'))


if __name__ == '__main__':
    # from configs.datasets import my_pre_train_pad_dataset, my_pre_train_dataset
    # dataset = get_my_pre_train_dataset(my_pre_train_dataset, tokenizer, 'valid')
    # dataset = get_my_pre_train_pad_dataset(my_pre_train_pad_dataset, tokenizer, 'valid')
    # for i in range(10):
    #     for k, v in dataset[i].items():
    #         print(k)
    #         print(len(v), v)
    pool = multiprocessing.Pool(32)
    work_dir = '/home/paperspace/xingguang/datasets/pre-training-ariticle'
    for input_file in os.listdir(f'{work_dir}/text/'):
        pool.apply_async(
            func=pre_tokenize,
            args=(
                f'{work_dir}/text/{input_file}',
                f'{work_dir}/tokenized.13b/{input_file.replace(".txt", ".bin")}',
            )
        )
    pool.close()
    pool.join()