import os
import sys
import json
import tqdm
import gzip
import pickle
import multiprocessing
import numpy as np


def pre_tokenize(input_file, output_file):
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    datas = []
    for data in gzip.open(input_file, 'rb'):
        obj = json.loads(str(data, 'utf8'))
        text = f'{obj["_id"]}\n'
        if 'title' in obj:
            text += obj['title'] + '\n'
        if 'content' in obj:
            text += obj['content'] + ' '
        datas.append(text)
    tokenized_datas = []
    for data in tqdm.tqdm(datas, colour='green'):
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
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    pool = multiprocessing.Pool(32)
    work_dir = '/home/paperspace/xingguang/datasets/pre-training-ariticle'
    for input_file in sorted(os.listdir(f'{work_dir}/text.gz/')):
        pool.apply_async(
            func=pre_tokenize,
            args=(
                f'{work_dir}/text.gz/{input_file}',
                f'{work_dir}/tokenized.7b/{input_file.replace(".gz", ".bin")}',
            )
        )
    pool.close()
    pool.join()



