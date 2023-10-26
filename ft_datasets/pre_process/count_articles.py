import os
import sys
import json
import tqdm
import gzip
import pickle
import multiprocessing
import numpy as np


def pre_tokenize(input_file):
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(f'meta-llama/Llama-2-{model_type}-hf')

    datas = []
    num_news, num_news_with_comment = 0, 0
    for data in gzip.open(input_file, 'rb'):
        obj = json.loads(str(data, 'utf8'))
        if 'comment' in obj and obj['comment']:
            num_news_with_comment += 1
        num_news += 1
    return num_news, num_news_with_comment

def process_aritilces():
    pool = multiprocessing.Pool(64)
    work_dir = '/home/paperspace/xingguang/datasets/pre-training-ariticle'

    os.makedirs(f'{work_dir}/tokenized.{model_type}', exist_ok=True)

    res_list = []
    for input_file in sorted(os.listdir(f'{work_dir}/text.comment.gz/')):
        res_list.append(pool.apply_async(
            func=pre_tokenize,
            args=(
                f'{work_dir}/text.comment.gz/{input_file}',
            )
        ))
    pool.close()
    pool.join()

    num_news, num_news_with_comment = 0, 0
    for res in res_list:
        n1, n2 = res.get()
        num_news += n1
        num_news_with_comment += n2
    print(num_news, num_news_with_comment)


if __name__ == '__main__':
    model_type = '13b'
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(f'meta-llama/Llama-2-{model_type}-hf')
    process_aritilces()




