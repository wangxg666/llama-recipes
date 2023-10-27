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
    tokenizer = LlamaTokenizer.from_pretrained(f'meta-llama/Llama-2-{model_type}-hf')

    datas = []
    for data in gzip.open(input_file, 'rb'):
        obj = json.loads(str(data, 'utf8'))
        text = f'{"-".join(obj["_id"])}, {obj["title"]}\n'
        if 'content' in obj:
            text += obj['content'] + '\n'
        if 'comment' in obj and obj['comment']:
            text += '\n'.join(obj['comment'])
        datas.append(text)
    tokenized_datas = []
    for data in tqdm.tqdm(datas, colour='green'):
        tokenized_datas.append(np.asarray(tokenizer.encode(data) + [tokenizer.eos_token_id], np.int64))
    pickle.dump(tokenized_datas, open(output_file, 'wb'))


def process_aritilces():
    pool = multiprocessing.Pool(64)
    work_dir = '/home/paperspace/xingguang/datasets/pre-training-ariticle'

    os.makedirs(f'{work_dir}/tokenized.{model_type}', exist_ok=True)

    for input_file in sorted(os.listdir(f'{work_dir}/text.comment.gz/')):
        pool.apply_async(
            func=pre_tokenize,
            args=(
                f'{work_dir}/text.comment.gz/{input_file}',
                f'{work_dir}/tokenized.{model_type}/{input_file.replace(".gz", ".bin")}',
            )
        )
    pool.close()
    pool.join()



if __name__ == '__main__':
    model_type = '13b'
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(f'meta-llama/Llama-2-{model_type}-hf')

    process_aritilces()




