import os
import sys
import json
import tqdm
import gzip
import pickle
import multiprocessing
import numpy as np


work_dir = '/home/paperspace/xingguang/datasets/pre-training-faq'
model_type = '13b'


def pre_tokenize(n_thread, i_thread, texts):
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(f'meta-llama/Llama-2-{model_type}-hf')

    tokenized_datas = []
    for i, text in tqdm.tqdm(enumerate(texts)):
        if i % n_thread != i_thread:
            continue
        tokenized_datas.append(np.asarray(tokenizer.encode(text) + [tokenizer.eos_token_id], np.int64))
    pickle.dump(tokenized_datas, open(f'{work_dir}/tokenized.{model_type}/faq.part-{str(1000+i_thread)[1:]}.bin', 'wb'))


def process_faq():

    input_files = []
    input_files.extend([f'{work_dir}/quora/{filename}' for filename in os.listdir(f'{work_dir}/quora/') if 'index' in filename])
    input_files.extend([f'{work_dir}/gpt_local_qa/{filename}' for filename in os.listdir(f'{work_dir}/gpt_local_qa/') if 'index' in filename])
    print(json.dumps(input_files, indent=2))

    os.makedirs(f'{work_dir}/tokenized.{model_type}', exist_ok=True)

    texts = []
    for input_file in input_files:
        for data in open(input_file):
            obj = json.loads(data)
            text = f'{obj["title"]}\n{obj["answer"]}\n'
            texts.append(text)

    pool = multiprocessing.Pool(16)
    for i in range(16):
        pool.apply_async(func=pre_tokenize, args=(16, i, texts))
    pool.close()
    pool.join()


if __name__ == '__main__':
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(f'meta-llama/Llama-2-{model_type}-hf')

    process_faq()



