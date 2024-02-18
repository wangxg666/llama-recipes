import collections
import copy
import multiprocessing
import os
import json
import time

import tqdm

from ft_datasets.agent_sft_gen_dataset import AgentSFTDataset
from ft_datasets.agent_sft_act_dataset import AgentActDataset

from text_generation import Client


def call_tgi(prompt, tgi_server="http://209.51.170.51:1309"):
    client = Client(tgi_server)
    response = client.generate(prompt=prompt,
                               temperature=1.0,
                               # best_of=1,
                               max_new_tokens=500,
                               repetition_penalty=1,
                               do_sample=True)
    return response.generated_text

def get_system_response(obj, tgi_server='http://209.51.170.51:1308'):
    prompt, _ = AgentSFTDataset.prompting(obj)
    tgi_output = call_tgi(prompt, tgi_server)
    return tgi_output


if __name__ == '__main__':
    dataset = 'agent_sft.v10.baseline.dst.limit_8k'
    input_dir = f'/home/paperspace/xingguang/datasets/{dataset}/'

    import sys, traceback

    objs = []
    for input_file in [
        f'{input_dir}/test.casual.json',
        f'{input_dir}/test.rag.json',
    ]:
        objs = [json.loads(data) for data in open(input_file)]
        pool = multiprocessing.Pool(2)
        tgi_servers = [
            'http://209.51.170.51:1308',
            'http://209.51.170.51:1309',
        ]
        reses = []
        for i, obj in enumerate(objs):
            reses.append(pool.apply_async(
                func=get_system_response,
                args=(obj, tgi_servers[i % 2])
            ))

        sout = open(input_file.replace('json', 'pred.json'), 'w')
        for i in tqdm.tqdm(range(len(objs))):
            try:
                res = reses[i]
                obj = objs[i]
                res = res.get()
                key = obj['dialog_id'].lower() + '_' + str(obj['turn_id'])
                real = obj['label']
                gen = get_system_response(obj)
                out = {
                    'key': key,
                    'real_resp': real,
                    'pred_resp': gen
                }
                sout.write(json.dumps(out) + '\n')
                sout.flush()
            except Exception as e:
                print('\nException', flush=True)
                traceback.print_exc(file=sys.stdout)
                print('', flush=True)

