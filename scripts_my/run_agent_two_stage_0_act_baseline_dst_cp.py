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
                               best_of=2,
                               max_new_tokens=500,
                               repetition_penalty=1,
                               do_sample=False)
    return response.generated_text


def is_valid_action_response(output):
    try:
        output = json.loads(output)
        if 'slots' not in output:
            return False
        for k, v in output['slots'].items():
            if not isinstance(v, list) and not isinstance(v, dict):
                return False
        return True
    except:
        return False


def is_valid_api_response(output):
    try:
        json.loads(output)
        return True
    except:
        return False


def run_one(obj, tgi_svr):
    key = f'{obj["dialog_id"]}_{obj["turn_id"]}'

    prompt, label = AgentActDataset.prompting(obj)

    output = call_tgi(prompt, tgi_svr)
    if not is_valid_action_response(output):
        return "{}"

    act_output = json.loads(output)
    return json.dumps({
            'key': key,
            'pred_act': act_output,
            'real_act': obj['label']
        })



if __name__ == '__main__':
    input_dir = '/home/paperspace/xingguang/datasets/agent_sft.woz.2.4.limit_8k.new/'
    input_dir = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/'

    tgi_servers = [
        'http://209.51.170.51:1300',
        'http://209.51.170.51:1301',
        'http://209.51.170.51:1302',
        'http://209.51.170.51:1303',
        'http://209.51.170.51:1304',
        'http://209.51.170.51:1305',
        'http://209.51.170.51:1306',
        'http://209.51.170.51:1307',
    ]

    split = 'test'

    pool = multiprocessing.Pool(len(tgi_servers))
    reses = []
    for i, data in enumerate(open(f'{input_dir}/{split}.act.json')):
        obj = json.loads(data)
        reses.append(pool.apply_async(
            func=run_one,
            args=(obj, tgi_servers[i % len(tgi_servers)])
        ))

    sout = open(f'{input_dir}/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.dedup.json', 'w')
    for res in tqdm.tqdm(reses):
        res = res.get()
        sout.write(res + '\n')
        sout.flush()
    pool.close()