import collections
import copy
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
        if output['action'] not in {'chat', 'search'}:
            return False
        for k, v in output['slots'].items():
            if not isinstance(v, list):
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


if __name__ == '__main__':
    input_dir = '/home/paperspace/xingguang/datasets/agent_sft.v09'

    # act_tgi_svr = 'http://209.51.170.51:1308'
    act_tgi_svr = 'http://172.83.13.53:1308'
    gen_tgi_svr = 'http://209.51.170.51:1309'

    counter = collections.defaultdict(float)

    key2sample = {}
    for filename in ['dev.api.json', 'dev.casual.json']:
        for data in open(f'{input_dir}/{filename}'):
            obj = json.loads(data)
            key = f'{obj["dialog_id"]}_{obj["turn_id"]}'
            key2sample[key] = obj

    key2prediction = {}

    sout = open(f'{input_dir}/dev.pred.7b.13b.s0.act.rl.799.json', 'w')

    datas = [data for data in open(f'{input_dir}/dev.act.json')]
    for data in tqdm.tqdm(datas):
        act_obj = json.loads(data)
        key = f'{act_obj["dialog_id"]}_{act_obj["turn_id"]}'

        if key not in key2sample:
            counter['sample_is_missing'] += 1
            continue

        prompt, label = AgentActDataset.prompting(act_obj)

        output = {}
        for _ in range(3):
            try:
                output = call_tgi(prompt, act_tgi_svr)
                if is_valid_action_response(output):
                    break
            except Exception as e:
                print(e)
                continue

        if not is_valid_action_response(output):
            counter['decision_maker_error'] += 1
            continue

        act_output = json.loads(output)

        sout.write(json.dumps({
            'key': key,
            'pred_act': act_output,
            'real_act': act_obj['label']
        }) + '\n')
        sout.flush()
        counter['success'] += 1
    print(json.dumps(counter, indent=2))