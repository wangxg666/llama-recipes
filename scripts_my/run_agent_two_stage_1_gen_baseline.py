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
        if 'current_service' not in output or 'slots' not in output:
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
    input_dir = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2/'

    # act_tgi_svr = 'http://209.51.170.51:1308'
    # act_tgi_svr = 'http://172.83.13.53:1308'
    gen_tgi_svr = 'http://209.51.170.51:1308'
    # gen_tgi_svr = 'http://172.83.13.53:1309'

    for split in ['dev', 'test']:
        counter = collections.defaultdict(float)

        key2sample = {}
        for filename in [f'{split}.api.json']:
            for data in open(f'{input_dir}/{filename}'):
                obj = json.loads(data)
                key = f'{obj["dialog_id"]}_{obj["turn_id"]}'
                key2sample[key] = obj

        key2prediction = {}

        key2act = {}
        for data in open(f'{input_dir}/{split}.act.pred.7b.json'):
            obj = json.loads(data)
            key2act[obj['key']] = obj['pred_act']

        sout = open(f'{input_dir}/{split}.gen.pred.7b_13b.json', 'w')

        datas = [data for data in open(f'{input_dir}/{split}.act.json')]
        for data in tqdm.tqdm(datas):
            act_obj = json.loads(data)
            key = f'{act_obj["dialog_id"]}_{act_obj["turn_id"]}'

            if key not in key2sample:
                counter['sample_is_missing'] += 1
                continue

            act_output = key2act[key]

            ttype = 'api_generation'
            gen_obj = copy.deepcopy(act_obj)
            gen_obj['type'] = ttype
            gen_obj['label'] = act_output['slots']

            prompt, _ = AgentSFTDataset.prompting(gen_obj)
            # print(prompt)
            output = call_tgi(prompt, gen_tgi_svr)
            # print(output)

            if not is_valid_api_response(output):
                counter['generation_api_error'] += 1
                continue
            gen_output = json.loads(output)

            sout.write(json.dumps({
                'real_gen': {
                    'type': key2sample[key]['type'],
                    'label': key2sample[key]['label'],
                },
                'pred_gen': {
                    'type': gen_obj['type'],
                    'label': gen_output,
                }
            }) + '\n')
            sout.flush()
            counter['success'] += 1
        print(json.dumps(counter, indent=2))