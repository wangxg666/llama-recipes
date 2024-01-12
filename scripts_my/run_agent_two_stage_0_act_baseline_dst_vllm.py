import collections
import json

import asyncio
import aiohttp
from tqdm.asyncio import tqdm

from ft_datasets.agent_sft_act_dataset import AgentActDataset

async def call_vllm(sem, pbar, idx, prompt, vllm_svr):
    req_obj = {
        "prompt": prompt,
        "best_of": 2,
        "temperature": 0.0,
        "max_tokens": 500,
        "use_beam_search": True
    }

    async with sem:
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{vllm_svr}/generate', json=req_obj) as response:
                pbar.update(1)
                return idx, await response.text()


def is_valid_action_response(output):
    try:
        output = json.loads(output)
        if 'current_service' not in output or 'slots' not in output:
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
    
async def run_generations(vllm_svr, datas):
    generation_tasks = []
    sem = asyncio.Semaphore(30)
    pbar = tqdm(total = len(datas))
    
    for idx, data in enumerate(datas):
        act_obj = json.loads(data)
        key = f'{act_obj["dialog_id"]}_{act_obj["turn_id"]}'

        if key not in key2sample:
            counter['sample_is_missing'] += 1
            continue

        prompt, _ = AgentActDataset.prompting(act_obj)

        generation_tasks.append(asyncio.create_task(call_vllm(sem, pbar, idx, prompt, vllm_svr)))
        
    generations = await asyncio.gather(*generation_tasks)
    generations = sorted(generations, key= lambda x : x[0])

    return generations


def run(vllm_svr, output_file):
    open(output_file, "w").close()
    sout = open(output_file, "a")
    # debug
    datas = [data for data in open(f'{input_dir}/{split}.act.json')][:10]
    generations = asyncio.run(run_generations(vllm_svr, datas))

    for data_idx, output in generations:
        if not is_valid_action_response(output):
            counter['decision_maker_error'] += 1
            continue

        act_output = json.loads(output)

        sout.write(json.dumps({
            'key': key,
            'pred_act': act_output,
            'real_act': datas[data_idx]['label']
        }) + '\n')
        sout.flush()
        counter['success'] += 1
    print(vllm_svr, json.dumps(counter, indent=2))
    sout.close()


if __name__ == '__main__':
    input_dir = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.1.dst/'

    for split in ['dev', 'test']:
        key2sample = {}
        for filename in [f'{split}.api.json']:
            for data in open(f'{input_dir}/{filename}'):
                obj = json.loads(data)
                key = f'{obj["dialog_id"]}_{obj["turn_id"]}'
                key2sample[key] = obj

        counter = collections.defaultdict(float)
        vllm_svr2output_file = {
            'http://0.0.0.0:8000': f'debug.1_12_2023.json',
            # 'http://209.51.170.51:1301': f'/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e02/{split}.act.pred.7b.json',
            # 'http://209.51.170.51:1302': f'/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e03/{split}.act.pred.7b.json',
            # 'http://209.51.170.51:1303': f'/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e04/{split}.act.pred.7b.json',
            # 'http://209.51.170.51:1304': f'/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k.e01/{split}.act.pred.7b.json',
            # 'http://209.51.170.51:1305': f'/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k.e02/{split}.act.pred.7b.json',
            # 'http://209.51.170.51:1306': f'/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k.e03/{split}.act.pred.7b.json',
            # 'http://209.51.170.51:1307': f'/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k.e04/{split}.act.pred.7b.json',
        }

        for vllm_svr, output_file in vllm_svr2output_file.items():
            run(vllm_svr, output_file)