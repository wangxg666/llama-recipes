import collections
import json
import os.path
import pickle
import random

import accelerate
import requests
import logging

import torch

from agent.gen_utils import *
from transformers import LlamaForCausalLM, LlamaTokenizer
from ppo_trainer import PPOTrainer
from typing import Tuple


logging.basicConfig(level=logging.WARN,
                    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', )

try:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
except:
    rank = 0
    world_size = 1
print('*' * 10, rank, '*' * 10, flush=True)


class FakePPOTrainner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.current_device = 'cuda'


def generate_dialog(ppo_trainer):
    random.shuffle(services)
    used_services = [services[0]]
    # if random.random() > 0.6:
    #     used_services.append(services[1])
    # print(f'current service is {json.dumps(used_services, indent=2)}')

    service2slots = {
        service: random.choice(db.service2db[service]) for service in used_services
    }
    service2preference = {
        service: {k: slot.get(k, '') for k in service2config[service]}
        for service, slot in service2slots.items() if service in service2config
    }

    simulater = GPTUserSimulator()
    turns = []
    turn_no, turn_no_eos = 0, -1

    for i in range(16):
        service = used_services[0]
        service2fields = {
            service: list(set(
                list(service2preference[service]) + [random.choice(service2field_config[service]) for _ in range(2)]
            )) if service in service2field_config else db.get_keys(service)
        }

        user_utterance = simulater(
            history=[turn['utterance'] for turn in turns if ':' not in turn['turn_id']],
            service2fields=service2fields,
            service2preference=service2preference,
            verbose=False
        )
        user_utterance = str(user_utterance)
        print(f'rank = {rank}, turn = {turn_no}, User: {user_utterance}', flush=True)

        turns.append({
            "turn_id": str(turn_no),
            "speaker": "USER",
            "actions": [f"{service.capitalize()}-Inform"],
            "utterance": user_utterance.replace('[EOF]', '')
        })
        turn_no += 1

        item = {
            'type': 'api_generation',
            'action': service,
            'history': [turn['utterance'] for turn in turns if ':' not in turn['turn_id']],
            'search_results': [],
            'label_type': '',
            'label': ''
        }
        prompt, label = prompting(item)

        batch = ppo_trainer.tokenizer(prompt, return_tensors="pt")
        batch = {k: v.to(ppo_trainer.current_device) for k, v in batch.items()}

        model = ppo_trainer.model
        if isinstance(ppo_trainer, PPOTrainer):
            model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
        outputs = model.generate(
            **batch,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.8,
            use_cache=True,
            pad_token_id=ppo_trainer.tokenizer.eos_token_id
        )
        api_response = ppo_trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]

        # from torch import distributed as dist
        # test 01 仅仅在这里放置一个 barrier 会死锁
        # dist.barrier()

        # test 02 删除其他dist配置，包括 import，使用 accelerator wait
        # 只使用这个配置同样会挂起
        # accelerator.wait_for_everyone()

        # test 03 使用 all reduce 显示同步
        # signal = torch.tensor([1.], device=f'cuda:{rank}')
        # print(f'rank = {rank}, {signal}')
        # dist.all_reduce(signal, op=dist.ReduceOp.SUM)
        # print(f'rank = {rank}, signal: {signal.item()}, out!!!!!', flush=True)

        parts = api_response.split('\n')
        if len(parts) != 2:
            return None, None, None
        else:
            action, answer = parts[0], '\n'.join(parts[1:]).replace('\n', ' ')
        print(f'rank = {rank}, turn = {turn_no}, action = {action}, answer = {answer}', flush=True)

        if action != 'Query':
            turns.append({
                "turn_id": str(turn_no),
                "speaker": "SYSTEM",
                "actions": [f"{service.capitalize()}-Inform"],
                "utterance": answer
            })
            # print(f'rank = {rank}, turn = {turn_no}, Agent (Casual): {answer}', flush=True)
            turn_no += 1
        else:
            # print(f'rank = {rank}, turn = {turn_no}, Agent (API): {answer}', flush=True)
            answer = parse_answer(answer)
            data = {
                "scenario": 'multiwoz',
                'api_configs': [{
                    'service': service,
                    'active_intent': f'find_{service}',
                    'slot_values': {f'{service}-{k}': [v] for k, v in answer.get(service, {}).items()}
                }]
            }
            items = requests.post(url='http://35.91.70.214:80/do_search', data=json.dumps(data)).json()

            rag_response = call_tgi_service('rag_generation', service, turns, items)
            # print(f'rank = {rank}, turn = {turn_no}, Agent (RAG): {rag_response}', flush=True)

            turns.extend([{
                    "turn_id": f"{turn_no-1}:follow_by_user_select_api",
                    "speaker": "SYSTEM",
                    "actions": [f"{service.capitalize()}-Inform"],
                    "utterance": "GenAPIConfig",
                    "reference": data['api_configs'],
                },{
                    "turn_id": f"{turn_no - 1}:follow_by_user_call_api",
                    "speaker": "SYSTEM",
                    "actions": [f"{service.capitalize()}-Inform"],
                    "utterance": "DoAPICall",
                    "reference": items,
                },{
                    "turn_id": str(turn_no),
                    "speaker": "SYSTEM",
                    "actions": [f"{service.capitalize()}-Inform"],
                    "utterance": rag_response,
            }])
            turn_no += 1

        if '[EOF]' in user_utterance:
            break

    dialog = [
        {'User' if turn['speaker'] == 'USER' else 'Agent': turn['utterance']}
        for turn in turns if ':' not in turn['turn_id']
    ]
    try:
        reward = requests.post(
            'http://35.91.70.214:80/do_reward', data=json.dumps({'dialog': dialog})
        ).json().get('report', [])
        reward = {} if not reward else reward[0]
    except:
        reward = {}
    return used_services, turns, reward


def get_data_size(data_file, num_sample):
    current_size = 0
    for r in range(world_size):
        if os.path.exists(f'{data_file}.{r}.bin'):
            current_size += len(pickle.load(open(f'{data_file}.{r}.bin', 'rb')))
    print(f'rank = {rank}, data size = {current_size}')
    return current_size >= num_sample


def generate_samples_from_dialog(ppo_trainer, out_turns, out_reward, batch_size):
    print(f'***** rank = {rank}, generation begin *****', flush=True)
    print(out_reward)
    samples = []
    reward_value = max(-1., out_reward.get('avg_score', 0.) - 5. / 5.)
    for key, datas in convert_dialog_to_task(out_turns).items():
        # if key == 'rag':
        #     continue
        for data in datas:
            prompt, label = prompting(data)
            prompt_ids = ppo_trainer.tokenizer.encode(prompt)
            label_ids = ppo_trainer.tokenizer.encode(label)[1:] + [ppo_trainer.tokenizer.eos_token_id]

            print(f'{key}, {len(prompt_ids)}, {len(label_ids)}, {ppo_trainer.tokenizer.decode(label_ids)}', flush=True)

            samples.append({
                'type': key,
                'prompt': prompt,
                'label': label,
                'prompt_input_ids': prompt_ids,       # .to(ppo_trainer.current_device),
                'label_input_ids': label_ids,         # .to(ppo_trainer.current_device),
                'rewards': [reward_value]             # .to(ppo_trainer.current_device)
            })
    return samples


if __name__ == '__main__':
    model_name_or_path = '/home/paperspace/xingguang/models/my_agent_sft_dataset.13b.2e-5.full.B4.E1.v07.all.hf'
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    print(tokenizer.decode([tokenizer.eos_token_id, tokenizer.bos_token_id]))

    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        return_dict=True,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model = model.half()

    fake_ppo_trainner = FakePPOTrainner(model, tokenizer)

    output_dir = '/home/paperspace/xingguang/datasets/ppo_test.ex'
    os.makedirs(output_dir, exist_ok=True)

    all_samples = collections.defaultdict(list)
    all_raw_datas = []
    sout_tok = open(f'{output_dir}/data.tok.json', 'a')
    sout_raw = open(f'{output_dir}/data.raw.json', 'a')
    for i in range(1000):
        out_services, out_turns, out_rewards = generate_dialog(fake_ppo_trainner)
        if not out_services:
            continue
        samples = generate_samples_from_dialog(fake_ppo_trainner, out_turns, out_rewards, -1)
        for sample in samples:
            sout_tok.write(json.dumps(sample) + '\n')
        sout_raw.write(json.dumps({'reward': out_rewards, 'dialog': out_turns}) + '\n')

        sout_raw.flush()
        sout_tok.flush()
