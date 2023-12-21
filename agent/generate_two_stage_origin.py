import collections
import datetime
import json
import math
import os.path
import pickle
import random
import sys

import accelerate
import numpy as np
import requests
import logging

import torch
import tqdm
import transformers.models.llama

from agent.gen_utils import *
from transformers import LlamaForCausalLM, LlamaTokenizer
from ppo_trainer import PPOTrainer

from ft_datasets.agent_sft_gen_dataset import AgentSFTDataset
from ft_datasets.agent_sft_act_dataset import AgentActDataset, agent_tokenize
import torch.distributed as dist


def print_rank_0(*args):
    try:
        if dist.get_rank() == 0:
            print(*args, flush=True)
    except:
        print(*args, flush=True)


def call_tgi(prompt, tgi_server="http://209.51.170.51:1309"):
    client = Client(tgi_server)
    response = client.generate(prompt=prompt,
                               temperature=1.0,
                               best_of=2,
                               max_new_tokens=500,
                               repetition_penalty=1,
                               do_sample=False)
    return response.generated_text


def call_tgi_random(prompt, tgi_server="http://209.51.170.51:1309"):
    client = Client(tgi_server)
    response = client.generate(prompt=prompt,
                               temperature=0.5,
                               max_new_tokens=500,
                               repetition_penalty=1,
                               do_sample=True)
    return response.generated_text


def validate_action_response(output):
    try:
        if output['action'] not in {'chat', 'search'}:
            return {"action": "chat", "slots": {}}
        if 'slots' not in output or not isinstance(output['slots'], dict):
            return {"action": "chat", "slots": {}}
        for k in output['slots']:
            vs = output['slots'][k]
            if not isinstance(vs, list):
                del output['slots'][k]
            else:
                output['slots'][k] = [v.split('-')[-1] for v in output['slots'][k]]
        return output
    except:
        return {"action": "chat", "slots": {}}


def is_valid_api_response(output):
    try:
        json.loads(output)
        return True
    except:
        return False


def is_gen_out_no_response(response:str):
    response = response.lower()
    if ('sorry' in response
            or 'there is no' in response
            or 'there are no' in response
            or 'do not have' in response
            or "don't have" in response
        or 'unfortunately' in response
        or 'sorry' in response
        or 'apologize' in response
        or 'unable' in response
        or ' no ' in response
        or ' not ' in response
    ):
        return True
    return False


act_tgi_svr = 'http://209.51.170.51:1308'
gen_tgi_svr = 'http://209.51.170.51:1309'


class Cache:
    def __init__(self):
        self.root = '/home/paperspace/xingguang/datasets/ppo_cache/'
        self.hour = datetime.datetime.now().strftime('%Y-%m-%d_%H')
        self.gpu = torch.cuda.current_device()
        self.sout = open(f'{self.root}/{self.hour}.{self.gpu}.txt', 'a')

    def write(self, data):
        hour = datetime.datetime.now().strftime('%Y-%m-%d_%H')
        if hour != self.hour:
            if self.sout:
                self.sout.close()
            self.hour = hour
            self.sout = open(f'{self.root}/{self.hour}.{self.gpu}.txt', 'a')
        self.sout.write(data)
        self.sout.flush()

cache = Cache()

def generate_dialog(policy_model: transformers.models.llama.LlamaForCausalLM=None,
                    policy_tokenizer: transformers.models.llama.LlamaTokenizer=None,
                    device=None):
    random.shuffle(services)
    used_services = [services[0]]
    print_rank_0(f'current service is {json.dumps(used_services, indent=2)}')

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

    n_act_call, n_act_tgi_call = 0., 0.
    for i in range(8):
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
        print_rank_0(f'rank = {rank}, turn = {turn_no}, User: {user_utterance}')

        turns.append({
            "turn_id": str(turn_no),
            "speaker": "USER",
            "actions": [f"{service.capitalize()}-Inform"],
            "utterance": user_utterance.replace('[EOF]', '')
        })
        turn_no += 1

        act_item = {
            'type': 'act_selection',
            'action': service,
            'history': [turn['utterance'] for turn in turns if ':' not in turn['turn_id']],
            'label': ''
        }
        act_prompt, _ = AgentActDataset.prompting(act_item)
        n_act_call += 1
        if policy_model is not None and policy_tokenizer is not None:
            current_device = device if device is not None else 'cuda'
            batch = policy_tokenizer(act_prompt, return_tensors="pt")
            batch = {k: v.to(current_device) for k, v in batch.items()}
            output = policy_model.generate(
                **batch,
                max_new_tokens=300,
                temperature=1.0,
                use_cache=True,
                repetition_penalty=1.,
                pad_token_id=policy_tokenizer.eos_token_id
            )[0]
            act_output = policy_tokenizer.decode(output, skip_special_tokens=True)[len(act_prompt):]

        else:
            act_output = call_tgi(act_prompt, act_tgi_svr)

        try:
            act_output = json.loads(act_output)
            print_rank_0(f'rank = {rank}, turn = {turn_no}', '*' * 10, act_output, '*' * 10)
        except:
            act_output_r = call_tgi(act_prompt, act_tgi_svr)
            print_rank_0(f'rank = {rank}, turn = {turn_no}', '*' * 10, act_output, '*' * 10)
            print_rank_0(f'rank = {rank}, turn = {turn_no}', '+' * 10, act_output_r, '+' * 10)
            act_output = json.loads(act_output_r)
            n_act_tgi_call += 1.

        act_output = validate_action_response(act_output)

        ttype = 'api_generation' if act_output['action'] == 'search' else (
            'casual_generation' if act_output['slots'] else 'casual_generation_no_slots'
        )
        print_rank_0(f'rank = {rank}, turn = {turn_no}, System Act: {ttype}, detail = {act_output}')

        gen_item = {
            'type': ttype,
            'action': service,
            'history': [turn['utterance'] for turn in turns if ':' not in turn['turn_id']],
            'label': act_output['slots'],
        }

        def get_gen_output(gen_item):
            gen_prompt, _ = AgentSFTDataset.prompting(gen_item)
            gen_output = call_tgi_random(gen_prompt, gen_tgi_svr)

            for _ in range(5):
                gen_output = call_tgi_random(gen_prompt, gen_tgi_svr)
                # if gen_output.endswith('?') and '.' in gen_output:
                #     gen_output = gen_output.split('.')[-1].strip()
                if not is_gen_out_no_response(gen_output.lower()):
                    break

            # casual 的数字都很危险，替换成不确定量词
            if gen_item['type'] in {'casual_generation', 'casual_generation_no_slots'}:
                words = gen_output.split(' ')
                for idx in range(len(words)):
                    if words[idx].isdigit() and 1 < int(words[idx]) < 20:
                        if idx + 1 < len(words) and words[idx+1] == 'minutes':
                            continue
                        words[idx] = random.choice(['some', 'many', 'several', 'various', 'multiple'])
                gen_output = ' '.join(words)
            return gen_output

        need_api = False
        if ttype != 'api_generation':
            # 反问或者闲聊
            gen_item['asked_slots'] = act_output['slots']
            print_rank_0(f'rank = {rank}, turn = {turn_no}, directly chatting')
            gen_output = get_gen_output(gen_item)

        else:
            gen_item['label'] = act_output['slots']
            api_prompt, _ = AgentSFTDataset.prompting(gen_item)
            api_output = call_tgi(api_prompt, gen_tgi_svr)
            api_output = json.loads(api_output)
            req_data = {
                "scenario": 'multiwoz',
                'api_configs': [{
                    'service': service,
                    'active_intent': f'find_{service}',
                    'slot_values': {f'{service}-{k}': [v] for k, v in api_output.get(service, {}).items()}
                }]
            }
            search_results = requests.post(url='http://35.91.154.68:80/do_search', data=json.dumps(req_data)).json()

            turns.append({
                "turn_id": f'{str(turn_no-1)}::follow_by_user_select_api',
                "speaker": "SYSTEM",
                "actions": [f"{service.capitalize()}-Inform"],
                "utterance": 'GenAPIConfig',
                'reference': req_data['api_configs']
            })
            turns.append({
                "turn_id": f"{turn_no - 1}:follow_by_user_call_api",
                "speaker": "SYSTEM",
                "actions": [f"{service.capitalize()}-Inform"],
                "utterance": "DoAPICall",
                "reference": search_results,
            })
            gen_item['type'] = 'rag_generation'
            gen_item['search_results'] = search_results
            gen_output = get_gen_output(gen_item)

        print_rank_0(f'rank = {rank}, turn = {turn_no}, System Gen: {gen_output}')

        turns.append({
            "turn_id": str(turn_no),
            "speaker": "SYSTEM",
            "actions": [f"{service.capitalize()}-Inform"],
            "utterance": gen_output,
            "turn_type": 'api' if need_api else 'casual'
        })
        if 'asked_slots' in gen_item:
            turns[-1]['asked_slots'] = gen_item['asked_slots']

        turn_no += 1

        if '[EOF]' in user_utterance:
            break

    dialog = [
        {'User' if turn['speaker'] == 'USER' else 'Agent': turn['utterance']}
        for turn in turns if ':' not in turn['turn_id']
    ]
    try:
        reward = requests.post(
            'http://35.91.154.68:80/do_reward', data=json.dumps({'dialog': dialog})
        ).json().get('report', [])
        reward = {} if not reward else reward[0]
    except:
        reward = {}
    print(f'rank = {rank}, {n_act_tgi_call} of {n_act_call} act generation failed')
    return used_services[0], turns, reward

def parse_dialog(turns, reward, batch_size, policy_tokenizer):
    dialog_reward = reward.get('avg_score', 0.)
    if dialog_reward > 0:
        dialog_reward = math.sqrt(dialog_reward)
    else:
        dialog_reward = -math.sqrt(-dialog_reward)

    turn2weight = {}
    n_gen_turn, n_api_turn = 0., 0.
    for i, turn in enumerate(turns):
        if ':' in turn['turn_id'] or 'turn_type' not in turn:
            continue

        if turn['turn_type'] == 'api':
            if i >= 2 and turns[i-2]['utterance'] == 'GenAPIConfig':
                slots = simplify_params(turns[i-2]['reference'])
                slots = {k: v for k, v in slots.items() if len(v) > 0}
                if not slots:
                    # 无槽位的搜索，属于严重错误，永远都是负的reward
                    weight = -0.8 if dialog_reward > 0 else 0.8
                else:
                    # 整体 reward 为负时，这一轮有极低的正向 reward 权重
                    weight = 0.8 if dialog_reward > 0 else -0.1
            else:
                weight = 1.
        else:
            asked_slots = turn.get('asked_slots', {})
            asked_slots = {k: v for k, v in asked_slots.items() if len(v) > 0}
            if not asked_slots and i != len(turns) - 1:
                # 最后一轮的槽位可空，其他都不应该为空，但 Chat 这个行为本身值得鼓励
                weight = -0.4 if dialog_reward > 0 else 0.4
            else:
                weight = 1.2 if dialog_reward else -0.2
        turn2weight[turn['turn_id']] = weight

        if turn['turn_type'] == 'api':
            n_api_turn += 1.
        else:
            n_gen_turn += 1.
    # 最后一轮的casual chat 不能占太多分，总分 -0.6
    factor_sum = math.sqrt(sum([x**2 for x in turn2weight.values()]))
    if factor_sum == 0.:
        return {}
    # scale = (0.5 * n_api_turn + n_gen_turn) / (n_api_turn + n_gen_turn)
    # dialog_reward = dialog_reward * scale
    # print_rank_0(f'reward is {round(reward.get("avg_score", 1.), 5)}, scale = {round(scale, 5)}, dialog reward = {round(dialog_reward, 5)}')
    # print_rank_0(f'reward is {round(reward.get("avg_score", 1.), 5)}')

    key2turns = collections.defaultdict(list)

    all_utterances = []
    idx = 0
    while idx < len(turns):
        speaker = turns[idx].get('speaker', '')
        utterance = turns[idx].get('utterance', '')

        if utterance == GEN_API_CONFIG:
            api_generate_turn = turns[idx]
            turn = turns[idx + 2]
            turn_id = turn.get('turn_id', '')
            turn_factor = turn2weight[turn_id]
            key2turns['api'].append(
                [turn_id, turn_factor / factor_sum * dialog_reward, api_generate_turn.get('reference', [])]
            )
            all_utterances.append(turn['speaker'] + ': ' + turn['utterance'])
            idx += 3
            # print(turn_id, turn_factor, turn_factor / factor_sum * dialog_reward)
        else:
            if speaker == SPEAKER_SYSTEM:
                turn = turns[idx]
                turn_id = turn.get('turn_id', '')
                turn_factor = turn2weight[turn_id]
                key2turns['casual'].append(
                    [turn_id, turn_factor / factor_sum * dialog_reward, turn.get('asked_slots', {})]
                )
                # print(turn_id, turn_factor, turn_factor / factor_sum * dialog_reward)
            all_utterances.append(speaker + ': ' + utterance)
            idx += 1

    prompts, labels, rewards = [], [], []
    from ft_datasets.agent_sft_act_dataset import AgentActDataset
    for key, turns in key2turns.items():
        for turn in turns:
            turn_id = int(turn[0])
            turn_reward = turn[1]

            if key == 'api':
                action = 'search'
                slots = {k: list(v) for k, v in simplify_params(turn[-1]).items()}
            else:
                action = 'chat'
                slots = {k: list(v) for k, v in turn[-1].items()}
            # print_rank_0(f'{turn_id}, {key} = {slots}')

            data = {
                'dialog_id': '',
                'turn_id': turn_id,
                'type': 'act_selection',
                'action': action,
                'history': all_utterances[0: turn_id],
                'label': {'action': action, 'slots': slots}
            }

            prompt, label = AgentActDataset.prompting(data)
            prompts.append(prompt)
            labels.append(label)
            rewards.append(turn_reward)

    # mean_reward = np.mean(rewards)
    # std_reward = np.std(rewards)
    # rewards = [(v - mean_reward) / std_reward for v in rewards]

    batch = {
        'query_tensors': [],
        'response_tensors': [],
        'reward_tensors': []
    }
    index_list = list(range(len(prompts)))
    if batch_size != -1:
        while len(index_list) <= batch_size:
            index_list += index_list
        random.shuffle(index_list)
        index_list = index_list[0: batch_size]

    for idx in index_list:
        prompt, label, reward = prompts[idx], labels[idx], rewards[idx]
        example = prompt + label

        prompt = policy_tokenizer.encode(prompt)
        example = policy_tokenizer.encode(example) + [policy_tokenizer.eos_token_id]

        # query_tensor = query_tensors[idx]
        # response_tensor = response_tensors[idx]
        # print('gen:', response_tensor)
        # print('gen:', policy_tokenizer.decode(response_tensor))
        # print('tok:', example[len(prompt):])
        # print('tok:', label)

        batch['query_tensors'].append(torch.tensor(prompt))
        batch['response_tensors'].append(torch.tensor(example[len(prompt):]))
        batch['reward_tensors'].append(torch.tensor([reward]))
    return batch


def get_batch(batch_size=4,
              policy_model=None,
              policy_tokenizer=None,
              device=None):
    service, turns, reward = generate_dialog(policy_model, policy_tokenizer, device)
    cache.write(json.dumps({
        'dialog': turns,
        'reward': reward
    }) + '\n')
    return parse_dialog(turns, reward, batch_size, policy_tokenizer)



if __name__ == '__main__':
    if len(sys.argv) > 1:
        output_dir = '/home/paperspace/xingguang/datasets/agent_two_stage_auto_gen'
        output_file = f'{output_dir}/{sys.argv[1]}'
        os.makedirs(output_dir, exist_ok=True)

        all_samples = collections.defaultdict(list)
        all_raw_datas = []
        sout = open(f'{output_file}', 'w')
        for i in tqdm.tqdm(range(500)):
            out_services, out_turns, out_rewards = generate_dialog()
            sout.write(json.dumps({
                'dialog': out_turns,
                'reward': out_rewards
            }) + '\n')
    else:
        model_name_or_path = 'meta-llama/Llama-2-13b-hf'
        policy_tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

        # policy_model = LlamaForCausalLM.from_pretrained('/home/paperspace/xingguang/models/agent_sft_act_dataset.7b.2e-5.full.B16.E1.hf')
        # policy_model.to('cuda')
        # get_batch(4, policy_model, policy_tokenizer)

        critic_pre_train_dir = '/home/paperspace/xingguang/datasets/ppo_cache'
        datas = []
        for filename in os.listdir(critic_pre_train_dir):
            datas.extend([json.loads(line) for line in open(f'{critic_pre_train_dir}/{filename}')])
        # for data in datas:
        #     parse_dialog(data['dialog'], data['reward'], batch_size=-1, policy_tokenizer=policy_tokenizer)

        train_datas = collections.defaultdict(list)
        for data in tqdm.tqdm(datas):
            batch_input = parse_dialog(data['dialog'], data['reward'], batch_size=-1, policy_tokenizer=policy_tokenizer)
            for key, val in batch_input.items():
                train_datas[key].extend(val)
        print_rank_0(f'load {len(train_datas["query_tensors"])} training datas')

        idxs = [i for i in range(len(train_datas['query_tensors']))]
        random.shuffle(idxs)

        rewards = [x.item() for x in train_datas["reward_tensors"]]
        print(f'reward mean = {np.mean(rewards)}, std = {np.std(rewards)}, max = {np.max(rewards)}, min = {np.min(rewards)}')

        lengths = [len(q.tolist() + r.tolist()) for q, r in zip(train_datas['query_tensors'], train_datas['response_tensors'])]
        print(f'length mean = {np.mean(lengths)}, std = {np.std(lengths)}, max = {np.max(lengths)}, min = {np.min(lengths)}')


        batch_size = 4
        eos = (len(idxs) // batch_size) * batch_size
        for bos in range(0, eos, batch_size):
            batch_input = {
                k: [v[idx] for idx in idxs[bos: bos + 4]]
                for k, v in train_datas.items()
            }
            query_tensors = batch_input['query_tensors']
            response_tensors = batch_input['response_tensors']
            reward_tensors = batch_input['reward_tensors']

