import collections
import json
import os.path
import pickle
import random
import sys

import accelerate
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


logging.basicConfig(level=logging.WARN,
                    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', )

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


def generate_dialog(policy_model: transformers.models.llama.LlamaForCausalLM=None,
                    policy_tokenizer: transformers.models.llama.LlamaTokenizer=None,
                    device=None):
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
        print(f'rank = {rank}, turn = {turn_no}, User: {user_utterance}', flush=True)

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
        if policy_model is not None and policy_tokenizer is not None:
            current_device = device if device is not None else 'cuda'
            batch = tokenizer(act_prompt, return_tensors="pt")
            batch = {k: v.to(current_device) for k, v in batch.items()}
            output = policy_model.generate(
                **batch,
                max_new_tokens=300,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
                pad_token_id=policy_tokenizer.eos_token_id
            )[0]
            act_output = policy_tokenizer.decode(output, skip_special_tokens=True)[len(act_prompt):]
            print(act_output, '*' * 10)
        else:
            act_output = call_tgi(act_prompt, act_tgi_svr)
        act_output = json.loads(act_output)

        ttype = 'api_generation' if act_output['action'] == 'search' else (
            'casual_generation' if act_output['slots'] else 'casual_generation_no_slots'
        )
        print(f'rank = {rank}, turn = {turn_no}, System Act: {ttype}, detail = {act_output}', flush=True)

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
                print('.', end='', flush=True)
            print()

            # casual 的数字都很危险，替换成不确定量词
            words = gen_output.split(' ')
            for idx in range(len(words)):
                if words[idx].isdigit() and 1 < int(words[idx]) < 20:
                    words[idx] = random.choice(['some', 'many', 'several', 'various', 'multiple'])
            gen_output = ' '.join(words)

            return gen_output

        need_api = False
        if ttype != 'api_generation':
            # 反问或者闲聊
            gen_item['asked_slots'] = act_output['slots']
            print(f'rank = {rank}, turn = {turn_no}, directly chatting')
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

            # 判断 missing 的slots，方便反问，missing 的
            asked_slots = {}
            for service, slot_keys in act_output['slots'].items():
                if service not in service2field_config:
                    continue
                asked_slots[service] = [slot_key for slot_key in service2field_config[service] if slot_key not in slot_keys]

            # 根据检索结果判断是否需要强制反问
            need_ask = False
            if len(search_results.get(service, [])) >= 3 or len(search_results.get(service, [])) == 0:
                need_ask = True
            if len(search_results.get(service, [])) == 0:
                # 如果检索结果为空，反问已有槽
                asked_slots[service] = act_output['slots']
            print(f'rank = {rank}, turn = {turn_no}, System API: {api_output}, '
                  f'result size = {len(search_results.get(service, []))}', flush=True)

            if turn_no <= 6 and need_ask:
                # turn num <= 6, 也就是前三轮，如果检索结果为空，或过多，都强转 Chat
                # asked slots 是当前轮次缺少的slots
                gen_item['type'] = 'casual_generation'
                gen_item['asked_slots'] = asked_slots
                print(f'rank = {rank}, turn = {turn_no}, System API2ASK: {asked_slots}', flush=True)

                gen_output = get_gen_output(gen_item)
                if is_gen_out_no_response(gen_output) and len(search_results.get(service, [])) > 0:
                    # 转回复生成错误，需要API
                    need_api = True
                    print(f'rank = {rank}, turn = {turn_no}, chatting is invalid')
            else:
                need_api = True

            if need_api:
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

        print(f'rank = {rank}, turn = {turn_no}, System Gen: {gen_output}', flush=True)

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
        print()

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
    return used_services[0], turns, reward


def get_batch(batch_size=4,
              policy_model=None,
              policy_tokenizer=None,
              device=None):
    service, turns, reward = generate_dialog(policy_model, policy_tokenizer, device)

    factor_sum = 0.
    for turn in turns:
        if ':' in turn['turn_id'] or 'turn_type' not in turn:
            continue
        factor_sum += 1.2 if turn['turn_type'] == 'api' else 0.8
    # 最后一轮的casual chat 不能占太多分，总分 -0.6
    factor_sum -= 0.6
    factor_accu = 0.
    dialog_reward = reward.get('avg_score', 1.0)

    action = f'find_{service}'

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
            factor_accu += 0.8
            key2turns['api'].append([turn_id, factor_accu / factor_sum * dialog_reward, api_generate_turn.get('reference', [])])
            all_utterances.append(turn['speaker'] + ': ' + turn['utterance'])
            idx += 3
        else:
            if speaker == SPEAKER_SYSTEM:
                turn = turns[idx]
                turn_id = turn.get('turn_id', '')
                factor_accu += 1.2
                if idx == len(turns) - 1:
                    factor_accu -= 0.6
                key2turns['casual'].append([turn_id, factor_accu / factor_sum * dialog_reward, turn.get('asked_slots', {})])

            all_utterances.append(speaker + ': ' + utterance)
            idx += 1

    prompts, labels, rewards = [], [], []
    from ft_datasets.agent_sft_act_dataset import AgentActDataset, agent_tokenize
    for key, turns in key2turns.items():
        for turn in turns:
            turn_id = int(turn[0])
            turn_reward = turn[1]
            print(turn, flush=True)

            if key == 'api':
                action = 'search'
                slots = {k: list(v) for k, v in simplify_params(turn[-1]).items()}
            else:
                action = 'chat'
                slots = {k: list(v) for k, v in turn[-1].items()}
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

            print(prompt + label)
            print(turn_reward)

    batch = {
        'query_tensors': [],
        'response_tensors': [],
        'reward_tensors': []
    }
    index_list = list(range(len(prompts)))
    while len(index_list) <= batch_size:
        index_list += index_list
    random.shuffle(index_list)

    for idx in index_list[0:batch_size]:
        prompt, label, reward = prompts[idx], labels[idx], rewards[idx]
        example = prompt + label
        # print(prompt+label)
        prompt = policy_tokenizer.encode(prompt)
        example = policy_tokenizer.encode(example) + [policy_tokenizer.eos_token_id]

        batch['query_tensors'].append(torch.tensor(prompt))
        batch['response_tensors'].append(torch.tensor(example[len(prompt):]))
        batch['reward_tensors'].append(torch.tensor([reward]))
    return batch


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
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

        model = LlamaForCausalLM.from_pretrained('/home/paperspace/xingguang/models/agent_sft_act_dataset.7b.2e-5.full.B16.E1.hf')
        model.to('cuda')
        get_batch(4, model, tokenizer)

