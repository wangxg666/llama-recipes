import argparse
import datetime
import math

import numpy as np
import requests
import tqdm
import transformers.models.llama

from agent.gen_utils import *
from transformers import LlamaForCausalLM, LlamaTokenizer

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


# act_tgi_svr = 'http://209.51.170.51:1308'
act_tgi_svr = 'http://172.83.13.53:1308'
# gen_tgi_svr = 'http://209.51.170.51:1309'
gen_tgi_svr = 'http://172.83.13.53:1309'


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


def compute_reward_weight(turns, dialog_reward):
    if dialog_reward > 0:
        dialog_reward = math.sqrt(dialog_reward)
    else:
        dialog_reward = -math.sqrt(-dialog_reward)
    turn2reward_weight = {}
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
        turn2reward_weight[turn['turn_id']] = weight
    return dialog_reward, turn2reward_weight


turn_type2count = collections.defaultdict(float)

def compute_reward_weight_v2(turns, dialog_reward):
    dialog_reward = (dialog_reward + 10) / 4
    turn2reward_weight = {}
    i = 0
    dialog_slot_keys = set()
    while i < len(turns):
        turn = turns[i]
        speaker = turn['speaker']
        utterance = turn['utterance']
        weight, record = 1., False
        if utterance == 'GenAPIConfig':
            record = True
            # search weight 固定减 0.2
            slots = simplify_params(turn['reference'])
            if i+2 >= len(turns):
                break
            turn = turns[i+2]
            turn_id = turn['turn_id']
            slots = {k: v for k, v in slots.items() if len(v) > 0}
            if not slots:
                # 检索必须要槽位，如果没有，reward惩罚最大
                weight -= 0.8
            elif len(slots) > 1:
                # 多个service检索槽位，惩罚略小
                weight -= 0.3
            else:
                # 检索，判断slot key数量
                slot_keys = list(slots.values())[0]
                missing_slot_keys = [slot_key for slot_key in dialog_slot_keys if slot_key not in slot_keys]
                if len(slot_keys) == 0:
                    # 有检索意图，但检索槽位为空
                    weight -= 0.5
                elif len(missing_slot_keys) > 0:
                    # 丢失历史检索槽位惩罚
                    weight -= 0.2 * len(missing_slot_keys)
                else:
                    # 正常检索
                    weight -= 0.1
                    dialog_slot_keys.update([x.split('-')[-1] for x in slot_keys])

                # 如果 slot key 带 `-`，降低reward
                if len([slot_key for slot_key in slot_keys if '-' in slot_key]) > 0:
                    weight -= 0.1
            turn2reward_weight[turn_id] = weight
            i += 2
        elif speaker == 'SYSTEM':
            record = True
            asked_slots = turn.get('asked_slots', {})
            asked_slots = {k: v for k, v in asked_slots.items() if len(v) > 0}
            if len(asked_slots) > 1:
                # 只有一个service，如果有多个service，reward为
                weight -= 0.5
            elif len(asked_slots) == 0:
                if i == len(turns) - 1:
                    # 最后一轮的槽位可空，其他都不应该为空，但 Chat 这个行为本身并没有过多价值
                    weight -= 0.05
                else:
                    # 中间流程，不鼓励闲聊
                    weight -= 0.2
            else:
                # 有反问意图，判断slot key数量
                slot_keys = list(asked_slots.values())[0]
                repeated_aks_slot_keys = [slot_key for slot_key in slot_keys if slot_key in dialog_slot_keys]

                if len(slot_keys) == 0:
                    # slot key 为空
                    weight -= 0.3
                elif len(repeated_aks_slot_keys) > 0:
                    # 重复反问的槽位，降低reward
                    weight -= 0.2 * len(repeated_aks_slot_keys)
                else:
                    # 正常反问，略正的 reward
                    weight += 0.5
                # 如果 slot key 带 `-`，降低reward
                if len([slot_key for slot_key in slot_keys if '-' in slot_key]) > 0:
                    weight -= 0.1
            weight = max(0., weight)
        i += 1
        if record:
            turn2reward_weight[turn['turn_id']] = weight
    return dialog_reward, turn2reward_weight


weight2count = collections.defaultdict(float)


def parse_dialog(turns, reward, batch_size, policy_tokenizer):
    dialog_reward, turn2reward_weight = compute_reward_weight_v2(turns, reward.get('avg_score', 0.))

    for weight in turn2reward_weight.values():
        weight2count[str(weight)] += 1

    weight_sum = math.sqrt(sum([x**2 for x in turn2reward_weight.values()]))
    if weight_sum == 0.:
        return {}

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
            turn_weight = turn2reward_weight[turn_id]
            key2turns['api'].append(
                [turn_id, turn_weight / weight_sum * dialog_reward, api_generate_turn.get('reference', [])]
            )
            all_utterances.append(turn['speaker'] + ': ' + turn['utterance'])
            idx += 3
            # print(turn_id, turn_weight, turn_weight / weight_sum * dialog_reward)
        else:
            if speaker == SPEAKER_SYSTEM:
                turn = turns[idx]
                turn_id = turn.get('turn_id', '')
                turn_weight = turn2reward_weight[turn_id]
                key2turns['casual'].append(
                    [turn_id, turn_weight / weight_sum * dialog_reward, turn.get('asked_slots', {})]
                )
                # print(turn_id, turn_weight, turn_weight / weight_sum * dialog_reward)
            all_utterances.append(speaker + ': ' + utterance)
            idx += 1

    chat_prompts, chat_labels, chat_rewards = [], [], []
    search_prompts, search_labels, search_rewards = [], [], []

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

            data = {
                'dialog_id': '',
                'turn_id': turn_id,
                'type': 'act_selection',
                'action': action,
                'history': all_utterances[0: turn_id],
                'label': {'action': action, 'slots': slots}
            }

            prompt, label = AgentActDataset.prompting(data)

            if key == 'casual':
                chat_prompts.append(prompt)
                chat_labels.append(label)
                chat_rewards.append(turn_reward)
            else:
                search_prompts.append(prompt)
                search_labels.append(label)
                search_rewards.append(turn_reward)

    batch = {
        'query_tensors': [],
        'response_tensors': [],
        'reward_tensors': []
    }

    # 强制replace过轮次的数据，不用首保chat
    search_labels.extend(chat_labels)
    search_prompts.extend(chat_prompts)
    search_rewards.extend(chat_rewards)

    index_list = list(range(len(search_prompts)))
    if batch_size != -1:
        need_size = batch_size - len(batch['query_tensors'])
        while len(index_list) < need_size:
            index_list += index_list
        random.shuffle(index_list)
        index_list = index_list[0: need_size]

    for idx in index_list:
        search_prompt, search_label, search_reward = search_prompts[idx], search_labels[idx], search_rewards[idx]
        example = search_prompt + search_label

        prompt = policy_tokenizer.encode(search_prompt)
        example = policy_tokenizer.encode(example) + [policy_tokenizer.eos_token_id]

        batch['query_tensors'].append(torch.tensor(prompt))
        batch['response_tensors'].append(torch.tensor(example[len(prompt):]))
        batch['reward_tensors'].append(torch.tensor([search_reward]))
    return batch


cache = Cache()


if __name__ == '__main__':
    pass
