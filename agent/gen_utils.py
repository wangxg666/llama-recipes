import json
import os
import torch
import collections
import random

from text_generation import Client
from agent.gpt_base import GPTBase
from agent.db import DataBase


try:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
except:
    rank = 0
    world_size = 1


GEN_API_CONFIG = 'GenAPIConfig'
DO_API_CALL = 'DoAPICall'

SPEAKER_USER = 'USER'
SPEAKER_SYSTEM = 'SYSTEM'


PERSONA_PROMPT_DICT = {
    'attraction': (
        'You are a community outreach coordinator, engaging with locals and tourists alike to promote the rich heritage and attractions of the area.'
    ),
    'hotel': (
        'You are a staff member responsible for hotel reservations at a local travel agency. You understand the unique features of each local hotel and can quickly find the hotel that meets users\' preferences based on their needs.'
    ),
    'train': (
        'You are a ticket seller at the local train ticket sales center. You work diligently, have a friendly personality, and are very skilled at assisting passengers inquiring about and purchasing train tickets.'
    ),
    'restaurant': (
        'You are a locally renowned food critic who has tried almost every restaurant in the area. Whenever someone consults you for restaurant information, you are always able to respond enthusiastically and accurately.'
    ),
    'default': (
        'You are a local 114 operator, primarily handling local services such as contacting the police, finding hospitals, calling taxis, and other convenient services. Your service is efficient and of high quality, earning widespread praise from the local community.'
    )
}

ANSWER_TYPE_PROMPT = {
    'casual_generation': (
        '{persona}\n'
        'Given the conversion history, please firstly to decide what to do next. Specifically:\n'
        '- Normal, if the user\' request can be answered without querying the database, please response directly.\n'
        '- Query, if a database query is needed to answer the user\'s request, please output the specific query parameters.\n'
        'The first word you provide should represent the action you are going to perform.\n'
        'Then, give your response based on your action decision.\n'
        'Here is the conversion history:\n{history}\n'
        'the user lastest utterence: \n{user_utterence}\n'
        'Please give your output:\n'
    ),
    'rag_generation': (
        '{persona}\n'
        'Given the conversion history, user utterance, and query result from database, '
        'please generate a appropriate answer based on the give conversion status.\n'
        'Here is the conversion history:\n{history}\n'
        'query result:\n{search_results}\n'
        'the user lastest utterence: \n{user_utterence}\n'
        'Please give your output:\n'
    )
}
ANSWER_TYPE_PROMPT['api_generation'] = ANSWER_TYPE_PROMPT['casual_generation']




service2config = {
    'attraction': ['area', 'type'],
    'restaurant': ['area', 'food', 'pricerange', 'type'],
    'hotel': ['area', 'internet', 'parking', 'pricerange', 'stars', 'type'],
    'train': ['departure', 'destination', 'day'],
}


service2field_config = {
    'attraction': ['area', 'type', 'name', 'entrance fee', 'openhours'],
    'restaurant': ['area', 'food', 'pricerange', 'type', 'address', 'introduction', 'phone', 'postcode'],
    'hotel': ['area', 'internet', 'parking', 'pricerange', 'stars', 'type', 'address', 'phone', 'postcode'],
    'train': ['departure', 'destination', 'day', 'arriveby', 'leaveat', 'price', 'trainid', 'duration'],
}

service2slot_ask = {
    'attraction': ['area', 'type', 'name'],
    'restaurant': ['area', 'food', 'pricerange', 'type'],
    'hotel': ['area', 'internet', 'parking', 'pricerange', 'stars', 'type'],
    'train': ['departure', 'destination', 'day', 'arriveby', 'leaveat'],
}

service2prompt = {
    'attraction': ['a place to go', 'a trip', 'local attractions'],
    'restaurant': ['a place to eat', 'a restaurant'],
    'hotel': ['some palces to stay', 'a hotel'],
    'train': ['a train to ']
}

services2ex_prompt = {
    'restaurant': (
        "- Do not ask the agent about the restaurant's dishes; he also doesn't know.\n"
        "- If you find this restaurant good, please try to make a reservation for a table and provide the reservation time and the number of people."
    )
}


services = [
    'attraction', 'restaurant', 'hotel', 'train',
]

db = DataBase()


def prompting(item):
    type = item['type']
    history = [x.replace('USER', 'user').replace('SYSTEM', 'you') for x in item['history']]
    history = [x.replace('Tom', 'user').replace('agent', 'you') for x in item['history']]

    if type == 'api_generation':
        persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
        prompt = ANSWER_TYPE_PROMPT[type].format(
            persona=persona,
            history=json.dumps(history[0:-1], indent=2),
            user_utterence=history[-1].replace('user: ', '')
        )
        label = item['label_type'] + '\n' + json.dumps(item['label'], separators=(',', ':'))

    elif type == 'casual_generation':
        persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
        prompt = ANSWER_TYPE_PROMPT[type].format(
            persona=persona,
            history=json.dumps(history[0:-1], indent=2),
            user_utterence=history[-1].replace('user: ', '')
        )
        label = item['label_type'] + '\n' + item['label']

    else:
        persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
        prompt = ANSWER_TYPE_PROMPT[type].format(
            persona=persona,
            history=json.dumps(history[0:-1], indent=2),
            user_utterence=history[-1].replace('user: ', ''),
            search_results=json.dumps(item['search_results'], separators=(',', ':'))
        )
        label = item['label']
    return prompt, label


def get_action(actions, all_actions):
    output = 'default'
    for action in actions:
        output = action.split('-')[0].lower()
        if output != 'booking' and output != 'general':
            break
        if 'Recommend' in action:
            # recommend 是最高优先级，排他
            break
    if output == 'booking':
        for actions in all_actions[::-1]:
            for action in actions:
                output = action.split('-')[0].lower()
                if output != 'booking' and output != 'general':
                    return output
    return output


def simplify_params(api_params_list):
    out = {}
    for api_params in api_params_list:
        service  = api_params['service']
        slots = {
            k.replace(f'{service}-', ''): v[0] for k, v in api_params['slot_values'].items() if v
        }
        out[service] = slots
    return out


def convert_dialog_to_task(turns):
    api_generate_turns = []
    rag_generate_turns = []
    casual_generate_turns = []

    # reward = max(-1, reward.get('history_scores', [0])[0] - 5 / 5)
    # api_turns = len([x for x in turns if x['utterence'] == 'GenAPIConfig'])
    # sys_turns = len([x for x in turns if x['speaker'] == 'USER'])

    all_actions = []
    all_utterances = []
    idx = 0
    while idx < len(turns):
        speaker = turns[idx].get('speaker', '')
        utterance = turns[idx].get('utterance', '')

        if utterance == GEN_API_CONFIG:
            api_generate_turn = turns[idx]
            api_call_turn = turns[idx+1]
            turn = turns[idx+2]
            turn_id = turn.get('turn_id', '')
            api_generate_turns.append([
                turn_id, turn.get('actions', []),
                api_generate_turn.get('reference', [])
            ])
            rag_generate_turns.append([
                turn_id, turn.get('actions', []),
                {
                    k: v.get('search results')
                    for k, v in api_call_turn.get('reference', {}).items()
                    if 'search results' in v
                },
            ])
            all_actions.append(turn.get('actions', []))
            all_utterances.append(turn['speaker'] + ': ' + turn['utterance'])
            idx += 3
        else:
            if speaker == SPEAKER_SYSTEM:
                turn = turns[idx]
                turn_id = turn.get('turn_id', '')
                casual_generate_turns.append([
                    turn_id, turn.get('actions', [])
                ])
                all_actions.append(turn.get('actions', []))
            all_utterances.append(speaker + ': ' + utterance)
            idx += 1

    api_generate_datas = []
    for api_generate_turn in api_generate_turns:
        turn_id, actions, api_params_list = api_generate_turn
        turn_id = int(turn_id)

        api_generate_datas.append({
            'turn_id': turn_id,
            'type': 'api_generation',
            'action': get_action(actions, all_actions),
            'label_type': 'Query',
            'history': all_utterances[0: turn_id],
            'label': simplify_params(api_params_list)
        })

    casual_generation_datas =[]
    for casual_generate_turn in casual_generate_turns:
        turn_id, actions = casual_generate_turn
        turn_id = int(turn_id)
        casual_generation_datas.append({
            'turn_id': turn_id,
            'type': 'casual_generation',
            'action': get_action(actions, all_actions),
            'label_type': 'Normal',
            'history': all_utterances[0: turn_id],
            'label': all_utterances[turn_id].replace("SYSTEM: ", "")
        })

    rag_generate_datas = []
    for rag_generate_turn in rag_generate_turns:
        turn_id, actions, search_results = rag_generate_turn
        turn_id = int(turn_id)
        rag_generate_datas.append({
            'turn_id': turn_id,
            'type': 'rag_generation',
            'action': get_action(actions, all_actions),
            'history': all_utterances[0: turn_id],
            'search_results': search_results,
            'label': all_utterances[turn_id].replace("SYSTEM: ", "")
        })

    return {
        'api': api_generate_datas,
        'rag': rag_generate_datas,
        'casual': casual_generation_datas,
    }


class GPTUserSimulator(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4')

    def prompting(self, service2fields, service2preference, history, **kwargs):
        service_templates = []
        for service, preference in service2preference.items():
            if service not in service2prompt:
                continue
            if service == 'train':
                template = service2prompt[service][0] + preference['destination']
                service_templates.append(template)
            else:
                service_templates.append(random.choice(service2prompt[service]))
        service_prompt = service_templates[-1]
        if len(service_templates) > 1:
            service_prompt = ', '.join(service_templates[0:-1]) + ' and ' + service_prompt

        return (
            f"Tom and his firends are the first time to here, and want to find {service_prompt}.\n"
            f"Tom is currently chatting with a local guide online. \n"
            f"Here is the Tom's preference:\n{json.dumps(service2preference, indent=2)}\n"
            f"and the fields that you can ask: {json.dumps(service2fields, indent=2)}\n"
            f"and the conversation history (may be empty) with local guide:\n{json.dumps(history, indent=2)}\n"
            f"If you were Tom, how would you initiate the inquiry or respond to the guide?\n"
            f"- Your responses should resemble an online chat as much as possible, and make them as brief as possible.\n"
            f"- Do not reveal all your needs at once; inquire them gradually.\n"
            f"- Suppose you don't know the your perference, take the given preferences as a script, and it is needs multiple communications (including the history) that you can describe them clearly.\n"
            f"- Please don't ask any thing that is not listed in the service fields.\n"
            f"- Please random provide some field for asking to narrow down the search scope. \n"
            f"- Each time, you can only output a single utterance based on the conversation history.\n"
            f"- Don't repeat asking if you have asked from the conversation history.\n"
            f"- Only output the newest utterance, don't output the conversation history.\n"
            f"- If all your preference is satisfied, please try booking it.\n"
            f"- If your preference and booking are both ready or the agent informs you that the booking is not avaliable, "
            f"please say good bye or other words to end the conversion, and output a extra mark `[EOF]`.\n"
            f"Please give your latest utterance:\n"
        )

    def parsing(self, res, **kwargs):
        res = str(res)
        if res == 'None':
            res = '[EOF]'
        if 'Tom:' in res:
            return res.replace('Tom:', '').replace('"', '').strip()
        return res.replace('"', '').strip()


class GPTSystemRAGSimulator(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4')

    def prompting(self, prompt, **kwargs):
        return prompt

    def parsing(self, res, **kwargs):
        return res


client = Client("http://209.51.170.51:1308")
def call_tgi_service(type, service, turns, search_results=[]):
    item = {
        'type': type,
        'action': service,
        'history': [turn['utterance'] for turn in turns if ':' not in turn['turn_id']],
        'search_results': search_results,
        'label_type': '',
        'label': ''
    }
    prompt, label = prompting(item)

    try:
        response = client.generate(prompt=prompt,
                                   temperature=0.8,
                                   max_new_tokens=500,
                                   repetition_penalty=1.1,
                                   do_sample=True,
                                   seed=0)
        return response.generated_text
    except:
        return ""



def call_local_generation(policy_model, policy_tokenizer, type, service, turns, search_results=[], gpt4=False):
    item = {
        'type': type,
        'action': service,
        'history': [turn['utterance'] for turn in turns if ':' not in turn['turn_id']],
        'search_results': search_results,
        'label_type': '',
        'label': ''
    }
    prompt, label = prompting(item)

    batch = policy_tokenizer(prompt, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}

    outputs = policy_model.generate(
        **batch,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.8,
        use_cache=True,
        pad_token_id=policy_tokenizer.eos_token_id
    )

    from torch import distributed as dist
    torch.cuda.set_device(rank)
    print(f'rank = {dist.get_rank()}, {outputs}', flush=True)

    # test 01 仅仅在这里放置一个 barrier 会死锁
    # dist.barrier()

    # test 02 删除其他dist配置，包括 import，使用 accelerator wait
    # 只使用这个配置同样会挂起
    # accelerator.wait_for_everyone()

    # test 03 使用 all reduce 显示同步
    signal = torch.tensor([1.], device=f'cuda:{rank}')
    print(f'rank = {rank}, {signal}')
    dist.all_reduce(signal, op=dist.ReduceOp.SUM)
    print(f'rank = {rank}, signal: {signal.item()}, out!!!!!', flush=True)

    output_text = policy_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
    return output_text



def parse_answer(answer):
    try:
        return json.loads(answer)
    except:
        return {}


if __name__ == '__main__':
    input_dir = '.'
    output_dir = '.'
    task = 'test'
    key2fout = {
        'api': open(f'{output_dir}/{task}.api.json', 'w'),
        'rag': open(f'{output_dir}/{task}.rag.json', 'w'),
        'casual': open(f'{output_dir}/{task}.casual.json', 'w')
    }
    for dialog in open(f'{input_dir}/test.json'):
        dialog = json.loads(dialog)
        for key, datas in convert_dialog_to_task(dialog).items():
            if not datas:
                continue
            for data in datas:
                key2fout[key].write(json.dumps(data) + '\n')
                key2fout[key].flush()
    [f.close() for f in key2fout.values()]