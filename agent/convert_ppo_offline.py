import json
import os
from ft_datasets.my_agent_sft_dataset import MyAgentSFTDataset


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


def enrich_api_params(api_params):
    api_params_list = []
    for service, slots in api_params.items():
        api_params = {
            'service': service,
            'active_intent': f'find_{service}',
            'slot_values': {f'{service}-{k}': [v] for k, v in slots.items()}
        }
        api_params_list.append(api_params)
    return api_params_list


def convert_sft_types(turns):
    api_generate_turns = []
    rag_generate_turns = []
    casual_generate_turns = []

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
                }
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


def prompting(turn):
    type = turn['type']
    history = [x.replace('USER', 'user').replace('SYSTEM', 'you') for x in turn['history']]
    history = [x.replace('Tom', 'user').replace('agent', 'you') for x in turn['history']]

    if type == 'api_generation':
        persona = PERSONA_PROMPT_DICT.get(turn['action'], PERSONA_PROMPT_DICT['default'])
        prompt = ANSWER_TYPE_PROMPT[type].format(
            persona=persona,
            history=json.dumps(history[0:-1], indent=2),
            user_utterence=history[-1].replace('user: ', '')
        )
        label = turn['label_type'] + '\n' + json.dumps(turn['label'], separators=(',', ':'))

    elif type == 'casual_generation':
        persona = PERSONA_PROMPT_DICT.get(turn['action'], PERSONA_PROMPT_DICT['default'])
        prompt = ANSWER_TYPE_PROMPT[type].format(
            persona=persona,
            history=json.dumps(history[0:-1], indent=2),
            user_utterence=history[-1].replace('user: ', '')
        )
        label = turn['label_type'] + '\n' + turn['label']

    else:
        persona = PERSONA_PROMPT_DICT.get(turn['action'], PERSONA_PROMPT_DICT['default'])
        prompt = ANSWER_TYPE_PROMPT[type].format(
            persona=persona,
            history=json.dumps(history[0:-1], indent=2),
            user_utterence=history[-1].replace('user: ', ''),
            search_results=json.dumps(turn['search_results'], separators=(',', ':'))
        )
        label = turn['label']
    return prompt, label


def tokenize_samples(reward, key2truns, tokenizer):
    samples = []
    for key, turns in key2truns.items():
        for turn in turns:
            prompt, label = prompting(turn)
            prompt = prompt.strip()
            label = label.strip()
            sample = {
                'key': key,
                'reward': reward,
                'query_tensor': tokenizer.encode(prompt),
            }
            input_ids = tokenizer.encode(prompt + label)[len(sample['query_tensor']):] + [tokenizer.eos_token_id]

            # print(input_ids)
            # print(tokenizer.encode(label))
            # print(tokenizer.encode(prompt + label))
            # MyAgentSFTDataset.inner_tokenize(turn, 1000, tokenizer, False)

            sample['response_tensor'] = input_ids
            samples.append(sample)
    return samples

if __name__ == '__main__':
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b")


    output_dir = '/home/paperspace/xingguang/datasets/agent_raft.v07/'
    os.makedirs(output_dir, exist_ok=True)
    key2sout = {
        'api': open(f'{output_dir}/train.api.json', 'w'),
        'rag': open(f'{output_dir}/train.rag.json', 'w'),
        'casual': open(f'{output_dir}/train.casual.json', 'w'),
    }
    sout = open(f'{output_dir}/ppo.train.jsonl', 'w')


    for line in open('/home/paperspace/xingguang/datasets/ppo_test.ex/data.raw.json'):
        obj = json.loads(line)
        reward = obj['reward']['avg_score']

        key2turns = convert_sft_types(obj['dialog'])
        if reward >= 4:
            for key, turns in key2turns.items():
                key2sout[key].write('\n'.join([json.dumps(turn) for turn in turns]) + '\n')
                key2sout[key].flush()
        samples = tokenize_samples(reward, key2turns, tokenizer)

        for sample in samples:
            sout.write(json.dumps(sample) + '\n')
            sout.flush()
    sout.close()

    for key, sout in key2sout.items():
        sout.close()