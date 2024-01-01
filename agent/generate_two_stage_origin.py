from agent.generate_two_stage_utils import *


def generate_dialog(step,
                    policy_model: transformers.models.llama.LlamaForCausalLM=None,
                    policy_tokenizer: transformers.models.llama.LlamaTokenizer=None,
                    device=None):
    # random.shuffle(services)
    # used_services = [services[0]]
    used_services = [services[step % len(services)]]
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
            act_output_raw = policy_tokenizer.decode(output, skip_special_tokens=True)[len(act_prompt):]
            act_output_raw = act_output_raw.split('\n')[0]

        else:
            act_output_raw = call_tgi(act_prompt, act_tgi_svr)

        try:
            act_output = json.loads(act_output_raw)
            print_rank_0(f'rank = {rank}, turn = {turn_no}, act = {act_output}')
        except:
            act_output_r = call_tgi(act_prompt, act_tgi_svr)
            print_rank_0(f'rank = {rank}, turn = {turn_no}, act = {act_output}, react = {act_output_r}')
            act_output = json.loads(act_output_r)
            n_act_tgi_call += 1.

        act_output = validate_action_response(act_output)
        print_rank_0(f'rank = {rank}, turn = {turn_no}, act = {act_output}')

        action2ttype = {
            'search': 'api_generation',
            'asking': 'casual_generation',
            'chat': 'casual_generation_no_slots',
        }
        ttype = action2ttype.get(act_output['action'], 'api_generation')
        # print_rank_0(f'rank = {rank}, turn = {turn_no}, System Act: {ttype}, detail = {act_output}')

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

        if ttype != 'api_generation':
            need_api = False
            # 反问或者闲聊
            gen_item['asked_slots'] = act_output['slots']
            # print_rank_0(f'rank = {rank}, turn = {turn_no}, directly chatting')
            gen_output = get_gen_output(gen_item)
        else:
            need_api = True
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
            search_results = requests.post(url='http://35.86.252.8:1201/do_search', data=json.dumps(req_data)).json()

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
            turns[-1]['direct_output'] = act_output_raw

        turn_no += 1

        if '[EOF]' in user_utterance:
            break

    dialog = [
        {'User' if turn['speaker'] == 'USER' else 'Agent': turn['utterance']}
        for turn in turns if ':' not in turn['turn_id']
    ]
    try:
        reward = requests.post(
            'http://35.86.252.8:1201/do_reward', data=json.dumps({'dialog': dialog})
        ).json().get('report', [])
        reward = {} if not reward else reward[0]
    except:
        reward = {}
    print(f'rank = {rank}, {n_act_tgi_call} of {n_act_call} act generation failed')
    return used_services[0], turns, reward


def get_batch(step,
              batch_size=4,
              policy_model=None,
              policy_tokenizer=None,
              device=None):
    service, turns, reward = generate_dialog(step, policy_model, policy_tokenizer, device)
    cache.write(json.dumps({
        'dialog': turns,
        'reward': reward
    }) + '\n')
    return parse_dialog(turns, reward, batch_size, policy_tokenizer)


if __name__ == '__main__':
    pass