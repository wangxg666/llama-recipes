from agent.generate_two_stage_utils import *




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--replace', action='store_true')
    args.add_argument('--gen_file', type=str)
    args.add_argument('--gen_batch', action='store_true')
    args.add_argument('--view_reward', action='store_true')
    args = args.parse_args()

    if args.replace:
        from agent.generate_two_stage_replace import generate_dialog, get_batch
    else:
        from agent.generate_two_stage_origin import generate_dialog, get_batch

    if args.gen_file:
        output_dir = '/home/paperspace/xingguang/datasets/agent_two_stage_auto_gen'
        output_file = f'{output_dir}/{args.gen_file}'
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

    if args.gen_batch:
        model_name_or_path = 'meta-llama/Llama-2-13b-hf'
        policy_tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

        policy_model = LlamaForCausalLM.from_pretrained('/home/paperspace/xingguang/models/agent_sft_act_dataset.7b.2e-5.full.B8.E1.agent_sft.v09.1.hf')
        policy_model.to('cuda')

        for i in range(10):
            get_batch(i, 4, policy_model, policy_tokenizer)

    if args.view_reward:
        model_name_or_path = 'meta-llama/Llama-2-13b-hf'
        policy_tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

        critic_pre_train_dir = '/home/paperspace/xingguang/datasets/ppo_cache'
        datas = []
        for filename in os.listdir(critic_pre_train_dir):
            try:
                datas.extend([json.loads(line) for line in open(f'{critic_pre_train_dir}/{filename}')])
            except:
                pass
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

        print(json.dumps(weight2count, indent=2))
        print(json.dumps(turn_type2count, indent=2))

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