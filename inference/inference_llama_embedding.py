import json
import sys

import numpy as np
import torch
import tqdm
from transformers.models.llama import LlamaTokenizer, LlamaForCausalLM
from ft_datasets.agent_sft_act_dataset import AgentActDataset, agent_tokenize


if __name__ == '__main__':
    # dataset = 'agent_sft.auto.gen.v08.37.1.template.8k.dst.ctx'
    # dataset = 'agent_sft.v10.baseline.dst.limit_8k'
    dataset = sys.argv[1]
    data_path = f'/home/paperspace/xingguang/datasets/{dataset}/train.act.json'
    # model_path = '/home/paperspace/xingguang/models/agent_sft_act_dataset.FT-8K-7b.2e-5.full.B8.E1.agent_sft.v10.baseline.dst.limit_8k.hf'
    model_path = "meta-llama/Llama-2-7b-chat-hf"

    did2data = {}
    for line in open(data_path):
        data = json.loads(line)
        did = data['dialog_id']
        tid = data['turn_id']
        did2data[did] = data

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.to('cuda:0')
    model = model.half()

    embeddings = []

    sout = open(f'/home/paperspace/xingguang/datasets/embedding.7b.chat.dst/{dataset}.ids', 'w')
    for did, data in tqdm.tqdm(did2data.items()):
        # data['type'] = 'act_selection_baseline_dst_emb'
        prompt, label = AgentActDataset.prompting(data)
        inputs = tokenizer(prompt, return_tensors="pt")

        for key, val in inputs.items():
            inputs[key] = val.to('cuda:0')

        model_output = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_hidden_states=True
        )
        # print(model_output.hidden_states)

        last_hidden_state = model_output.hidden_states[-1]
        last_hidden_state = last_hidden_state.squeeze().detach().cpu().numpy()
        last_hidden_state = last_hidden_state[-1]

        if last_hidden_state.shape[0] != 4096:
            print('error')
            continue
        else:
            sout.write(f'{did}\n')
            embeddings.append(last_hidden_state)
    sout.close()
    np.save(open(f'/home/paperspace/xingguang/datasets/embedding.7b.chat.dst/{dataset}.npy', 'wb'), np.asarray(embeddings))

