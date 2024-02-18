import collections
import random

import numpy as np
import json

import tqdm


def diff_slots(states_a, states_b):
    diff = 0
    for service, slots in states_a.items():
        if service not in states_b:
            diff += len(slots)
        else:
            for k, v in slots.items():
                if k not in states_b[service] or v != states_b[service][k]:
                    diff += 1

    for service, slots in states_b.items():
        if service not in states_a:
            diff += len(slots)
        else:
            for k, v in slots.items():
                if k not in states_a[service]:
                    diff += 1
    return diff


if __name__ == '__main__':
    dataset = 'agent_sft.auto.gen.v08.37.1.template.16k.dst.ctx.deduped.dst_prompt'
    data_path = f'/home/paperspace/xingguang/datasets/{dataset}/train.act.json'
    did2data = {}
    for line in open(data_path):
        data = json.loads(line)
        did = data['dialog_id']
        tid = data['turn_id']
        did2data[did] = data

    data_dir = '/home/paperspace/xingguang/datasets/embedding.7b.chat.dst/'
    dialog_ids = [x.strip() for x in open(f'{data_dir}/{dataset}.ids')]
    dialog_embs = np.load(open(f'{data_dir}/{dataset}.npy', 'rb'))
    print(dialog_embs[0])
    print(np.linalg.norm(dialog_embs, axis=1))
    print(np.linalg.norm(dialog_embs, axis=1).shape)

    dialog_embs /= np.linalg.norm(dialog_embs, axis=1, keepdims=True)

    remove_dids = set()
    dids_list = []

    sout = open(f'/home/paperspace/xingguang/datasets/{dataset}/train.act.dids', 'w')
    for i, did in enumerate(tqdm.tqdm(dialog_ids)):
        if i % 10 == 0:
            print(len(dids_list), len(remove_dids))

        if did in remove_dids:
            continue
        emb = dialog_embs[i]
        scores = np.dot(dialog_embs, emb).tolist()

        did2score = {}
        for j, socre in enumerate(scores):

            if i == j or dialog_ids[j] in remove_dids:
                continue
            did2score[dialog_ids[j]] = scores[j]
        did_socres = sorted(did2score.items(), key=lambda x:x[1], reverse=True)

        same_did2score = {did: 1.}
        same_did2history = {}
        for same_did, same_score in did_socres[0:10]:
            if same_score > 0.99: # and diff_slots(did2data[did]['label']['slots'], did2data[same_did]['label']['slots']) <= 3:
                same_did2score[same_did] = same_score
                same_did2history[same_did] = did2data[same_did]['history']
        same_did2history[did] = did2data[did]['history']

        remove_dids.update(same_did2history)
        dids_list.append(list(same_did2history))

        sout.write(json.dumps(same_did2score) + '\n')
        sout.flush()
    sout.close()

    save_dids = set()
    for data in open(f'/home/paperspace/xingguang/datasets/{dataset}/train.act.dids'):
        data = json.loads(data)
        did = random.choice(list(data.keys()))
        save_dids.add(did)
    print(len(save_dids))

    sout = open(f'/home/paperspace/xingguang/datasets/{dataset}/train.act.json.dedup', 'w')
    for data in open(f'/home/paperspace/xingguang/datasets/{dataset}/train.act.json'):
        obj = json.loads(data)
        if obj['dialog_id'] in save_dids:
            sout.write(data)
    sout.close()
