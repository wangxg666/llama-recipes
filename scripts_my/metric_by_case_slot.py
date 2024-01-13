import collections
import json

if __name__ == '__main__':
    from woz_name_config import update_slots

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.1.dst/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.2.dst/dev.act.pred.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.3.dst/dev.act.pred.vllm.7b.json'
    error2count = collections.defaultdict(float)
    did2sample = {}
    # for data in open(input_file.replace('.pred.7b.json', '.json')):
    for data in open(input_file.replace('.pred.vllm.7b.json', '.json')):
        obj = json.loads(data)
        did = f'{obj["dialog_id"]}'
        if did not in did2sample or len(did2sample[did]) <= len(obj['history']):
            did2sample[did] = obj['history']

    error2did2tid = collections.defaultdict(dict)

    service2slot_keys = json.load(open('woz_valid_slot_keys.json'))

    slot_key2preds = collections.defaultdict(list)
    slot_key2reals = collections.defaultdict(list)

    for data in open(input_file):
        data = json.loads(data)

        key = data['key']
        did, tid = key.split('_')
        pred_slots = data['pred_act']['slots']
        real_slots = data['real_act']['slots']

        pred_slots = update_slots(pred_slots)

        errors = []

        for pred_service, pred_slot_keys in pred_slots.items():
            if pred_service not in service2slot_keys:
                continue
            real_slot_keys = real_slots.get(pred_service, {})
            slot_keys = list(set(real_slot_keys) | set(pred_slot_keys))
            for slot_key in slot_keys:
                if slot_key not in service2slot_keys[pred_service]:
                    continue
                slot_key2preds[f'{pred_service}-{slot_key}'].append(int(slot_key in pred_slot_keys))
                slot_key2reals[f'{pred_service}-{slot_key}'].append(int(slot_key in real_slot_keys))

        for real_service, real_slot_keys in real_slots.items():
            if real_service not in service2slot_keys:
                continue
            pred_slot_keys = pred_slots.get(real_service, {})
            for real_slot_key in real_slot_keys:
                if real_slot_key in pred_slot_keys or real_slot_key not in service2slot_keys[real_service]:
                    continue
                slot_key2preds[f'{real_service}-{real_slot_key}'].append(0)
                slot_key2reals[f'{real_service}-{real_slot_key}'].append(1)

    from sklearn.metrics import precision_score, recall_score, f1_score
    for slot_key, preds in sorted(slot_key2preds.items(), key=lambda x:len(x[1]), reverse=True):
        print(f"{slot_key}, {len(preds)}, "
              f"precision = {round(precision_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)}, "
              f"recall_score = {round(recall_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)}, "
              f"f1_score = {round(f1_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)}, ")
