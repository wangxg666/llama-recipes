import collections
import json

if __name__ == '__main__':
    service2slot_keys = json.load(open('woz_valid_slot_keys.json'))

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.1.dst/dev.act.pred.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.2.dst/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e01/dev.act.pred.7b.json'
    error2count = collections.defaultdict(float)
    did2sample = {}
    for data in open(input_file.replace('.pred.7b.json', '.json')):
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

        errors = []

        services = set(list(pred_slots) + list(real_slots))

        for service in services:
            if service not in service2slot_keys:
                continue

            pred_slot_kvs = pred_slots.get(service, {})
            pred_combined_keys = {
                f'{service}-{slot_key}-{slot_val.replace(" ", "_")}'
                for slot_key, slot_val in pred_slot_kvs.items()
                if slot_key in service2slot_keys[service]
            }

            real_slot_kvs = real_slots.get(service, {})
            real_combined_keys = {
                f'{service}-{slot_key}-{slot_val.replace(" ", "_")}'
                for slot_key, slot_val in real_slot_kvs.items()
                if slot_key in service2slot_keys[service]
            }

            all_combined_keys = pred_combined_keys | real_combined_keys
            for combined_key in all_combined_keys:
                slot_key2preds[combined_key].append(int(combined_key in pred_combined_keys))
                slot_key2reals[combined_key].append(int(combined_key in real_combined_keys))

    from sklearn.metrics import precision_score, recall_score, f1_score
    for slot_key, preds in sorted(slot_key2preds.items(), key=lambda x:len(x[1]), reverse=True):
        if len(preds) > 10:
            print(f"{input_file.split('/')[-2]}, {slot_key}, {len(preds)}, "
                  f"{round(precision_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)}, "
                  f"{round(recall_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)}, "
                  f"{round(f1_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)}, ")
