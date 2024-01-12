import collections
import json

if __name__ == '__main__':
    service2slot_keys = json.load(open('woz_valid_slot_keys.json'))

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.1.dst/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.2.dst/dev.act.pred.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline/dev.act.pred.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e01/dev.act.pred.7b.json'
    error2count = collections.defaultdict(float)
    did2sample = {}
    for data in open(input_file.replace('.pred.7b.json', '.json')):
        obj = json.loads(data)
        did = f'{obj["dialog_id"]}'
        if did not in did2sample or len(did2sample[did]) <= len(obj['history']):
            did2sample[did] = obj['history']

    error2did2tid = collections.defaultdict(dict)

    service2slot_keys = json.load(open('woz_valid_slot_keys.json'))

    service2preds = collections.defaultdict(list)
    service2reals = collections.defaultdict(list)

    for data in open(input_file):
        data = json.loads(data)

        key = data['key']
        did, tid = key.split('_')
        pred_slots = data['pred_act']['slots']
        real_slots = data['real_act']['slots']

        errors = []

        services = list(set(pred_slots) | set(real_slots))
        for service in services:
            service2preds[service].append(int(service in pred_slots))
            service2reals[service].append(int(service in real_slots))

    from sklearn.metrics import precision_score, recall_score, f1_score
    for service, preds in sorted(service2preds.items(), key=lambda x:len(x[1]), reverse=True):
        print(f"{service}, {len(preds)}, "
              f"precision = {round(precision_score(y_true=service2reals[service], y_pred=preds), 5)}, "
              f"recall_score = {round(recall_score(y_true=service2reals[service], y_pred=preds), 5)}, "
              f"f1_score = {round(f1_score(y_true=service2reals[service], y_pred=preds), 5)}, ")
