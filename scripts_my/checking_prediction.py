import collections
import json

import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz


# def get_slots(obj):
#     slots = {}
#     for t in obj:
#         slots.update(t.get('slot_values', {}))
#     slots = {k: v[0] for k, v in slots.items() if v}
#     return slots

def get_slots(obj):
    slots = {}
    for service, ks in obj.items():
        slots[service] = [k.split('-')[-1] for k in ks]
    return slots


def pretty_percent(num):
    return f'{round(num * 100, 2)}%'


def print_action_metric(pred_actions, real_actions):
    actions = sorted(list(set(pred_actions) | set(real_actions)))
    action2id = {t: i for i, t in enumerate(actions)}
    y_true = [action2id[t] for t in real_actions]
    y_pred = [action2id[t] for t in pred_actions]

    from sklearn.metrics import classification_report, confusion_matrix
    print(f'classification_report\n', classification_report(y_true=y_true, y_pred=y_pred, target_names=actions))

    print(f'confusion matrix\n', confusion_matrix(y_true=y_true, y_pred=y_pred))

if __name__ == '__main__':
    for input_file in [
        '/mnt/share16t/xingguang/datasets/agent_sft.auto.gen.v02/test.act.pred.7b.auto_gen.json',
    ]:
        print(input_file)
        pred_actions, real_actions = [], []

        def get_action(obj):
            if obj['action'] == 'search':
                return 'search'
            elif obj['action'] == 'chat':
                return 'chat'
            elif obj['action'] == 'asking':
                return 'asking'
            else:
                return 'error'

        error = collections.defaultdict(float)

        for data in open(input_file):
            obj = json.loads(data)
            if obj['real_act']['action'] != 'search' or obj['pred_act']['action'] != 'search':
                continue

            real_slots = get_slots(obj['real_act']['slots'])
            pred_slots = get_slots(obj['pred_act']['slots'])

            for pred_service, pred_slot_keys in pred_slots.items():
                if pred_service not in real_slots:
                    for pred_slot_key in pred_slot_keys:
                        error[f'extra_service_{pred_service}'] += 1
                else:
                    for pred_slot_key in pred_slot_keys:
                        if pred_slot_key not in real_slots[pred_service]:
                            error[f'extra_slot_{pred_service}_{pred_slot_key}'] += 1

            for real_service, real_slot_keys in real_slots.items():
                if real_service not in pred_slots:
                    for real_slot_key in real_slot_keys:
                        error[f'missing_service_{real_service}'] += 1
                else:
                    for real_slot_key in real_slot_keys:
                        if real_slot_key not in pred_slots[real_service]:
                            error[f'missing_slot_{real_service}_{real_slot_key}'] += 1

        print(json.dumps(error, indent=2, sort_keys=True))




