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
        # '/mnt/share16t/xingguang/datasets/agent_sft.auto.gen.v03/test.act.pred.7b.auto_gen.restaurant.json',
        '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline/dev.act.pred.7b.json',
        '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline/test.act.pred.7b.json',
    ]:
        print(input_file)
        pred_actions, real_actions = [], []

        for data in open(input_file):
            obj = json.loads(data)
            real_actions.append(obj['real_act'].get('current_service', 'error'))
            pred_actions.append(obj['pred_act'].get('current_service', 'error'))
        print_action_metric(pred_actions, real_actions)

        total_tp, total_fp, total_fn, joint_match, total_n = 0., 0., 0., 0., 0.
        for data in open(input_file):
            obj = json.loads(data)

            total_n += 1
            real_slots = get_slots(obj['real_act']['slots'])
            pred_slots = get_slots(obj['pred_act']['slots'])

            has_fn, has_fp = False, False
            for service, real_slot_keys in real_slots.items():
                real_slot_keys = set(real_slot_keys)
                pred_slot_keys = set(pred_slots.get(service, []))
                for real_slot_key in real_slot_keys:
                    if real_slot_key not in pred_slot_keys:
                        total_fn += 1
                        has_fn = True
                    else:
                        total_tp += 1
                for pred_slot_key in pred_slot_keys:
                    if pred_slot_key not in real_slot_keys:
                        total_fp += 1
                        has_fp = True
            if not has_fn and not has_fp:
                joint_match += 1

        slot_p = total_tp / (total_tp + total_fp + 1e-10)
        slot_r = total_tp / (total_tp + total_fn + 1e-10)
        slot_f1 = 2 * slot_p * slot_r / (slot_p + slot_r + 1e-10)

        print(json.dumps({
            'SlotF': pretty_percent(slot_f1),
            'SlotP': pretty_percent(slot_p),
            'SlotR': pretty_percent(slot_r),
            'JGA': pretty_percent(joint_match / total_n)
        }))



