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

    # print(f'confusion matrix\n', confusion_matrix(y_true=y_true, y_pred=y_pred))

if __name__ == '__main__':
    service2slot_keys = json.load(open('woz_valid_slot_keys.json'))

    for input_file in [
        '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst/dev.act.pred.7b.json',
        '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst.lower/dev.act.pred.7b.json',
        '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.2.dst/dev.act.pred.7b.json',
    ]:
        print(input_file)
        pred_actions, real_actions = [], []

        objs = [json.loads(data) for data in open(input_file)]

        for obj in objs:
            real_action = obj['real_act'].get('current_service', 'error')
            if real_action not in ['hotel', 'attraction', 'restaurant', 'train', 'taxi']:
                real_action = 'other'
            real_actions.append(real_action)

            pred_action = obj['pred_act'].get('current_service', 'error')
            if pred_action not in ['hotel', 'attraction', 'restaurant', 'train', 'taxi']:
                pred_action = 'other'
            pred_actions.append(pred_action)
        print_action_metric(pred_actions, real_actions)

        total_tp, total_fp, total_fn, joint_match, total_n = 0., 0., 0., 0., 0.
        for obj in objs:
            total_n += 1
            real_slots = get_slots(obj['real_act']['slots'])
            pred_slots = get_slots(obj['pred_act']['slots'])

            has_fn, has_fp = False, False
            for service, real_slot_keys in real_slots.items():
                real_slot_keys = set(real_slot_keys)

                pred_slot_keys = set(pred_slots.get(service, []))
                pred_slot_keys = {pred_slot_key for pred_slot_key in pred_slot_keys if pred_slot_key in service2slot_keys[service]}

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



