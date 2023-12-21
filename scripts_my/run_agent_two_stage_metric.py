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
    for service, kv in obj.items():
        for k, v in kv.items():
            slots[f'{service}-{k}'] = v
    return slots



def is_matching(hyp, ref, fuzzy_ratio=95):
    hyp_k = hyp.keys()
    ref_k = ref.keys()
    if hyp_k != ref_k:
        return False
    for k in ref_k:
        if fuzz.partial_ratio(hyp[k], ref[k]) <= fuzzy_ratio:
            return False
    return True


def get_slots(obj):
    slots = {}
    for service, kv in obj.items():
        for k, v in kv.items():
            k = k.split('-')[-1]
            slots[f'{service}-{k}'] = v
    return slots


def compare(hyp, ref, fuzzy_ratio=95):
    hyp = get_slots(hyp)
    ref = get_slots(ref)
    # tp ... those mentioned in both and matching
    # tn ... those not mentioned in both (this inflates results for slot acc., thus reporting F1)
    # fn ... those not mentioned in hyp but mentioned in ref
    # fp ... those mentioned in hyp but not mentioned in ref OR mentioned in hyp but not matching
    tp, fp, fn = 0, 0, 0
    for slot, value in hyp.items():
        if slot in ref and fuzz.partial_ratio(value, ref[slot]) > fuzzy_ratio:
            tp += 1
        elif slot not in ref:
            # print(f'ref {slot} = None, hyp = {value}')
            fp += 1
        else:
            # print(f'ref {slot} = {ref[slot]}, hyp = {value}')
            fp += 1
    for slot, value in ref.items():
        if slot not in hyp:
            fn += 1
            # print(f'ref {slot} = {value}, hyp = None')
        elif fuzz.partial_ratio(hyp[slot], value) <= fuzzy_ratio:
            # print(f'ref {slot} = {value}, hyp = {hyp[slot]}')
            fn += 1
    return tp, fp, fn, is_matching(hyp, ref)


def pretty_percent(num):
    return f'{round(num * 100, 2)}%'


if __name__ == '__main__':
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.v08/dev.pred.7b.13b-rl.json'

    real_types, pred_types = [], []
    total_tp, total_fp, total_fn, joint_match, total_api_casual = 0., 0., 0., 0., 0.

    for data in open(input_file):
        obj = json.loads(data)
        real_types.append(obj['real_gen']['type'])
        pred_types.append(obj['pred_gen']['type'])

        if obj['real_gen']['type'] == 'api_generation' or obj['pred_gen']['type'] == 'api_generation':
            api_config_real = obj['real_gen']['label'] if obj['real_gen']['type'] == 'api_generation' else {}
            api_config_pred = obj['pred_gen']['label'] if obj['pred_gen']['type'] == 'api_generation' else {}
            tp, fp, fn, is_match = compare(api_config_pred, api_config_real)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            joint_match += int(is_match)
            total_api_casual += 1.


    slot_p = total_tp / (total_tp + total_fp + 1e-10)
    slot_r = total_tp / (total_tp + total_fn + 1e-10)
    slot_f1 = 2 * slot_p * slot_r / (slot_p + slot_r + 1e-10)

    print(json.dumps({
        'SlotF': pretty_percent(slot_f1),
        'SlotP': pretty_percent(slot_p),
        'SlotR': pretty_percent(slot_r),
        'JGA': pretty_percent(joint_match / total_api_casual)
    }))

    types = list(set(real_types) | set(pred_types))
    type2id = {t: i for i, t in enumerate(types)}
    y_true = [type2id[t] for t in real_types]
    y_pred = [type2id[t] for t in pred_types]

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=types)
    disp.plot()
    plt.savefig('cm.jpg')
    n2n = {
        'casual_generation_no_slots': 'casual',
        'casual_generation': 'ask',
        'api_generation': 'api',

    }
    target_names = [n2n.get(tt, tt) for tt in types]
    print(f'classification_report\n', classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names))

