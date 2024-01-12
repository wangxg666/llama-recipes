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
    for service, kvs in obj.items():
        slots[service] = {k.split('-')[-1]: v.lower() for k, v in kvs.items()}
    return slots


def pretty_percent(num):
    return f'{round(num * 100, 2)}%'


def is_matching(hyp, ref, fuzzy_ratio=95):
    hyp_k = hyp.keys()
    ref_k = ref.keys()
    if hyp_k != ref_k:
        return False
    for k in ref_k:
        if fuzz.partial_ratio(hyp[k], ref[k]) <= fuzzy_ratio:
            return False
    return True


def compare(hyp, ref, fuzzy_ratio=95):
    # hyp = get_slots(hyp)
    # ref = get_slots(ref)
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
        # '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.6/dev.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.6/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst/dev.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst.lower/dev.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst.lower/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst/dev.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e01/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e02/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e03/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e04/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k.e01/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k.e02/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k.e03/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k.e04/test.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst/dev.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst.lower/dev.act.pred.7b.json',
        # '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.1.dst/dev.act.pred.7b.json',
        '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.2.dst/dev.act.pred.7b.json',
        '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e01/dev.act.pred.7b.json',
        '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.2.dst/test.act.pred.7b.json',
        '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e01/test.act.pred.7b.json',
    ]:
        print(input_file)
        pred_actions, real_actions = [], []

        objs = [json.loads(data) for data in open(input_file)][:]

        for obj in objs:
            real_action = obj['real_act'].get('current_service', 'error')
            if real_action not in ['hotel', 'attraction', 'restaurant', 'train', 'taxi']:
                real_action = 'other'
            real_actions.append(real_action)

            pred_action = obj['pred_act'].get('current_service', 'error')
            if pred_action not in ['hotel', 'attraction', 'restaurant', 'train', 'taxi']:
                pred_action = 'other'
            pred_actions.append(pred_action)
        # print_action_metric(pred_actions, real_actions)

        total_tp, total_fp, total_fn, joint_match, total_n = 0., 0., 0., 0., 0.
        for obj in objs:
            real_slots = get_slots(obj['real_act']['slots'])
            pred_slots = get_slots(obj['pred_act']['slots'])

            is_match = 1
            for service, real_slot_kvs in real_slots.items():
                # 直接生成会有错误，需要对一遍schema
                pred_slot_kvs = pred_slots.get(service, {})
                pred_slot_kvs = {
                    pred_k: pred_v for pred_k, pred_v in pred_slot_kvs.items()
                    if pred_k in service2slot_keys[service] and pred_v
                }

                tp, fp, fn, _ = compare(pred_slot_kvs, real_slot_kvs)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                if fp > 0 or fn > 0:
                    is_match = 0

            if is_match:
                joint_match += 1
            total_n += 1

        slot_p = total_tp / (total_tp + total_fp + 1e-10)
        slot_r = total_tp / (total_tp + total_fn + 1e-10)
        slot_f1 = 2 * slot_p * slot_r / (slot_p + slot_r + 1e-10)

        print(total_n)
        print(json.dumps({
            'SlotF': pretty_percent(slot_f1),
            'SlotP': pretty_percent(slot_p),
            'SlotR': pretty_percent(slot_r),
            'JGA': pretty_percent(joint_match / total_n),
        }))



