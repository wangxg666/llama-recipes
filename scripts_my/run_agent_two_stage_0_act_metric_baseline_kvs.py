import json

from fuzzywuzzy import fuzz



def get_slots(obj):
    slots = {}
    for service, kvs in obj.items():
        try:
            kvs = {k.split('-')[-1]: v.lower() for k, v in kvs.items() if v}
            kvs_clean = {}
            for k, v in kvs.items():
                if not v or v in {"dontcare":1, "":1, 'none': 1}:
                    continue
                kvs_clean[k] = v.lower()
            slots[service] = kvs_clean
        except:
            print(kvs)

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
        else:
            fp += 1
    for slot, value in ref.items():
        if slot not in hyp or fuzz.partial_ratio(hyp[slot], value) <= fuzzy_ratio:
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

    for split in ['test']:
        for input_file in [
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.attraction.full.dst.ctx/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.hotel.full.dst.ctx/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.restaurant.full.dst.ctx/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.taxi.full.dst.ctx/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.train.full.dst.ctx/{split}.act.pred.vllm.7b.2e-5.json',

            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.with.gen.37.2k/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.with.gen.37.4k/{split}.act.pred.vllm.7b.2e-5.json',

            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k.e01/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k.e01.with.gen.8k/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.hotel.full.fix_taxi.dst.ctx/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.restaurant.full.fix_taxi.dst.ctx/{split}.act.pred.vllm.7b.2e-5.json',

            # 2.2 蒸馏实验 7B
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',
            # # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k/{split}.act.pred.vllm.7b.2e-5.pre-train-16k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',
            # # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k/{split}.act.pred.vllm.7b.2e-5.pre-train-16k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',
            # # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k/{split}.act.pred.vllm.7b.2e-5.pre-train-16k.json',
            f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',
            # # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.only.json',
            f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.7b.2e-5.pre-train-16k.dedup.json',
            f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.7b.2e-5.pre-train-16k.dedup.v2.json',
            # # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.7b.2e-5.pre-train-16k.json',

            # 2.2 蒸馏实验 13B
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k/{split}.act.pred.vllm.13b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k/{split}.act.pred.vllm.13b.2e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k/{split}.act.pred.vllm.13b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k/{split}.act.pred.vllm.13b.2e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k/{split}.act.pred.vllm.13b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k/{split}.act.pred.vllm.13b.2e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.13b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.13b.1e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.13b.2e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.13b.5e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.13b.2e-5.pre-train-16k.json',

            # 2.4 蒸馏实验 7B
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_1k.new/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_1k.new/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_2k.new/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_2k.new/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_4k.new/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_4k.new/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_8k.new/{split}.act.pred.vllm.7b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_8k.new/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',

            # 2.4 蒸馏实验 13B
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_1k.new/{split}.act.pred.vllm.13b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_1k.new/{split}.act.pred.vllm.13b.2e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_2k.new/{split}.act.pred.vllm.13b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_2k.new/{split}.act.pred.vllm.13b.2e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_4k.new/{split}.act.pred.vllm.13b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_4k.new/{split}.act.pred.vllm.13b.2e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_8k.new/{split}.act.pred.vllm.13b.2e-5.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_8k.new/{split}.act.pred.vllm.13b.1e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_8k.new/{split}.act.pred.vllm.13b.2e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_8k.new/{split}.act.pred.vllm.13b.5e-5.pre-train-8k.json',
            # f'/mnt/share16t/xingguang/datasets/agent_sft.woz.2.4.limit_8k.new/{split}.act.pred.vllm.13b.2e-5.pre-train-16k.json',
        ]:
            parts = input_file.split('/')
            print(input_file)
            print('/'.join(parts[-2:]))
            pred_actions, real_actions = [], []

            objs = [json.loads(data) for data in open(input_file)][:]

            for obj in objs:
                if 'real_act' not in obj or 'pred_act' not in obj:
                    continue
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
                if 'real_act' not in obj or 'pred_act' not in obj:
                    continue
                real_slots = get_slots(obj['real_act']['slots'])
                pred_slots = get_slots(obj['pred_act']['slots'])

                is_match = 1
                for service, real_slot_kvs in real_slots.items():
                    # 直接生成会有错误，需要对一遍schema
                    pred_slot_kvs = pred_slots.get(service, {})
                    pred_slot_kvs = {
                        pred_k: pred_v for pred_k, pred_v in pred_slot_kvs.items()
                        if service in service2slot_keys and pred_k in service2slot_keys[service] and pred_v
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

            print(json.dumps({
                'SlotF': pretty_percent(slot_f1),
                'SlotP': pretty_percent(slot_p),
                'SlotR': pretty_percent(slot_r),
                'JGA': pretty_percent(joint_match / total_n),
            }) + '\n')



