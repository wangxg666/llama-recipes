import collections
import json
from fuzzywuzzy import fuzz

if __name__ == '__main__':
    from woz_name_config import update_slots

    split = 'test'
    for input_file in [
        # 2.2 蒸馏实验 7B
        f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k/{split}.act.pred.vllm.7b.2e-5.json',
        f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',
        # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k/{split}.act.pred.vllm.7b.2e-5.pre-train-16k.json',
        f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k/{split}.act.pred.vllm.7b.2e-5.json',
        f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',
        # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k/{split}.act.pred.vllm.7b.2e-5.pre-train-16k.json',
        f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k/{split}.act.pred.vllm.7b.2e-5.json',
        f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',
        # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k/{split}.act.pred.vllm.7b.2e-5.pre-train-16k.json',
        f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.7b.2e-5.json',
        f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.7b.2e-5.pre-train-8k.json',
        # f'/mnt/share16t/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k/{split}.act.pred.vllm.7b.2e-5.pre-train-16k.json',
    ]:
        print(input_file)
        error2count = collections.defaultdict(float)

        error2did2tid = collections.defaultdict(dict)

        service2slot_keys = json.load(open('woz_valid_slot_keys.json'))

        slot_key2preds = collections.defaultdict(list)
        slot_key2reals = collections.defaultdict(list)

        slot_key2fp = collections.defaultdict(float)
        slot_key2tp = collections.defaultdict(float)
        slot_key2fn = collections.defaultdict(float)
        slot_key2total = collections.defaultdict(float)

        from fuzzywuzzy import fuzz

        for data in open(input_file):
            data = json.loads(data)

            key = data['key']
            did, tid = key.split('_')
            pred_slots = data['pred_act']['slots']
            real_slots = data['real_act']['slots']
            pred_slots = update_slots(pred_slots)

            errors = []

            services = list(set(pred_slots) | set(real_slots))
            for service in services:
                if service not in service2slot_keys:
                    continue

                real_slot_keys = real_slots.get(service, {})
                pred_slot_keys = pred_slots.get(service, {})

                slot_keys = list(set(real_slot_keys) | set(pred_slot_keys))

                total, tp, fp, tn, fn = 0., 0., 0., 0., 0.

                for slot_key in slot_keys:
                    if slot_key not in service2slot_keys[service]:
                        continue

                    full_slot_key = f'{service}-{slot_key}'
                    full_slot_key = service
                    if slot_key in real_slot_keys and slot_key in pred_slot_keys:
                        if fuzz.partial_ratio(pred_slot_keys.get(slot_key, ''), real_slot_keys.get(slot_key, '')) > 95:
                            # 预测对
                            slot_key2tp[full_slot_key] += 1.
                        else:
                            # 预测错
                            slot_key2fp[full_slot_key] += 1.
                    elif slot_key not in real_slot_keys:
                        # 多预测
                        slot_key2fp[full_slot_key] += 1.
                    else:
                        # 未预测
                        slot_key2fn[full_slot_key] += 1.
                    # 没有 TN

                    slot_key2total[full_slot_key] += 1.

        for full_slot_key, total in sorted(slot_key2total.items(), key=lambda x:x[0]):
            # print(full_slot_key, slot_key2tp[full_slot_key], slot_key2fp[full_slot_key], slot_key2fn[full_slot_key])
            p = slot_key2tp[full_slot_key] / (slot_key2tp[full_slot_key] + slot_key2fp[full_slot_key])
            r = slot_key2tp[full_slot_key] / (slot_key2tp[full_slot_key] + slot_key2fn[full_slot_key])
            f = 2 * p * r / (p + r)
            print(f'{full_slot_key} '
                  f'{round(p, 5)} '
                  f'{round(r, 5)} '
                  f'{round(f, 5)} '
                  )

            # acc = slot_key2tp[full_slot_key] / (slot_key2tp[full_slot_key] + slot_key2fp[full_slot_key] + slot_key2fn[full_slot_key])
            # print(f'{full_slot_key} {round(acc, 5)}')