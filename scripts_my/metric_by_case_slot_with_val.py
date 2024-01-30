import collections
import json
from fuzzywuzzy import fuzz

if __name__ == '__main__':
    from woz_name_config import update_slots

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.1.dst/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.2.dst/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.4.dst/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.5.dst/dev.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.1.dst.ctx/dev.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.2.dst.ctx/dev.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.3.dst.ctx/dev.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.merge.1_3.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.4.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.7.dst.ctx/dev.act.pred.vllm.7b.5e-5.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.8.dst.ctx/dev.act.pred.vllm.7b.5e-5.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.8.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.10.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.10.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.10.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.10.dst.ctx/dev.act.pred.vllm.7b.replace.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.11.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.12.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.13.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.13.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.14.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.15.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.13.dst.ctx/dev.act.pred.7b.single.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.17.dst.ctx/dev.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.18.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.19.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.20.1.dst.ctx/test.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.21.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.23.dst.ctx/dev.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.23.2.dst.ctx/dev.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.24.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.25.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.26.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.27.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.25.ultra.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.28.1.dst.ctx/test.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.35.1.dst.ctx/dev.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.36.1.dst.ctx/dev.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.33.1.dst.ctx/dev.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst/dev.act.pred.7b.json'

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e01/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k.e01/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace_restaurant.dst.ctx/dev.act.pred.vllm.7b.2e-5.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.37.1.template.2k.dst.ctx/dev.act.pred.vllm.7b.2e-5.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.restaurant.dst.ctx/dev.act.pred.vllm.7b.2e-5.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.attraction.dst.ctx/dev.act.pred.vllm.7b.2e-5.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.hotel.dst.ctx/dev.act.pred.vllm.7b.2e-5.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.train.2k.dst.ctx/dev.act.pred.vllm.7b.2e-5.json'
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

            # print(set(real_slot_keys))
            # print(set(pred_slot_keys))
            # print(slot_keys)
            # print()

            total, tp, fp, tn, fn = 0., 0., 0., 0., 0.

            for slot_key in slot_keys:
                if slot_key not in service2slot_keys[service]:
                    continue

                full_slot_key = f'{service}-{slot_key}'
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

    from sklearn.metrics import *
    # for slot_key, preds in sorted(slot_key2preds.items(), key=lambda x:x[0], reverse=True):
    #     print(f"{slot_key} {len(preds)} "
    #           f"{round(precision_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)} "
    #           f"{round(recall_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)} "
    #           f"{round(accuracy_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)}"
    #           )
    for full_slot_key, total in slot_key2total.items():
        # print(full_slot_key, slot_key2tp[full_slot_key], slot_key2fp[full_slot_key], slot_key2fn[full_slot_key])
        p = slot_key2tp[full_slot_key] / (slot_key2tp[full_slot_key] + slot_key2fp[full_slot_key])
        r = slot_key2tp[full_slot_key] / (slot_key2tp[full_slot_key] + slot_key2fn[full_slot_key])
        f = 2 * p * r / (p + r)
        print(f'{full_slot_key} '
              f'{round(p, 5)} '
              f'{round(r, 5)} '
              f'{round(f, 5)} '
              )