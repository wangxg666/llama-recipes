import collections
import json

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
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.28.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.29.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.35.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.36.1.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.36.1.4k.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.37.1.template.2k.dst.ctx/dev.act.pred.vllm.7b.2e-5.json'

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst/dev.act.pred.7b.json'

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e01/test.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_2k.e01/test.act.pred.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.28.1.replace.hotel.dst.ctx/dev.act.pred.vllm.7b.2e-5.json'

    error2count = collections.defaultdict(float)

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
        pred_slots = update_slots(pred_slots)

        errors = []

        services = list(set(pred_slots) | set(real_slots))
        for service in services:
            if service not in service2slot_keys:
                continue

            real_slot_keys = real_slots.get(service, {})
            pred_slot_keys = pred_slots.get(service, {})

            slot_keys = list(set(real_slot_keys) | set(pred_slot_keys))

            for slot_key in slot_keys:
                if slot_key not in service2slot_keys[service]:
                    continue
                slot_key2preds[f'{service}-{slot_key}'].append(int(slot_key in pred_slot_keys))
                slot_key2reals[f'{service}-{slot_key}'].append(int(slot_key in real_slot_keys))

    from sklearn.metrics import precision_score, recall_score, f1_score
    for slot_key, preds in sorted(slot_key2preds.items(), key=lambda x:len(x[1]), reverse=True):
        print(f"{slot_key} {len(preds)} "
              f"{round(precision_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)} "
              f"{round(recall_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)} "
              f"{round(f1_score(y_true=slot_key2reals[slot_key], y_pred=preds), 5)} ")
