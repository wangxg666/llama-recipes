import collections
import json


def load_errors(input_file):
    did2tid2errors = collections.defaultdict(dict)

    target_services = ['hotel', 'attraction', 'restaurant', 'train', 'taxi']

    for data in open(input_file):
        data = json.loads(data)

        key = data['key']
        did, tid = key.split('_')
        pred_slots = data['pred_act']['slots']
        real_slots = data['real_act']['slots']
        pred_slots = update_slots(pred_slots)

        if tid not in did2tid2errors[did]:
            did2tid2errors[did][tid] = []

        for pred_service, pred_slot_keys in pred_slots.items():
            if pred_service not in target_services:
                continue
            if pred_service not in real_slots:
                for pred_slot_key in pred_slot_keys:
                    error = f'{pred_service}_{pred_slot_key}_extra'
                    did2tid2errors[did][tid].append(error)
            else:
                pred_slot_keys = {pred_slot_key: pred_slot_val for pred_slot_key, pred_slot_val in
                                  pred_slot_keys.items() if
                                  pred_slot_key in service2slot_keys[pred_service]}

                for pred_slot_key in pred_slot_keys:
                    if pred_slot_key not in real_slots[pred_service]:
                        error = f'{pred_service}_{pred_slot_key}_extra, {pred_slot_keys[pred_slot_key]}'
                        did2tid2errors[did][tid].append(error)
                    elif real_slots[pred_service][pred_slot_key] != pred_slot_keys[pred_slot_key]:
                        error = f'{pred_service}_{pred_slot_key}_different, {pred_slot_keys[pred_slot_key]} ||| {real_slots[pred_service][pred_slot_key]}'
                        did2tid2errors[did][tid].append(error)

        for real_service, real_slot_keys in real_slots.items():
            if real_service not in target_services:
                continue

            if real_service not in pred_slots:
                for real_slot_key in real_slot_keys:
                    error = f'{real_service}_{real_slot_key}_miss'
                    did2tid2errors[did][tid].append(error)

            else:
                for real_slot_key in real_slot_keys:
                    pred_slot_keys = {pred_slot_key for pred_slot_key in pred_slots[real_service] if
                                      pred_slot_key in service2slot_keys[real_service]}
                    if real_slot_key not in pred_slot_keys:
                        error = f'{real_service}_{real_slot_key}_missing, {real_slot_keys[real_slot_key]}'
                        did2tid2errors[did][tid].append(error)
    return did2tid2errors


if __name__ == '__main__':
    from woz_name_config import update_slots, service2slot_keys

    input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k/test.act.pred.vllm.7b.2e-5.json'

    error2count = collections.defaultdict(float)
    did2sample = {}
    # for data in open(input_file.replace('.pred.7b.json', '.json')):
    for data in open(input_file.replace('.pred.vllm.7b.2e-5.json', '.json')):
        obj = json.loads(data)
        did = f'{obj["dialog_id"]}'
        if did not in did2sample or len(did2sample[did]) <= len(obj['history']):
            did2sample[did] = obj['history']

    base_did2tid2errors = load_errors('/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k/test.act.pred.vllm.7b.2e-5.json')
    exp_did2tid2errors = load_errors('/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_4k/test.act.pred.vllm.7b.2e-5.pre-train-8k.json')

    base_error2count, exp_error2count = collections.defaultdict(float), collections.defaultdict(float)
    for did, tid2errors in sorted(exp_did2tid2errors.items(), key=lambda x:sum([len(y) for y in x[1].values()]), reverse=True):
        # print(did)
        # print(json.dumps([f'{i + 1}: {utt}' for i, utt in enumerate(did2sample[did])], indent=2))
        # for tid, errors in tid2errors.items():
        #     print(tid, errors, base_did2tid2errors[did][tid])
        # print('\n\n')
        for tid, errors in tid2errors.items():
            for error in errors:
                exp_error2count[error.split(',')[0]] += 1
            if tid not in base_did2tid2errors[did]:
                continue
            for error in base_did2tid2errors[did][tid]:
                base_error2count[error.split(',')[0]] += 1

    for error, exp_count in sorted(exp_error2count.items(), key=lambda x:x[0]):
        print(error, exp_count, base_error2count.get(error, 0))
