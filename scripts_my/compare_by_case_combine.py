import collections
import json

if __name__ == '__main__':
    service2slot_keys = json.load(open('woz_valid_slot_keys.json'))

    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.1.dst/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline/dev.act.pred.7b.json'

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline/dev.act.pred.7b.json'
    error2count = collections.defaultdict(float)
    did2sample = {}
    for data in open(input_file.replace('.pred.7b.json', '.json')):
        obj = json.loads(data)
        did = f'{obj["dialog_id"]}'
        if did not in did2sample or len(did2sample[did]) <= len(obj['history']):
            did2sample[did] = obj['history']

    error2did2tid = collections.defaultdict(dict)

    for data in open(input_file):
        data = json.loads(data)

        key = data['key']
        did, tid = key.split('_')
        pred_slots = data['pred_act']['slots']
        real_slots = data['real_act']['slots']

        errors = []

        for pred_service, pred_slot_keys in pred_slots.items():
            if pred_service not in real_slots:
                errors.append(f'extra_{pred_service}')
            else:
                pred_slot_keys = {pred_slot_key for pred_slot_key in pred_slot_keys if
                                  pred_slot_key in service2slot_keys[pred_service]}

                for pred_slot_key in pred_slot_keys:
                    if pred_slot_key not in real_slots[pred_service]:
                        error = f'extra_{pred_service}_{pred_slot_key}'
                        errors.append(error)

        for real_service, real_slot_keys in real_slots.items():
            if real_service not in pred_slots:
                error = f'missing_{real_service}'
                errors.append(error)
            else:
                for real_slot_key in real_slot_keys:
                    pred_slot_keys = {pred_slot_key for pred_slot_key in pred_slots[real_service] if pred_slot_key in service2slot_keys[real_service]}
                    if real_slot_key not in pred_slot_keys:
                        error = f'missing_{real_service}_{real_slot_key}'
                        errors.append(error)

        error = ','.join(sorted(errors))
        if error:
            error2count[error] += 1
            if did not in error2did2tid[error]:
                error2did2tid[error][did] = []
            error2did2tid[error][did].append(tid)

    for error, count in sorted(error2count.items(), key=lambda x:x[1], reverse=True):
        print(error, 'auto', count)
        did2tid = error2did2tid[error]
        # print(did2tid)
        for did, tids in sorted(did2tid.items(), key=lambda x:len(x[1]), reverse=True)[0:10]:
            print(error, did, tids)
            print(json.dumps([f'{i+1}: {utt}' for i, utt in enumerate(did2sample[did])], indent=2))
