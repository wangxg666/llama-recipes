import collections
import json

if __name__ == '__main__':
    service2slot_keys = json.load(open('woz_valid_slot_keys.json'))

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst/dev.act.pred.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.1.dst/dev.act.pred.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.2.dst/dev.act.pred.7b.json'
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

        services = list(set(pred_slots) | set(real_slots))
        for service in services:
            if service in pred_slots and service in real_slots:
                continue
            if service not in pred_slots:
                errors.append(f'missing_{service}')
            else:
                errors.append(f'extra_{service}')
            error = errors[-1]
            if did not in error2did2tid[error]:
                error2did2tid[error][did] = []
            error2did2tid[error][did].append(tid)

        for error in errors:
            error2count[error] += 1

    for error, count in sorted(error2count.items(), key=lambda x:x[1], reverse=True):
        print(error, 'auto', count)
        did2tid = error2did2tid[error]
        # print(did2tid)
        # if 'taxi' not in error:
        #     continue
        for did, tids in sorted(did2tid.items(), key=lambda x:len(x[1]), reverse=True)[0:10]:
            print(error, did, tids)
            print(json.dumps([f'{i+1}: {utt}' for i, utt in enumerate(did2sample[did])], indent=2))
