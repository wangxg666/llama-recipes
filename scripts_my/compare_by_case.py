import collections
import json

if __name__ == '__main__':
    from woz_name_config import update_slots, service2slot_keys

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2.dst/dev.act.pred.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.1.dst/dev.act.pred.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.2.dst/dev.act.pred.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.3.dst/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v07.5.dst/dev.act.pred.vllm.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.1.dst.ctx/test.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.2.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.3.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.4.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.8.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.10.dst.ctx/dev.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.17.dst.ctx/dev.act.pred.vllm.7b.json'
    # input_file = f'/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_1k.e01/dev.act.pred.7b.json'

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline/dev.act.pred.7b.json'
    error2count = collections.defaultdict(float)
    did2sample = {}
    # for data in open(input_file.replace('.pred.7b.json', '.json')):
    for data in open(input_file.replace('.pred.vllm.7b.json', '.json')):
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
        pred_slots = update_slots(pred_slots)

        errors = []

        for pred_service, pred_slot_keys in pred_slots.items():
            if pred_service not in real_slots:
                errors.append(f'{pred_service}_extra')
            else:
                pred_slot_keys = {pred_slot_key: pred_slot_val for pred_slot_key, pred_slot_val in pred_slot_keys.items() if
                                  pred_slot_key in service2slot_keys[pred_service]}

                for pred_slot_key in pred_slot_keys:
                    if pred_slot_key not in real_slots[pred_service]:
                        error = f'{pred_service}_{pred_slot_key}_extra'
                        errors.append(error)

                        if did not in error2did2tid[error]:
                            error2did2tid[error][did] = []
                        error2did2tid[error][did].append(tid)
                    elif real_slots[pred_service][pred_slot_key] == pred_slot_keys[pred_slot_key]:
                        error = f'{pred_service}_{pred_slot_key}_value_same'
                        # errors.append(error)
                        # if did not in error2did2tid[error]:
                        #     error2did2tid[error][did] = []
                        # error2did2tid[error][did].append(tid)

        for real_service, real_slot_keys in real_slots.items():
            if real_service not in pred_slots:
                error = f'{real_service}_missing'
                errors.append(error)
                if did not in error2did2tid[error]:
                    error2did2tid[error][did] = []
                error2did2tid[error][did].append(tid)
            else:
                for real_slot_key in real_slot_keys:
                    pred_slot_keys = {pred_slot_key for pred_slot_key in pred_slots[real_service] if pred_slot_key in service2slot_keys[real_service]}
                    if real_slot_key not in pred_slot_keys:
                        error = f'{real_service}_{real_slot_key}_missing'
                        errors.append(error)
                        if did not in error2did2tid[error]:
                            error2did2tid[error][did] = []
                        error2did2tid[error][did].append(tid)

        for error in errors:
            error2count[error] += 1

    for error, count in sorted(error2count.items(), key=lambda x:x[1], reverse=True):
        print(error, 'auto', count)
        did2tid = error2did2tid[error]
        # print(did2tid)
        for did, tids in sorted(did2tid.items(), key=lambda x:len(x[1]), reverse=True)[0:100]:
            # print(error)
            print(error, did, tids)
            print(json.dumps([f'{i+1}: {utt}' for i, utt in enumerate(did2sample[did])], indent=2))
