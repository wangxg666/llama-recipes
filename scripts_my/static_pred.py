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
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.1.dst.ctx/test.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.2.dst.ctx/test.act.pred.vllm.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v08.3.dst.ctx/test.act.pred.vllm.7b.json'
    error2count = collections.defaultdict(float)
    did2sample = {}
    # for data in open(input_file.replace('.pred.7b.json', '.json')):
    for data in open(input_file.replace('.pred.vllm.7b.json', '.json')):
        obj = json.loads(data)
        did = f'{obj["dialog_id"]}'
        if did not in did2sample or len(did2sample[did]) <= len(obj['history']):
            did2sample[did] = obj['history']

    error2did2tid = collections.defaultdict(dict)

    service2slot_keys = json.load(open('woz_valid_slot_keys.json'))


    service2count = collections.defaultdict(float)
    service2slot_key2count = {}

    for data in open(input_file):
        data = json.loads(data)
        real_slots = data['real_act']['slots']

        for service, slots in real_slots.items():
            if not slots:
                continue
            service2count[service] += 1
            if service not in service2slot_key2count:
                service2slot_key2count[service] = collections.defaultdict(float)
            slot_keys = ','.join(sorted([x for x in list(slots) if 'book' not in x]))
            service2slot_key2count[service][slot_keys] += 1

    for sevice, count in service2count.items():
        for slot_key, sk_count in service2slot_key2count[sevice].items():
            print(sevice, slot_key, count, sk_count, round(float(sk_count) / count))