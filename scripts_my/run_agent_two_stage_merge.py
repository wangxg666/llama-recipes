import collections
import json

if __name__ == '__main__':
    input_dir = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline.dst.limit_8k'

    key2pred = {}
    for data in open(f'{input_dir}/test.act.pred.vllm.7b.2e-5.pre-train-8k.json'):
        obj = json.loads(data)
        key = obj['key']
        key2pred[key] = {
            'state': obj['pred_act']['slots'],
            'active_domains': [obj['pred_act']['current_service']]
        }

    for input_file in [
        f'{input_dir}/test.rag.pred.json',
        f'{input_dir}/test.casual.pred.json',
    ]:
        for obj in open(input_file):
            obj = json.loads(obj)
            key = obj['key']
            key2pred[key]['response'] = obj['pred_resp']

    did2preds = collections.defaultdict(list)
    for key, pred in key2pred.items():
        did, tid = key.split('_')
        tid = int(tid)
        did2preds[did].append([tid, pred])

    for did in list(did2preds.keys()):
        preds = sorted(did2preds[did], key=lambda x:x[0])
        preds = [pred[1] for pred in preds]
        did2preds[did] = preds

    print(json.dumps(did2preds, indent=2), file=open(f'{input_dir}/test.pred.merge.json', 'w'), flush=True)