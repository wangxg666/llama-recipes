import collections
import json

datas = [json.loads(line) for line in open('/home/paperspace/xingguang/datasets/agent_sft.v08/train.rag.json')]

labels = []
for data in datas:
    label = str(data['label'])
    if label.startswith('Wh') and 'about' not in label and label.endswith('?'):
        labels.append(label)

label2count = collections.defaultdict(float)
for label in labels:
    label2count[' '.join(label.split(' ')[0:4])] += 1
for label, count in sorted(label2count.items(), key=lambda x:x[1], reverse=True):
    print(f'{count}\t{label}')