import json

objs = [
    json.loads(data) for data in open('/home/paperspace/xingguang/datasets/answer_extractor.v016/train.txt')
]
answers = {obj['label'] for obj in objs if obj['type'] == "FAQ_ANSWER_EXTRACT"}


objs = [
    json.loads(data) for data in open('/home/paperspace/xingguang/datasets/answer_extractor.v017/train.txt')
]
answers1 = {obj['label'] for obj in objs if obj['type'] == "FAQ_ANSWER_EXTRACT"}


for a in answers1:
    if a not in answers:
        print(a)