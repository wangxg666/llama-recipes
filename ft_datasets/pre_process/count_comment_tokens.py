import collections
import json

import numpy, pickle

if __name__ == '__main__':
    datas = pickle.load(open('/home/paperspace/xingguang/datasets/comment.v02/train.13b.bin', 'rb'))
    l2c = collections.defaultdict(int)

    datas = [data for data in datas if data['input_ids'].shape[0] <= 4096]
    pickle.dump(datas, open('/home/paperspace/xingguang/datasets/comment.v02/train.13b.bin.v1', 'wb'))

    for data in datas:
        input_ids = data['input_ids']
        l2c[input_ids.shape[0]] += 1

    print(json.dumps(l2c, indent=2, sort_keys=True))
