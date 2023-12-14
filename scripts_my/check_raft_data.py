import collections
import json

if __name__ == '__main__':
    input_file = '/home/paperspace/xingguang/datasets/ppo_test.ex/data.raw.json'

    reward2count = collections.defaultdict(int)

    for data in open(input_file):
        data = json.loads(data)
        reward = data['reward']['avg_score']
        reward = int(reward)
        reward2count[reward] += 1

    print(json.dumps(reward2count, indent=2, sort_keys=True))