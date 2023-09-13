import argparse
import json
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import classification_report
import numpy as np


def is_default_ans(ans):
    if 'not be ans' in ans:
        return True
    for mark in [
        'the knowledge provided does not',
        'Therefore, based on this information',
        'information provided is limited',
        'not mentioned in the given',
        'not specifically mentioned in the given',
        'not mentioned in the provided'
        'the provided knowledge does not',
        'is not provided in the given knowledge',
        'is not directly provided in the given knowledge',
        'is not specified in the provided',
        'not be determined from the given',
        'based on the given knowledge'
    ]:
        if mark in ans:
            return True
    return False


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_file', type=str)
    args = args.parse_args()

    error = 0
    default = {
        'pred': 0.,
        'real': 0.,
        'correct': 0.
    }

    y_real, y_pred = [], []
    smooth = SmoothingFunction()

    blue_scores = []
    real_lengths, pred_lengths = [], []


    for data in open(args.input_file):
        try:
            obj = json.loads(data)
            real = json.loads(obj['real'])['answer']
            pred = json.loads(obj['pred'])['answer']
            real_is_default = is_default_ans(real)
            pred_is_default = is_default_ans(pred)

            y_real.append(real_is_default)
            y_pred.append(pred_is_default)
            if not real_is_default and not pred_is_default:
                score = sentence_bleu([pred], real, smoothing_function=smooth.method1)
                blue_scores.append(score)
                real_lengths.append(len(real.split(' ')))
                pred_lengths.append(len(pred.split(' ')))
        except:
            pass

    print(classification_report(y_real, y_pred, target_names=['has_answer', 'default_answer']))
    print(f'avg_blue = {round(np.average(blue_scores), 5)}')
    print(f'avg_real length = {round(np.mean(real_lengths), 5)}')
    print(f'avg_pred length = {round(np.mean(pred_lengths), 5)}')
