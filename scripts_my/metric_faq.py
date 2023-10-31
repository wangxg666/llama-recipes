import argparse
import json
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import classification_report
import numpy as np


def is_default_answer(ans):
    if 'not be ans' in ans:
        return True
    for mark in [
        'not directly mentioned in the retrieved information',
        'the information provided does not',
        'the information provided is not',
        'the information is not',
        'the information does not',
        'the retrieved information provided does not',
        'the retrieved information provided is not',
        'the retrieved information is not',
        'the retrieved information does not',
        'the knowledge provided is not',
        'the knowledge provided does not',
        'Therefore, based on this information',
        'the information provided may not',
        'information provided is limited',
        'not mentioned in the given',
        'not specifically mentioned in the given',
        'not mentioned in the provided'
        'the provided knowledge does not',
        'is not provided in the given knowledge',
        'is not directly provided in the given knowledge',
        'is not specified in the provided',
        'not be determined from the given',
        'based on the given knowledge',
        'is no relevant information in the knowledge',
        'retrieved information does not provide',
        'the retrieved information provided does not',
        'the provided retrieved information does not',
        'is not provided in the given retrieved information',
        'is not directly provided in the given retrieved information',
        'based on the given retrieved information',
        'is not provided in the retrieved information',
        'is not mentioned in the retrieved information'
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
        obj = json.loads(data, strict=False)
        real = obj['real']
        pred = obj['pred']
        real_is_default = is_default_answer(real)
        pred_is_default = is_default_answer(pred)

        y_real.append(real_is_default)
        y_pred.append(pred_is_default)
        if not real_is_default and not pred_is_default:
            score = sentence_bleu([pred], real, smoothing_function=smooth.method1)
            blue_scores.append(score)
            real_lengths.append(len(real.split(' ')))
            pred_lengths.append(len(pred.split(' ')))

    print(classification_report(y_real, y_pred, target_names=['has_answer', 'default_answer']))
    print(f'avg_blue = {round(np.average(blue_scores), 5)}')
    print(f'avg_real length = {round(np.mean(real_lengths), 5)}')
    print(f'avg_pred length = {round(np.mean(pred_lengths), 5)}')
