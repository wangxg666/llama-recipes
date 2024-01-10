import collections
import copy
import json
from typing import Dict, Any

from fuzzywuzzy import fuzz

def add_idx_prefix_to_bullet_para(para:str, idx:int):
    beautified_para = []
    prefix = f'{idx}: '
    for idx, line in enumerate(para.splitlines()):
        if idx == 0:
            beautified_para.append(prefix + line)
        else:
            beautified_para.append(' '*len(prefix) + line)
            
    return "\n".join(beautified_para).strip()

def squeeze_dict_of_dict(target_dict: Dict[str, Dict[str, Any]]):
    squuezed_dict = dict()
    
    for outer_key, outer_value in target_dict.items():
        for inner_key, inner_value in outer_value.items():
            squuezed_dict[f'{outer_key}.{inner_key}'] = inner_value
            
    return squuezed_dict

def compare(squeezed_dict_pred: Dict[str, str], squeezed_dict_label: Dict[str, str]):
    find_error = False
    logs = []
    
    for k, pred_v in squeezed_dict_pred.items():
        if k in squeezed_dict_label and fuzz.partial_ratio(pred_v, squeezed_dict_label[k]) <= 90:
            find_error = True
            logs.append(f'[{k}] values mismatch: PRED:[{pred_v}] LABEL:[{squeezed_dict_label[k]}], SCORE:{fuzz.partial_ratio(pred_v, squeezed_dict_label[k])}')
        # elif k in squeezed_dict_label and pred_v != squeezed_dict_label[k]:
        #     print(pred_v, "||", squeezed_dict_label[k], fuzz.partial_ratio(pred_v, squeezed_dict_label[k]), "*" * 10)
            
    return find_error, "\n".join(logs).strip()

if __name__ == '__main__':
    service2slot_keys = json.load(open('woz_valid_slot_keys.json'))

    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.5.2/dev.act.pred.7b.json'
    # input_file = '/home/paperspace/xingguang/datasets/agent_sft.v10.baseline/dev.act.pred.7b.json'
    input_file = '/home/paperspace/xingguang/datasets/agent_sft.auto.gen.v05.6/dev.act.pred.7b.json'
    error2count = collections.defaultdict(float)
    d_id_to_sample_hist = {}
    for data in open(input_file.replace('.pred.7b.json', '.json')):
        obj = json.loads(data)
        d_id = f'{obj["dialog_id"]}'
        if d_id not in d_id_to_sample_hist or len(d_id_to_sample_hist[d_id]) <= len(obj['history']):
            d_id_to_sample_hist[d_id] = obj['history']

    error2did2tid = collections.defaultdict(dict)

    for data in open(input_file):
        data = json.loads(data)

        key = data['key']
        d_id, turn_id = key.split('_')
        pred_slots = data['pred_act']['slots']
        real_slots = data['real_act']['slots']
        
        squeezed_pred_slots = squeeze_dict_of_dict(pred_slots)
        squeezed_real_slots = squeeze_dict_of_dict(real_slots)
        
        find_error, logs = compare(squeezed_pred_slots, squeezed_real_slots)
        
        if find_error:
            error2did2tid[d_id][turn_id] = logs
            
    for dialog_id, turns_record in error2did2tid.items():
        dialog_report = "\n".join([add_idx_prefix_to_bullet_para(turn_logs, turn_id) for turn_id, turn_logs in turns_record.items()]).strip()
            
        print("*" * 10)
        print(f"Dialogue ID: {dialog_id}")
        print(dialog_report)
        
        marked_history = copy.deepcopy(d_id_to_sample_hist[dialog_id])
        marked_history = "\n".join([f'{idx+1}: {utt}' for idx, utt in enumerate(marked_history)]).strip()
        
        print(f'\n{marked_history}')
        print("-" * 10, "\n\n")