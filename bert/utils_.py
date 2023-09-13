import json
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import wandb
from sklearn.metrics import f1_score, accuracy_score


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def encode_fn(text_list):
    all_input_ids = []
    for text_a, text_b in text_list:
        input_ids = tokenizer.encode(
                        f'{text_a} [SEP] {text_b}',
                        add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                        max_length = 160,           # 设定最大文本长度
                        padding = 'max_length',        # pad到最大的长度
                        return_tensors = 'pt'       # 返回的类型为pytorch tensor
                   )
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids

labels = ['Relevant', 'Different', 'Same']
label2index = {l: i for i, l in enumerate(labels)}
index2label = {i: l for l, i in label2index.items()}

def load_data(input_file):
    text_list = []
    label_list = []
    for data in open(input_file):
        obj = json.loads(data)
        if obj['label'] not in label2index:
            continue
        text_list.append([obj['text_a'], obj['text_b']])
        label_list.append(label2index[obj['label']])
    return text_list, label_list


def flat_accuracy(preds, labels):
    """A function for calculating accuracy scores"""

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


class WanDBWriter:
    def __init__(self, name, rank):
        if rank == 0:
            wandb.init(
                project="llama",
                entity="canon",
                name=name
            )

            wandb.define_metric('step')
            wandb.define_metric('learning_rate', step_metric='step')
            wandb.define_metric('step_loss', step_metric='step')
            wandb.define_metric('valid_loss', step_metric='step')
            wandb.define_metric('valid_accuracy', step_metric='step')

    def log(self, rank, info):
        if rank == 0:
            wandb.log(info)
