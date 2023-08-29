# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import pathlib
from dataclasses import dataclass
import os

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"


class hallucination_dataset:
    dataset: str = "hallucination_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/hallucination_data.json"



def get_data_root():
    for path in [
        '/mnt/nlp/xingguang/llama/datasets/nb_training',
        '/home/cpp/xingguang/datasets',
        '/home/paperspace/datasets',
    ]:
        if os.path.exists(path):
            return path
    return ""


class my_allin_one_dataset:
    root: str = get_data_root()
    dataset: str = "my_allin_one_dataset"
    train_split: str = "train"
    test_split: str = "valid"
    train_data_path: str = f""
    valid_data_path: str = f""


class my_pre_train_dataset:
    root: str = get_data_root()
    dataset: str = "my_pre_train_dataset"
    train_split: str = "train"
    test_split: str = "valid"
    train_data_path: str = f""
    valid_data_path: str = f""


class my_pre_train_dataset_padding:
    root: str = get_data_root()
    dataset: str = "my_pre_train_dataset"
    train_split: str = "train"
    test_split: str = "valid"
    train_data_path: str = f""
    valid_data_path: str = f""