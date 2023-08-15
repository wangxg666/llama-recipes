# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
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


class my_grammar_dataset:
    root = '/mnt/nlp/xingguang/mac_desk/husky-go/hallucination/data_scripts_grammar'
    dataset: str = "my_grammar_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = f"{root}/datas/train.txt"
    valid_data_path: str = f"{root}/datas/valid.txt"


class my_clickbait_dataset:
    root = '/mnt/nlp/xingguang/mac_desk/husky-go/hallucination/data_scripts_clickbaity'
    dataset: str = "my_clickbait_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = f"{root}/datas/train.txt"
    valid_data_path: str = f"{root}/datas/valid.txt"