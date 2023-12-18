# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from functools import partial

from ft_datasets import *
from typing import Optional


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=224),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    'my_allin_one_dataset': get_my_allin_one_dataset,
    'my_pre_train_dataset': get_my_pre_train_dataset,
    'my_news_comment_dataset': get_my_news_comment_dataset,
    'my_news_comment_tokenized_dataset': get_my_news_comment_tokenized_dataset,
    'my_news_comment_dpo_dataset': get_my_news_comment_dpo_dataset,
    'my_agent_sft_dataset': get_my_agent_sft_dataset,
    'agent_sft_gen_dataset': get_agent_sft_gen_dataset,
    'agent_sft_act_dataset': get_agent_sft_act_dataset,
    'agent_sft_gen_whitening_dataset': get_agent_sft_gen_whitening_dataset,
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )


def get_processed_dpo_dataset(name):
    return DATASET_PREPROC[name]