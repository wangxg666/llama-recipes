# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

PROMPT_DICT = {
    'dialog': (
        "Below is an instruction that describes a task, paired with an input that provides further context. \n"
        "Write a comment that appropriately reflect the news's `title` and `content`\n\n"
        "### Title:\n{title}\n\n"
        "### Content:\n{content}\n\n"
        "### Comment:\n"
    )
}


class NewsCommentDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        self.items = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.items = self.items[200:]
        else:
            self.items = self.items[:200]

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return NewsCommentDataset.process_item(item, self.tokenizer, self.max_words)

    @staticmethod
    def process_item(item, tokenizer, max_words):
        owner = item['dialog'][0]['user']
        u2nick = {owner: 'You'}
        for round in item['dialog']:
            if round['user'] not in u2nick:
                u2nick[round['user']] = f'User {len(u2nick)}'

        input_txt = PROMPT_DICT["dialog"].format_map(item)
        input_ids = tokenizer.encode(input_txt)
        labels = [IGNORE_INDEX for _ in input_ids]

        for round in item['dialog']:
            user = round['user']
            comment = round['comment']
            round_txt = f'\n{u2nick[user]}: \n{comment}'
            round_ids = tokenizer.encode(round_txt)[1:]  # remove first token id `1`

            if user == owner:
                round_labels = copy.deepcopy(round_ids)
            else:
                round_labels = [IGNORE_INDEX for _ in round_ids]

            input_ids.extend(round_ids)
            labels.extend(round_labels)

        input_ids = torch.tensor(input_ids + [tokenizer.eos_token_id], dtype=torch.int64)
        labels = torch.tensor(labels + [tokenizer.eos_token_id], dtype=torch.int64)

        padding = max_words - input_ids.shape[0]
        if padding > 0:
            input_ids = torch.cat((input_ids, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat((labels, torch.zeros(padding, dtype=torch.int64) - 1))

        return {
            "input_ids": input_ids[: max_words],
            "labels": labels[: max_words],
            "attention_mask": input_ids[: max_words].ge(0),
        }



if __name__ == '__main__':
    item = {
        "docid": "0mDBslHA",
        "title": "Is Gabapentin a Narcotic or Controlled Substance ?",
        "content": "Gabapentin is n't a narcotic or federally controlled substance , but it is regulated and recognized as a controlled substance in certain states . Gabapentin is approved by the Food and Drug Administration -LRB- FDA -RRB- to treat seizure disorders and neuropathic pain . Some people misuse the prescription medication alongside opioids to boost their effects , though this the risk of unintentional opioid poisoning and death . This has led several U.S. states to classify gabapentin as a controlled substance , with more potentially looking to do the same . There have also been calls for the Drug Enforcement Administration -LRB- DEA -RRB- to classify the medication as a federally controlled substance , though some doctors disagree with such a move . Read on to find out more about gabapentin 's current classification status across the United States and the various side effects and risks of the medication . What class of drug is gabapentin ? Gabapentin has been a federally noncontrolled substance since its FDA approval in 1993 . It 's typically used for epilepsy and nerve pain , a severe symptom that other prescription medications can often not manage . But some states do control its use , labeling gabapentin as a Schedule 5 controlled substance . Why does gabapentin 's drug class vary from state to state ? Although gabapentin is n't controlled federally , some states have listed it as a controlled substance and therefore regulate its use . That 's because there have been increasing reports of gabapentin being misused , whether by being combined with opioids or used alone for nonprescribed reasons . Some neurologists believe that stricter gabapentin regulation may lead to greater opioid use and make it harder for people with neuropathic pain to receive proper care . The following states classify gabapentin as a controlled substance : Alabama Kentucky Michigan North Dakota Virginia West Virginia Several other states require gabapentin prescriptions to be monitored , allowing authorities to detect potential misuse : Connecticut Indiana Kansas Massachusetts Minnesota Nebraska New Jersey Ohio Oregon Utah Washington , D.C. Wisconsin Wyoming These lists may be subject to change . What side effects are possible when using gabapentin ? Gabapentin is generally well tolerated and safe for most people to use . But as with any medication , there 's a risk of side effects . Misuse can increase the risk of side effects . Potential side effects include : In rare cases , more serious side effects include : long lasting stomach pain or nausea and vomiting new or worsening depression , anxiety , or irritability unusual bruising or bleeding If you experience any of the above symptoms , seek immediate medical attention or contact your local emergency services . Before taking gabapentin , tell your doctor if you : are pregnant or planning to become pregnant currently take opioids , sleep medication , or anxiety medication have diabetes , myasthenia gravis , or myoclonus have difficulty breathing or a history of respiratory conditions have a history of kidney conditions have a history of suicidal thoughts or self-harm What risks are possible when using gabapentin ? When first taking gabapentin , it 's best to be cautious when driving , using machinery , or drinking alcohol . The medication can cause drowsiness , which may affect your ability to do certain things , or have an adverse reaction when mixed with alcohol . But the biggest risks of gabapentin come when people take the medication with opioids , or if a person already has a substance use disorder . In these cases , there may be an increased risk of dependence or overdose . Serious breathing troubles can in people with respiratory conditions , like chronic obstructive pulmonary disease -LRB- COPD -RRB- or asthma , or related risk factors . Finally , there may be a higher risk of fetal cardiac abnormalities in pregnant people , according to a 2020 study . But the same study did not find evidence of a link between gabapentin use and major fetal abnormalities overall . When to consult a doctor or other healthcare professional Before taking any new medication , it 's a good idea to talk with a healthcare professional . Let them know if you currently take any opioid medication or medications for anxiety or sleep , or if you have any health conditions , such as breathing disorders , kidney disease , or diabetes . It 's important to be honest about any drug or alcohol use or misuse . This will help your clinician determine whether gabapentin is safe for you , or if there 's a better alternative . The bottom line While there have been calls to make gabapentin a controlled substance across the United States , there are currently only limitations in some states . Concerns revolve around its use alongside opioids and the potentially dangerous effects of this combination . Lauren Sharkey is a U.K.-based journalist and author specializing in women 's issues . When she is n't trying to discover a way to banish migraines , she can be found uncovering the answers to your lurking health questions . She has also written a book profiling young female activists across the globe and is currently building a community of such resisters . Catch her on Twitter .",
        "dialog": [
            {
                "cid":"s2e4rz19z541",
                "user":"Mary Dowdell",
                "comment":"So the damn answer is YES..IT IS A NARCOTIC ",
                "is_label":True
            },
            {
                "cid":"s2mlfi084qfh",
                "user":"Laura Stclair",
                "comment":"NO!ITS NOT",
                "is_label":False
            },
            {
                "cid":"s2mmju19z541",
                "user":"Mary Dowdell",
                "comment":"WHERE IM FROM YES TF IT IS ",
                "is_label":True
            }
        ]
    }

    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    NewsCommentDataset.process_item(item, tokenizer, 2048)