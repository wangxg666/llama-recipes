# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import gzip
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
from typing import List

IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

YOU = 'You'
PROMPT = (
    "<<SYS>>\n"
    "You are an excellent Agent for local bussiness.\n"
    "Here is a local business conversation sence, please response to the user based on the dialogure history.\n"
    "<</SYS>>"
)

class AgentSFTDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=4096):
        input_dir = f'{dataset_config.root}/{dataset_config.dataset_dir}/'
        input_file = 'train.json' if partition == 'train' else 'dev.json'

        self.items = [json.loads(data) for data in open(f'{input_dir}/{input_file}')]

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return AgentSFTDataset.process_item(item, self.tokenizer, self.max_words)

    @staticmethod
    def process_item(item, tokenizer, max_words, do_padding=True):
        input_txt = PROMPT
        input_ids = tokenizer.encode(input_txt)

        labels = [IGNORE_INDEX for _ in input_ids]

        for turn in item['turns']:
            turn_id = turn['turn_id']
            speaker = turn['speaker']
            utterance = turn['utterance']
            round_label = ''

            if speaker == 'SYSTEM':
                round_label = f'{utterance}</s>'
                round_prompt = f'\n\n{speaker}:\n'
            else:
                round_prompt = f'\n\n[INST]\n{speaker}: \n{utterance}\n[/INST]'

            """
                input: \n\nYou:\n
                tokenize: ['▁', '<0x0A>', '<0x0A>', 'You', ':', '<0x0A>']
                input_ids: [1, 29871, 13, 13, 3492, 29901, 13]
                需要去掉的 [1, 29871, 13, 13] -> "<s> \n\n"
                加两个回车，然后通过[3:] 操作跳过前4个操作符，能够确保结果跟一次性tokenize尽可能一致
            """
            round_prompt_ids = tokenizer.encode(round_prompt)[3:]

            if round_label:
                round_input_ids = tokenizer.encode(round_prompt + round_label)[3:]
                round_label_ids = round_input_ids[len(round_prompt_ids):]
            else:
                round_label_ids = []

            input_ids.extend(round_prompt_ids + round_label_ids)
            labels.extend([IGNORE_INDEX for _ in round_prompt_ids] + round_label_ids)

        if input_ids[-1] != tokenizer.eos_token_id:
            input_ids.append(tokenizer.eos_token_id)
            labels.append(tokenizer.eos_token_id)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        padding = max_words - input_ids.shape[0]
        if padding > 0 and do_padding:
            input_ids = torch.cat((input_ids, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat((labels, torch.zeros(padding, dtype=torch.int64) + IGNORE_INDEX))

        input_mask = input_ids.ge(0)
        input_ids[~input_mask] = 0

        return {
            "input_ids": input_ids[: max_words],
            "labels": labels[: max_words],
            "attention_mask": input_mask[: max_words],
        }



if __name__ == '__main__':
    item = {
        "dialogue_id":"PMUL4398.json",
        "services":[
            "restaurant",
            "hotel"
        ],
        "turns":[
            {
                "turn_id":"0",
                "speaker":"USER",
                "utterance":"i need a place to dine in the center thats expensive"
            },
            {
                "turn_id":"1",
                "speaker":"SYSTEM",
                "utterance":"I have several options for you; do you prefer African, Asian, or British food?"
            },
            {
                "turn_id":"2",
                "speaker":"USER",
                "utterance":"Any sort of food would be fine, as long as it is a bit expensive. Could I get the phone number for your recommendation?"
            },
            {
                "turn_id":"3",
                "speaker":"SYSTEM",
                "utterance":"There is an Afrian place named Bedouin in the centre. How does that sound?"
            },
            {
                "turn_id":"4",
                "speaker":"USER",
                "utterance":"Sounds good, could I get that phone number? Also, could you recommend me an expensive hotel?"
            },
            {
                "turn_id":"5",
                "speaker":"SYSTEM",
                "utterance":"Bedouin's phone is 01223367660. As far as hotels go, I recommend the University Arms Hotel in the center of town."
            },
            {
                "turn_id":"6",
                "speaker":"USER",
                "utterance":"Yes. Can you book it for me?"
            },
            {
                "turn_id":"7",
                "speaker":"SYSTEM",
                "utterance":"Sure, when would you like that reservation?"
            },
            {
                "turn_id":"8",
                "speaker":"USER",
                "utterance":"i want to book it for 2 people and 2 nights starting from saturday."
            },
            {
                "turn_id":"9",
                "speaker":"SYSTEM",
                "utterance":"Your booking was successful. Your reference number is FRGZWQL2 . May I help you further?"
            },
            {
                "turn_id":"10",
                "speaker":"USER",
                "utterance":"That is all I need to know. Thanks, good bye."
            },
            {
                "turn_id":"11",
                "speaker":"SYSTEM",
                "utterance":"Thank you so much for Cambridge TownInfo centre. Have a great day!"
            }
        ]
    }
    feature = {
        'title': 'Auto Parts Carry Millions in Smuggled Meth and Other Drugs Into the U.S. and Canada',
        'content': "The opioid crisis has made headlines in recent years . But fentanyl is n't the only drug fueling overdose deaths across the nation . Methamphetamine use has also grown , as have fatal overdoses . Law enforcement is cracking down on the manufacturing and distribution of these and other controlled substances . But transnational criminal organizations -LRB- TCOs -RRB- use increasingly sophisticated trafficking techniques to thwart detection . For example , TCO drug traffickers stash meth in auto parts and vehicles transported via semi-trucks to the United States and Canada . The auto parts smuggling scheme For the past few years , authorities in the United States and Canada have stopped tractor-trailers transporting auto parts or vehicles from Mexico . In June 2019 , Canadian border patrol agents seized a 400-pound shipment of methamphetamine hidden in the spare tires of 14 Ford Fusion sedans . Apparently , the Sinaloa Cartel had failed to extract the drugs from the cars before they went to nine Ontario Ford dealerships . The cartel had accessed the vehicles after their assembly at a plant in Hermosillo , Mexico , according to The Drive . And in June 2020 , U.S. Customs and Border Protection -LRB- CBP -RRB- agents seized $ 420,000 worth of meth from a tractor-trailer hauling auto parts from Nuevo Laredo , Tamaulipas , to Laredo , Texas , News4SA reports . April 19-21 , 2021 : CBP officers in Laredo foiled four separate smuggling attempts resulting in the seizure of $ 11M in narcotics . In total : ▪ 432lbs of meth ▪ 134lbs of cocaine ▪ 41lbs of heroin ▪ 16lbs of fentanyl #CBPTop5 RELEASE : https://t.co/GdMYEOyiTL pic.twitter.com/f1Cbn1yghz -- CBP -LRB- @CBP -RRB- December 28 , 2021 In addition , CBP agents seized a record-breaking amount of meth and fentanyl in a tractor-trailer at the U.S.-Mexico border in November 2021 . Carlos Martin Quintana-Arias attempted to sneak almost 18,000 pounds of the drugs through the Otay Mesa Port of Entry in San Diego . According to The Guardian , most of the drugs were meth -LRB- 17,500 pounds -RRB- , with the remaining 389 pounds consisting of fentanyl . Most recently , a Mexican national was arrested for trying to drive a semi-truck filled with meth into the U.S. His trailer , carrying auto parts , was inspected by a canine unit that detected the illicit drugs . Per CBP , the auto parts contained more than 3,280 pounds of meth bound for Arizona . How common is auto parts drug trafficking ? CBP officers at Laredo Port of Entry seized $ 2.9 M in meth and cocaine in back-to-back seizures on Feb. 16 . Details via @CBPSouthTexas : https://t.co/G2CRirIR3P pic.twitter.com/sSZWmfBRw4 -- CBP -LRB- @CBP -RRB- February 24 , 2020 According to The National Drug Intelligence Center , smuggling meth and other drugs in auto parts is common in the United States . Generally , traffickers use ground transportation . However , smuggled meth has been seized from commercial air and water transport . Mexico-based TCOs typically try to move drugs to staging areas primarily in Arizona , California , and Texas before distributing them nationally . Usually , smugglers use one of seven points of entry : Nogales in Arizona ; Calexico , Otay Mesa , and San Ysidro in California ; and Hidalgo , Laredo , and Pharr in Texas . From there , the drugs are often distributed to markets such as Phoenix , Los Angeles , San Diego , and San Francisco . Mexico-based TCOs are the primary producers and suppliers of meth available in the U.S. , and the Southwest border is the main entry point . Some domestic producers manufacture meth , but none can match the quantity , purity , or potency of the meth produced by Mexico-based TCOs . According to the DEA 's 2020 National Drug Threat Assessment , `` Commonly , traffickers transport multi-kilogram shipments of methamphetamine in privately owned vehicles . Fuel tank concealment remains a widely used technique ... Methamphetamine concealed in tires and other natural voids in vehicles are other popular methods for smuggling . '' Why is meth so dangerous ? Over three months sober after a year and a half long relapse on heroin and some short term meth use , which ravaged my face . Even weeks before I got sober I was absolutely sure I had no chance . Change can come when you least expect it to . Do n't ever believe you 're hopeless pic.twitter.com/0xxUacA1Ud -- taylor nicole dean -LRB- @taylorndean -RRB- August 14 , 2021 If you 're unfamiliar with meth -LRB- and the TV show Breaking Bad did n't scare you enough -RRB- , it 's a relatively cheap drug that can release as much as four times that amount of dopamine as cocaine . Dopamine is a neurotransmitter that sends signals of pleasure to the body . Ingesting meth stimulates the production of dopamine , which is then sent through the body 's nervous system , creating a euphoric feeling . According to the Addiction Center , a single dose costs as little as $ 5 . However , methamphetamine overdoses occur at nearly twice the rate of heroin overdoses , making meth quite deadly . Further , chronic meth use alters the areas of the brain associated with emotion and memory and can result in anxiety , confusion , insomnia , hallucinations , delusions , and violent behavior . These symptoms can last long after a user has quit meth . Weight loss , tooth decay , and skin sores are also common symptoms . Meth has historically had a high prevalence in the American West , Midwest , and Southeast . However , the Drug Enforcement Administration has noted a growing market for the drug in the Northeast . Unfortunately , its migration is also fueling a growth in overdose deaths , which , according to WebMD , nearly tripled between 2015 and 2019 . How to get help : In the U.S. , contact the Substance Abuse and Mental Health Services Administration helpline at 1-800-662-4357 .",
    }
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    out = AgentSFTDataset.process_item(item, tokenizer, 4096)

    print(out['labels'].tolist())
    print(out['input_ids'].tolist())
    print(out['attention_mask'].tolist())
    print(tokenizer.decode([x for x in out['input_ids'] if x != -1 and x != 0]))
    # print(tokenizer.decode([x for x in out['labels'] if x != -1 and x != -100]).replace('</s>', '</s>\n'))
