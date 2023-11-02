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

from configs.datasets import get_data_root


IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

YOU = 'You'
PROMPT = (
    "<<SYS>>\n"
    "You are a senior news commentator, that is good at understanding the key points of news and making accurate comments.\n"
    "Here is a news data with `title`, `content` and some comment history.\n"
    "Please make a new comment or reply to the specified user based on the command.\n"
    "### title:\n{title}\n"
    "### content:\n{content}\n"
    "<</SYS>>"
)

class NewsCommentDPODataset(Dataset):
    def __init__(self, dataset_dir, partition="train"):
        input_dir = f'{get_data_root()}/{dataset_dir}/'
        input_file = 'train' if partition == 'train' else 'valid'

        features = []
        for data in open(f'{input_dir}/{input_file}.feature'):
            features.append(json.loads(data))
        self.doc2feature = {obj['docid']: obj for obj in features}

        self.items = [json.loads(data) for data in open(f'{input_dir}/{input_file}.dialog')]
        self.items = [item for item in self.items if item['docid'] in self.doc2feature]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        feature = self.doc2feature[item['docid']]
        return NewsCommentDPODataset.process_item(item, feature)

    @staticmethod
    def process_item(item, feature):
        prompt = PROMPT.format_map(feature)
        for round in item['history']:
            user = round['user']
            reply_to = round['reply_to']
            comment = round['comment']

            round_prompt = ''
            if user == YOU:
                if reply_to:
                    round_prompt += f'r reply to {reply_to}'
                round_prompt = f'\n{user}{round_prompt}:\n{comment}'

            else:
                if reply_to:
                    round_prompt += f'\'s reply to {reply_to}'
                round_prompt += f':\n{comment}'
                round_prompt = f'\n[INST]\n{user}{round_prompt}\n[/INST]'
            prompt += round_prompt

        reply_w = item['reply_w']
        reply_l = item['reply_l']

        reply_to = reply_w['reply_to']
        if reply_to:
            prompt += f'\nYour reply to {reply_to}:\n'
        else:
            prompt += '\n You:\n'

        return {
            "prompt": prompt,
            "chosen": reply_w['comment'],
            "rejected": reply_l['comment'],
        }


if __name__ == '__main__':
    item = {
        "docid":"0pIZFupi",
        "history":[
            {
                "user":"User A",
                "reply_to":"",
                "comment":"It's okay to cut a little. But why isn't it okay to make the top pay a little more to get the budget balanced? Why should grandma take a cut on her ssi check that barley pays utilities when the mega rich don't pay theor fair share? America is going down fast and gets just what she deserves. Ignorance is bliss. Whatever happened to politicians who worked for the good of the American people. now it's all about work for your donor. Not even representing your own party anymore. Come on China, Russia, Iran, and get this over with. "
            }
        ],
        "reply_w":{
            "user":"You",
            "reply_to":"User A",
            "comment":"The majority of the country is working. The smallest class of people in America are poor and rich. The working pay into Medicare, SSI, State and Federal taxes. Some poor people paid more in taxes than president Trump. He said himself it's not his fault, he played by the ridiculous rules we have. I agree with him. When Joe the plumber pays in more than a billionaire, we have a problem. Rich shouldn't pay it all by themselves either. But realistically, they need to pay their share and some poor people may loose a little. These things are what caused the Babylonian, Persian, British, and Roman empire to fall. History repeating when the poor amd working people get tired......"
        },
        "reply_l":{
            "user":"You",
            "reply_to":"User A",
            "comment":"The top 1% all ready pays 40% of all taxes. They pay their share. Why deincentivise the largest producers"
        }
    }
    feature = {
        'title': 'Auto Parts Carry Millions in Smuggled Meth and Other Drugs Into the U.S. and Canada',
        'content': "The opioid crisis has made headlines in recent years . But fentanyl is n't the only drug fueling overdose deaths across the nation . Methamphetamine use has also grown , as have fatal overdoses . Law enforcement is cracking down on the manufacturing and distribution of these and other controlled substances . But transnational criminal organizations -LRB- TCOs -RRB- use increasingly sophisticated trafficking techniques to thwart detection . For example , TCO drug traffickers stash meth in auto parts and vehicles transported via semi-trucks to the United States and Canada . The auto parts smuggling scheme For the past few years , authorities in the United States and Canada have stopped tractor-trailers transporting auto parts or vehicles from Mexico . In June 2019 , Canadian border patrol agents seized a 400-pound shipment of methamphetamine hidden in the spare tires of 14 Ford Fusion sedans . Apparently , the Sinaloa Cartel had failed to extract the drugs from the cars before they went to nine Ontario Ford dealerships . The cartel had accessed the vehicles after their assembly at a plant in Hermosillo , Mexico , according to The Drive . And in June 2020 , U.S. Customs and Border Protection -LRB- CBP -RRB- agents seized $ 420,000 worth of meth from a tractor-trailer hauling auto parts from Nuevo Laredo , Tamaulipas , to Laredo , Texas , News4SA reports . April 19-21 , 2021 : CBP officers in Laredo foiled four separate smuggling attempts resulting in the seizure of $ 11M in narcotics . In total : ▪ 432lbs of meth ▪ 134lbs of cocaine ▪ 41lbs of heroin ▪ 16lbs of fentanyl #CBPTop5 RELEASE : https://t.co/GdMYEOyiTL pic.twitter.com/f1Cbn1yghz -- CBP -LRB- @CBP -RRB- December 28 , 2021 In addition , CBP agents seized a record-breaking amount of meth and fentanyl in a tractor-trailer at the U.S.-Mexico border in November 2021 . Carlos Martin Quintana-Arias attempted to sneak almost 18,000 pounds of the drugs through the Otay Mesa Port of Entry in San Diego . According to The Guardian , most of the drugs were meth -LRB- 17,500 pounds -RRB- , with the remaining 389 pounds consisting of fentanyl . Most recently , a Mexican national was arrested for trying to drive a semi-truck filled with meth into the U.S. His trailer , carrying auto parts , was inspected by a canine unit that detected the illicit drugs . Per CBP , the auto parts contained more than 3,280 pounds of meth bound for Arizona . How common is auto parts drug trafficking ? CBP officers at Laredo Port of Entry seized $ 2.9 M in meth and cocaine in back-to-back seizures on Feb. 16 . Details via @CBPSouthTexas : https://t.co/G2CRirIR3P pic.twitter.com/sSZWmfBRw4 -- CBP -LRB- @CBP -RRB- February 24 , 2020 According to The National Drug Intelligence Center , smuggling meth and other drugs in auto parts is common in the United States . Generally , traffickers use ground transportation . However , smuggled meth has been seized from commercial air and water transport . Mexico-based TCOs typically try to move drugs to staging areas primarily in Arizona , California , and Texas before distributing them nationally . Usually , smugglers use one of seven points of entry : Nogales in Arizona ; Calexico , Otay Mesa , and San Ysidro in California ; and Hidalgo , Laredo , and Pharr in Texas . From there , the drugs are often distributed to markets such as Phoenix , Los Angeles , San Diego , and San Francisco . Mexico-based TCOs are the primary producers and suppliers of meth available in the U.S. , and the Southwest border is the main entry point . Some domestic producers manufacture meth , but none can match the quantity , purity , or potency of the meth produced by Mexico-based TCOs . According to the DEA 's 2020 National Drug Threat Assessment , `` Commonly , traffickers transport multi-kilogram shipments of methamphetamine in privately owned vehicles . Fuel tank concealment remains a widely used technique ... Methamphetamine concealed in tires and other natural voids in vehicles are other popular methods for smuggling . '' Why is meth so dangerous ? Over three months sober after a year and a half long relapse on heroin and some short term meth use , which ravaged my face . Even weeks before I got sober I was absolutely sure I had no chance . Change can come when you least expect it to . Do n't ever believe you 're hopeless pic.twitter.com/0xxUacA1Ud -- taylor nicole dean -LRB- @taylorndean -RRB- August 14 , 2021 If you 're unfamiliar with meth -LRB- and the TV show Breaking Bad did n't scare you enough -RRB- , it 's a relatively cheap drug that can release as much as four times that amount of dopamine as cocaine . Dopamine is a neurotransmitter that sends signals of pleasure to the body . Ingesting meth stimulates the production of dopamine , which is then sent through the body 's nervous system , creating a euphoric feeling . According to the Addiction Center , a single dose costs as little as $ 5 . However , methamphetamine overdoses occur at nearly twice the rate of heroin overdoses , making meth quite deadly . Further , chronic meth use alters the areas of the brain associated with emotion and memory and can result in anxiety , confusion , insomnia , hallucinations , delusions , and violent behavior . These symptoms can last long after a user has quit meth . Weight loss , tooth decay , and skin sores are also common symptoms . Meth has historically had a high prevalence in the American West , Midwest , and Southeast . However , the Drug Enforcement Administration has noted a growing market for the drug in the Northeast . Unfortunately , its migration is also fueling a growth in overdose deaths , which , according to WebMD , nearly tripled between 2015 and 2019 . How to get help : In the U.S. , contact the Substance Abuse and Mental Health Services Administration helpline at 1-800-662-4357 .",
    }
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    out = NewsCommentDPODataset.process_item(item, feature)

    for k, v in out.items():
        print(k)
        print(v)
