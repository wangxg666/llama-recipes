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
    "You are a senior news commentator, that is good at understanding the key points of news and making accurate comments.\n"
    "Here is a news data with `title`, `content` and some comment history.\n"
    "Please make a new comment or reply to the specified user based on the command.\n"
    "### title:\n{title}\n"
    "### content:\n{content}\n"
    "<</SYS>>"
    # "You:\n{comment} </s>\n"
    # "[INST] User A reply to You: {other_coments} [/INST]\n" # 如果有多轮，重复多行
    # "[INST] User A reply to User B: {other_coments} [/INST]\n" # 如果有多轮，重复多行
    # "Your reply to User A: {reply} </s>"
)

class NewsCommentDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=4096):
        input_dir = f'{dataset_config.root}/{dataset_config.dataset_dir}/'
        input_file = 'train' if partition == 'train' else 'valid'

        features = []
        for data in gzip.open(f'{input_dir}/{input_file}.feature.gz', 'rb'):
            features.append(json.loads(str(data, 'utf8')))
        self.doc2feature = {obj['docid']: obj for obj in features}

        self.items = [json.loads(data) for data in open(f'{input_dir}/{input_file}.dialog.txt')]
        self.items = [item for item in self.items if item['docid'] in self.doc2feature]

        if partition == 'train':
            self.items = self.items[0:500]

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        feature = self.doc2feature[item['docid']]
        return NewsCommentDataset.process_item(item, feature, self.tokenizer, self.max_words)

    @staticmethod
    def process_item(item, feature, tokenizer, max_words):
        input_txt = PROMPT.format_map(feature)
        input_ids = tokenizer.encode(input_txt)

        labels = [IGNORE_INDEX for _ in input_ids]

        for round in item['dialog']:
            user = round['user']
            reply_to = round['reply_to']
            comment = round['comment']

            round_prompt = ''
            round_label = ''
            if user == YOU:
                if reply_to:
                    round_prompt += f'r reply to {reply_to}'
                round_prompt = f'\n\n{user}{round_prompt}:\n'
                round_label = f'{comment} </s>'

            else:
                if reply_to:
                    round_prompt += f'\'s reply to {reply_to}'
                round_prompt += ':\n'
                round_prompt += comment
                round_prompt = f'\n\n[INST]\n{user}{round_prompt}\n[/INST]'

            """
                input: \n\nYou:\n
                tokenize: ['▁', '<0x0A>', '<0x0A>', 'You', ':', '<0x0A>']
                input_ids: [1, 29871, 13, 13, 3492, 29901, 13]
                需要去掉的 [1, 29871, 13, 13] -> "<s> \n\n"
                加两个回车，然后通过[3:] 操作跳过前4个操作符，能够确保结果跟一次性tokenize尽可能一致
            """
            # print(tokenizer.tokenize(round_prompt))
            # print(tokenizer(round_prompt)['input_ids'])
            round_prompt_ids = tokenizer.encode(round_prompt)[3:]

            if round_label:
                # print(tokenizer.tokenize(round_prompt + round_label))
                # print(tokenizer(round_prompt + round_label)['input_ids'])
                round_prompt_label_ids = tokenizer.encode(round_prompt + round_label)[3:]
                round_label_ids = round_prompt_label_ids[len(round_prompt_ids):]
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
        if padding > 0:
            input_ids = torch.cat((input_ids, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat((labels, torch.zeros(padding, dtype=torch.int64) - 1))

            labels[~input_ids.ge(0)] = -100
            input_ids[~input_ids.ge(0)] = 0

        return {
            "input_ids": input_ids[: max_words],
            "labels": labels[: max_words],
            "attention_mask": input_ids[: max_words].ge(0),
        }



if __name__ == '__main__':
    item = {
        "docid":"0fEbGsKB",
        "dialog":[
            {
                "cid":"ranmj129o951",
                "pre_cid":"",
                "user":"You",
                "comment":"am sitting at the bar right now on my day off and there's a guy in here having a beer working from home, maybe that's why companies want you in the office",
                "reply_to":""
            },
            {
                "cid":"ranndj22e5qv",
                "pre_cid":"ranmj129o951",
                "user":"User A",
                "comment":"if that guy is being productive and getting his work done then who cares? I'm sure his boss doesn't of that's the case.",
                "reply_to":"You"
            },
            {
                "cid":"rannmv29o951",
                "pre_cid":"ranndj22e5qv",
                "user":"You",
                "comment":"I know lots of alcoholic who can function at work and still get fired what's the difference",
                "reply_to":"User A"
            },
            {
                "cid":"rannp522e5qv",
                "pre_cid":"rannmv29o951",
                "user":"User A",
                "comment":"I used to have a beer at lunch every now and then when I used to go into the office and I still got my job done. this is no different.",
                "reply_to":"You"
            },
            {
                "cid":"rannrt29o951",
                "pre_cid":"rannp522e5qv",
                "user":"You",
                "comment":"who says his boss knows he's at the bar drinking on company time 99% of companies don't let there employees drink on the clock",
                "reply_to":"User A"
            },
            {
                "cid":"rano4t22e5qv",
                "pre_cid":"rannrt29o951",
                "user":"User A",
                "comment":"and way to throw a random stat about drinking on company time which technically you're not when you're on your lunch break since most companies don't pay for the time you're out to lunch",
                "reply_to":"You"
            },
            {
                "cid":"ranodd29o951",
                "pre_cid":"rano4t22e5qv",
                "user":"You",
                "comment":"so the guy who is at the bar for a couple hours on company time is ok if your the owner of a business is what your telling me",
                "reply_to":"User A"
            },
            {
                "cid":"ranp7t22e5qv",
                "pre_cid":"ranodd29o951",
                "user":"User A",
                "comment":"so you know how much this guy was drinking? were you there for hours counting how many beers he was drinking? like I said if the guy is getting his job done I doubt he company cares but once it effects his work then the guy is going to get in trouble and possibly fired",
                "reply_to":"You"
            },
            {
                "cid":"ranpag29o951",
                "pre_cid":"ranp7t22e5qv",
                "user":"You",
                "comment":"actually I got here before the guy we talked he told me he was at work we laughed had 3 beers and 2 shots together and then he left",
                "reply_to":"User A"
            }
        ],
        "num_labels":5
    }
    feature = {
        'title': 'Auto Parts Carry Millions in Smuggled Meth and Other Drugs Into the U.S. and Canada',
        'content': "The opioid crisis has made headlines in recent years . But fentanyl is n't the only drug fueling overdose deaths across the nation . Methamphetamine use has also grown , as have fatal overdoses . Law enforcement is cracking down on the manufacturing and distribution of these and other controlled substances . But transnational criminal organizations -LRB- TCOs -RRB- use increasingly sophisticated trafficking techniques to thwart detection . For example , TCO drug traffickers stash meth in auto parts and vehicles transported via semi-trucks to the United States and Canada . The auto parts smuggling scheme For the past few years , authorities in the United States and Canada have stopped tractor-trailers transporting auto parts or vehicles from Mexico . In June 2019 , Canadian border patrol agents seized a 400-pound shipment of methamphetamine hidden in the spare tires of 14 Ford Fusion sedans . Apparently , the Sinaloa Cartel had failed to extract the drugs from the cars before they went to nine Ontario Ford dealerships . The cartel had accessed the vehicles after their assembly at a plant in Hermosillo , Mexico , according to The Drive . And in June 2020 , U.S. Customs and Border Protection -LRB- CBP -RRB- agents seized $ 420,000 worth of meth from a tractor-trailer hauling auto parts from Nuevo Laredo , Tamaulipas , to Laredo , Texas , News4SA reports . April 19-21 , 2021 : CBP officers in Laredo foiled four separate smuggling attempts resulting in the seizure of $ 11M in narcotics . In total : ▪ 432lbs of meth ▪ 134lbs of cocaine ▪ 41lbs of heroin ▪ 16lbs of fentanyl #CBPTop5 RELEASE : https://t.co/GdMYEOyiTL pic.twitter.com/f1Cbn1yghz -- CBP -LRB- @CBP -RRB- December 28 , 2021 In addition , CBP agents seized a record-breaking amount of meth and fentanyl in a tractor-trailer at the U.S.-Mexico border in November 2021 . Carlos Martin Quintana-Arias attempted to sneak almost 18,000 pounds of the drugs through the Otay Mesa Port of Entry in San Diego . According to The Guardian , most of the drugs were meth -LRB- 17,500 pounds -RRB- , with the remaining 389 pounds consisting of fentanyl . Most recently , a Mexican national was arrested for trying to drive a semi-truck filled with meth into the U.S. His trailer , carrying auto parts , was inspected by a canine unit that detected the illicit drugs . Per CBP , the auto parts contained more than 3,280 pounds of meth bound for Arizona . How common is auto parts drug trafficking ? CBP officers at Laredo Port of Entry seized $ 2.9 M in meth and cocaine in back-to-back seizures on Feb. 16 . Details via @CBPSouthTexas : https://t.co/G2CRirIR3P pic.twitter.com/sSZWmfBRw4 -- CBP -LRB- @CBP -RRB- February 24 , 2020 According to The National Drug Intelligence Center , smuggling meth and other drugs in auto parts is common in the United States . Generally , traffickers use ground transportation . However , smuggled meth has been seized from commercial air and water transport . Mexico-based TCOs typically try to move drugs to staging areas primarily in Arizona , California , and Texas before distributing them nationally . Usually , smugglers use one of seven points of entry : Nogales in Arizona ; Calexico , Otay Mesa , and San Ysidro in California ; and Hidalgo , Laredo , and Pharr in Texas . From there , the drugs are often distributed to markets such as Phoenix , Los Angeles , San Diego , and San Francisco . Mexico-based TCOs are the primary producers and suppliers of meth available in the U.S. , and the Southwest border is the main entry point . Some domestic producers manufacture meth , but none can match the quantity , purity , or potency of the meth produced by Mexico-based TCOs . According to the DEA 's 2020 National Drug Threat Assessment , `` Commonly , traffickers transport multi-kilogram shipments of methamphetamine in privately owned vehicles . Fuel tank concealment remains a widely used technique ... Methamphetamine concealed in tires and other natural voids in vehicles are other popular methods for smuggling . '' Why is meth so dangerous ? Over three months sober after a year and a half long relapse on heroin and some short term meth use , which ravaged my face . Even weeks before I got sober I was absolutely sure I had no chance . Change can come when you least expect it to . Do n't ever believe you 're hopeless pic.twitter.com/0xxUacA1Ud -- taylor nicole dean -LRB- @taylorndean -RRB- August 14 , 2021 If you 're unfamiliar with meth -LRB- and the TV show Breaking Bad did n't scare you enough -RRB- , it 's a relatively cheap drug that can release as much as four times that amount of dopamine as cocaine . Dopamine is a neurotransmitter that sends signals of pleasure to the body . Ingesting meth stimulates the production of dopamine , which is then sent through the body 's nervous system , creating a euphoric feeling . According to the Addiction Center , a single dose costs as little as $ 5 . However , methamphetamine overdoses occur at nearly twice the rate of heroin overdoses , making meth quite deadly . Further , chronic meth use alters the areas of the brain associated with emotion and memory and can result in anxiety , confusion , insomnia , hallucinations , delusions , and violent behavior . These symptoms can last long after a user has quit meth . Weight loss , tooth decay , and skin sores are also common symptoms . Meth has historically had a high prevalence in the American West , Midwest , and Southeast . However , the Drug Enforcement Administration has noted a growing market for the drug in the Northeast . Unfortunately , its migration is also fueling a growth in overdose deaths , which , according to WebMD , nearly tripled between 2015 and 2019 . How to get help : In the U.S. , contact the Substance Abuse and Mental Health Services Administration helpline at 1-800-662-4357 .",
    }
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    out = NewsCommentDataset.process_item(item, feature, tokenizer, 4096)

    print(out['labels'].tolist())
    print(out['input_ids'].tolist())
    print(tokenizer.decode([x for x in out['input_ids'] if x != -1]))
    # print(tokenizer.decode([x for x in out['labels'] if x != -1 and x != -100]).replace('</s>', '</s>\n'))
