# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import json
from torch.utils.data import Dataset
from configs.datasets import get_data_root


IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

YOU = 'You'
PROMPT = "{prompt}"

class NewsCommentDPODataset(Dataset):
    def __init__(self, dataset_dir, partition="train"):
        input_dir = f'{get_data_root()}/{dataset_dir}/'
        input_file = 'train' if partition == 'train' else 'valid'
        self.items = [json.loads(data) for data in open(f'{input_dir}/{input_file}.txt')]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return NewsCommentDPODataset.process_item(item)

    @staticmethod
    def process_item(item):
        prompt = PROMPT.format_map(item)
        comment_w = item['comment_w']
        comment_l = item['comment_l']
        return {
            "prompt": prompt,
            "chosen": comment_w,
            "rejected": comment_l,
        }


if __name__ == '__main__':
    item = {
        "docid":"0pQonDTU",
        "prompt":"<<SYS>>\nYou are a senior news commentator, that is good at understanding the key points of news and making accurate comments.\nHere is a news data with `title`, `content` and some comment history.\nPlease make a new comment or reply to the specified user based on the command.\n### title:\nAn ad showing Christmas party hats burning in a fire was taken down after people compared it to the Palestinian flag\n### content:\nUK retailer M&S deleted an Instagram post of an outtake from its 2023 Christmas ad after backlash . The image , showing Christmas party hats on fire , was compared online to the Palestinian flag . M&S has since apologized . British retailer Marks & Spencer deleted an Instagram post of an outtake from its 2023 Christmas ad showing party hats with the same colors as the Palestinian flag on fire , after facing intense backlash on social media . The image -- posted on Wednesday -- and since deleted , showed hats in the traditional festive colors of red , green , and silver being burned in a fireplace , according to screenshots of the post from the BBC and The Times of London . The image was captioned : `` This Christmas , do only what you love ... like saying no to paper hats ( although if we 're honest , we 're partial . ) '' The company received a backlash on social media about the post with users drawing similarities between these hats and the colors of the Palestinian flag which is red , white , green , and black . Israel and Gaza have been embroiled in a bitter conflict since October 7 when Palestinian militant group Hamas staged a surprise attack on Israel and took about 240 hostages . The war has claimed the lives of around 9,000 Palestinians and 1,400 Israelis . Comments have flooded M&S 's other posts even after it deleted the original image saying : `` Took you 9 hours to delete a post that caused offence to thousands . The damage has been done . Do n't think many people will be shopping at M&S this Christmas . '' Another user commented : `` What a Stupid ad . The message your ` trying ' to portray could 've been portrayed so differently without upsetting so many people . '' M&S has confirmed to Insider that the image was an outtake from its Clothing & Homes advert filmed in August . `` While the intent was to playfully show that some people just do n't enjoy wearing paper Christmas hats over the festive season , we have removed the post following feedback and apologise for any unintentional hurt caused , '' the spokesperson said . The advert , which was released Wednesday featured actors Hannah Waddingham , Zawe Ashton , and Queer Eye presenter Tan France . The theme of the video appeared to be about saying no to Christmas traditions people do n't love anymore . France took to Instagram to defend the advert saying : `` The ad was shot in AUGUST so ... maybe you 're reaching with your ridiculous comments ? '' Read the original article on Business Insider\n### instructions:\n- Please make your comment with a **more cute** manner.\n<</SYS>>\nYou:\n",
        "comment_w":"But feel free to wave that American flag all day, cutie pie!",
        "comment_l":"But burn that American flag all day"
    }
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')
    out = NewsCommentDPODataset.process_item(item)

    for k, v in out.items():
        print(k)
        print(v)
