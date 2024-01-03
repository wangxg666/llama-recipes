import copy
import os
import json
import openai
import tiktoken
import requests
import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', )

class GPTBase:
    """
    model: gpt-4, gpt-3.5-turbo, gpt-4-1106-preview'
    """
    def __init__(self, model='gpt-3.5-turbo', retry_times=3, temperature=0.5):
        self.model = model
        self.temperature = temperature
        self.retry_times = retry_times

    def prompting(self, **kwargs):
        raise NotImplementedError

    def parsing(self, res, **kwargs):
        return json.loads(res.replace('```json\n', '').replace('```', ''))

    def post_processing(self, res, **kwargs):
        try:
            return self.parsing(res, **kwargs)
        except Exception as e:
            return None

    def get(self, **kwargs):
        prompt = self.prompting(**kwargs)
        if kwargs.get('verbose', False):
            logging.info(f'prompt is, \n {prompt}')

        content = ''
        try:
            resp = requests.post('http://35.86.252.8:1201/do', data=json.dumps({'prompt': prompt})).json()
            content = resp.get('content', '')

            if kwargs.get('verbose', False):
                logging.info(f'output is \"{json.dumps(content)}\"')

            output = self.post_processing(res=content, **kwargs)
            if kwargs.get('verbose', False):
                logging.info(f'output is \"{json.dumps(output)}\"')

            return output
        except Exception as e:
            logging.info(str(content))
            logging.info(str(e))
            pass
        return None

    def __call__(self, **kwargs):
        for i in range(self.retry_times):
            res = self.get(**kwargs)
            if res is not None:
                return res

import json
class GPTTest(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4-1106-preview')

    def prompting(self, **kwargs):
        prompt = """
You are a helpful QA robot.
Now is Dec 06, 2023, Wed, 11:36AM.
Here are a conversation between a user from Lawrenceville, Georgia(A) and you(B):
EMPTY
and the latest question from the user: Any recommendations for a dishwasher?
answer the question based on the following information you find:
[1]
source quality: unknown
[{"title": "Finish Dishwasher Detergent Gel Liquid, Lemon Scent, 75oz", "link": "title_link_0", "image": "image_link_0", "rating": "4.4 out of 5 stars", "reviews": "14,027", "price": "$6.38", "origin_price": "$0.09", "deal_info": ["$6.06 with Subscribe & Save discount\nSave 5% on 4 select item(s)"], "delivery": "FREE delivery for Prime members", "stock": "", "attrs": {}, "label": "", "prime": false}, {"title": "Finish - All in 1 - Dishwasher Detergent - Powerball - Dishwashing Tablets - Dish Tabs - Fresh Scent, 94 Count (Pack of 1) - Packaging May Vary", "link": "title_link_1", "image": "image_link_1", "rating": "4.6 out of 5 stars", "reviews": "73,063", "price": "$18.99", "origin_price": "$0.20 $21.28", "deal_info": ["$18.04 with Subscribe & Save discount"], "delivery": "FREE delivery Fri, Dec 8 ", "stock": "", "attrs": {}, "label": "", "prime": true}, {"title": "Cascade Platinum Dishwasher Cleaner, 3 count", "link": "title_link_2", "image": "image_link_2", "rating": "4.8 out of 5 stars", "reviews": "3,202", "price": "$5.44", "origin_price": "$1.81 $5.99", "deal_info": ["$5.17 with Subscribe & Save discount\nGet a $15 promotional credit when you spend at least $50.00 in promotional item(s)"], "delivery": "FREE delivery Fri, Dec 8 ", "stock": "", "attrs": {}, "label": "", "prime": true}, {"title": "Finish - Quantum - 82ct - Dishwasher Detergent - Powerball - Ultimate Clean & Shine - Dishwashing Tablets - Dish Tabs (Packaging May Vary)", "link": "title_link_3", "image": "image_link_3", "rating": "4.8 out of 5 stars", "reviews": "26,623", "price": "$16.38", "origin_price": "$0.20 $19.26", "deal_info": ["$15.56 with Subscribe & Save discount\nSave $1.63 with coupon"], "delivery": "FREE delivery for Prime members", "stock": "", "attrs": {}, "label": "", "prime": false}, {"title": "Cascade Platinum Dishwasher Pods, Detergent, Soap Pods, Actionpacs with Dishwasher Cleaner and Deodorizer Action, Fresh, 62 Count", "link": "title_link_4", "image": "image_link_4", "rating": "4.8 out of 5 stars", "reviews": "71,511", "price": "$13.96", "origin_price": "$0.23 $20.99", "deal_info": ["$13.26 with Subscribe & Save discount\nExtra $3.00 off when you subscribe"], "delivery": "FREE delivery for Prime members", "stock": "", "attrs": {}, "label": "", "prime": false}]

When you respond, follow the following rules:
1. Use only the information you've discovered to answer the question. Do not make up information.
2. Do not use times, numbers, person names, and locations that are not mentioned in the provided information.
3. If the answer isn't in the found information, inform the user and suggest related questions that the data can answer.
4. Refer to the data as "information I found" rather than "information provided".
5. Your output should be summarize the property title, sale and origin prices, rating, delivery, prime, and source of the product into one sentence.
6. Please cite your source of information, formatted as an article number enclosed in <cite></cite>, for example, <cite>1</cite>. If citing multiple sources at once, separate the numbers with a comma, like <cite>3,4</cite>.
7. Please read the the search result carefully, select the related options and output your recommendations with the search results.
- You should be summarize the property title, sale and origin prices, rating,  delivery , prime, and source of the product into one sentence.
- Then output another line with the image following each summary.
8. The title you output should be a hyper-link with the property `link` which is provided.
9. You also need to filter the non-relevant items base on the users's intent, and the dimension `gender`, `brand`, `price` are the most important filter conditions that you need take into consideration.
10. The content retrieved may contain some that is not exactly match the latest question needs. Please filter the irrelevant content when output your answer.
Only output your answer. Do not start with 'B:'. Your answer:
        """
        return prompt

    def parsing(self, res, **kwargs):
        return res

if __name__ == '__main__':
    test = GPTTest()
    print(test(verbose=True))