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
    model: gpt-4, gpt-3.5-turbo, gpt-3.5-turbo'
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
        super().__init__(model='gpt-3.5-turbo')

    def prompting(self, **kwargs):
        return kwargs['prompt']

    def parsing(self, res, **kwargs):
        return res

if __name__ == '__main__':
    test = GPTTest()
    prompt = """
Translate the following into English:
下面是状态以及对应的行为定义
Your translated English:    
"""
    print(test(prompt=prompt, verbose=True))