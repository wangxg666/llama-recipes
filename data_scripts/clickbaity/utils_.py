import logging

import pandas as pd

from llama.data_scripts.utils_ import *


def generate_clickbait(title):
    openai.api_key = os.environ['OPENAI_API_KEY']
    openai.api_base = "http://cooper.k8s.nb-prod.com/v1"

    model = "gpt-3.5-turbo"
    prompt = """
Here is an short title, please rewrite it with another style, 
which is attractive, exaggerate, curious, firghtened or even angry. 
you should try replacing the original words with new one, and must not add more than 3 words, 
and try not to use exaggerated adjectives, 
please answer in json format like {'rewrote text': '', 'phrase': []},

- rewrote text, for rewrote text
- phrase, for the key phrase you used to change the style

the title is 
    """ + title

    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        choice = response['choices'][0]
        content = choice['message']['content']

        # messages = [
        #     {"role": "user", "content": prompt},
        #     {'role': 'assistant', "content": content},
        #     {"role": "user", "content": "One more?"},
        # ]
        # response = openai.ChatCompletion.create(
        #     model=model,
        #     messages=messages,
        #     temperature=0,
        # )

        obj = json.loads(content)
        if 'rewrote text' not in obj or 'phrase' not in obj:
            return '', []

        rewrote_title = obj['rewrote text']
        rewrote_phrase = obj['phrase']

        if len(rewrote_title.split(' ')) >= len(title.split(' ')) * 2:
            return '', []
        return rewrote_title, rewrote_phrase
    except:
        return '', []


def rewrite_titles(day, doc2title):
    output = defaultdict(list)
    for doc, title in tqdm.tqdm(doc2title.items(), desc=f'rewrite title of {day}'):
        rewrote_title, rewrote_phrase = generate_clickbait(title)
        if not rewrote_title:
            continue
        output['doc_id'].append(doc)
        output['raw_title'].append(title)
        output['rewrote_title'].append(rewrote_title)
        output['rewrote_phrase'].append('|||'.join(rewrote_phrase))
    logging.info(f'load {len(output["doc_id"])} rewrote titles')

    return pd.DataFrame(output)