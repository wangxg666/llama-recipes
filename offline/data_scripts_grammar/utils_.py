import random

from utils.db_utils import *
from utils.str_utils import *
from utils.time_utils import *
from utils.file_utils import *
from utils.openai_utils import *

from collections import defaultdict


def generate_next_word(sentence, word):
    openai.api_key = 'nb-8jh4Y8klEqhMTXjWn-LdfUmCYhVUIVwL4VHncJWjhNmekG1MIY55DgfIs1_iCxkL2HM'
    openai.api_base = "http://cooper.k8s.nb-prod.com/v1"

    model = "gpt-3.5-turbo"
    prompt = """
Given a sentence prefix, please generate the next word,
the sentence prefix is:
    {sentence}
the next word should not be "{word}" but have similar spelling with "{word}" and start with {char}
please answer in json format like {format} with 3 candidate
    """.format(word=word, sentence=sentence, char=word[0], format='{"candidate": []}')

    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )

        choice = response['choices'][0]
        content = choice['message']['content']
        resp = json.loads(content)
        return resp['candidate']

    except:
        return []


def generate_grammatical_error_sent(sentence, fix_words = []):
    openai.api_key = 'nb-8jh4Y8klEqhMTXjWn-LdfUmCYhVUIVwL4VHncJWjhNmekG1MIY55DgfIs1_iCxkL2HM'
    openai.api_base = "http://cooper.k8s.nb-prod.com/v1"

    model = "gpt-3.5-turbo"

    other = '' if not fix_words else f'you should not change the following words: {",".join(fix_words)}'
    prompt = """
Please rewrite the sentence with grammatical errors, {other}
the sentence is:
    {sentence}
    """.format(sentence=sentence, other=other)

    print(prompt)

    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        choice = response['choices'][0]
        content = choice['message']['content']
        return content

    except:
        return sentence


def generate_data(words, retry=3):
    for i in range(retry):
        start_pos = max(3, int(random.random() * 0.8 * len(words)))
        prefix = words[0: start_pos]
        word = ''

        for title_word in words[start_pos:]:
            if title_word in stop_tokens:
                prefix.append(title_word)
            elif len(title_word) <= 3 or not title_word.isalpha():
                prefix.append(title_word)
            elif random.random() <= 0.5:
                prefix.append(title_word)
            else:
                word = title_word
                break
        if not word:
            continue
        else:
            return len(prefix), word
    return -1, ''


def collect_feature(day, n=200):
    doc2click = defaultdict(float)

    for doc, click in get_doc_by_day(day, is_push=True).items():
        doc2click[doc] += click

    for doc, click in get_doc_by_day(day, is_push=False).items():
        doc2click[doc] += click

    doc2click = {doc: click for doc, click in sorted(doc2click.items(), key=lambda x: x[1], reverse=True)[0:n]}
    doc2feature: Dict[str, StaticDocument] = pull_features_async(doc2click.keys(), return_dict=True, return_item=True)
    return doc2feature


def collect_titles(day, doc2feature):
    output = defaultdict(list)

    for doc, feature in tqdm.tqdm(doc2feature.items(), desc=f'collection title of {day}'):
        title = clean_text(feature.seg_title).lower()
        title_words = title.lower().split(' ')

        if len(title_words) < 5:
            continue

        pos, word = generate_data(title_words)
        if not word:
            continue
        up_words = generate_next_word(' '.join(title_words[0: pos]), word)
        if not up_words:
            continue

        output['doc_id'].append(doc)
        output['change_pos_list'].append(pos)
        output['change_word_list'].append('|||'.join(up_words))
        output['full_text'].append(' '.join(title_words))

    return pd.DataFrame(output)


def get_content_words(seg_content):
    content_list = seg_content.split('.')
    content_list = [clean_text(content) for content in content_list]

    content = ''
    for _c in content_list:
        content += _c + '. '
        if len(content.split(' ')) >= 100:
            break
    content_words = content.split(' ')
    return content_words


def collect_paragraphs(day, doc2feature):
    output = defaultdict(list)

    for doc, feature in tqdm.tqdm(doc2feature.items(), desc=f'collection paragraph of {day}'):
        content_words = get_content_words(feature.seg_content)
        if len(content_words) < 30:
            continue

        n = int(random.random() * 3 + 0.5)

        change_pos_list = []
        change_word_list = []
        for i in range(n):
            pos, word = generate_data(content_words)
            if not word:
                continue
            if pos in change_pos_list:
                continue
            up_words = generate_next_word(' '.join(content_words[0: pos]), word)
            if not up_words:
                continue

            change_pos_list.append(pos)
            change_word_list.append('|||'.join(up_words))

        if not change_pos_list:
            continue

        output['doc_id'].append(doc)
        output['change_pos_list'].append(','.join([str(int(x)) for x in change_pos_list]))
        output['change_word_list'].append(','.join([str(x) for x in change_word_list]))
        output['full_text'].append(' '.join(content_words))

    return pd.DataFrame(output)