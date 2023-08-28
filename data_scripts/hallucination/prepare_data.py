import json

from utils import *
from hallucination.utils_ import *


def get_content(doc):
    obj = g_collection_document_static.find_one({'_id': doc}, {'seg_content': 1})
    if not obj:
        return ''
    content = obj.get('seg_content')
    content = normalize_content(content)
    content = cut_by_sentence(content, 512)
    return content


if __name__ == '__main__':
    col = g_mongo_rs3['aigc']['quality_check']

    # print(json.dumps(col.find_one({})))
    datas = []
    for obj in col.find({}).limit(1000):
        raw_obj = obj
        if 'request_json' not in obj:
            continue
        obj = obj['request_json']
        if 'original' not in obj or 'ai_generated' not in obj:
            continue

        raw_title = obj['original'].get('title', '')
        raw_content = obj['original'].get('content', '')

        aigc_title = obj['ai_generated'].get('title', '')
        aigc_content = obj['ai_generated'].get('content', '')

        n_words_raw = len(raw_content.strip().split(' '))
        n_words_aigc = len(aigc_content.strip().split(' '))

        if not (n_words_raw * 0.5 <= n_words_aigc <= n_words_raw * 4) or n_words_raw <= 32:
            continue

        if 'Effective: ' in raw_content or \
            'red flag' in raw_content:
            continue

        raw_content = normalize_content(raw_content)

        datas.append({
            'article_a': raw_content,
            'article_b': aigc_content,
            'output': 'Yes, the content is the same.'
        })

        if len(datas) >= 100:
            break

    for line in open('./data_pair.txt'):
        parts = line.strip().split('\t')
        doc_a, doc_b, status = parts

        content_a = get_content(doc_a)
        content_b = get_content(doc_b)
        datas.append({
            'article_a': content_a,
            'article_b': content_a,
            'output': 'Yes, the content is the same.'
        })

    human_datas = json.load(open('./human_created_data.json'))
    datas.extend(human_datas)

    with open('/mnt/nlp/xingguang/mac_desk/llama-recipes/ft_datasets/hallucination_data.json', 'w') as sout:
        sout.write(json.dumps(datas))

    with open('./data.json', 'w') as sout:
        sout.write(json.dumps(datas))