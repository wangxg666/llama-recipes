import argparse
import json
import os
from pymongo import MongoClient
from utils import *


def yelp_entity_formatter(entity):
    def get_first_n_words(raw_text, n=50):
        split_text = raw_text.split()
        if len(split_text) > n:
            return ' '.join(split_text[:n]) + '...'
        return raw_text

    def clean_desc_text(raw_desc):
        return " ".join([x for x in raw_desc.splitlines() if len(x.strip()) > 0])

    default_keys = ['name', 'cuisine', 'price_range', 'address', 'rating', 'telephone']
    ret = ''
    for k in default_keys:
        if k in entity and entity[k] is not None:
            if k == 'address' and entity[k][0]:
                ret += f"{k.replace('_', ' ').capitalize()}: {entity[k][0]}.\n"
            else:
                ret += f"{k.replace('_', ' ').capitalize()}: {entity[k]}.\n"

    if 'hours' in entity and entity['hours'] is not None:
        hours = []
        for k, v in entity['hours'].items():
            hours.append(f"{k}: {v}")
        if hours:
            ret += 'Hours:\n' + '\n'.join(hours) + '\n'

    if 'categories' in entity and entity['categories'] is not None:
        categories = [x['title'] for x in entity['categories'] if 'title' in x]
        if categories:
            ret += f"Categories:\n{', '.join(categories)}.\n"

    if 'popular_dishes' in entity and entity['popular_dishes'] is not None and len(entity['popular_dishes']) > 0:
        ret += f"Popular dishes:\n{', '.join(x['name'] for x in entity['popular_dishes'])}.\n"

    # if 'specialties' in entity and entity['specialties']:
    #     text = [x.strip() for x in entity['specialties'].split('\n') if x.strip()]
    #     if text:
    #         ret += f'Specialities:\n' + '\n'.join(text)

    if 'reviews' in entity and isinstance(entity['reviews'], list):
        ret += "Reviews:\n"
        for r in entity['reviews'][:3]:
            ret += get_first_n_words(f"{r['date']}, {r['comment_text']}", n=50) + "\n"

    ret = ' '.join(ret.strip().replace('\n', '__N__').split())
    return ret


def save_file(out_file, datas):
    sout = open(out_file, 'w')
    for data in datas:
        sout.write(data + '\n')
    sout.close()


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--state', type=str, default='CA')
    args = args.parse_args()

    output_dir = f'/mnt/nlp/xingguang/llama/datasets/nb_raw_datas/yelp/{args.state}'
    os.makedirs(output_dir, exist_ok=True)

    sql = """
    select name, yelp_categories, city, state, yelp_rating from local_entity.dim_yelp_entity_daily_full
    where yelp_categories is not NULL and size(yelp_categories) >= 2.0 and yelp_rating > 0 and state = '{state}'
        """.format(state=args.state)
    # get_result(sql, output_file=f'{output_dir}/names.csv')

    names = set(pd.read_csv(f'{output_dir}/names.csv')['name'].tolist())

    mongo = MongoClient('mongodb://rs0.mongo.nb.com:27017/customized_crawler?readPreference=secondaryPreferred&replicaSet=rs0')
    col = mongo['yelp']['business']

    raws = []
    texts = []

    projection = {
        k: 1 for k in [
            'name', 'cuisine', 'price_range', 'address', 'rating', 'telephone', 'website', 'description', 'hours', 'categories', 'popular_dishes', 'reviews', 'specialties'
        ]
    }
    for obj in col.find({'format_address.addressRegion': args.state}, projection):
        if obj['name'] not in names:
            continue
        raws.append(json.dumps(obj))
        texts.append(yelp_entity_formatter(obj))

    words = [len(text.replace('\n', '').split(' ')) for text in texts]
    print(f'raw texts size = {len(texts)}, max_word = {max(words)}, min_word = {min(words)}')

    texts = [text for text in texts if len(text.replace('\n', '').split(' ')) > 20]

    save_file(f'{output_dir}/jsons.txt', raws)
    save_file(f'{output_dir}/texts.txt', texts)
