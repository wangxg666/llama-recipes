import argparse
import json
import os

import tqdm
from pymongo import MongoClient
from utils import *
import html


def yelp_entity_formatter(entity, reviews):
    ret = ''
    if entity.get('name', None):
        ret += f"{entity['name']}, "
    if entity.get('address', None):
        ret += f"located in {entity['address'][0]}, "
    if entity.get('telephone', None):
        ret += f"telephone number is {entity['telephone']}, "
    if entity.get('price_range', None):
        ret += f"price ranges with {entity['price_range'].lower()}, "
    if entity.get('rating', None):
        ret += f"and rating with {entity['rating']}. "

    # if 'hours' in entity and entity['hours']:
    #     hours = []
    #     for k, v in entity['hours'].items():
    #         hours.append(f"{k}: {v}")
    #     if hours:
    #         ret += 'Hours: ' + ' '.join(hours) + ' '
    if 'categories' in entity and entity['categories']:
        categories = [x['title'] for x in entity['categories'] if 'title' in x]
        if categories:
            ret += f"It's service {'category is' if len(categories) == 1 else 'categories are'} {', '.join(categories)} . "

    if 'popular_dishes' in entity and entity['popular_dishes'] and len(entity['popular_dishes']) > 0:
        ret += f"Popular {'dish is' if len(entity['popular_dishes']) == 1 else 'dishes are'}, {', '.join(x['name'] for x in entity['popular_dishes'])} . "

    if entity.get('history', None):
        history = entity['history'].strip()
        if history:
            if history[-1] != '.':
                history += '.'
            ret += f"It's history: " + history + ' '

    if reviews:
        ret += "Consumers' review(s): "
        for i, (r, _) in enumerate(sorted(reviews, key=lambda x:x[1], reverse=True)[0:5]):
            ret += f'{i+1}, ' + r.replace('<br>', '') + " "

    ret = ' '.join(ret.strip().replace(' ', ' ').split()).strip()
    ret = html.unescape(ret)
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

    names = list(set(pd.read_csv(f'{output_dir}/names.csv')['name'].tolist()))
    names.sort()

    mongo = MongoClient('mongodb://rs0.mongo.nb.com:27017/customized_crawler?readPreference=secondaryPreferred&replicaSet=rs0')
    col = mongo['yelp']['business']
    col_comm = mongo['yelp']['review']

    raws = []
    texts = []

    projection = {
        k: 1 for k in [
            'name', 'cuisine', 'price_range', 'address', 'rating', 'telephone', 'website', 'description', 'hours', 'categories', 'popular_dishes', 'reviews', 'specialties', 'history'
        ]
    }
    for name in tqdm.tqdm(list(names)):
        for obj in col.find({'name': name, 'format_address.addressRegion': args.state}, projection):
            if not obj:
                continue
            if obj['name'] not in names:
                continue

            review_objs = [review_obj for review_obj in col_comm.find({'business.id': obj['_id']}, {'comment.text': 1, 'feedback.counts.useful': 1})]
            reviews = [
                (review_obj.get('comment', {}).get('text'), review_obj.get('feedback', {}).get('counts', {}).get('useful', 0))
                for review_obj in review_objs
                if 'comment' in review_obj and review_obj.get('feedback', {}).get('counts', {}).get('useful', 0) > 0
            ]
            reviews = [review for review in reviews if len(review[0]) >= 3 and '<' not in review[0]]

            obj['reviews'] = [x[0] for x in reviews]
            raws.append(json.dumps(obj))
            texts.append(yelp_entity_formatter(obj, reviews))

    words = [len(text.replace(' ', ' ').split(' ')) for text in texts]
    print(f'raw texts size = {len(texts)}, max_word = {max(words)}, min_word = {min(words)}')

    texts = [text for text in texts if len(text.replace(' ', ' ').split(' ')) > 20]

    save_file(f'{output_dir}/jsons.txt', raws)
    save_file(f'{output_dir}/texts.txt', texts)
