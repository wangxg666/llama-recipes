import logging

from pymongo import MongoClient, UpdateOne, errors
from datetime import datetime,timedelta
import pandas as pd
from pymongo import MongoClient
from collections import defaultdict


def pull_comments():
    url = 'https://newsbreak.com/n/{}'
    reddit_coll = MongoClient(
        "mongodb://rs0.mongo.nb.com:27017/?readPreference=secondaryPreferred&replicaSet=rs0"
    ).customized_crawler.reddit_posts
    date_start_str = (datetime.utcnow() - timedelta(hours=24 * 2)).strftime("%Y-%m-%d %H:%M:%S")
    doc = list(reddit_coll.find({"date": {'$gte': date_start_str}, 'is_bad': {"$ne": 0}}))
    df = pd.DataFrame(doc)
    df = df.drop(columns=['_id'])
    df = df.sort_values(by=['date', 'template_id'], ascending=False).reset_index(drop=True)
    df['link'] = df['doc_id'].apply(lambda x: url.format(x))
    df['removed'] = df['removed'].fillna(0).astype(int)
    return df[['doc_id', 'is_bad', 'template_id', 'removed', 'date', 'comment', 'title', 'link']]


def dump_exist_doc_comments():
    mongo = MongoClient("mongodb://video.mongo.nb.com/?replicaSet=rs_video&readPreference=secondaryPreferred")
    col_label_data = mongo['aigc']['hallucination']

    doc2comments = defaultdict(set)

    for obj in col_label_data.find({}):
        doc2comments[obj.get('doc_id', '')].add(obj.get('comment', ''))

    return doc2comments


def main():
    df_comments = pull_comments()
    doc2comments = dump_exist_doc_comments()

    mongo = MongoClient(
        'mongodb://mongos.content.xingguang_wang:oq73xA7M55pDakL4TSFq97kuAoFKg0a9@content-mongos.mongos.nb.com/?readPreference=secondaryPreferred&authSource=admin&tls=false&connectTimeoutMS=1000&socketTimeoutMS=300000')
    col_static = mongo["staticFeature"]["document"]

    media_ids = {
                    int(x) for x in open('./datas/municipal.media_ids.txt')
                } | {
                    int(x) for x in open('./datas/youtube.media_ids.txt')
                }

    output = defaultdict(list)
    for row in df_comments.itertuples():
        obj = col_static.find_one({'_id': row.doc_id}, {'media_id': 1})
        if not obj:
            continue
        if obj.get('media_id', -1) not in media_ids and row.template_id != 'municipal_v1':
            continue
        doc = row.doc_id
        comment = str(row.comment)
        if doc in doc2comments and comment in doc2comments[doc]:
            continue
        if row.removed == 1:
            continue

        output['doc_id'].append(str(row.doc_id))
        output['template_id'].append(str(row.template_id))
        output['removed'].append(str(row.removed))
        output['comment'].append(str(row.comment))
        output['title'].append(str(row.title))
        output['link'].append(str(row.link))
        output['date'].append(str(row.date))

    from utils.time_utils import now
    day = now(True)[0:10]
    df = pd.DataFrame(output)
    logging.info(f'total load {len(df_comments)} comments, with {len(df)} datas for moderation')
    df.to_csv(f'/mnt/nlp/xingguang/mac_desk/husky-go/llama/data_scripts/moderation/datas/{day}.csv', index=False)


if __name__ == '__main__':
    main()