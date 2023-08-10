from pymongo import MongoClient, UpdateOne, errors
from datetime import datetime,timedelta
import pandas as pd
url = 'https://newsbreak.com/n/{}'
reddit_coll = MongoClient(
            "mongodb://rs0.mongo.nb.com:27017/?readPreference=secondaryPreferred&replicaSet=rs0"
        ).customized_crawler.reddit_posts
date_start_str = (datetime.utcnow()-timedelta(hours=24*50)).strftime("%Y-%m-%d %H:%M:%S")
doc = list(reddit_coll.find({"date":{'$gte':date_start_str},'is_bad':{"$ne":0}}))
df = pd.DataFrame(doc)
df = df.drop(columns=['_id'])
df = df.sort_values(by=['date','template_id'],ascending=False).reset_index(drop=True)
df['link'] = df['doc_id'].apply(lambda x: url.format(x))
df['removed'] = df['removed'].fillna(0).astype(int)
df = df[['doc_id', 'is_bad', 'template_id', 'removed', 'date', 'delete_reason', 'comment', 'title', 'link']]
df.to_csv('comment_feedback.csv', index=False, sep='\t')

print(len(df))
from pymongo import MongoClient

mongo = MongoClient(
    'mongodb://mongos.content.xingguang_wang:oq73xA7M55pDakL4TSFq97kuAoFKg0a9@content-mongos.mongos.nb.com/?readPreference=secondaryPreferred&authSource=admin&tls=false&connectTimeoutMS=1000&socketTimeoutMS=300000')
col_static = mongo["staticFeature"]["document"]

mongo = MongoClient("mongodb://video.mongo.nb.com/?replicaSet=rs_video&readPreference=secondaryPreferred")
col_label_data = mongo['aigc']['hallucination']

media_ids = {
    1665603
} | {
    1727941,
    1671184,
    1599052,
    1727920,
    1599088,
    1671149,
    1599070,
    1727946,
    1689908,
    1688949,
    1671183,
    1731255,
    1729765,
    1733291,
    1731372,
    1729676,
    1735965,
    1733296,
    1745966,
    1745972,
    1745980,
    1745983,
    1745955,
    1745961
}

from collections import defaultdict

doc2comments = defaultdict(set)

for obj in col_label_data.find({}):
    doc2comments[obj.get('doc_id', '')].add(obj.get('comment', ''))

# for row in pd.read_csv('./datas/2023-08-09.csv').itertuples():
#     doc2comments[row.doc_id].add(row.comment)

output = defaultdict(list)
for row in df.itertuples():
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
pd.DataFrame(output).to_csv(f'./datas/{day}.csv', index=False)

