import os

from pymongo import MongoClient
import pandas as pd
from collections import defaultdict


def md5_str(text) -> str:
    import hashlib
    m = hashlib.md5()
    m.update(text.encode("utf-8"))
    return m.hexdigest()


if __name__ == '__main__':
    mongo = MongoClient("mongodb://video.mongo.nb.com/?replicaSet=rs_video&readPreference=secondaryPreferred")
    col_label_data = mongo['aigc']['hallucination']

    doc2comments = defaultdict(set)

    for obj in col_label_data.find({}):
        doc2comments[obj.get('doc_id', '')].add(obj.get('comment', ''))

    datas = []

    for filename in os.listdir('./datas'):
        if 'info.csv' not in filename:
            continue
        for row in pd.read_csv(f'./datas/{filename}').itertuples():
            iid = md5_str(row.doc_id + ' ' + str(row.comment))

            if row.doc_id in doc2comments and row.comment in doc2comments[row.doc_id]:
                continue

            if str(row.doc_id) == 'nan' or str(row.link) == 'nan' or str(row.template_id) == 'nan':
                continue
            datas.append({
                '_id': iid,
                'doc_id': str(row.doc_id),
                'template_id': str(row.template_id),
                'comment': str(row.comment),
                'title': str(row.title),
                'link': str(row.link),
                'youtube_title': '' if str(row.youtube_title) == 'nan' else str(row.youtube_title),
                'youtube_descp': '' if str(row.youtube_descp) == 'nan' else str(row.youtube_descp),
                'youtube_trans': '' if str(row.youtube_trans) == 'nan' else str(row.youtube_trans),
            })

    for data in datas:
        mongo['aigc']['hallucination'].update_one(
            filter={
                '_id': data['_id']
            },
            update={
                '$set': data
            },
            upsert=True
        )