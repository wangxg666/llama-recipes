import datetime
import os

from pymongo import MongoClient
import pandas as pd
from collections import defaultdict

from utils import *


def md5_str(text) -> str:
    import hashlib
    m = hashlib.md5()
    m.update(text.encode("utf-8"))
    return m.hexdigest()


def post_slack(tid2count):
    from slack_sdk import WebClient

    member_ids = {
        "canon": "U02EV139CPK"
    }
    client = WebClient("xoxb-4818648140-5733850261559-AIJf14wtYagBcMuOVCgxYg7a")

    def send_message_to_channel(channel, text):
        return client.chat_postMessage(
            channel="#{}".format(channel),
            text=text
        )

    text = f"*Daily Document Update* \n"
    datas = sorted(tid2count.items(), key=lambda x:x[0])
    for tid, count in datas:
        text += f'    {tid}: {count}\n'

    send_message_to_channel("aigc-notifications", text)


if __name__ == '__main__':
    mongo = MongoClient("mongodb://video.mongo.nb.com/?replicaSet=rs_video&readPreference=secondaryPreferred")
    col_label_data = mongo['aigc']['hallucination']

    doc2comments = defaultdict(set)

    for obj in col_label_data.find({}):
        doc2comments[obj.get('doc_id', '')].add(obj.get('comment', ''))

    datas = []

    day = datetime.datetime.now().strftime('%Y-%m-%d')

    filename = f'/mnt/nlp/xingguang/mac_desk/husky-go/llama/data_scripts/moderation/datas/{day}.info.csv'
    tid2count = defaultdict(float)

    if os.path.exists(filename):
        for row in pd.read_csv(filename).itertuples():
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
                'insert_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            tid2count[str(row.template_id)] += 1

    logging.info(f'data size = {len(datas)}')

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

    post_slack(tid2count)