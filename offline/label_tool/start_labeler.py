import logging
import os.path

import gradio as gr
import pandas as pd
from utils.db_utils import *
from utils.log_utils import *

from pymongo import MongoClient


class RecordItem:
    def __init__(self, obj):
        self.iid = obj.get('_id')
        self.doc = obj.get('doc_id', '')
        self.link = obj.get('link', '')
        self.title = obj.get('title', '')
        self.content = ''
        self.comment = obj.get('comment', '')
        self.youtube_descp = obj.get('youtube_descp', '').replace('___N___', '\n')
        self.youtube_title = obj.get('youtube_title', '')
        self.youtube_trans = obj.get('youtube_trans', '').replace('___N___', '\n')
        self.template_id = obj.get('template_id', '')

        self.label = obj.get('moderation', {}).get('label', '')
        self.reason = obj.get('moderation', {}).get('reason', '')

        self.index = -1


class LabelManager:
    def __init__(self):
        self.items = []
        self.iid2item = {}

        self.template2iids = defaultdict(list)
        self.template2moderated_iids = defaultdict(set)
        self.reload('municipal_v1')

    def reload(self, template_id):
        mongo = MongoClient("mongodb://video.mongo.nb.com/?replicaSet=rs_video&readPreference=secondaryPreferred")

        filter = {}
        objs = mongo['aigc']['hallucination'].find(filter)
        self.items = [RecordItem(obj) for obj in objs]
        self.iid2item = {item.iid: item for item in self.items}

        self.template2iids = defaultdict(list)
        for item in self.items:
            self.template2iids[item.template_id].append(item.iid)
            if item.label:
                self.template2moderated_iids[item.template_id].add(item.iid)
        for tid in self.template2iids.keys():
            self.template2iids[tid].sort()

    def get_templates(self):
        return sorted(list(self.template2iids.keys()))

    def prev_doc(self, cur_idx, template_id):
        cur_idx = max(0, cur_idx - 1)
        return self.display(cur_idx, template_id)

    def next_doc(self, cur_idx, template_id):
        iids = self.template2iids.get(template_id, [])
        cur_idx = min(len(iids) - 1, cur_idx + 1)
        return self.display(cur_idx, template_id)

    def first_doc(self, template_id):
        iids = self.template2iids.get(template_id, [])
        moderated_iids = self.template2moderated_iids.get(template_id, set())

        for cur_idx, iid in enumerate(iids):
            if iid not in moderated_iids:
                return self.display(cur_idx, template_id)

    def display(self, cur_idx, template_id) -> RecordItem:
        iid = self.template2iids.get(template_id, [])[cur_idx]
        item = self.iid2item[iid]

        display_col = MongoClient(
            "mongodb://rs2.mongo.nb.com:27017/?readPreference=secondaryPreferred&replicaSet=rs2"
        )['serving']['displayDocument']

        feat = display_col.find_one({'_id': item.doc})
        if feat['content']:
            item.content = feat['content'].replace('image1.hipu.com', 'img.particlenews.com')
        item.index = cur_idx
        return item

    def progress(self, cur_idx, template_id):
        iids = self.template2iids.get(template_id, [])
        moderated_iids = self.template2moderated_iids.get(template_id, set())

        return f'Moderation progress: {len(moderated_iids)}/{len(iids)}'


label_manager = LabelManager()
valid_labels = ['Single', 'Pair', 'Other', 'Good']


with gr.Blocks() as demo:

    gr.Markdown("Hallucination Label Manager")

    with gr.Blocks():
        with gr.Row():
            slider = gr.Slider(0, 100, value=0, step=1, label=f"Progress -1/-1 ")
            btn_start = gr.Button('start')

    with gr.Blocks():

        with gr.Row():
            drop_template_id = gr.Dropdown(label_manager.get_templates(), label="template id", visible=False)
            label_iid = gr.Label(label="iid", visible=False)
            label_doc = gr.Label(label='doc_id', visible=False)
            label_cur_idx = gr.Label(label='cur_idx', visible=False)

        txt_title = gr.Text(label="title", visible=False)
        txt_comment = gr.Text(label="comment", visible=False)

        txt_youtube_title = gr.Text(label='youtube_title', visible=False)
        txt_youtube_descp = gr.Text(label='youtube_description', visible=False)

        with gr.Row():
            html_content = gr.HTML(label='content', visible=False)
            txt_youtube_trans = gr.Text(label='youtube_transcript', visible=False)

    with gr.Blocks():
        txt_label = gr.Dropdown(label="Your label", choices=valid_labels)
        txt_reason = gr.Textbox(label="Your Reason")

        with gr.Row():
            btn_prev = gr.Button('prev')
            btn_next = gr.Button('next')
            btn_submit = gr.Button('submit label')

    def update_display(record_item: RecordItem):
        visible_youtube = record_item.youtube_title != ''
        return {
            slider: gr.update(value=len(label_manager.template2moderated_iids[record_item.template_id]),
                              label=label_manager.progress(record_item.index + 1, record_item.template_id)),
            label_iid: gr.update(value=record_item.iid, visible=True),
            label_doc: gr.update(value=f'[{record_item.index + 1}]: {record_item.doc}', visible=True),
            label_cur_idx: gr.update(value=record_item.index, visible=False),

            txt_title: gr.update(value=record_item.title, visible=True),
            txt_comment: gr.update(value=record_item.comment, visible=True),
            html_content: gr.update(value=record_item.content, visible=True),

            txt_youtube_title: gr.update(value=record_item.youtube_title, visible=visible_youtube),
            txt_youtube_descp: gr.update(value=record_item.youtube_descp, visible=visible_youtube),
            txt_youtube_trans: gr.update(value=record_item.youtube_trans, visible=visible_youtube),

            txt_label: gr.update(value=record_item.label, visible=True),
            txt_reason: gr.update(value=record_item.reason, visible=True),

            btn_start: gr.update(visible=False),

            drop_template_id: gr.update(value=record_item.template_id, visible=True),
        }

    def prev_doc(cur_idx, template_id):
        idx = int(cur_idx['label'])
        logging.info(f'template = {template_id}, request prev, cur index = {idx}')
        return update_display(label_manager.prev_doc(idx, template_id))

    def next_doc(cur_idx, template_id):
        idx = int(cur_idx['label'])
        logging.info(f'template = {template_id}, request next, cur index = {idx}')
        return update_display(label_manager.next_doc(idx, template_id))

    def start_label(template_id):
        if not template_id:
            template_id = 'municipal_v1'
        logging.info(f'template = {template_id}, start_label')
        return update_display(label_manager.first_doc(template_id))

    def submit_label(iid, label, reason, template_id):
        if label not in valid_labels:
            return update_display(label_manager.first_doc(template_id))

        item = label_manager.iid2item[iid['label']]
        logging.info(f'template = {template_id}, iid = {iid["label"]}, doc = {item.doc}, label = {label}, reason = {reason}')

        mongo = MongoClient("mongodb://video.mongo.nb.com/?replicaSet=rs_video&readPreference=secondaryPreferred")

        data = {
            'moderation': {
                'label': label,
                'reason': reason
            }
        }
        mongo['aigc']['hallucination'].update_one(
            filter={
                '_id': item.iid
            },
            update={
                '$set': data
            },
            upsert=True
        )

        label_manager.reload(template_id)
        return update_display(label_manager.first_doc(template_id))

    def change_template(template_id):
        logging.info(f'template = {template_id}, change template')
        label_manager.reload(template_id)
        return start_label(template_id)


    controls = [
        slider,
        label_iid,
        label_doc,
        label_cur_idx,

        txt_title,
        txt_comment,
        txt_youtube_title,
        txt_youtube_descp,
        txt_youtube_trans,
        txt_label,
        txt_reason,

        html_content,

        drop_template_id,

        btn_start,
    ]

    btn_prev.click(prev_doc, inputs=[label_cur_idx, drop_template_id], outputs=controls)
    btn_next.click(next_doc, inputs=[label_cur_idx, drop_template_id], outputs=controls)

    btn_start.click(start_label, inputs=[drop_template_id], outputs=controls)
    btn_submit.click(submit_label, inputs=[label_iid, txt_label, txt_reason, drop_template_id], outputs=controls)

    drop_template_id.change(change_template, inputs=[drop_template_id], outputs=controls)

demo.launch(server_name='0.0.0.0', server_port=1201)