import os, argparse

import numpy as np
import tqdm

import torch
import json
from urllib import request
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from pymongo import MongoClient


class ImageDataset(Dataset):
    def __init__(self, images, preprocess):
        self.images = images
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.preprocess(Image.open(self.images[idx]))  # preprocess from clip.load
        return images


def get_title(doc_id):
    mongo = MongoClient(
        'mongodb://mongos.content.xingguang_wang:oq73xA7M55pDakL4TSFq97kuAoFKg0a9@content-mongos.mongos.nb.com/?readPreference=secondaryPreferred&authSource=admin&tls=false&connectTimeoutMS=1000&socketTimeoutMS=300000')
    col_document_static = mongo["staticFeature"]["document"]

    return col_document_static.find_one({'_id': doc_id}).get('stitle', '')


def get_image_id(doc_id):
    from urllib import request

    image_id_url_template = 'http://docenter.ha.nb.com:8010/docenter/ids/%s?fields=image,_id'

    image_id_url = image_id_url_template % doc_id
    response = request.urlopen(image_id_url).read()
    meta = json.loads(response.decode("utf8"))[0]

    if 'image' not in meta:
        return ''
    image_id = meta['image']
    return image_id


def download_image(image_id):
    os.makedirs('./caches/', exist_ok=True)

    image_download_url_template = 'https://img.particlenews.com/image.php?url=%s'
    image_download_url = image_download_url_template % image_id
    filename = os.path.join('./caches/' + image_id + '.jpeg')
    request.urlretrieve(image_download_url, filename)
    return './caches/' + image_id + '.jpeg'


def norm_embed(emb):
    return emb / np.linalg.norm(emb)


if __name__ == '__main__':
    root = '/mnt/nlp/xingguang/mac_desk/nlu_embedding_new/demo-data/'

    args = argparse.ArgumentParser()
    args.add_argument('--clip_name', type=str, default='ViT-L/14')
    args.add_argument('--input_doc_id', type=str)
    args.add_argument('--input_image_id', type=str)
    args.add_argument('--input_image_file', type=str)
    args = args.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # print(torch.cuda.device_count())
    # torch.multiprocessing.set_start_method('spawn')
    global_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    import clip
    model, preprocess = clip.load("ViT-L/14", device=global_device)


    def encode_text(doc_id):
        title = get_title(doc_id)
        title = clip.tokenize([title], truncate=False).to(global_device)
        embedding = model.encode_text(title)
        embedding = embedding.detach().cpu().numpy()[0]
        return norm_embed(embedding)


    def encode_image(doc_id, image_id=None, image_file=None):
        if not image_file:
            if not image_id:
                image_id = get_image_id(doc_id)
                image_file = download_image(image_id)

        if not image_file:
            return None

        image = preprocess(Image.open(image_file)).unsqueeze(0).to(global_device)
        embedding = model.encode_image(image)
        embedding = embedding.detach().cpu().numpy()[0]

        return norm_embed(embedding)

    docs = [
        ['0npJdoDz', 0],
        ['0npc85Ty', 0],
        ['0niQMnET', 0],
        ['0niOTRZu', 0],
        ['0nh3M3uz', 0],
        ['0ngq2cXt', 0],
        ['0ng6Zr87', 0],
        ['0ndaXl8B', 0],
        ['0ngxq2sS', 1],
        ['0ngxq2sS', 1],
        ['0ncMNRea', 1],
        ['0npFRu3X', 1],
        ['0nlmbNdz', 1],
        ['0njPtU5h', 1],
        ['0nh5SzW0', 1],
        ['0ndU7bYy', 1],
        ['0nwxyBRo', 0],
        ['0nx7TBiB', 0],
        ['0nwxshiC', 0],
        ['0nx0IoGI', 0],
        ['0nxBP2zO', 0],
        ['0nxBrZZk', 0],
        ['0nx04W8n', 0],
        ['0nxA3Y0d', 0],
        ['0nxKccjn', 0],
        ['0nxNQHBB', 0],
        ['0nxMtF5x', 0],
        ['0nxNOhrf', 0],
        ['0nxLGMoA', 0],
        ['0nxPBGcF', 0],
    ]

    for doc in [
        '0nwxyBRo',
        '0nx7TBiB',
        '0nwxshiC',
        '0nx0IoGI',
        '0nxBP2zO',
        '0nxBrZZk',
        '0nx04W8n',
        '0nxA3Y0d',
        '0nxKccjn',
        '0nxNQHBB',
        '0nxMtF5x',
        '0nxNOhrf',
        '0nxPBGcF',
        '0nxLGMoA',
        '0nxNffCy',
        '0nxL7BoF',
        '0nwznnyR',
        '0nwXbxzX',
        '0nwUrV2S',
        '0nx5lhlX',
        '0nwkSYrS',
        '0nwhE3Bh',
        '0nwl1qlX',
        '0nwO8UEF',
        '0nwW4GFl',
        '0nwNjPtR',
        '0nwbccjJ',
        '0nwf3poE',
        '0nwodIAJ',
        '0nwARu3X',
        '0nwE1Q18',
        '0nwEGEkK',
        '0nwQPNR4',
        '0nwO8TLW',
        '0nwXiTWo',
        '0nwipIsb',
        '0nweuJeV',
        '0nwGZGu3',
        '0nwJYopT',
        '0nwFaJEU',
        '0nxCvPbf',
        '0nxH9XuV',
        '0nx3tvjT',
        '0nwcFdZg',
        '0nwwaNkb',
        '0nwKq5X6',
        '0nx1UYnl',
        '0nwp4dLx',
        '0nwqTm9l',
        '0nwmpPoO',
        '0nx99dGG',
        '0nwzVTGX',
        '0nwQPVFe',
        '0nx37nA6',
        '0nx0ENld',
        '0nwKrbOM',
        '0nwyjFIi',
        '0nwyjMTn',
        '0nx1TSTC',
        '0nwsyLGt',
        '0nwyqxez',
        '0nwsisOA',
        '0nx0lrLU',
        '0nwOcVVf',
        '0nwSPqal',
        '0nx9VzKV',
        '0nwrgv6m',
        '0nwfIWJD',
        '0nwxpt5t',
        '0nwYGCyo',
        '0nxAiA0P',
        '0nwHIxxS',
        '0nwNupT2',
        '0nxDo3hM',
        '0nwxnjRZ',
        '0nx4CmiA',
        '0nxJlzOw',
        '0nx7Lg1Y',
        '0nxBr84c',
        '0nx5jCfY',
        '0nwkSgyO',
        '0nxFiHBZ',
        '0nxAtYca',
        '0nx74EV4',
        '0nx2Paiz',
        '0nwa40DE',
        '0nwjHarE',
        '0nwf4EU4',
        '0nwSvzBK',
        '0nx8EK7K',
        '0nxC1A8t',
        '0nx7rrH3',
        '0nxALMYm',
        '0nx8ICug',
        '0nxDQPzr',
        '0nxAZetW',
        '0nxBZl4c',
        '0nxH9t8N',
        '0nwqUQis',
        '0nxHvetV'
    ]:
        docs.append([doc, 0])

    for doc_id, label in docs:
        try:
            txt_emb = encode_text(doc_id)
            img_emb = encode_image(doc_id)
            score = np.dot(txt_emb, img_emb)
            print(doc_id, label, score < 0.2, score)
        except:
            pass