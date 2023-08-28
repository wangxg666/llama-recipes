import re
from utils import *


def normalize_content(content):
    content = content.replace('\\n', ' ')
    content = content.replace('\n', ' ')

    content = content.replace("', '", '. ')
    for rep_token_f, re_token_t in rep_tokens.items():
        content = content.replace(rep_token_f, re_token_t)

    content = re.sub(' +', ' ', content)

    if content.startswith('[\'') or content.startswith('["'):
        content = content[2:]

    if content.endswith('\']') or content.endswith('"]'):
        content = content[:-2]

    return content


def collect_feature(day, n=200):
    doc2click = defaultdict(float)

    for doc, click in get_doc_by_day(day, is_push=True).items():
        doc2click[doc] += click

    for doc, click in get_doc_by_day(day, is_push=False).items():
        doc2click[doc] += click

    doc2click = {doc: click for doc, click in sorted(doc2click.items(), key=lambda x: x[1], reverse=True)[0:n]}
    doc2feature: Dict[str, StaticDocument] = pull_features_async(doc2click.keys(), return_dict=True, return_item=True)
    return doc2feature


def collect_titles(doc2feature):
    doc2title = {}
    for doc, feature in doc2feature.items():
        title = clean_text(feature.seg_title).lower()
        doc2title[doc] = title
    return doc2title


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


def collect_paragraphs(doc2feature):
    doc2content = {}
    for doc, feature in doc2feature.items():
        content_words = get_content_words(feature.seg_content)
        if len(content_words) < 30:
            continue

        doc2content[doc] = ' '.join(content_words)
    return doc2content