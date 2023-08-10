import re
from utils.str_utils import *


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