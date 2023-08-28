from llama.data_scripts.utils_ import *
import copy


def generate_next_word(sentence, word):
    openai.api_key = os.environ['OPENAI_API_KEY']
    openai.api_base = "http://cooper.k8s.nb-prod.com/v1"

    model = "gpt-3.5-turbo"
    prompt = """
Given a sentence prefix, please generate the next word,
the sentence prefix is:
    {sentence}
the next word should not be "{word}" but have similar spelling with "{word}" and start with {char}
please answer in json format like {format} with 3 candidate
    """.format(word=word, sentence=sentence, char=word[0], format='{"candidate": []}')

    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )

        choice = response['choices'][0]
        content = choice['message']['content']
        resp = json.loads(content)
        return resp['candidate']

    except:
        return []


def generate_grammatical_error_sent(sentence, fix_words = []):
    openai.api_key = os.environ['OPENAI_API_KEY']
    openai.api_base = "http://cooper.k8s.nb-prod.com/v1"

    model = "gpt-3.5-turbo"

    other = '' if not fix_words else f'you should not change the following words: {",".join(fix_words)}'
    prompt = """
Please rewrite the sentence with grammatical errors, {other}
the sentence is:
    {sentence}
    """.format(sentence=sentence, other=other)

    print(prompt)

    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        choice = response['choices'][0]
        content = choice['message']['content']
        return content

    except:
        return sentence


def generate_data(words, retry=3):
    for i in range(retry):
        start_pos = max(3, int(random.random() * 0.8 * len(words)))
        prefix = words[0: start_pos]
        word = ''

        for title_word in words[start_pos:]:
            if title_word in stop_tokens:
                prefix.append(title_word)
            elif len(title_word) <= 3 or not title_word.isalpha():
                prefix.append(title_word)
            elif random.random() <= 0.5:
                prefix.append(title_word)
            else:
                word = title_word
                break
        if not word:
            continue
        else:
            return len(prefix), word
    return -1, ''


def rewrite_titles(day, doc2title):
    output = defaultdict(list)
    for doc, title in tqdm.tqdm(doc2title.items(), desc=f'rewrite title of {day}'):
        title_words = title.lower().split(' ')

        if len(title_words) < 5:
            continue

        pos, word = generate_data(title_words)
        if not word:
            continue
        up_words = generate_next_word(' '.join(title_words[0: pos]), word)
        if not up_words:
            continue

        output['doc_id'].append(doc)
        output['change_pos_list'].append(pos)
        output['change_word_list'].append('|||'.join(up_words))
        output['full_text'].append(' '.join(title_words))

    return pd.DataFrame(output)


def rewrite_content(day, doc2content):
    output = defaultdict(list)
    for doc, content in tqdm.tqdm(doc2content.items(), desc=f'rewrite content of {day}'):
        content_words = content.split(' ')
        change_pos_list = []
        change_word_list = []

        n = int(random.random() * 3 + 0.5)

        for i in range(n):
            pos, word = generate_data(content_words)
            if not word:
                continue
            if pos in change_pos_list:
                continue
            up_words = generate_next_word(' '.join(content_words[0: pos]), word)
            if not up_words:
                continue

            change_pos_list.append(pos)
            change_word_list.append('|||'.join(up_words))

        if not change_pos_list:
            continue

        output['doc_id'].append(doc)
        output['change_pos_list'].append(','.join([str(int(x)) for x in change_pos_list]))
        output['change_word_list'].append(','.join([str(x) for x in change_word_list]))
        output['full_text'].append(' '.join(content_words))
    return pd.DataFrame(output)


def save_file(out_file, datas):
    sout = open(out_file, 'w')
    for data in datas:
        sout.write(json.dumps(data) + '\n')
    sout.close()


class Candidate:
    def __init__(self, source_sent, target_sent, updates=None):
        self.source_sent = source_sent
        self.target_sent = target_sent
        self.source_words = source_sent.split(' ')
        self.updates = updates


valid_alphas = set(
    'A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z'.split(',')
)

valid_years = set([str(x) for x in range(1900, 2030)])


def is_standard_en_sent(sent: str):
    puncs = {
        ',', '.', ':', '?', '-', '_', '!', '@', '/', '\\',
        '[', ']', '{', '}', '(', ')', '=', '+', '|', '#', '$', '%',
        '^', '&', '\'', '"', '’'
    }
    if sent[0] in puncs:
        return False

    if "__" in sent or '---' in sent or '...' in sent or '^^^' in sent or '///' in sent:
        return False

    sent_copy = copy.deepcopy(sent)
    for punc in puncs:
        sent_copy = sent_copy.replace(punc, '')
    sent_copy = sent_copy.replace(' ', '')
    return sent_copy.encode('utf8').isalnum()


def norm_text(text: str):
    puncs = {
        ',', '.', ':', '?', '-', '_', '!', '@', '/', '\\',
        '[', ']', '{', '}', '(', ')', '=', '+', '|', '#', '$', '%',
        '^', '&', '\'', '"', '’'
    }
    text_copy = copy.deepcopy(text)
    for punc in puncs:
        text_copy = text_copy.replace(punc, '')
    return text_copy