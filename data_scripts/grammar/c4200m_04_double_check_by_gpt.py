import json
import os
import openai


def request_gpt(text):
    prompt = """
The following **text** might have some grammatical errors or typos.
Please correct these errors and provide the revised text if it contains any mistakes, 
or simply provide the text as is if it is correct.
Your answer should be the json format like
{
  "revised text": ""
}
text:
    """
    openai.api_key = os.environ['OPENAI_API_KEY']
    openai.api_base = "http://cooper.k8s.nb-prod.com/v1"

    model = "gpt-3.5-turbo"

    try:
        messages = [{"role": "user", "content": prompt + text}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        choice = response['choices'][0]
        content = choice['message']['content']
        content = json.loads(content)
        if 'revised text' not in content:
            return ''
        return content['revised text']
    except:
        return ''


if __name__ == '__main__':
    root = '/mnt/nlp/xingguang/llama/datasets/nb_raw_datas/grammar_c4200m/'
    exist_srcs = {
        x.strip().split('\t')[0] for x in open(f'{root}/checked.txt')
    }

    total, same = 0, 0
    with open(f'{root}/checked.txt', 'a') as fout:
        for i, data in enumerate(open(f'{root}/raw.txt')):
            parts = data.strip().split('\t')
            if len(parts) != 5:
                continue
            src, trg = parts[2:4]
            if src in exist_srcs:
                continue
            gpt_res = request_gpt(src)
            fout.write(f'{src}\t{trg}\t{gpt_res}\n')
            fout.flush()

            same += int(trg.replace(' ', '') == gpt_res.replace(' ', ''))
            total += 1

            print(i, total, same)
