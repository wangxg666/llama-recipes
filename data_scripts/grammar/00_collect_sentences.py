from llama.data_scripts.grammar.utils_ import *


if __name__ == '__main__':
    for day in time_range('2023-07-01', '2023-08-08'):
        doc2feature = collect_feature(day)

        doc2title = collect_titles(doc2feature)
        df_title = rewrite_titles(day, doc2title)

        doc2content = collect_titles(doc2feature)
        df_paragraph = rewrite_content(day, doc2content)

        df = pd.concat([df_title, df_paragraph])
        df.to_csv(f'./datas/gpt_res/{day}.csv', index=False)
