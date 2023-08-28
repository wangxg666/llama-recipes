from llama.data_scripts.clickbaity.utils_ import *


if __name__ == '__main__':
    for day in time_range('2023-08-01', '2023-08-07'):
        doc2feature = collect_feature(day, 300)
        doc2title = collect_titles(doc2feature)

        df_title = rewrite_titles(day, doc2title)
        df_title.to_csv(f'./datas/gpt_res/{day}.csv', index=False)
