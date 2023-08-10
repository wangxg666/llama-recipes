from hallucination.data_scripts_grammar.utils_ import *


if __name__ == '__main__':
    for day in time_range('2023-07-01', '2023-08-08'):
        doc2feature = collect_feature(day)
        df_title = collect_titles(day, doc2feature)
        df_paragraph = collect_paragraphs(day, doc2feature)
        df = pd.concat([df_title, df_paragraph])
        df.to_csv(f'./{day}.csv', index=False)
