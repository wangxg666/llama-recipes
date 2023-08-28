## NB Training Data

保存路径 `/mnt/nlp/xingguang/llama/datasets/nb_training`，
子目录名是对应语料名
- `clickbait/`: 对应脚本 `clickbait/01_format_dataset_nb_training.py`
- `grammar_c4200m/`: 对应脚本 `grammar/01_format_dataset_c4200m_nb_training.py`


### Clickbiat

将此前人工标注的Clickbiat语料和Chat GPT生成语料合并，同时保留 500 的验证集，训练集大小  `48787`，验证集 `500`


### Gramar C4 200M

挑选过程中，做了一些必要的语料清洗操作，比如：
- 全英文检测
- 长度检测
- 首词过度修改（比如删除首名词等）

同时，区分了不同的任务类型，大致数量分布如下

- `GRAMMAR_SEQ2SEQ` 301934
- `GRAMMAR_SINGLE / No.` 62458
- `GRAMMAR_SINGLE / One.` 63297
- `GRAMMAR_SINGLE / Multi.` 29685
```