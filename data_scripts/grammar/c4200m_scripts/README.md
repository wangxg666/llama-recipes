### C4

>通过 Allenai 下载

数据集规模，链接 https://github.com/allenai/allennlp/discussions/5056

- `en`: 800GB in TFDS format, 305GB in JSON format
- `en.noclean`: 6.3TB in TFDS format, 2.3TB in JSON format
- `en.noblocklist`: 1003G in TFDS format, 380GB in JSON format
- `realnewslike`: 38GB in TFDS format, 15GB in JSON format
- `multilingual`: 27TB in TFDS format, 9.7TB in JSON format

```
$env GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
# 指定下载内容
git lfs pull --include "en/*.json.gz"
```

离线数据位置 `/mnt/nlp/xingguang/llama/c4200m/c4`

### C4 200M

链接 https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction.git

离线数据位置  `/mnt/nlp/xingguang/llama/c4200m`

### 数据格式化

```shell
#cd /mnt/nlp/xingguang/llama/c4200m/C4_200M-synthetic-dataset-for-grammatical-error-correction
cd /mnt/nlp/xingguang/mac_desk/husky-go/llama/data_scripts/grammar/c4200m_scripts

# 抽取一个part，非常耗时
python c4200m_get_target_sentences_json.py \
    /mnt/nlp/xingguang/llama/c4200m/edits.tsv-00000-of-00010 \
    /mnt/nlp/xingguang/llama/c4200m/c4/en \
    /mnt/nlp/xingguang/llama/c4200m/target_sentences.tsv-00000-of-00010
    
# 合并成 Sentence Pair
python c4200m_make_sentence_pairs_my.py \
    /mnt/nlp/xingguang/llama/c4200m/target_sentences.tsv-00000-of-00010 \
    /mnt/nlp/xingguang/llama/c4200m/edits.tsv-00000-of-00010 \
    /mnt/nlp/xingguang/llama/c4200m/sentence_pairs.tsv-00000-of-00010
```

