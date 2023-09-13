#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES="0" python svr_bert_cls.py \
  --port 1302 \
  --model_name "bert-base-uncased" \
  --fine_tuning_model "/home/paperspace/xingguang/bert/faq.relevance/model.epoch.5.bin"
