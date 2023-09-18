#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES="0" python svr_bert_cls.py \
  --port 1302 \
  --model_name "bert-base-cased" \
  --fine_tuning_model "/home/paperspace/xingguang/bert/faq.relevance.cased.2Class.cased.WithQT.Instruct.MoreData/model.epoch.9.bin"
