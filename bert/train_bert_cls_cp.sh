#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES="2" python train_bert_cls.py \
  --model_name "bert-large-cased" \
  --input_data_file "/home/paperspace/xingguang/datasets/faq.relevance/data.txt" \
  --output_model_dir "/home/paperspace/xingguang/bert/faq.relevance.bert.large.cased" \
  --batch_size 64 \
  --evaluation_steps 500 \
  --epochs 5 \
  --valid_split 0.05 \
  --learning_rate 1e-5 \
  --warmup_steps 10 \
  --wandb_name 'bert-faq-relevance-large-cased'