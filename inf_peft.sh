#!/bin/bash
set -x

DATA_DIR="/mnt/nlp/xingguang/llama/datasets/nb_training/clickbait"
DATA_DIR="/mnt/nlp/xingguang/llama/datasets/nb_training/grammar_c4200m"

CUDA_VISIBLE_DEVICES="0,1,2,3" python inference/inference_my.py \
  --length_penalty 3 \
  --model_name meta-llama/Llama-2-7b-hf \
  --peft_model ./llama-2-7b-allin-one-peft/epoch_000/ \
  --dataset my_allin_one_dataset \
  --input_file ${DATA_DIR}/valid.txt \
  --output_file ${DATA_DIR}/valid.txt.pred
