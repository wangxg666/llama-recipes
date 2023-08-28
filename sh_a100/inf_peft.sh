#!/bin/bash
set -x

DATA_DIR="/mnt/nlp/xingguang/llama/datasets/nb_training/clickbait"
DATA_DIR="/mnt/nlp/xingguang/llama/datasets/nb_training/grammar_c4200m_seq2seq"
#DATA_DIR="/mnt/nlp/xingguang/llama/datasets/nb_training/grammar_c4200m_single"
DATA_DIR="/home/paperspace/datasets/grammar_c4200m_seq2seq"


WORK_DIR="/home/paperspace/llama/ckpt.peft"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="my_allin_one_dataset"
TAG="grammar-seq2seq-a100"
#TAG="grammar-single"

cd ../
CUDA_VISIBLE_DEVICES="0,1" python llama_inf.py \
  --length_penalty 1 \
  --model_name ${MODEL_NAME} \
  --peft_model ${WORK_DIR}/${MODEL_NAME}/${DATASET_NAME}/${TAG}-peft/step_00005k \
  --dataset my_allin_one_dataset \
  --input_file ${DATA_DIR}/valid.txt \
  --output_file ${DATA_DIR}/valid.txt.pred
