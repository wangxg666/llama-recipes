#!/bin/bash
set -x

#INPUT_FILE="clickbait/valid.txt"
#INPUT_FILE="grammar_c4200m_single/valid.txt"
#INPUT_FILE="grammar_c4200m_seq2seq/valid.txt"
INPUT_FILE="answer_extractor/valid.txt"

WORK_DIR="/home/cpp/xingguang/llama/model_checkpoints.peft"
WORK_DIR="/home/paperspace/xingguang/llama/ckpt.peft"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="my_allin_one_dataset"
#TAG="grammar-seq2seq"
#TAG="grammar-single"
TAG="answer_extractor"

cd ..

CUDA_VISIBLE_DEVICES="0,1" python llama_inf.py \
  --length_penalty 1 \
  --model_name ${MODEL_NAME} \
  --peft_model ${WORK_DIR}/${MODEL_NAME}/${DATASET_NAME}/${TAG}-peft/best_model \
  --dataset my_allin_one_dataset \
  --input_file ${INPUT_FILE}
