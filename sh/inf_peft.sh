#!/bin/bash
set -x

#DATA_DIR="/mnt/nlp/xingguang/llama/datasets/nb_training/clickbait"
#DATA_DIR="/mnt/nlp/xingguang/llama/datasets/nb_training/grammar_c4200m_seq2seq"
#DATA_DIR="/mnt/nlp/xingguang/llama/datasets/nb_training/grammar_c4200m_single"
DATA_DIR="/mnt/nlp/xingguang/llama/datasets/nb_training/hallucination"

WORK_DIR="/home/cpp/xingguang/llama/model_checkpoints.peft"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="my_allin_one_dataset"
#TAG="grammar-seq2seq"
#TAG="grammar-single"
TAG="hallucination"

cd ..

CUDA_VISIBLE_DEVICES="0,1,2,3" python llama_inf.py \
  --length_penalty 3 \
  --model_name ${MODEL_NAME} \
  --peft_model ${WORK_DIR}/${MODEL_NAME}/${DATASET_NAME}/${TAG}-peft/best_model \
  --dataset my_allin_one_dataset \
  --input_file ${DATA_DIR}/valid.txt \
  --output_file ${DATA_DIR}/valid.txt.pred
