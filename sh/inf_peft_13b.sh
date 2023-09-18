#!/bin/bash
set -x


MODEL_TYPE="13b"
WORK_DIR="/home/paperspace/xingguang/llama/ckpt.peft/${MODEL_TYPE}"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
DATASET_NAME="my_allin_one_dataset"
DATASET_SUB_DIR="answer_extractor.v018"
INPUT_FILE="${DATASET_SUB_DIR}/valid_online.txt"
TAG="${MODEL_TYPE}.2e-5"

cd ..

CUDA_VISIBLE_DEVICES="4,5,6,7" python llama_inf.py \
  --length_penalty 1 \
	--num_beams 1 \
	--do_sample 0 \
  --max_new_tokens 1000 \
  --model_name ${MODEL_NAME} \
  --peft_model ${WORK_DIR}/${DATASET_SUB_DIR}-${TAG}-peft/epoch_004/ \
  --dataset ${DATASET_NAME} \
  --input_file ${INPUT_FILE}
