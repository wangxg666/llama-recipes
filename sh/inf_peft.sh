#!/bin/bash
set -x


WORK_DIR="/home/paperspace/xingguang/llama/ckpt.peft"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="my_allin_one_dataset"
TAG="answer_extractor.v009.2e-5"
INPUT_FILE="answer_extractor.v009/valid.txt"


cd ..

CUDA_VISIBLE_DEVICES="2,3" python llama_inf.py \
  --length_penalty 0.5 \
	--num_beams 10 \
	--do_sample 0 \
  --model_name ${MODEL_NAME} \
  --peft_model ${WORK_DIR}/${MODEL_NAME}/${DATASET_NAME}/${TAG}-peft/best_model \
  --dataset my_allin_one_dataset \
  --input_file ${INPUT_FILE}
