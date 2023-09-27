#!/bin/bash
set -x

MODEL_TYPE="13b"
HOME_DIR="/home/paperspace/xingguang"
WORK_DIR="llama/ckpt.full/${MODEL_TYPE}"
DATASET_SUB_DIR="answer_extractor.v022"
TAG="${MODEL_TYPE}.2e-5.full"
ts=$(date +"%Y-%m-%d")


sudo docker run \
  --gpus "device=0" \
  --shm-size 4g -p 1308:80 -v ${HOME_DIR}:/data ghcr.io/huggingface/text-generation-inference:1.0.3 \
  --model-id /data/${WORK_DIR}/${DATASET_SUB_DIR}-${TAG}/step_004000-hf \
  --dtype bfloat16 \
  --max-total-tokens 4096 \
  --cuda-memory-fraction 0.4 \
  --max-input-length 3000 \
  --sharded false