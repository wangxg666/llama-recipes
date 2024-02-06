#!/bin/bash
set -x

HOME_DIR="/home/paperspace/xingguang"
GPU=${1}
PORT=${2}
MODEL=${3}

echo ${PORT}

sudo docker run \
  --gpus "device=${GPU}" \
  --shm-size 4g -p ${PORT}:80 -v ${HOME_DIR}:/data ghcr.io/huggingface/text-generation-inference:1.4 \
  --model-id /data/models/${MODEL}/ \
  --dtype bfloat16 \
  --max-total-tokens 4096 \
  --cuda-memory-fraction 0.4 \
  --max-input-length 3000 \
  --sharded false