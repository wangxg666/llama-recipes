#!/bin/bash
set -x

HOME_DIR="/home/paperspace/xingguang"
#HOME_DIR="/mnt/share16t/xingguang/"


sudo docker run \
  --gpus "device=${1}" \
  --shm-size 4g -p ${2}:80 -v ${HOME_DIR}:/data ghcr.io/huggingface/text-generation-inference:1.4 \
  --model-id /data/models/agent_sft_gen_dataset.7b.2e-5.full.B8.E1.agent_sft.v10.baseline.dst.limit_8k.hf \
  --dtype bfloat16 \
  --max-total-tokens 4096 \
  --cuda-memory-fraction 0.4 \
  --max-input-length 3000 \
  --sharded false