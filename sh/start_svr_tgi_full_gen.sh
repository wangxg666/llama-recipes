#!/bin/bash
set -x

HOME_DIR="/home/paperspace/xingguang"


sudo docker run \
  --gpus "device=4" \
  --shm-size 4g -p 1309:80 -v ${HOME_DIR}:/data ghcr.io/huggingface/text-generation-inference:1.0.3 \
  --model-id /data/models/agent_sft_gen_dataset.13b.2e-5.full.B16.E1.hf \
  --dtype bfloat16 \
  --max-total-tokens 4096 \
  --cuda-memory-fraction 0.4 \
  --max-input-length 3000 \
  --sharded false