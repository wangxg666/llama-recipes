#!/bin/bash
set -x

HOME_DIR="/home/paperspace/xingguang"
GPU_IDS=${1}
GPU_NUMS=${2}
PORT=${3}
MODEL=${4}

# bash start_svr_vllm_full_act.sh "0,2" 2 8000 "agent_sft_act_dataset.7b.2e-5.full.B8.E1.agent_sft.auto.gen.v07.1.dst.hf"

echo ${PORT}

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -O -u -m vllm.entrypoints.api_server \
  --host=0.0.0.0 \
  --port=${PORT} \
  --model="${HOME_DIR}/models/${MODEL}" \
  --tokenizer="hf-internal-testing/llama-tokenizer" \
  --dtype="bfloat16" \
  --tensor-parallel-size=${GPU_NUMS}





