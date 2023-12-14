#!/bin/bash
set -x

HOME="/home/paperspace/xingguang"
REF_MODEL="${HOME}/models/my_agent_sft_dataset.13b.2e-5.full.B4.E1.v07.all.hf"
QUERY_DATASET="${HOME}/datasets/agent_raft.v07/ppo.train.jsonl"
OUTPUT_CHECKPOINT_DIR="${HOME}/models/rl/agent.ppo.v07"

set -x NCCL_P2P_LEVEL "NVL"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" accelerate launch \
  --config_file ${HOME}/llama-recipes/sh/ds_config.2.yaml \
  ../llama_ppo_offline.py \
  --ppo_config.model_name "${REF_MODEL}" \
  --ppo_config.query_dataset "${QUERY_DATASET}" \
  --output_checkpoint_dir "${OUTPUT_CHECKPOINT_DIR}"