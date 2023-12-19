#!/bin/bash
set -x

HOME="/home/paperspace/xingguang"
REF_MODEL="${HOME}/models/agent_sft_act_dataset.7b.2e-5.full.B16.E1.hf"
QUERY_DATASET=""
OUTPUT_CHECKPOINT_DIR="${HOME}/models/rl/agent.ppo.v08"

set -x NCCL_P2P_LEVEL "NVL"

CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
  --config_file ${HOME}/llama-recipes/sh/ds_config.2.yaml \
  ../llama_ppo_online.py \
  --ppo_config.model_name "${REF_MODEL}" \
  --ppo_config.query_dataset "${QUERY_DATASET}" \
  --output_checkpoint_dir "${OUTPUT_CHECKPOINT_DIR}"