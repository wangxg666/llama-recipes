#!/bin/bash
set -x

HOME="/home/paperspace/xingguang"
REF_MODEL="${HOME}/models/agent_sft_act_dataset.7b.2e-5.full.B16.E1.hf"
CRITIC_DATA_DIR="${HOME}/datasets/ppo_cache/"
OUTPUT_CHECKPOINT_DIR="${HOME}/models/rl/agent.ppo.v08"

set -x NCCL_P2P_LEVEL "NVL"

mkdir -p ../logs/
hour=$(date +"%Y-%m-%d_%H")


CUDA_VISIBLE_DEVICES="0" accelerate launch \
  --config_file ${HOME}/llama-recipes/sh/ds_config.2.critic.yaml \
  ../llama_ppo_online.py \
  --ppo_config.model_name "${REF_MODEL}" \
  --use_critic_pre_train \
  --critic_pre_train_dir "${CRITIC_DATA_DIR}" \
  --output_checkpoint_dir "${OUTPUT_CHECKPOINT_DIR}"