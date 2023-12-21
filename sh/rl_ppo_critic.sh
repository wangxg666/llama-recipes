#!/bin/bash
set -x

HOME="/home/paperspace/xingguang"
REF_MODEL="${HOME}/models/agent_sft_act_dataset.7b.2e-5.full.B16.E1.hf"
PRE_TRAIN_CRITIC_DATA_DIR="${HOME}/datasets/ppo_cache/"
PRE_TRAIN_CRITIC_CHECKPOINT_DIR="${HOME}/models/rl/agent.ppo.v08/critic"

set -x NCCL_P2P_LEVEL "NVL"

mkdir -p ../logs/
hour=$(date +"%Y-%m-%d_%H")


CUDA_VISIBLE_DEVICES="0" accelerate launch \
  --config_file ${HOME}/llama-recipes/sh/ds_config.2.critic.yaml \
  ../llama_ppo_online.py \
  --ppo_config.model_name "${REF_MODEL}" \
  --ppo_config.batch_size 8 \
  --ppo_config.mini_batch_size 8 \
  --pre_train_critic \
  --pre_train_critic_data_dir "${PRE_TRAIN_CRITIC_DATA_DIR}" \
  --pre_train_critic_checkpoint_dir "${PRE_TRAIN_CRITIC_CHECKPOINT_DIR}"