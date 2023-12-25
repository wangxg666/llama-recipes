#!/bin/bash
set -x

HOME="/home/paperspace/xingguang"
REF_MODEL="${HOME}/models/agent_sft_act_dataset.v09.7b.2e-5.full.B16.E1.hf"
QUERY_DATASET=""
OUTPUT_CHECKPOINT_DIR="${HOME}/models/rl/agent.ppo.v09.v02"
PRE_TRAIN_CRITIC_CHECKPOINT_DIR="${HOME}/models/rl/agent.ppo.v09.v02/critic"

set -x NCCL_P2P_LEVEL "NVL"

mkdir -p ../logs/
hour=$(date +"%Y-%m-%d_%H")


CUDA_VISIBLE_DEVICES="0" accelerate launch \
  --config_file ${HOME}/llama-recipes/sh/ds_config.2.ppo.yaml \
  ../llama_ppo_online.py \
  --ppo_config.model_name "${REF_MODEL}" \
  --ppo_config.query_dataset "${QUERY_DATASET}" \
  --output_checkpoint_dir "${OUTPUT_CHECKPOINT_DIR}" \
  --pre_train_critic_checkpoint_dir "${PRE_TRAIN_CRITIC_CHECKPOINT_DIR}" \
  > ../logs/ppo.train.${hour}.log &