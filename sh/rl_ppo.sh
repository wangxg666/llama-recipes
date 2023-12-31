#!/bin/bash
set -x

HOME="/home/paperspace/xingguang"
CKPT_HOME="/mnt/share16t/xingguang/"
CKPT_HOME="${HOME}"

REF_MODEL="${HOME}/models/agent_sft_act_dataset.7b.2e-5.full.B8.E1.agent_sft.v09.1.hf"
PRE_TRAIN_CRITIC_CHECKPOINT_DIR="${HOME}/models/rl/agent.ppo.v09.1.v01/critic"

QUERY_DATASET=""
3OUTPUT_CHECKPOINT_DIR="${CKPT_HOME}/models/rl/agent.ppo.v09.1.v01/"

set -x NCCL_P2P_LEVEL "NVL"

mkdir -p ../logs/
hour=$(date +"%Y-%m-%d_%H")


nohup accelerate launch \
  --config_file ${HOME}/llama-recipes/sh/ds_config.2.ppo.yaml \
  ../llama_ppo_online.py \
  --gpus "4" \
  --ppo_config.model_name "${REF_MODEL}" \
  --ppo_config.query_dataset "${QUERY_DATASET}" \
  --ppo_config.ppo_epochs 2 \
  --output_checkpoint_dir "${OUTPUT_CHECKPOINT_DIR}" \
  --pre_train_critic_checkpoint_dir "${PRE_TRAIN_CRITIC_CHECKPOINT_DIR}" \
  >../logs/ppo.train.${hour}.log &