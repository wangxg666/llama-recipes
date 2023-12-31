#!/bin/bash
set -x

HOME="/home/paperspace/xingguang"
CKPT_HOME="/mnt/share16t/xingguang/"
CKPT_HOME="${HOME}"

REF_MODEL="${HOME}/models/agent_sft_act_dataset.v09.7b.2e-5.full.B16.E1.hf"
PRE_TRAIN_CRITIC_CHECKPOINT_DIR="${HOME}/models/rl/agent.ppo.v09.v05/critic"

QUERY_DATASET=""
OUTPUT_CHECKPOINT_DIR="${CKPT_HOME}/models/rl/agent.ppo.v09.v05.origin"

set -x NCCL_P2P_LEVEL "NVL"

mkdir -p ../logs/
hour=$(date +"%Y-%m-%d_%H")


CUDA_VISIBLE_DEVICES="5" accelerate launch --main_process_port 29501 \
  --config_file ${HOME}/llama-recipes/sh/ds_config.2.ppo_origin.yaml \
  ../llama_ppo_online_origin.py \
  --ppo_config.model_name "${REF_MODEL}" \
  --ppo_config.query_dataset "${QUERY_DATASET}" \
  --ppo_config.ppo_epochs 2 \
  --output_checkpoint_dir "${OUTPUT_CHECKPOINT_DIR}" \
  --pre_train_critic_checkpoint_dir "${PRE_TRAIN_CRITIC_CHECKPOINT_DIR}"
#  \
#  > ../logs/ppo.train.${hour}.log &