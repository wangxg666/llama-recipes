#!/bin/bash
set -x

cd ..

set -x NCCL_P2P_LEVEL "NVL"

CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
  --config_file /home/paperspace/xingguang/llama-recipes/sh/ds_config.yaml ./llama_ppo.py \
  --model_name "/home/paperspace/xingguang/models/my_agent_sft_dataset.13b.2e-5.full.B4.E1.v07.all.hf"

