#!/bin/bash
set -x

MODEL_TYPE="13b"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"

python ../inference/hf-text-generation-inference/merge_lora_weights.py \
  --base_model ${MODEL_NAME} \
  --peft_model /home/paperspace/xingguang/llama/ckpt.peft/13b/answer_extractor.v021-13b.2e-5-peft/epoch_004/ \
  --output_dir /home/paperspace/xingguang/llama/ckpt.peft/13b/answer_extractor.v021-13b.2e-5-peft/epoch_004_merged