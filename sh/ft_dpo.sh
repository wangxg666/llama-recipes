#!/bin/bash
set -x

MODEL_TYPE="13b"
WORK_DIR="/home/paperspace/xingguang/models/${MODEL_TYPE}"
SFT_MODEL_PATH=""
LR=1e-6
BETA=0.1
LR_SCHEDULER="cosine"
BATCH_SIZE=4
DATASET_NAME="my_news_comment_dpo_dataset"
DATASET_VERSION='comment.dpo.stage-3.v02'

TAG="${MODEL_TYPE}.${LR}.beta${BETA}.${LR_SCHEDULER}.B${BATCH_SIZE}"

cd ..

CUDA_VISIBLE_DEVICES="3,4,6,7" accelerate launch \
  ./llama_dpo.py \
  --model_name_or_path "${SFT_MODEL_PATH}"  \
  --dataset_name "${DATASET_NAME}" \
  --dataset_version "${DATASET_VERSION}" \
  --beta ${BETA} \
  --learning_rate ${LR} \
  --lr_scheduler_type ${LR_SCHEDULER} \
  --warmup_steps 300 \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${BATCH_SIZE} \
  --max_prompt_length 1800 \
  --max_length 2048 \
  --max_steps 0 \
  --save_steps 2000000 \
  --eval_steps 200 \
  --output_dir "${WORK_DIR}-${DATASET_NAME}-${TAG}"/ \
  --wandb_name "${TAG}"

