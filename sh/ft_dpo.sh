#!/bin/bash
set -x

MODEL_TYPE="13b"
WORK_DIR="/home/paperspace/xingguang/models/${MODEL_TYPE}"
SFT_MODEL_PATH=""
LR=1e-5
LR_SCHEDULER="cosine"
BATCH_SIZE=4

TAG="${MODEL_TYPE}.${LR}.${LR_SCHEDULER}.B${BATCH_SIZE}.DPO"

cd ..

CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch \
  ./llama_dpo.py \
  --model_name_or_path "${SFT_MODEL_PATH}"  \
  --learning_rate ${LR} \
  --lr_scheduler_type ${LR_SCHEDULER} \
  --warmup_steps 300 \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${BATCH_SIZE} \
  --max_prompt_length 2000 \
  --max_length 2048 \
  --max_steps 0 \
  --save_steps 2000 \
  --eval_steps 200 \
  --output_dir "${WORK_DIR}/${DATASET_NAME}.${TAG}"/ \
  --wandb_name ${MODEL_TYPE}-${TAG}

