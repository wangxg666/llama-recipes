#!/bin/bash
set -x

MODEL_TYPE="13b"
WORK_DIR="/home/paperspace/xingguang/models/${MODEL_TYPE}"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
DATASET_NAME="my_allin_one_dataset"
DATASET_DIR="comment.sft.stage-2.v01"
# PRE_TRAIN_MODEL="/home/paperspace/xingguang/models/13b/my_news_comment_tokenized_dataset.13b.1e-5.full.B4.E1.Tokenized.Partition/step_097248.hf"
PRE_TRAIN_MODEL="/home/paperspace/xingguang/models/13b/my_news_comment_tokenized_dataset.13b.1e-5.full.B4.E1.Tokenized.Partition/optimizer_step_114413.hf"

LR=2e-5
BATCH_SIZE=4
EPOCH=1

TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.withSFT"
# ts=$(date +"%Y-%m-%d")

cd ..

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --master_port=1202 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --model_name "${MODEL_NAME}" \
  -- "${PRE_TRAIN_MODEL}" \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --save_model \
  --pure_bf16 \
  --output_dir "${WORK_DIR}/${DATASET_NAME}.${TAG}"/ \
  --lr ${LR} \
  --valid_batch_size ${BATCH_SIZE} \
  --train_batch_size ${BATCH_SIZE} \
  --micro_batch_size 4 \
  --num_epochs ${EPOCH} \
  --evaluation_steps 100 \
  --check_point_steps 2000000 \
  --wandb_name "${MODEL_TYPE}-${DATASET_DIR}-${TAG}" \
  --wandb_project "llama-comment"


python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path "${WORK_DIR}/${DATASET_NAME}.${TAG}/epoch_000" \
    --consolidated_model_path "${WORK_DIR}/${DATASET_NAME}.${TAG}/epoch_000.hf" \
    --HF_model_path_or_name "${MODEL_NAME}"


SFT_MODEL_PATH="${WORK_DIR}/${DATASET_NAME}.${TAG}/epoch_000.hf"
LR=1e-6
BETA=0.1
LR_SCHEDULER="cosine"
BATCH_SIZE=4
DATASET_NAME="my_news_comment_dpo_dataset"
DATASET_VERSION='comment.dpo.stage-3.v02'

TAG="${MODEL_TYPE}.${LR}.beta${BETA}.${LR_SCHEDULER}.B${BATCH_SIZE}"

cd ..

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
  --config_file /home/paperspace/xingguang/config.yaml\
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
  --gradient_accumulation_steps 1 \
  --max_prompt_length 1800 \
  --max_length 2048 \
  --max_steps 0 \
  --save_steps 2000000 \
  --eval_steps 200 \
  --output_dir "${WORK_DIR}-${DATASET_NAME}-${TAG}"/ \
  --wandb_name "${TAG}"

