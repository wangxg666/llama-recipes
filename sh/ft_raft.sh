#!/bin/bash
set -x

MODEL_TYPE="13b"
WORK_DIR="/home/paperspace/xingguang/models/${MODEL_TYPE}"

HOME="/home/paperspace/xingguang"
MODEL_NAME="${HOME}/models/my_agent_sft_dataset.${MODEL_TYPE}.2e-5.full.B4.E1.v07.all.hf"

DATASET_NAME="my_agent_sft_dataset"
DATASET_DIR="agent_raft.v07"

LR=2e-5
BATCH_SIZE=4
EPOCH=1

TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.RAFT"
ts=$(date +"%Y-%m-%d")

cd ..

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --master_port=1201 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --model_name "${MODEL_NAME}" \
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
  --evaluation_steps 500 \
  --check_point_steps 1000000 \
  --wandb_name ${MODEL_TYPE}-${DATASET_DIR}-${TAG} \
  --wandb_project "llama-pre-train-cmp"


