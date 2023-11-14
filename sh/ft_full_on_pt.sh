#!/bin/bash
set -x

MODEL_TYPE="13b"
WORK_DIR="/home/paperspace/xingguang/models/${MODEL_TYPE}"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
DATASET_NAME="my_allin_one_dataset"
DATASET_DIR="comment.sft.stage-2.v01"
PRE_TRAIN_MODEL="/home/paperspace/xingguang/models/13b/my_news_comment_tokenized_dataset.13b.1e-5.full.B4.E1.Tokenized.Partition/step_097248.hf"

LR=2e-5
BATCH_SIZE=4
EPOCH=1

TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.withSFT.097248"
ts=$(date +"%Y-%m-%d")

cd ..

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --master_port=1202 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --model_name "${MODEL_NAME}" \
  --pre_train_model_path "${PRE_TRAIN_MODEL}" \
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
  --wandb_name ${MODEL_TYPE}-${DATASET_DIR}-${TAG} \
  --wandb_project "llama-comment"

cd ../