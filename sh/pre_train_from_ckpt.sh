#!/bin/bash
set -x

MODEL_TYPE="7b"
WORK_DIR="/home/paperspace/xingguang/models/"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
DATASET_NAME="my_pre_train_dataset"
DATASET_SUB_DIR="pre-training-ariticle/tokenized.${MODEL_TYPE}"
PRE_TRAIN_CKPT_PATH=""

LR=3e-5
BATCH_SIZE=4
EPOCH=1

TAG="${MODEL_TYPE}.${LR}.B${BATCH_SIZE}.E${EPOCH}.full"
ts=$(date +"%Y-%m-%d")

cd ..

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --master_port=1201 \
  ./llama_pre_training.py \
  --enable_fsdp  \
  --model_name "${MODEL_NAME}" \
  --pre_train_model_path "${PRE_TRAIN_CKPT_PATH}" \
  --dataset "${DATASET_NAME}" \
  --dataset_sub_dir_prefix "${DATASET_SUB_DIR}" \
  --save_model \
  --pure_bf16 \
  --output_dir "${WORK_DIR}/${DATASET_NAME}.${TAG}"/ \
  --lr ${LR} \
  --valid_batch_size ${BATCH_SIZE} \
  --train_batch_size ${BATCH_SIZE} \
  --micro_batch_size 4 \
  --num_epochs ${EPOCH} \
  --evaluation_steps 100 \
  --check_point_steps 2000 \
  --wandb_name ${DATASET_NAME}-${DATASET_SUB_DIR}-${TAG}-${ts}

cd ../

#python inference/checkpoint_converter_fsdp_hf.py \
#  --fsdp_checkpoint_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-${MODEL_NAME}/ \
#  --consolidated_model_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-hf/ \
#  --HF_model_path_or_name ${MODEL_NAME}