#!/bin/bash
set -x

MODEL_TYPE="7b"
WORK_DIR="/home/paperspace/xingguang/models/${MODEL_TYPE}"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
DATASET_NAME="my_allin_one_dataset"
DATASET_TYPE=""
DATASET_SUB_DIR="answer_extractor.v027"

LR=3e-5
BATCH_SIZE=8
EPOCH=1

TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}"
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
  --dataset_tag "${DATASET_TYPE}" \
  --dataset_sub_dir_prefix "${DATASET_SUB_DIR}" \
  --save_model \
  --pure_bf16 \
  --output_dir "${WORK_DIR}/${DATASET_NAME}.${TAG}"/ \
  --lr ${LR} \
  --val_batch_size ${BATCH_SIZE} \
  --batch_size_training ${BATCH_SIZE} \
  --micro_batch_size 8 \
  --num_epochs ${EPOCH} \
  --evaluation_steps 100 \
  --check_point_steps 2000 \
  --wandb_name ${MODEL_TYPE}-${DATASET_SUB_DIR}-${TAG}

cd ../

python inference/checkpoint_converter_fsdp_hf.py \
  --fsdp_checkpoint_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-${MODEL_NAME}/ \
  --consolidated_model_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-hf/ \
  --HF_model_path_or_name ${MODEL_NAME}