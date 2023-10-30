#!/bin/bash
set -x

MODEL_TYPE="13b"
WORK_DIR="/home/paperspace/xingguang/models/${MODEL_TYPE}"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
DATASET_NAME="my_news_comment_tokenized_dataset"
DATASET_DIR="comment.v02"

LR=1e-5
BATCH_SIZE=4
EPOCH=1

TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.Tokenized"
ts=$(date +"%Y-%m-%d")

cd ..

CUDA_VISIBLE_DEVICES="4,5,6,7" torchrun \
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
  --check_point_steps 5000 \
  --wandb_name ${MODEL_TYPE}-${DATASET_DIR}-${TAG}

cd ../

python inference/checkpoint_converter_fsdp_hf.py \
  --fsdp_checkpoint_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-${MODEL_NAME}/ \
  --consolidated_model_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-hf/ \
  --HF_model_path_or_name ${MODEL_NAME}