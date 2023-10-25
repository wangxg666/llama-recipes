#!/bin/bash
set -x

MODEL_TYPE="13b"
WORK_DIR="/home/paperspace/xingguang/models/ckpt.peft/${MODEL_TYPE}"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
DATASET_NAME="my_allin_one_dataset"
DATASET_DIR="answer_extractor.v023"
TAG="${MODEL_TYPE}.2e-5"
ts=$(date +"%Y-%m-%d")

cd ..

CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" torchrun \
  --nnodes 1 \
  --nproc_per_node 6 \
  --master_port=1201 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --use_peft \
  --peft_method lora \
  --model_name ${MODEL_NAME} \
  --dataset ${DATASET_NAME} \
  --dataset_dir ${DATASET_DIR} \
  --save_model \
  --pure_bf16 \
  --output_dir "${WORK_DIR}/${DATASET_NAME}.${TAG}-peft/" \
  --lr 2e-5 \
  --valid_batch_size 8 \
  --train_batch_size 8 \
  --micro_batch_size 8 \
  --num_epochs 5 \
  --evaluation_steps 100 \
  --check_point_steps 100 \
  --wandb_name ${MODEL_NAME}-${DATASET_NAME}-${DATASET_DIR}-${TAG}-${ts}

cd ../

