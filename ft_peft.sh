#!/bin/bash
set -x

WORK_DIR="/home/cpp/xingguang/llama/model_checkpoints.peft"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="my_allin_one_dataset"
TAG="grammar-single"

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
  --nnodes 1 \
  --nproc_per_node 4  \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --use_peft \
  --peft_method lora \
  --model_name ${MODEL_NAME} \
  --dataset ${DATASET_NAME} \
  --save_model \
  --pure_bf16 \
  --output_dir ${WORK_DIR}/${MODEL_NAME}/${DATASET_NAME}/${TAG}-peft/ \
  --batch_size_training 8 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --check_point_steps 3000