#!/bin/bash
set -x

MODEL_TYPE="7b"
WORK_DIR="/home/paperspace/xingguang/llama/ckpt.peft/${MODEL_TYPE}"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
DATASET_NAME="my_allin_one_dataset"
DATASET_SUB_DIR="answer_extractor.v014"
TAG="${MODEL_TYPE}.2e-5"
ts=$(date +"%Y-%m-%d")

cd ..

CUDA_VISIBLE_DEVICES="2,3" torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  --master_port=1202 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --use_peft \
  --peft_method lora \
  --model_name ${MODEL_NAME} \
  --dataset ${DATASET_NAME} \
  --dataset_sub_dir_prefix ${DATASET_SUB_DIR} \
  --save_model \
  --pure_bf16 \
  --output_dir ${WORK_DIR}/${DATASET_SUB_DIR}-${TAG}-peft/ \
  --lr 2e-5 \
  --val_batch_size 16 \
  --batch_size_training 16 \
  --micro_batch_size 16 \
  --num_epochs 10 \
  --evaluation_steps 100 \
  --check_point_steps 100 \
  --wandb_name ${MODEL_NAME}-${DATASET_NAME}-${DATASET_SUB_DIR}-${TAG}-${ts}

cd ../

