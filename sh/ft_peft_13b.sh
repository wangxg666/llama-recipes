#!/bin/bash
set -x

WORK_DIR="/home/paperspace/xingguang/llama/ckpt.peft"
MODEL_NAME="meta-llama/Llama-2-13b-hf"
DATASET_NAME="my_allin_one_dataset"
TAG="answer_extractor.v013.13b.2e-5"
ts=$(date +"%Y-%m-%d")

cd ..

CUDA_VISIBLE_DEVICES="4,5,6,7" torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --master_port=1202 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --use_peft \
  --peft_method lora \
  --model_name ${MODEL_NAME} \
  --dataset ${DATASET_NAME} \
  --save_model \
  --pure_bf16 \
  --output_dir ${WORK_DIR}/${MODEL_NAME}/${DATASET_NAME}/${TAG}-peft/ \
  --lr 2e-5 \
  --val_batch_size 8 \
  --batch_size_training 8 \
  --micro_batch_size 8 \
  --num_epochs 10 \
  --evaluation_steps 50 \
  --check_point_steps 300 \
  --wandb_name ${MODEL_NAME}-${DATASET_NAME}-${TAG}-${ts}

cd ../

