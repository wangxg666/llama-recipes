#!/bin/bash
set -x

WORK_DIR="/home/paperspace/xingguang/llama/ckpt.peft"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="my_allin_one_dataset"
#TAG="grammar-seq2seq"
#TAG="grammar-single"
TAG="answer_extractor.v002"
ts=$(date +"%Y-%m-%d")

cd ..

CUDA_VISIBLE_DEVICES="0,1" torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --use_peft \
  --peft_method lora \
  --model_name ${MODEL_NAME} \
  --dataset ${DATASET_NAME} \
  --save_model \
  --pure_bf16 \
  --output_dir ${WORK_DIR}/${MODEL_NAME}/${DATASET_NAME}/${TAG}-peft/ \
  --lr 0.00005 \
  --val_batch_size 4 \
  --batch_size_training 4 \
  --micro_batch_size 1 \
  --num_epochs 3 \
  --check_point_steps 1000 \
  --wandb_name ${DATASET_NAME}-${TAG}-${ts}

cd ../

