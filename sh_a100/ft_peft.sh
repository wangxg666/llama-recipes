#!/bin/bash
set -x

WORK_DIR="/home/cpp/xingguang/llama/model_checkpoints.peft"
WORK_DIR="/home/paperspace/llama/ckpt.peft"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="my_allin_one_dataset"
TAG="grammar-seq2seq-a100"
#TAG="grammar-single"

ts=`date +"%Y-%m-%d_%H-%M-%S"`
LOG_FILE="./logs/peft-${DATASET_NAME}-${TAG}-${ts}.txt"
echo "" > "${LOG_FILE}"

cd ../
CUDA_VISIBLE_DEVICES="0,1" torchrun \
  --nnodes 1 \
  --nproc_per_node 2  \
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
  --num_epochs 10 \
  --check_point_steps 1000 \
  --wandb_name ${DATASET_NAME}-${TAG}-${ts} \
  > ${LOG_FILE} &

echo ${LOG_FILE}
tail -f ${LOG_FILE}
