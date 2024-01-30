#!/bin/bash
set -x

MODEL_TYPE="7b"
WORK_DIR="/home/paperspace/xingguang/models/${MODEL_TYPE}"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
DATASET_NAME="agent_sft_act_dataset"
DATASET_DIR="agent_sft.auto.gen.v08.28.1.replace_restaurant.dst.ctx"
DATASET_DIR="agent_sft.auto.gen.v08.38.1.template.2k.dst.ctx"
DATASET_DIR="agent_sft.auto.gen.v08.37-38.1.template.4k.dst.ctx"
DATASET_DIR="agent_sft.auto.gen.v08.28.1.replace.hotel.dst.ctx"
DATASET_DIR="agent_sft.auto.gen.v08.28.1.replace.restaurant.dst.ctx"
DATASET_DIR="agent_sft.auto.gen.v08.28.1.replace.attraction.dst.ctx"
DATASET_DIR="agent_sft.auto.gen.v08.39.1.template.2k.dst.ctx"
DATASET_DIR="agent_sft.auto.gen.v08.28.1.replace.hotel.full.dst.ctx"
DATASET_DIR="agent_sft.auto.gen.v08.28.1.replace.restaurant.full.dst.ctx"
DATASET_DIR="agent_sft.auto.gen.v08.28.1.replace.attraction.full.dst.ctx"
#DATASET_DIR="agent_sft.v10.baseline.dst.with.gen"

LR=2e-5
BATCH_SIZE=8
EPOCH=1

TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.${DATASET_DIR}"
ts=$(date +"%Y-%m-%d")

cd ..

CUDA_VISIBLE_DEVICES="2,3,4,5" torchrun \
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
  --output_dir "/mnt/share16t/xingguang/models/${DATASET_NAME}.${TAG}"/ \
  --lr ${LR} \
  --valid_batch_size ${BATCH_SIZE} \
  --train_batch_size ${BATCH_SIZE} \
  --micro_batch_size ${BATCH_SIZE} \
  --num_epochs ${EPOCH} \
  --evaluation_steps 25 \
  --check_point_steps 1000000 \
  --wandb_name ${MODEL_TYPE}-${DATASET_DIR}-${TAG} \
  --wandb_project "llama-pre-train-cmp"


