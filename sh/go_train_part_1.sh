#!/bin/bash
set -x

MODEL_TYPE="7b"
WORK_DIR="/home/paperspace/xingguang/models/${MODEL_TYPE}"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
DATASET_NAME="agent_sft_act_dataset"

cd ..

input_dirs=(
"agent_sft.v10.baseline.dst.limit_1k.e04"
"agent_sft.v10.baseline.dst.limit_2k.e03"
"agent_sft.v10.baseline.dst.limit_2k.e04"
)

for dir in "${input_dirs[@]}"

do
DATASET_DIR=${dir}

LR=2e-5
BATCH_SIZE=8
EPOCH=1

TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.${DATASET_DIR}"
ts=$(date +"%Y-%m-%d")


CUDA_VISIBLE_DEVICES="4,5,6,7" torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --master_port=1203 \
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
  --evaluation_steps 200 \
  --check_point_steps 1000000 \
  --wandb_name ${MODEL_TYPE}-${DATASET_DIR}-${TAG} \
  --wandb_project "llama-pre-train-cmp"
done
