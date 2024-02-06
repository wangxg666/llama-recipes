#!/bin/bash
set -x
DATASET_NAME="agent_sft_act_dataset"

MODEL_TYPE="13b"
LR=2e-5
BATCH_SIZE=8
EPOCH=1

input_dirs=(
#"agent_sft.woz.2.4.limit_1k.new"
#"agent_sft.woz.2.4.limit_2k.new"
#"agent_sft.woz.2.4.limit_4k.new"
"agent_sft.woz.2.4.limit_8k.new"
)

for dir in "${input_dirs[@]}"

do
  DATASET_DIR=${dir}

#  TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.${DATASET_DIR}"
#  MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
#
#  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
#    --nnodes 1 \
#    --nproc_per_node 8 \
#    --master_port=1201 \
#    ../llama_finetuning.py \
#    --enable_fsdp  \
#    --model_name "${MODEL_NAME}" \
#    --dataset "${DATASET_NAME}" \
#    --dataset_dir "${DATASET_DIR}" \
#    --save_model \
#    --pure_bf16 \
#    --output_dir "/mnt/share16t/xingguang/models/${DATASET_NAME}.${TAG}"/ \
#    --lr ${LR} \
#    --valid_batch_size ${BATCH_SIZE} \
#    --train_batch_size ${BATCH_SIZE} \
#    --micro_batch_size ${BATCH_SIZE} \
#    --num_epochs ${EPOCH} \
#    --evaluation_steps 200 \
#    --check_point_steps 1000000 \
#    --wandb_name ${TAG} \
#    --wandb_project "agent"

  # train 2.2 with pre-train 8k
  TAG="FT-16K-${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.${DATASET_DIR}"
  MODEL_NAME="/home/paperspace/xingguang/models/agent_sft_act_dataset.13b.1e-5.full.B8.E1.agent_sft.auto.gen.v08.37.1.template.16k.dst.ctx.for2.4.hf/"

  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    --master_port=1201 \
    ../llama_finetuning.py \
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
    --wandb_name ${TAG} \
    --wandb_project "agent"
done