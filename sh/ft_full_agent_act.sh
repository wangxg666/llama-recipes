#!/bin/bash
set -x
DATASET_NAME="agent_sft_act_dataset"

MODEL_TYPE="7b"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"

input_dirs=(
"agent_sft.auto.gen.v08.37.1.template.8k.dst.ctx/"
)

for dir in "${input_dirs[@]}"

do
  DATASET_DIR=${dir}

  LR=2e-5
  BATCH_SIZE=2
  EPOCH=1

  TAG="${MODEL_TYPE}.Mistral.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.${DATASET_DIR}"

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