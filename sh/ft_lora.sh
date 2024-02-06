#!/bin/bash
set -x

MODEL_TYPE="7b"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
DATASET_NAME="agent_sft_act_dataset"

input_dirs=(
"agent_sft.v10.baseline.dst.limit_8k"
)

for dir in "${input_dirs[@]}"

do
  DATASET_DIR=${dir}

  cd ../
  LR=1e-5
  BATCH_SIZE=16
  EPOCH=1

  TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.${DATASET_DIR}.LORA"

  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
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
    --output_dir "/mnt/share16t/xingguang/models//${DATASET_NAME}.${TAG}-peft/" \
    --lr 2e-5 \
    --valid_batch_size ${BATCH_SIZE} \
    --train_batch_size ${BATCH_SIZE} \
    --micro_batch_size ${BATCH_SIZE} \
    --num_epochs 5 \
    --evaluation_steps 500 \
    --check_point_steps 100 \
    --wandb_name ${TAG} \
    --wandb_project "agent"

done

