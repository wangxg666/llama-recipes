#!/bin/bash
set -x
DATASET_NAME="agent_sft_act_dataset"

MODEL_TYPE="7b"
LR=2e-5
BATCH_SIZE=8
EPOCH=1

input_dirs=(
#"agent_sft.v10.baseline.dst.limit_1k"
#"agent_sft.v10.baseline.dst.limit_2k"
#"agent_sft.v10.baseline.dst.limit_4k"
"agent_sft.v10.baseline.dst.limit_8k"
)

for dir in "${input_dirs[@]}"

do
  DATASET_DIR=${dir}

  # train 2.2 with pre-train 8k
  TAG="FT-8K-${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.${DATASET_DIR}.DEDUP"
  # MODEL_NAME="/home/paperspace/xingguang/models/agent_sft_act_dataset.13b.1e-5.full.B8.E1.agent_sft.auto.gen.v08.37.1.template.16k.dst.ctx.hf/"
  MODEL_NAME="/home/paperspace/xingguang/models/agent_sft_act_dataset.7b.2e-5.full.B8.E1.agent_sft.auto.gen.v08.37.1.template.8k.dst.ctx.deduped.hf"

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