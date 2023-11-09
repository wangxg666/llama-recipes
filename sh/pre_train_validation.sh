#!/bin/bash
set -x

function run_validation() {
  LR=$1
  PRE_TRAIN_MODEL_PATH=$2
  DATA_SET_DIR=$3
  TAG=$4

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
        --nnodes 1 \
        --nproc_per_node 8 \
        --master_port=1202 ./llama_finetuning.py \
        --enable_fsdp \
        --model_name "meta-llama/Llama-2-13b-hf" \
        --pre_train_model_path "${PRE_TRAIN_MODEL_PATH}" \
        --dataset "my_allin_one_dataset" \
        --dataset_dir "${DATA_SET_DIR}" \
        --save_model \
        --pure_bf16 \
        --output_dir "/home/paperspace/xingguang/models/13b/${DATA_SET_DIR}.Test" \
        --lr ${LR} \
        --valid_batch_size 4 \
        --train_batch_size 4 \
        --micro_batch_size 4 \
        --num_epochs 1 \
        --evaluation_steps 100 \
        --check_point_steps 2000 \
        --wandb_name "13b-answer_extractor.v029-13b.${LR}.${TAG}" \
        --wandb_project "llama-pre-train-cmp"
}

cd ../

CUR_PRE_TRAIN_MODEL_PATH="/home/paperspace/xingguang/models/my_pre_train_dataset.7b.3e-5.B8.E1.full.step_062978.hf"

run_validation 3e-5 ${CUR_PRE_TRAIN_MODEL_PATH} "answer_extractor.v028" "withPreTrain"
run_validation 3e-5 "" "answer_extractor.v028" "base"

run_validation 3e-5 ${CUR_PRE_TRAIN_MODEL_PATH} "answer_extractor.v029" "withPreTrain"
run_validation 3e-5 "" "answer_extractor.v029" "base"