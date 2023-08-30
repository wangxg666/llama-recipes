WORK_DIR="/home/cpp/xingguang/llama/model_checkpoints"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="my_pre_train_yelp_ins_dataset"
TAG="pt-yelp-ca-25w-v02-mask-ins-a100"
WANDB_TAG="fullCA"
ts=$(date +"%Y-%m-%d")

MODEL_NAME_OR_PATH=""
OPTIMIZER_CHECKPOINT_PATH="./optimizer.pt"

cd ..

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --model_name "${MODEL_NAME}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset ${DATASET_NAME} \
  --save_model \
  --dist_checkpoint_root_folder "${WORK_DIR}" \
  --dist_checkpoint_folder "${DATASET_NAME}/${TAG}"  \
  --pure_bf16 \
  --num_epochs 5 \
  --lr 0.00001 \
  --val_batch_size 8 \
  --batch_size_training 16 \
  --micro_batch_size 16 \
  --max_grad_norm 1.0 \
  --evaluation_steps 100 \
  --check_point_steps 1000 \
  --save_optimizer \
  --optimizer_checkpoint_path "${OPTIMIZER_CHECKPOINT_PATH}" \
  --wandb_name "${ts}-${TAG}-${WANDB_TAG}"

#python inference/checkpoint_converter_fsdp_hf.py \
#  --fsdp_checkpoint_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-${MODEL_NAME}/ \
#  --consolidated_model_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-hf/ \
#  --HF_model_path_or_name ${MODEL_NAME}