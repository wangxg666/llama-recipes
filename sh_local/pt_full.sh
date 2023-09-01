WORK_DIR="/home/paperspace/llama/ckpt.full"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="my_pre_train_pad_dataset"
TAG="pt-yelp-ca-25w-v02.merge-pad256-h100"
WANDB_TAG="fullCA"
ts=$(date +"%Y-%m-%d")

MODEL_NAME_OR_PATH=""
OPTIMIZER_CHECKPOINT_PATH=""


cd ../
CUDA_VISIBLE_DEVICES="4,5,6,7" torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --master_port=25641 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --model_name "${MODEL_NAME}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset ${DATASET_NAME} \
  --save_model \
  --dist_checkpoint_root_folder "${WORK_DIR}" \
  --dist_checkpoint_folder "${DATASET_NAME}/${TAG}"  \
  --lr 2e-5 \
  --pure_bf16 \
  --num_epochs 5 \
  --batch_size_training 32 \
  --val_batch_size 32 \
  --micro_batch_size 32 \
  --max_grad_norm 1.0 \
  --evaluation_steps 100 \
  --check_point_steps 1000 \
  --optimizer_checkpoint_path "${OPTIMIZER_CHECKPOINT_PATH}" \
  --wandb_name "${ts}-${TAG}-${WANDB_TAG}"

#python inference/checkpoint_converter_fsdp_hf.py \
#  --fsdp_checkpoint_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-${MODEL_NAME}/ \
#  --consolidated_model_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-hf/ \
#  --HF_model_path_or_name ${MODEL_NAME}
