WORK_DIR="/home/cpp/xingguang/llama/model_checkpoints"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="my_allin_one_dataset"
TAG="grammar-single"

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --model_name ${MODEL_NAME} \
  --dataset ${DATASET_NAME} \
  --save_model \
  --dist_checkpoint_root_folder ${WORK_DIR} \
  --dist_checkpoint_folder ${DATASET_NAME}/${TAG}  \
  --pure_bf16 \
  --batch_size_training 4 \
  --micro_batch_size 2 \
  --check_point_steps 10000

python inference/checkpoint_converter_fsdp_hf.py \
  --fsdp_checkpoint_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-${MODEL_NAME}/ \
  --consolidated_model_path ${WORK_DIR}/${DATASET_NAME}/${TAG}-hf/ \
  --HF_model_path_or_name ${MODEL_NAME}