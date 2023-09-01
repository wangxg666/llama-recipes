WORK_DIR="/home/paperspace/llama/ckpt.full"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="my_pre_train_dataset"
TAG="pt-yelp-ny-a100"

cd ../
python inference/inference_pt.py \
  --model_name ${WORK_DIR}/${DATASET_NAME}/${TAG}-hf/
