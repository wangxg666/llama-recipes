DATA_DIR="/mnt/nlp/xingguang/llama/datasets/nb_training/clickbait"
DATA_DIR="/mnt/nlp/xingguang/llama/datasets/nb_training/grammar_c4200m_seq2seq"

WORK_DIR="/home/cpp/xingguang/llama/model_checkpoints"
DATASET_NAME="my_allin_one_dataset"
TAG="grammar-seq2seq"

python llama_inf.py \
  --model_name ${WORK_DIR}/${DATASET_NAME}/${TAG}-hf/ \
  --dataset my_allin_one_dataset \
  --input_file ${DATA_DIR}/valid.txt \
  --output_file ${DATA_DIR}/valid.txt.pred
