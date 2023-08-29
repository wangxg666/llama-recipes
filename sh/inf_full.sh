#INPUT_FILE="clickbait/valid.txt"
INPUT_FILE="grammar_c4200m_seq2seq/valid.txt"

WORK_DIR="/home/cpp/xingguang/llama/model_checkpoints"
DATASET_NAME="my_allin_one_dataset"
TAG="grammar-seq2seq"

cd ..

python llama_inf.py \
  --model_name ${WORK_DIR}/${DATASET_NAME}/${TAG}-hf/ \
  --dataset my_allin_one_dataset \
  --input_file ${INPUT_FILE}
