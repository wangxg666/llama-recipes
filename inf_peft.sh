#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES="2,3" python inference/inference_my.py \
  --length_penalty 3 \
  --model_name meta-llama/Llama-2-7b-hf \
  --peft_model ./llama-2-7b-allin-one-peft/epoch_000/ \
  --dataset my_allin_one_dataset \
  --input_file /mnt/nlp/xingguang/llama/datasets/nb_training/grammar_c4200m/valid.toy.txt \
  --output_file /mnt/nlp/xingguang/llama/datasets/nb_training/grammar_c4200m/valid.toy.pred.txt
