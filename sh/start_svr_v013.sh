#!/bin/bash
set -x

cd ../

CUDA_VISIBLE_DEVICES=1 \
python llama_svr.py \
  --port 1301 \
        --length_penalty 0 \
        --num_beams 10 \
        --max_new_tokens 1000 \
        --do_sample 1 \
        --model_name meta-llama/Llama-2-7b-hf \
        --peft_model /home/paperspace/xingguang/llama/ckpt.peft/meta-llama/Llama-2-7b-hf/my_allin_one_dataset/answer_extractor.v013.7b.2e-5-peft/best_model