#!/bin/bash
set -x

cd ../

CUDA_VISIBLE_DEVICES=0,1 \
python llama_svr.py \
  --port 1301 \
  --length_penalty 1 \
  --num_beams 1 \
  --max_new_tokens 1000 \
  --model_name meta-llama/Llama-2-13b-hf \
  --peft_model /home/paperspace/xingguang/llama/ckpt.peft/13b/answer_extractor.v016-13b.2e-5-peft/epoch_004/