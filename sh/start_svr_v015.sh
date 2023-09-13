#!/bin/bash
set -x

cd ../

CUDA_VISIBLE_DEVICES=1 \
python llama_svr.py \
  --port 1301 \
  --length_penalty 1 \
  --num_beams 1 \
  --max_new_tokens 1000 \
  --model_name meta-llama/Llama-2-7b-hf \
  --peft_model /home/paperspace/xingguang/llama/ckpt.peft/7b/answer_extractor.v015-7b.2e-5-peft/step_008000/