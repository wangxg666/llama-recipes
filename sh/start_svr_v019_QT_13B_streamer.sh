#!/bin/bash
set -x

cd ../

CUDA_VISIBLE_DEVICES=2,3 \
python llama_svr_stream.py \
  --port 1308 \
  --length_penalty 1 \
  --num_beams 1 \
  --max_new_tokens 1000 \
  --model_name meta-llama/Llama-2-13b-hf \
  --peft_model /home/paperspace/xingguang/llama/ckpt.peft/13b/answer_extractor.v019-13b.2e-5-peft/epoch_004/