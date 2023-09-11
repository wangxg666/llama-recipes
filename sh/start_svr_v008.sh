#!/bin/bash
set -x

cd ../

CUDA_VISIBLE_DEVICES=2,3 \
python llama_svr.py \
  --port 1201 \
	--length_penalty 0.5 \
	--num_beams 10 \
	--do_sample 0 \
	--model_name meta-llama/Llama-2-7b-hf \
	--peft_model /mnt/nlp/xingguang/llama/answer_extractor.v008.2e-5-peft/best_model