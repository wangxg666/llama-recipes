cd ../

CUDA_VISIBLE_DEVICES=2,3 \
python llama_svr.py \
  --port 1202 \
	--length_penalty 1 \
	--num_beams 5 \
	--do_sample 0 \
	--model_name meta-llama/Llama-2-7b-hf \
	--peft_model /mnt/nlp/xingguang/llama/answer_extractor.v003-peft/best_model