cd ../

CUDA_VISIBLE_DEVICES=0,1 \
python llama_svr.py \
	--length_penalty 1 \
	--model_name meta-llama/Llama-2-7b-hf \
	--peft_model /mnt/nlp/xingguang/llama/answer_extractor-peft/best_model