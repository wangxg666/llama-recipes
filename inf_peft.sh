CUDA_VISIBLE_DEVICES="2,3" python inference/inference_my.py \
  --length_penalty 3 \
  --model_name meta-llama/Llama-2-7b-hf \
  --peft_model ./llama-2-7b-clickbait-peft.epoch3/ \
  --dataset my_clickbait_dataset \
  --input_file /mnt/nlp/xingguang/mac_desk/husky-go/hallucination/data_scripts_clickbaity/datas/valid.ex.v1.txt \
  --output_file /mnt/nlp/xingguang/mac_desk/husky-go/hallucination/data_scripts_clickbaity/datas/valid.ex.v1.pred.txt
