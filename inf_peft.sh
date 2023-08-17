CUDA_VISIBLE_DEVICES="2,3" python inference/inference_my.py \
  --length_penalty 3 \
  --model_name meta-llama/Llama-2-7b-hf \
  --peft_model ./llama-2-7b-grammar-peft/step_00009k/ \
  --dataset my_grammar_dataset \
  --input_file /mnt/nlp/xingguang/mac_desk/husky-go/llama/data_scripts/grammar/datas/valid.c4200m.txt \
  --output_file /mnt/nlp/xingguang/mac_desk/husky-go/llama/data_scripts/grammar/datas/valid.c4200m.pred.txt
