CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
  --nnodes 1 \
  --nproc_per_node 2  \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset my_grammar_dataset \
  --save_model \
  --dist_checkpoint_root_folder model_checkpoints \
  --dist_checkpoint_folder fine-tuned  \
  --pure_bf16 \
  --output_dir ./llama-2-7b-grammar-full/ \
  --batch_size_training 4 \
  --micro_batch_size 1

python inference/checkpoint_converter_fsdp_hf.py \
  --fsdp_checkpoint_path model_checkpoints/fine-tuned-meta-llama/Llama-2-7b-hf/ \
  --consolidated_model_path ./llama-2-7b-grammar-full-hf/ \
  --HF_model_path_or_name meta-llama/Llama-2-7b-hf