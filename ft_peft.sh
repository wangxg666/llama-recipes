CUDA_VISIBLE_DEVICES="2,3" torchrun \
  --nnodes 1 \
  --nproc_per_node 2  \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --use_peft \
  --peft_method lora \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset my_grammar_dataset \
  --save_model \
  --dist_checkpoint_root_folder model_checkpoints \
  --dist_checkpoint_folder fine-tuned  \
  --pure_bf16 \
  --output_dir ./llama-2-7b-grammar-peft/ \
  --batch_size_training 4 \
  --micro_batch_size 4 \
  --num_epochs 5 \
  --check_point_steps 1000