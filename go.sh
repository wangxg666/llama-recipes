CUDA_VISIBLE_DEVICES="0" python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path /mnt/share16t/xingguang/models/agent_sft_act_dataset.FT-8K-13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_1k.new/epoch_000 \
    --consolidated_model_path /home/paperspace/xingguang/models/agent_sft_act_dataset.FT-8K-13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_1k.new.hf \
    --HF_model_path_or_name "meta-llama/Llama-2-13b-hf" &


CUDA_VISIBLE_DEVICES="1" python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path /mnt/share16t/xingguang/models/agent_sft_act_dataset.FT-8K-13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_2k.new/epoch_000 \
    --consolidated_model_path /home/paperspace/xingguang/models/agent_sft_act_dataset.FT-8K-13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_2k.new.hf \
    --HF_model_path_or_name "meta-llama/Llama-2-13b-hf" &


CUDA_VISIBLE_DEVICES="2" python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path /mnt/share16t/xingguang/models/agent_sft_act_dataset.FT-8K-13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_4k.new/epoch_000 \
    --consolidated_model_path /home/paperspace/xingguang/models/agent_sft_act_dataset.FT-8K-13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_4k.new.hf \
    --HF_model_path_or_name "meta-llama/Llama-2-13b-hf" &


CUDA_VISIBLE_DEVICES="3" python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path /mnt/share16t/xingguang/models/agent_sft_act_dataset.FT-8K-13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_8k.new/epoch_000 \
    --consolidated_model_path /home/paperspace/xingguang/models/agent_sft_act_dataset.FT-8K-13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_8k.new.hf \
    --HF_model_path_or_name "meta-llama/Llama-2-13b-hf" &


CUDA_VISIBLE_DEVICES="4" python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path /mnt/share16t/xingguang/models/agent_sft_act_dataset.13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_1k.new/epoch_000 \
    --consolidated_model_path /home/paperspace/xingguang/models/agent_sft_act_dataset.13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_1k.new.hf \
    --HF_model_path_or_name "meta-llama/Llama-2-13b-hf" &


CUDA_VISIBLE_DEVICES="5" python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path /mnt/share16t/xingguang/models/agent_sft_act_dataset.13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_2k.new/epoch_000 \
    --consolidated_model_path /home/paperspace/xingguang/models/agent_sft_act_dataset.13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_2k.new.hf \
    --HF_model_path_or_name "meta-llama/Llama-2-13b-hf" &


CUDA_VISIBLE_DEVICES="6" python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path /mnt/share16t/xingguang/models/agent_sft_act_dataset.13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_4k.new/epoch_000 \
    --consolidated_model_path /home/paperspace/xingguang/models/agent_sft_act_dataset.13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_4k.new.hf \
    --HF_model_path_or_name "meta-llama/Llama-2-13b-hf" &


CUDA_VISIBLE_DEVICES="7" python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path /mnt/share16t/xingguang/models/agent_sft_act_dataset.13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_8k.new/epoch_000 \
    --consolidated_model_path /home/paperspace/xingguang/models/agent_sft_act_dataset.13b.2e-5.full.B8.E1.agent_sft.woz.2.4.limit_8k.new.hf \
    --HF_model_path_or_name "meta-llama/Llama-2-13b-hf" &

