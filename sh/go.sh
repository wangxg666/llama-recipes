ckpt_prefix="/mnt/share16t/xingguang/models/agent_sft_act_dataset.7b.2e-5.full.B8.E1.agent_sft.v10.baseline.dst"
hf_ckpt_prefix="/home/paperspace/xingguang/models/agent_sft_act_dataset.7b.2e-5.full.B8.E1.agent_sft.v10.baseline.dst"

cd ../

CUDA_VISIBLE_DEVICES="0" \
python inference/checkpoint_converter_fsdp_hf.py \
    --HF_model_path_or_name "meta-llama/Llama-2-7b-hf" \
    --fsdp_checkpoint_path ${ckpt_prefix}.limit_1k.e01/epoch_000 \
    --consolidated_model_path ${hf_ckpt_prefix}.limit_1k.e01.hf &

CUDA_VISIBLE_DEVICES="1" \
python inference/checkpoint_converter_fsdp_hf.py \
    --HF_model_path_or_name "meta-llama/Llama-2-7b-hf" \
    --fsdp_checkpoint_path ${ckpt_prefix}.limit_1k.e02/epoch_000 \
    --consolidated_model_path ${hf_ckpt_prefix}.limit_1k.e02.hf &

CUDA_VISIBLE_DEVICES="2" \
python inference/checkpoint_converter_fsdp_hf.py \
    --HF_model_path_or_name "meta-llama/Llama-2-7b-hf" \
    --fsdp_checkpoint_path ${ckpt_prefix}.limit_1k.e03/epoch_000 \
    --consolidated_model_path ${hf_ckpt_prefix}.limit_1k.e03.hf &

CUDA_VISIBLE_DEVICES="3" \
python inference/checkpoint_converter_fsdp_hf.py \
    --HF_model_path_or_name "meta-llama/Llama-2-7b-hf" \
    --fsdp_checkpoint_path ${ckpt_prefix}.limit_1k.e04/epoch_000 \
    --consolidated_model_path ${hf_ckpt_prefix}.limit_1k.e04.hf &

CUDA_VISIBLE_DEVICES="4" \
python inference/checkpoint_converter_fsdp_hf.py \
    --HF_model_path_or_name "meta-llama/Llama-2-7b-hf" \
    --fsdp_checkpoint_path ${ckpt_prefix}.limit_2k.e01/epoch_000 \
    --consolidated_model_path ${hf_ckpt_prefix}.limit_2k.e01.hf &

CUDA_VISIBLE_DEVICES="5" \
python inference/checkpoint_converter_fsdp_hf.py \
    --HF_model_path_or_name "meta-llama/Llama-2-7b-hf" \
    --fsdp_checkpoint_path ${ckpt_prefix}.limit_2k.e02/epoch_000 \
    --consolidated_model_path ${hf_ckpt_prefix}.limit_2k.e02.hf &

CUDA_VISIBLE_DEVICES="6" \
python inference/checkpoint_converter_fsdp_hf.py \
    --HF_model_path_or_name "meta-llama/Llama-2-7b-hf" \
    --fsdp_checkpoint_path ${ckpt_prefix}.limit_2k.e03/epoch_000 \
    --consolidated_model_path ${hf_ckpt_prefix}.limit_2k.e03.hf &

CUDA_VISIBLE_DEVICES="7" \
python inference/checkpoint_converter_fsdp_hf.py \
    --HF_model_path_or_name "meta-llama/Llama-2-7b-hf" \
    --fsdp_checkpoint_path ${ckpt_prefix}.limit_2k.e04/epoch_000 \
    --consolidated_model_path ${hf_ckpt_prefix}.limit_2k.e04.hf &