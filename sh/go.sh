cd ../

CUDA_VISIBLE_DEVICES="2" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /mnt/share16t/xingguang/models/rl/agent.ppo.v09.v04/step_0200 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.v04/step_0200.hf &
    
CUDA_VISIBLE_DEVICES="2" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /mnt/share16t/xingguang/models/rl/agent.ppo.v09.v04/step_0400 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.v04/step_0400.hf &
    
CUDA_VISIBLE_DEVICES="5" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /mnt/share16t/xingguang/models/rl/agent.ppo.v09.v04/step_0600 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.v04/step_0600.hf &
    
CUDA_VISIBLE_DEVICES="5" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /mnt/share16t/xingguang/models/rl/agent.ppo.v09.v04/step_0800 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.v04/step_0800.hf &
    
CUDA_VISIBLE_DEVICES="6" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /mnt/share16t/xingguang/models/rl/agent.ppo.v09.v04/step_1000 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.v04/step_1000.hf &
    
CUDA_VISIBLE_DEVICES="6" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /mnt/share16t/xingguang/models/rl/agent.ppo.v09.v04/step_1200 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.v04/step_1200.hf &
    
CUDA_VISIBLE_DEVICES="7" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /mnt/share16t/xingguang/models/rl/agent.ppo.v09.v04/step_1400 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.v04/step_1400.hf &
    
CUDA_VISIBLE_DEVICES="7" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /mnt/share16t/xingguang/models/rl/agent.ppo.v09.v04/step_1600 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.v04/step_1600.hf &
    