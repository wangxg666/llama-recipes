cd ../

CUDA_VISIBLE_DEVICES="0" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0100 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0100.hf &



CUDA_VISIBLE_DEVICES="1" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0200 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0200.hf &


CUDA_VISIBLE_DEVICES="2" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0300 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0300.hf &


CUDA_VISIBLE_DEVICES="5" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0400 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0400.hf &


CUDA_VISIBLE_DEVICES="6" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0500 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0500.hf &


CUDA_VISIBLE_DEVICES="7" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0600 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0600.hf &