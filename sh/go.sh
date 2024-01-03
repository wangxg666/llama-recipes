cd ../

CUDA_VISIBLE_DEVICES="0" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0600 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0600.hf &



CUDA_VISIBLE_DEVICES="1" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0700 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0700.hf &


CUDA_VISIBLE_DEVICES="2" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0800 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0800.hf &


CUDA_VISIBLE_DEVICES="5" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0900 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_0900.hf &


CUDA_VISIBLE_DEVICES="6" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_1000 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/origin/step_1000.hf &

