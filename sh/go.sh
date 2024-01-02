cd ../

CUDA_VISIBLE_DEVICES="4" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/replace/step_0200 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/replace/step_0200.hf &



CUDA_VISIBLE_DEVICES="5" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/replace/step_0400 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/replace/step_0400.hf &


CUDA_VISIBLE_DEVICES="6" \
python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
    "meta-llama/Llama-2-7b-hf" \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/replace/step_0600 \
    /home/paperspace/xingguang/models/rl/agent.ppo.v09.1.v01/replace/step_0600.hf &