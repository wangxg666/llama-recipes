cd ../


CUDA_VISIBLE_DEVICES="4" python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
     "meta-llama/Llama-2-7b-hf" \
     /home/paperspace/xingguang/models/rl/agent.ppo.v09.v03/step_0399/ \
     /home/paperspace/xingguang/models/rl/agent.ppo.v09.v03/step_0399.hf &


CUDA_VISIBLE_DEVICES="5" python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
     "meta-llama/Llama-2-7b-hf" \
     /home/paperspace/xingguang/models/rl/agent.ppo.v09.v03/step_0799/ \
     /home/paperspace/xingguang/models/rl/agent.ppo.v09.v03/step_0799.hf &


CUDA_VISIBLE_DEVICES="6" python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
     "meta-llama/Llama-2-7b-hf" \
     /home/paperspace/xingguang/models/rl/agent.ppo.v09.v03/step_1199/ \
     /home/paperspace/xingguang/models/rl/agent.ppo.v09.v03/step_1199.hf &


CUDA_VISIBLE_DEVICES="7" python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
     "meta-llama/Llama-2-7b-hf" \
     /home/paperspace/xingguang/models/rl/agent.ppo.v09.v03/step_1599/ \
     /home/paperspace/xingguang/models/rl/agent.ppo.v09.v03/step_1599.hf &

