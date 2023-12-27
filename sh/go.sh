cd ../



CUDA_VISIBLE_DEVICES="1" python inference/checkpoint_conveter_deepspeed_zero_2_hf.py \
     "meta-llama/Llama-2-7b-hf" \
     /home/paperspace/xingguang/models/rl/agent.ppo.v09.v02/step_2199/ \
     /home/paperspace/xingguang/models/rl/agent.ppo.v09.v02/step_2199.hf
