python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_1k.new" --split "dev" --host "http://0.0.0.0:8000" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_2k.new" --split "dev" --host "http://0.0.0.0:8001" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_4k.new" --split "dev" --host "http://0.0.0.0:8002" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_8k.new" --split "dev" --host "http://0.0.0.0:8003" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_1k.new" --split "dev" --tag ".pre-train-8k" --host "http://0.0.0.0:8004" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_2k.new" --split "dev" --tag ".pre-train-8k" --host "http://0.0.0.0:8005" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_4k.new" --split "dev" --tag ".pre-train-8k" --host "http://0.0.0.0:8006" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_8k.new" --split "dev" --tag ".pre-train-8k" --host "http://0.0.0.0:8007" &

wait


python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_1k.new" --split "test" --host "http://0.0.0.0:8000" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_2k.new" --split "test" --host "http://0.0.0.0:8001" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_4k.new" --split "test" --host "http://0.0.0.0:8002" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_8k.new" --split "test" --host "http://0.0.0.0:8003" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_1k.new" --split "test" --tag ".pre-train-8k" --host "http://0.0.0.0:8004" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_2k.new" --split "test" --tag ".pre-train-8k" --host "http://0.0.0.0:8005" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_4k.new" --split "test" --tag ".pre-train-8k" --host "http://0.0.0.0:8006" &
python run_agent_two_stage_0_act_baseline_dst_vllm.py --dataset "agent_sft.woz.2.4.limit_8k.new" --split "test" --tag ".pre-train-8k" --host "http://0.0.0.0:8007" &

