MODEL=/home/paperspace/xingguang/models/pre-train/step_028489.hf


if [ ! -d "./lm-evaluation-harness" ];then
  git clone https://github.com/EleutherAI/lm-evaluation-harness
  cd lm-evaluation-harness && exit
  pip install -e .
else
  echo "文件夹已经存在"
fi

function run_task() {
tasks=$1
num_fewshot=$2
output_path=$3
cuda_index=$4
echo "python main.py\
  --model hf-causal-experimental\
  --model_args pretrained=${MODEL},dtype=\"float16\"\
  --batch_size 2\
  --tasks ${tasks}\
  --num_fewshot ${num_fewshot}\
  --output_path ${output_path}\
  --device cuda:${cuda_index}"
}

#run_task hellaswag 10 hellaswag.out 4
#run_task truthfulqa_mc 0 truthfulqa_mc.out 5
#run_task arc_challenge 25 arc_challenge.out 6


tasks="hendrycksTest-abstract_algebra hendrycksTest-anatomy hendrycksTest-astronomy hendrycksTest-business_ethics hendrycksTest-clinical_knowledge hendrycksTest-college_biology hendrycksTest-college_chemistry hendrycksTest-college_computer_science hendrycksTest-college_mathematics hendrycksTest-college_medicine hendrycksTest-college_physics hendrycksTest-computer_security hendrycksTest-conceptual_physics hendrycksTest-econometrics hendrycksTest-electrical_engineering hendrycksTest-elementary_mathematics hendrycksTest-high_school_chemistry hendrycksTest-high_school_computer_science hendrycksTest-high_school_microeconomics hendrycksTest-high_school_european_history hendrycksTest-high_school_geography hendrycksTest-high_school_physics hendrycksTest-high_school_government_and_politics hendrycksTest-high_school_macroeconomics hendrycksTest-formal_logic hendrycksTest-high_school_mathematics hendrycksTest-global_facts hendrycksTest-high_school_biology hendrycksTest-high_school_psychology hendrycksTest-high_school_statistics hendrycksTest-high_school_us_history hendrycksTest-high_school_world_history hendrycksTest-human_aging hendrycksTest-human_sexuality hendrycksTest-international_law hendrycksTest-jurisprudence hendrycksTest-logical_fallacies hendrycksTest-machine_learning hendrycksTest-management hendrycksTest-marketing hendrycksTest-medical_genetics hendrycksTest-miscellaneous hendrycksTest-moral_disputes hendrycksTest-moral_scenarios hendrycksTest-nutrition hendrycksTest-world_religions hendrycksTest-philosophy hendrycksTest-prehistory hendrycksTest-professional_accounting hendrycksTest-professional_law hendrycksTest-professional_medicine hendrycksTest-professional_psychology hendrycksTest-public_relations hendrycksTest-security_studies hendrycksTest-sociology hendrycksTest-us_foreign_policy hendrycksTest-virology"
for task in $tasks
do
  run_task "${task}" 5 "${task}.out" 7
done