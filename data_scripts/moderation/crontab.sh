#!/bin/bash

SYS_PATH=${PATH}
CONDA_PATH="/mnt/nlp/xingguang/Anaconda3/bin"
export PATH=${CONDA_PATH}:${SYS_PATH}

export PATH=$ANACONDA_HOME/bin:$PATH


source  /mnt/nlp/xingguang/.bash_profile.openai
echo ${OPENAI_API_KEY}

export PYTHONPATH='/mnt/nlp/xingguang/mac_desk/husky-go'
cd /mnt/nlp/xingguang/mac_desk/husky-go/llama/data_scripts/moderation
python 00_pull_all_training_data.py


echo "start dump transcript"
export PYTHONPATH='/mnt/nlp/xingguang/mac_desk/content-generation/content_generation'
cd /mnt/nlp/xingguang/mac_desk/content-generation/content_generation/aigc/utils
python youtube_transcript_run.py


export PYTHONPATH='/mnt/nlp/xingguang/mac_desk/husky-go'
cd /mnt/nlp/xingguang/mac_desk/husky-go/llama/data_scripts/moderation
python 02_backfill_mongo.py

