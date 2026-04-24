#!/usr/bin/env bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

source .venv/bin/activate

MASTER_PORT=29500
DISTRIBUTED_ARGS=`python scripts/selective_launch.py ${MASTER_PORT}`
if [[ -z "$DISTRIBUTED_ARGS" ]]; then
    exit 0
fi

# NLTK 下载（cite eval 中需要）
# python -c "import ssl; import nltk; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('punkt_tab')"

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINERS_NUM
export PADDLE_TRAINERS_NUM=1
export FLAGS_flash_attn_version=3

export PYTHONPATH="`pwd`:${PYTHONPATH:-}"

cd "eval/HELMET"

model_name_or_paths=${model_name_or_paths:-}
if [[ -z "$model_name_or_paths" ]]; then
    echo "Please set model_name_or_paths to one or more local checkpoint paths."
    echo "Example: model_name_or_paths=/path/to/local-model-or-paddle-checkpoint bash ./scripts/run_helmet.sh"
    exit 1
fi
data_root_dir=${data_root_dir:-"."}
qa_model_name_or_path=${qa_model_name_or_path:-"./models/roberta-large-squad"}
autoais_model_name_or_path=${autoais_model_name_or_path:-"./models/t5_xxl_true_nli_mixture"}

methods="rrattn"
tags="v1"
thresholds="0.9 0.95"
tasks="recall rag longqa icl rerank cite"

for model_name_or_path in $model_name_or_paths; do
    for tag in $tags; do
        for method in $methods; do
            for threshold in $thresholds; do
                for task in $tasks; do
                    # this will run the 8k to 64k versions
                    torchrun $DISTRIBUTED_ARGS eval.py \
                        --model_name_or_path $model_name_or_path \
                        --qa_model_name_or_path $qa_model_name_or_path \
                        --autoais_model_name_or_path $autoais_model_name_or_path \
                        --data_root_dir $data_root_dir \
                        --config configs/${task}_short.yaml \
                        --tag $tag \
                        --method $method \
                        --threshold $threshold \
                        --rrattn_version $tag
                    if [ $? -ne 0 ]; then
                        echo "评估进程失败，终止"
                        exit 1
                    fi

                    # this will run the 128k versions
                    torchrun $DISTRIBUTED_ARGS eval.py \
                        --model_name_or_path $model_name_or_path \
                        --data_root_dir $data_root_dir \
                        --qa_model_name_or_path $qa_model_name_or_path \
                        --autoais_model_name_or_path $autoais_model_name_or_path \
                        --config configs/${task}.yaml \
                        --tag $tag \
                        --method $method \
                        --threshold $threshold \
                        --rrattn_version $tag
                    if [ $? -ne 0 ]; then
                        echo "评估进程失败，终止"
                        exit 1
                    fi

                done
            done
        done
    done
done

methods="full"
tags="v1"
thresholds="1.0"
tasks="recall rag longqa icl rerank cite"

for model_name_or_path in $model_name_or_paths; do
    for tag in $tags; do
        for method in $methods; do
            for threshold in $thresholds; do
                for task in $tasks; do
                    # this will run the 8k to 64k versions
                    torchrun $DISTRIBUTED_ARGS eval.py \
                        --model_name_or_path $model_name_or_path \
                        --qa_model_name_or_path $qa_model_name_or_path \
                        --autoais_model_name_or_path $autoais_model_name_or_path \
                        --data_root_dir $data_root_dir \
                        --config configs/${task}_short.yaml \
                        --tag $tag \
                        --method $method \
                        --threshold $threshold \
                        --rrattn_version $tag
                    if [ $? -ne 0 ]; then
                        echo "评估进程失败，终止"
                        exit 1
                    fi

                    # this will run the 128k versions
                    torchrun $DISTRIBUTED_ARGS eval.py \
                        --model_name_or_path $model_name_or_path \
                        --data_root_dir $data_root_dir \
                        --qa_model_name_or_path $qa_model_name_or_path \
                        --autoais_model_name_or_path $autoais_model_name_or_path \
                        --config configs/${task}.yaml \
                        --tag $tag \
                        --method $method \
                        --threshold $threshold \
                        --rrattn_version $tag
                    if [ $? -ne 0 ]; then
                        echo "评估进程失败，终止"
                        exit 1
                    fi

                done
            done
        done
    done
done

echo "评估完成, 正常退出"
