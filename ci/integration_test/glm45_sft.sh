# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

set -exo pipefail
export root_dir=$(pwd)

cd $root_dir/PaddleFormers
git reset --hard HEAD
cd -

apt-get update
apt-get install jq -y

source PaddleFleet/.venv/bin/activate

cd $root_dir/glm45_fleet
export cur_dir=$(pwd)

config_sft_yaml=$cur_dir/tp4_sp_ep4_pp2_sft.yaml
config_lora_yaml=$cur_dir/tp4_sp_ep4_pp2_lora.yaml

config_json=$CACHE_DIR/glm45/GLM-4.5-Air/config.json

yq '.recompute_granularity = ""
    | .moe_token_dispatcher_type = "deepep"
    | .gated_linear_unit = true
    | .train_dataset_path = strenv(cur_dir) + "/data/sft/train_gsm8k.jsonl"
    | .eval_dataset_path = strenv(cur_dir) + "/data/sft/test_gsm8k.jsonl"
    | .model_name_or_path = strenv(CACHE_DIR) + "/glm45/GLM-4.5-Air"
    | .logging_dir = strenv(cur_dir) + "/vdl_log"
    | .output_dir = strenv(cur_dir) + "/checkpoints"' \
   $config_sft_yaml > ${config_sft_yaml}.tmp
mv ${config_sft_yaml}.tmp $config_sft_yaml

yq 'del(.recompute_granularity)
    | .train_dataset_path = strenv(cur_dir) + "/data/sft/train_gsm8k.jsonl"
    | .eval_dataset_path = strenv(cur_dir) + "/data/sft/test_gsm8k.jsonl"
    | .model_name_or_path = strenv(CACHE_DIR) + "/glm45/GLM-4.5-Air"
    | .logging_dir = strenv(cur_dir) + "/vdl_log"
    | .output_dir = strenv(cur_dir) + "/checkpoints"' \
   $config_lora_yaml > ${config_lora_yaml}.tmp
mv ${config_lora_yaml}.tmp $config_lora_yaml

jq '.num_hidden_layers = 4' \
    $config_json > ${config_json}.tmp
mv ${config_json}.tmp $config_json

rm -rf ./outputs
rm -rf paddleformers_dist_log
master=$(hostname -i)
port=36677

export FLAGS_use_stride_compute_kernel=False
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

unset http_proxy https_proxy

NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_sft_yaml 2>&1 | tee ./glm45_sft.log

NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_lora_yaml 2>&1 | tee ./glm45_lora.log
