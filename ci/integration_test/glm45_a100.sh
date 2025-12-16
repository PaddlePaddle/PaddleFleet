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

wget https://github.com/mikefarah/yq/releases/download/v4.44.1/yq_linux_amd64 -O /usr/local/bin/yq
chmod +x /usr/local/bin/yq
apt-get update
apt-get install jq -y

source PaddleFleet/.venv/bin/activate

wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/glm45/glm45_fleet_pt.1214.tar --no-check-certificate
tar -xf glm45_fleet_pt.1214.tar # glm45_fleet_pt
cd $root_dir/glm45_fleet_pt
export cur_dir=$(pwd)

config_yaml=$cur_dir/glm45.yaml
config_json=${cur_dir}/GLM-4.5-Air/config.json

yq eval '.expert_model_parallel_size = 1
    | .per_device_train_batch_size = 1
    | .use_expert_parallel = false
    | .train_dataset_path = strenv(cur_dir) + "/data/pre-training/train.jsonl"
    | .eval_dataset_path = strenv(cur_dir) + "/data/pre-training/eval.jsonl"
    | .model_name_or_path = strenv(cur_dir) + "/GLM-4.5-Air"
    | .logging_dir = strenv(cur_dir) + "/vdl_log"
    | .output_dir = strenv(cur_dir) + "/checkpoints"' \
  $config_yaml > ${config_yaml}.tmp
mv ${config_yaml}.tmp $config_yaml

# jq --arg cur_dir "$cur_dir" \
#     '.first_k_dense_replace = 0' \
#     $config_json > ${config_json}.tmp
# mv ${config_json}.tmp $config_json

sed -i 's/config.num_hidden_layers = 10/config.num_hidden_layers = 1/g' /workspace/PaddleFormers/paddleformers/transformers/glm4_moe/modeling.py
sed -i 's/\[0\] \* 1 + \[1\] \* 9/\[1\] \* 1/g' /workspace/PaddleFormers/paddleformers/transformers/glm4_moe/modeling.py

rm -rf checkpoints/
rm -rf vdl_log/
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

unset http_proxy https_proxy
# coverage run -m paddle.distributed.launch \
#    --log_dir ./log \
#    --master $master:$port \
#    --nnodes 1 \
#    --rank 0 \
#    --run_mode=collective \
#    run_pretrain.py $config_json \
#    --output_dir ./checkpoint | tee ./glm45_a100.log

FLAGS_use_stride_compute_kernel=False NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 coverage run $(which paddleformers-cli) train $cur_dir/glm45.yaml 2>&1 | tee ./glm45_a100.log
