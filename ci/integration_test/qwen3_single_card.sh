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

source PaddleFleet/.venv/bin/activate

export root_dir=$(pwd)
cd $root_dir/PaddleFormers/examples/experiments/paddlefleet

config_json="qwen_single_card.json"

jq --arg cache "$CACHE_DIR" \
   '.save_steps = 100
    | .input_dir = "1.0 \($cache)/glm45/data/pre-training/llama_openwebtext_100k"
    | .model_name_or_path = "\($cache)/qwen/Qwen3-30B-A3B-Base"' \
   $config_json > $config_json.tmp
mv $config_json.tmp $config_json

ls -lah $CACHE_DIR/qwen/Qwen3-30B-A3B-Base
cat $config_json

rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

unset http_proxy https_proxy
coverage run run_pretrain.py $config_json | tee ./qwen3_single_card.log
