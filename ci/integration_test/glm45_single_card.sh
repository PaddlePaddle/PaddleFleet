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

config_json="glm45_single_card.json"

jq --arg cache "$CACHE_DIR" \
   '.save_steps = 100
    | .input_dir = "1.0 \($cache)/glm45/data/pre-training/llama_openwebtext_100k"
    | .model_name_or_path = "\($cache)/glm45/GLM-4.5-Air"' \
   $config_json > $config_json.tmp
mv $config_json.tmp $config_json

# hotfix
sed -i 's/n_shared_experts: int = 1408/n_shared_experts: int = 1/' glm45_provider.py

rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

unset http_proxy https_proxy
coverage run run_pretrain.py $config_json 2>&1 | tee ./glm45_single_card.log

echo "
1 12.08998299
2 12.04966068
3 12.04866123
4 12.02987957
5 12.02322197
6 11.99096870
7 11.96842384
8 11.95415974
9 11.96843243
10 11.94553185
" > ./glm45_single_card_gt_loss.txt

python $root_dir/PaddleFleet/ci/integration_test/check_loss.py \
   --log_file ./glm45_single_card.log \
   --gt_file ./glm45_single_card_gt_loss.txt
