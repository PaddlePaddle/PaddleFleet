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

rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

unset http_proxy https_proxy
python run_pretrain.py $config_json | tee ./glm45_single_card.log

cat "
1 12.03771591
2 12.02239990
3 11.95531750
4 11.96506405
5 11.90386009
6 11.87367058
7 11.92392540
8 11.78950691
9 11.82711029
10 11.80441380
" > ./glm45_gt_loss.txt

SCRIPT_DIR=`dirname $(readlink -f $0)`
python $SCRIPT_DIR/check_loss.py --log_file ./glm45.log --gt_file ./glm45_gt_loss.txt --tolerance 1e-2
