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

cd $root_dir/PaddleFleet
git pull --no-edit origin pull/181/head
cd -

cd $root_dir/PaddleFormers
git log -5 --oneline
git reset --hard HEAD~
git pull --no-edit origin pull/3188/head
cd -

source PaddleFleet/.venv/bin/activate

cd $root_dir/glm45_fleet
export cur_dir=$(pwd)

config_dpo_yaml=$cur_dir/dpo.yaml

config_json=$CACHE_DIR/glm45/GLM-4.5-Air/config.json

yq '.recompute_granularity = ""
    | .train_dataset_path = strenv(cur_dir) + "/data/dpo/dpo_train.jsonl"
    | .eval_dataset_path = strenv(cur_dir) + "/data/dpo/dpo_eval.jsonl"
    | .model_name_or_path = strenv(CACHE_DIR) + "/glm45/GLM-4.5-Air"
    | .logging_dir = strenv(cur_dir) + "/dpo_log"
    | .output_dir = strenv(cur_dir) + "/checkpoints/glm_dpo_ckps"' \
   $config_dpo_yaml > ${config_dpo_yaml}.tmp
mv ${config_dpo_yaml}.tmp $config_dpo_yaml

jq '.num_hidden_layers = 4' \
    $config_json > ${config_json}.tmp
mv ${config_json}.tmp $config_json


python -c "
infile = '/workspace/PaddleFormers/paddleformers/transformers/auto/modeling.py'
outfile = infile + '.new'
with open(infile) as fin, open(outfile, 'w') as fout:
    lines = list(fin)
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.strip() == 'model_class = getattr(import_class, init_class)' and
            i + 1 < len(lines) and
            lines[i + 1].strip() == 'return model_class'
        ):
            pad = line[:len(line) - len(line.lstrip())]
            fout.write(pad + 'model_class = getattr(import_class, init_class + \"Fleet\")\n')
            fout.write(lines[i + 1])
            i += 2
        elif (
            line.strip() == 'model_class = getattr(import_class, init_class + \"Fleet\")' and
            i + 1 < len(lines) and
            lines[i + 1].strip() == 'return model_class'
        ):
            pad = line[:len(line) - len(line.lstrip())]
            fout.write(pad + 'model_class = getattr(import_class, init_class)\n')
            fout.write(lines[i + 1])
            i += 2
        else:
            fout.write(line)
            i += 1
"
mv /workspace/PaddleFormers/paddleformers/transformers/auto/modeling.py.new /workspace/PaddleFormers/paddleformers/transformers/auto/modeling.py

rm -rf ./outputs
rm -rf paddleformers_dist_log
master=$(hostname -i)
port=36677

export FLAGS_use_stride_compute_kernel=False
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

unset http_proxy https_proxy

NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port coverage run $(which paddleformers-cli) train $config_dpo_yaml 2>&1 | tee ./glm45_dpo.log
