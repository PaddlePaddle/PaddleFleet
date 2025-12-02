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

jq --arg cache "$CACHE_DIR" \
   '.expert_parallel_degree = 1
    | .use_expert_parallel = false
    | .save_steps = 100
    | .input_dir = "1.0 \($cache)/glm45/data/pre-training/llama_openwebtext_100k"
    | .model_name_or_path = "\($cache)/glm45/GLM-4.5-Air"' \
   glm45.json > glm45_tmp.json
mv glm45_tmp.json glm45.json

rm -rf checkpoint/
rm -rf outputs/
master=$(hostname -i)
port=36677

unset http_proxy https_proxy
python -m paddle.distributed.launch \
   --log_dir ./log \
   --master $master:$port \
   --nnodes 1 \
   --rank 0 \
   --run_mode=collective \
   run_pretrain.py glm45.json \
   --output_dir ./checkpoint
