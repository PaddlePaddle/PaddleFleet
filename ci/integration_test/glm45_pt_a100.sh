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
git pull --no-edit origin pull/3200/head
cd -

source PaddleFleet/.venv/bin/activate

wget -q --tries=5 --no-proxy https://xly-devops.cdn.bcebos.com/PaddleFleet/glm45/glm45_fleet.12-18.tar --no-check-certificate
tar -xf glm45_fleet.12-18.tar # glm45_fleet
cd $root_dir/glm45_fleet
export cur_dir=$(pwd)

config_yaml=$cur_dir/glm45_pt.yaml
# config_json=${cur_dir}/GLM-4.5-Air/config.json

yq eval 'del(.sharding_parallel_config)
    | .split_param = true
    |.moe_router_force_load_balancing = true
    | .expert_model_parallel_size = 1
    | .gated_linear_unit = true
    | .num_hidden_layers = 2
    | .apply_rope_fusion = true
    | .moe_router_fusion = true
    | .router_aux_loss_coef = 0.001
    | .gradient_accumulation_steps = 1
    | .per_device_train_batch_size = 1
    | .use_expert_parallel = false
    | .train_dataset_path = strenv(cur_dir) + "/data/pre-training/train.jsonl"
    | .eval_dataset_path = strenv(cur_dir) + "/data/pre-training/eval.jsonl"
    | .model_name_or_path = strenv(cur_dir) + "/GLM-4.5-Air"
    | .logging_dir = strenv(cur_dir) + "/vdl_log"
    | .output_dir = strenv(cur_dir) + "/checkpoints"' \
  $config_yaml > ${config_yaml}.tmp
mv ${config_yaml}.tmp $config_yaml

sed -i 's/config.num_hidden_layers = 10/config.num_hidden_layers = 2/g' /workspace/PaddleFormers/paddleformers/transformers/glm4_moe/modeling.py
sed -i 's/\[0\] \* 1 + \[1\] \* 9/\[0\] \* 1 + \[1\] \* 1/g' /workspace/PaddleFormers/paddleformers/transformers/glm4_moe/modeling.py

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

set +e
FLAGS_use_stride_compute_kernel=False NNODES=1 MASTER_ADDR=$master MASTER_PORT=$port CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 coverage run $(which paddleformers-cli) train $config_yaml 2>&1 | tee ./glm45_pt_a100.log


exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Test failed with exit code $exit_code, check the log: ./glm45_a100.log"
    python $root_dir/PaddleFleet/ci/check_log_for_exitcode.py ./glm45_a100.log
    check_log_exit_code=$?
    if [ $check_log_exit_code -ne 0 ]; then
        echo "Failed to find 'Training completed' in log file."
        exit 1
    else
        echo "Log check passed"
    fi
else
    echo "Test passed."
fi

set -e

echo "
10 12.65496635
" > ./glm45_pt_multi_card_gt_loss.txt

python $root_dir/PaddleFleet/ci/integration_test/check_loss.py \
   --compare_step 10 \
   --log_file ./glm45_pt_a100.log \
   --gt_file ./glm45_pt_multi_card_gt_loss.txt
