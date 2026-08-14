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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 新增用例列表: "用例名 paddle脚本 torch脚本"
CASES=(
    "MinimaxV2.5_EP2 ./MinimaxV2.5_EP2/run_paddle_minimax.sh ./MinimaxV2.5_EP2/run_torch_minimax.sh"
    # "transformer ./paddlepaddle_transformer/run_paddle_minimax.sh ./pytorch_transformer/run_torch_minimax.sh"
)

failed_cases=()

run_case() {
    local name="$1" paddle_script="$2" torch_script="$3"

    echo "==================== [${name}] 开始 ===================="
    rm -rf logs
    bash "${paddle_script}"
    bash "${torch_script}"

    if python3 compare_loss.py logs/paddle logs/torch; then
        echo "==================== [${name}] PASS ===================="
    else
        echo "==================== [${name}] FAIL ===================="
        failed_cases+=("${name}")
    fi
    rm -rf logs
}

echo "==================== 统一配置环境 ===================="
bash setup_venvs.sh

for case_line in "${CASES[@]}"; do
    run_case ${case_line}
done

echo
if [ "${#failed_cases[@]}" -eq 0 ]; then
    echo "全部用例通过 ✅"
    exit 0
else
    echo "失败用例: ${failed_cases[*]} ❌"
    exit 1
fi
