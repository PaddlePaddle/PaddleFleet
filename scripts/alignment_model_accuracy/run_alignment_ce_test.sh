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
    "DSV4_8_Cards ./DSV4_8_Cards/run_fleet_10step.sh ./DSV4_8_Cards/run_megatron_10step.sh"
    # "transformer ./paddlepaddle_transformer/run_paddle_minimax.sh ./pytorch_transformer/run_torch_minimax.sh"
)

failed_cases=()

run_case() {
    local name="$1" paddle_script="$2" torch_script="$3"
    # 训练脚本把日志写在各自所在目录下的 logs/{paddle,torch}
    local log_root="./DSV4_8_Cards/logs"

    echo "==================== [${name}] 开始 ===================="
    rm -rf "${log_root}"
    if ! bash "${paddle_script}" || ! bash "${torch_script}"; then
        echo "==================== [${name}] FAIL (训练异常退出) ===================="
        failed_cases+=("${name}")
        return
    fi

    # -m 2: yaml 的 gradient_accumulation_steps 与 megatron 的 GBS/(MBS*DP) 均为 2
    if python3 ./DSV4_8_Cards/compare_loss.py "${log_root}/paddle" "${log_root}/torch" -m 2; then
        echo "==================== [${name}] PASS ===================="
        rm -rf "${log_root}"
    else
        echo "==================== [${name}] FAIL ===================="
        failed_cases+=("${name}")
        echo "日志保留在 ${log_root} 供排查"
    fi
}

echo "==================== 统一配置环境 ===================="
bash ./DSV4_8_Cards/install_envs.sh

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
