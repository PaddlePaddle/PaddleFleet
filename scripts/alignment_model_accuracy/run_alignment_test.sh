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

# name paddle_script torch_script [comparison_mode] [required_steps]
CASES=(
    "MinimaxV2.5_EP2 ./MinimaxV2.5_EP2/run_paddle_minimax.sh ./MinimaxV2.5_EP2/run_torch_minimax.sh"
    "GLM45Air_EP2 ./GLM45Air_EP2/run_paddle_glm45.sh ./GLM45Air_EP2/run_torch_glm45.sh"
    "GLM52_EP2_TP1_PP2 ./GLM52_EP2_TP1_PP2/run_paddle_glm52.sh ./GLM52_EP2_TP1_PP2/run_torch_glm52.sh json 100"
)

selected_case=""
if [[ $# -gt 0 ]]; then
    if [[ $# -ne 2 || "$1" != "--case" ]]; then
        echo "usage: $0 [--case CaseName]" >&2
        exit 2
    fi
    selected_case="$2"
    matched=false
    for case_line in "${CASES[@]}"; do
        [[ "${case_line%% *}" != "${selected_case}" ]] || matched=true
    done
    if [[ "${matched}" != true ]]; then
        echo "unknown alignment case: ${selected_case}" >&2
        exit 2
    fi
fi

failed_cases=()
export ALIGNMENT_RUN_TAG="$(date -u +%Y%m%d-%H%M%S)-$$"
RESULT_DIR="${SCRIPT_DIR}/results/${ALIGNMENT_RUN_TAG}"
mkdir -p "${RESULT_DIR}"
if [[ -e logs ]]; then
    mv logs "${RESULT_DIR}/previous-logs"
fi

run_case() {
    local name="$1" paddle_script="$2" torch_script="$3"
    local mode="${4:-md5}" required_steps="${5:-}" status=0
    local compare_args=()
    echo "[${name}] begin"
    if bash "${paddle_script}"; then
        if bash "${torch_script}"; then
            if [[ "${mode}" == json ]]; then
                compare_args=(--loss-json --required-steps "${required_steps}"
                    "logs/paddle/${ALIGNMENT_RUN_TAG}/loss.json"
                    "logs/torch/${ALIGNMENT_RUN_TAG}/loss.json")
            else
                compare_args=(logs/paddle logs/torch)
            fi
            python3 compare_loss.py "${compare_args[@]}" 2>&1 | tee "${RESULT_DIR}/${name}.comparison.log" || status=$?
        else
            status=$?
        fi
    else
        status=$?
    fi
    if [[ -e logs ]]; then
        mv logs "${RESULT_DIR}/${name}"
    fi
    if [[ "${status}" -eq 0 ]]; then
        echo "[${name}] PASS"
    else
        echo "[${name}] FAIL (exit ${status})"
        failed_cases+=("${name}")
    fi
}

bash setup_venvs.sh
for case_line in "${CASES[@]}"; do
    read -r name paddle_script torch_script mode required_steps <<< "${case_line}"
    if [[ -z "${selected_case}" || "${selected_case}" == "${name}" ]]; then
        run_case "${name}" "${paddle_script}" "${torch_script}" "${mode:-md5}" "${required_steps}"
    fi
done

echo "alignment artifacts: ${RESULT_DIR}"
if [[ "${#failed_cases[@]}" -ne 0 ]]; then
    echo "failed cases: ${failed_cases[*]}" >&2
    exit 1
fi
echo "all selected alignment cases passed"
