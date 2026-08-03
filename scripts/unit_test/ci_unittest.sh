#!/usr/bin/env bash

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

set -e
export paddle=$1
export FLAGS_enable_CE=${2-false}
export nlp_dir=/workspace/PaddleFleet
export log_path=/workspace/PaddleFleet/unittest_logs
unset http_proxy && unset https_proxy
cd $nlp_dir
if [ ! -d "unittest_logs" ];then
    mkdir unittest_logs
fi
mkdir -p $log_path
export PYTEST_EXECUTE_FLAG_FILE=${3}
echo "PYTEST_EXECUTE_FLAG_FILE is ${PYTEST_EXECUTE_FLAG_FILE}"
if [ -f "${PYTEST_EXECUTE_FLAG_FILE}" ]; then
    rm "${PYTEST_EXECUTE_FLAG_FILE}"
fi
dir_name=$(dirname "${PYTEST_EXECUTE_FLAG_FILE}")
mkdir -p "${dir_name}"
set_env() {
    export NVIDIA_TF32_OVERRIDE=0 
    export FLAGS_cudnn_deterministic=1
    export FLAGS_use_cuda_managed_memory=true
    export HF_ENDPOINT=https://hf-mirror.com

    # for CE
    # if [[ ${FLAGS_enable_CE} == "true" ]];then
    #     export CE_TEST_ENV=1
    #     export RUN_SLOW_TEST=1
    #     unset PF_HOME
    #     export PYTHONPATH=${nlp_dir}:${nlp_dir}/llm:${PYTHONPATH}
    # fi
}

print_info() {
    if [ $1 -ne 0 ]; then
        # Extract only failures + short summary from full log
        python3 - "${log_path}/unittest.log" "${log_path}/unittest_FAIL.log" <<'PYEOF'
import sys, re

src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8", errors="replace") as f:
    content = f.read()

lines = content.splitlines()
output = []

# 1. FAILURES section (full tracebacks)
in_failures = False
for line in lines:
    if re.match(r"={3,}\s+FAILURES\s+={3,}", line):
        in_failures = True
    elif in_failures and re.match(r"={3,}", line) and "FAILURES" not in line:
        output.append(line)
        in_failures = False
    if in_failures:
        output.append(line)

# 2. Short test summary (FAILED lines)
summary_lines = [l for l in lines if l.startswith("FAILED ")]
if summary_lines:
    output.append("")
    output.append("=" * 60)
    output.append("SHORT TEST SUMMARY")
    output.append("=" * 60)
    output.extend(summary_lines)

# 3. Final stats line
for line in reversed(lines):
    if re.search(r"\d+ failed", line) or re.search(r"\d+ passed", line):
        output.append("")
        output.append(line)
        break

with open(dst, "w", encoding="utf-8") as f:
    f.write("\n".join(output) + "\n")
PYEOF

        cat "${log_path}/unittest_FAIL.log"
        echo ""
        echo -e "\033[31m========================================\033[0m"
        echo -e "\033[31m  FAILED TESTS\033[0m"
        echo -e "\033[31m========================================\033[0m"
        grep "^FAILED " "${log_path}/unittest_FAIL.log" | while read -r line; do
            echo -e "\033[31m  ✗ ${line#FAILED }\033[0m"
        done || true
        echo -e "\033[31m========================================\033[0m"
        tail -n 1 "${log_path}/unittest.log"
        if [ $1 -eq 124 ]; then
            echo -e "\033[33m [failed-timeout] Test execution exceeded time limit.\033[0m"
        fi
    else
        tail -n 1 ${log_path}/unittest.log
        echo -e "\033[32m All tests passed \033[0m"
    fi
}

export FLAGS_enable_CI=true
set_env
if [[ ${FLAGS_enable_CI} == "true" ]] || [[ ${FLAGS_enable_CE} == "true" ]];then
    cd ${nlp_dir}
    echo ' Testing all unittest cases '
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
    set +e
    export PYTHONFAULTHANDLER=1
    
    DOWNLOAD_SOURCE=aistudio WAIT_UNTIL_DONE=True PADDLEFLEET_TESTING=True \
    PYTHONPATH=$(pwd)/src:$(pwd) \
    COVERAGE_SOURCE=paddlefleet \
    timeout 60m \
    python -m pytest -v -n 1 test/formers \
        --dist no \
        --maxfail=10 \
        --timeout 200 --durations 20 \
        --alluredir=result \
        --cov=paddlefleet \
        --cov-report=xml:coverage.xml > ${log_path}/unittest.log 2>&1
    exit_code=$?
    print_info $exit_code unittest
    echo -e "\033[35m ---- Set PYTEST_EXECUTE_FLAG_FILE  \033[0m"
    touch ${PYTEST_EXECUTE_FLAG_FILE}
else
    echo -e "\033[32m Changed Not CI case, Skips \033[0m"
    exit_code=0
fi
exit $exit_code