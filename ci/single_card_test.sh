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

disable_file="$work_dir/tests/single_card_tests/disable_single_card_uts.txt"
test_dir="tests/single_card_tests"

disabled=()
if [ -f "$disable_file" ]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        disabled+=("$line")
    done < "$disable_file"
fi

echo -e "\033[34mDisabled tests:\033[0m ${disabled[@]}"

is_disabled() {
    local test=$1
    for d in "${disabled[@]}"; do
        if [[ "$test" == "$d" ]]; then
            return 0
        fi
    done
    return 1
}

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

# pytest-xdist worker count. The job holds a single GPU, so the workers share
# it -- 4 is the value this change measures against the previous serial run.
workers="${PYTEST_WORKERS:-4}"

python -c "import xdist" 2>/dev/null || pip install pytest-xdist

test_files=()
for test_file in $(find $test_dir -type f -name "test_*.py"); do
    filename=$(basename "$test_file")
    if is_disabled "$filename"; then
        echo "Skipping disabled test: $filename"
        continue
    fi
    test_files+=("$test_file")
done

run_count=${#test_files[@]}
echo -e "\033[34mRunning $run_count single card test files on $workers workers\033[0m"

if [ "$run_count" -eq 0 ]; then
    echo -e "\033[32mNo single card test to run.\033[0m"
    exit 0
fi

# One pytest invocation over every file, instead of one invocation per file:
# the files are what parallelises. ``--dist loadfile`` keeps all tests of a file
# inside one worker, because these suites set process-global paddle/fleet state
# at import time and cannot be split mid-file.
#
# ``-s`` is dropped -- xdist workers do not stream stdout back live -- and
# ``-rA`` takes its place so the captured output of passing tests stays in the
# report, which is where the printed MD5 baselines are read from.
pytest_args=(
    -n "$workers"
    --dist loadfile
    -rA
    --junitxml=single_card.xml
    "${test_files[@]}"
)
if [[ "${WITH_COVERAGE:-OFF}" == "ON" ]]; then
    coverage run -m pytest "${pytest_args[@]}"
else
    pytest "${pytest_args[@]}"
fi
exit_code=$?

echo "======================================"
echo -e "\033[34mTest files executed: $run_count\033[0m"
if [ $exit_code -eq 0 ]; then
    echo -e "\033[32mAll single card tests passed!\033[0m"
    echo "======================================"
else
    echo -e "::error:: \033[31mSome single card tests failed, see the pytest summary above.\033[0m"
    echo "======================================"
    exit 1
fi
