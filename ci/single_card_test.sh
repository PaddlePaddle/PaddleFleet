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

# How many test files run at the same time. Each one still gets its own pytest
# process, exactly as before -- only the loop is parallel. The job holds a
# single GPU, so the concurrent files share it.
workers="${PYTEST_WORKERS:-4}"

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
echo -e "\033[34mRunning $run_count single card test files, $workers at a time\033[0m"

if [ "$run_count" -eq 0 ]; then
    echo -e "\033[32mNo single card test to run.\033[0m"
    exit 0
fi

# One pytest process per test file, unchanged -- these suites set
# process-global paddle/fleet state at import time and cannot share a process.
# Only the loop is parallel now: `xargs -P` keeps $workers files in flight.
#
# Output of each file goes to its own log instead of straight to stdout,
# because $workers concurrent processes would interleave into something
# unreadable. Logs of failing files are dumped at the end; passing files just
# print one status line, and their log is left on disk.
status_dir=$(mktemp -d)
trap 'rm -rf "$status_dir"' EXIT

run_one_test() {
    local test_file="$1"
    local log="./$(basename "${test_file%.*}")_single_card.log"
    echo "Running single card test: $test_file"
    # --parallel-mode: several `coverage run` processes are alive at once, so
    # each needs its own data file. Workflows that consume the data already run
    # `coverage combine`.
    if [[ "${WITH_COVERAGE:-OFF}" == "ON" ]]; then
        coverage run --parallel-mode -m pytest -s "$test_file" >"$log" 2>&1
    else
        pytest -s "$test_file" >"$log" 2>&1
    fi
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "$test_file" >"$status_dir/$(basename "$test_file").failed"
        echo "Test FAILED: $test_file, see log for details..."
    fi
    return $exit_code
}
export -f run_one_test
export status_dir WITH_COVERAGE

printf '%s\n' "${test_files[@]}" |
    xargs -P "$workers" -n 1 bash -c 'run_one_test "$0"'

failed_tests=()
for marker in "$status_dir"/*.failed; do
    [ -e "$marker" ] || continue
    failed_tests+=("$(cat "$marker")")
done

for test_file in "${failed_tests[@]}"; do
    echo "--------------------------------------"
    echo -e "\033[31mLog of failed test: $test_file\033[0m"
    echo "--------------------------------------"
    cat "./$(basename "${test_file%.*}")_single_card.log"
done

echo "======================================"
echo -e "\033[34mTest files executed: $run_count\033[0m"
if [ ${#failed_tests[@]} -eq 0 ]; then
    echo -e "\033[32mAll single card tests passed!\033[0m"
    echo "======================================"
else
    echo -e "::error:: \033[31m${#failed_tests[@]} single card test files failed:\033[0m"
    for test_file in "${failed_tests[@]}"; do
        echo -e "::error:: \033[31m  $test_file\033[0m"
    done
    echo "======================================"
    exit 1
fi
