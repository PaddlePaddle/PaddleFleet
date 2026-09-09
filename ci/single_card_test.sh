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

# Validated before it reaches xargs. A non-numeric value makes xargs fail
# immediately without running anything, and 0 means "no concurrency limit" in
# GNU xargs -- 600 pytest processes at once on one GPU.
if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "::error:: \033[31mPYTEST_WORKERS must be a positive integer, got '$workers'\033[0m"
    exit 1
fi

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
status_dir=$(mktemp -d) || exit 1
trap 'rm -rf "$status_dir"' EXIT

# Per-file log and marker names are derived from the whole path, not the
# basename: 22 basenames occur in more than one directory under
# $test_dir, and two of those running at once would write the same file.
slug_for() {
    local p="${1#./}"
    p="${p#"$test_dir/"}"
    p="${p%.py}"
    printf '%s' "${p//\//_}"
}

run_one_test() {
    local test_file="$1"
    local slug
    slug=$(slug_for "$test_file")
    local log="./${slug}_single_card.log"
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
    # Written for every file, pass or fail, so the parent can tell "all green"
    # apart from "xargs never got round to this file".
    echo "$test_file" >"$status_dir/$slug.done"
    if [ $exit_code -ne 0 ]; then
        echo "$test_file" >"$status_dir/$slug.failed"
        echo "Test FAILED: $test_file, see log for details..."
    fi
    return $exit_code
}
export -f run_one_test slug_for
export status_dir WITH_COVERAGE test_dir

printf '%s\n' "${test_files[@]}" |
    xargs -P "$workers" -n 1 bash -c 'run_one_test "$0"'
xargs_code=$?

ran_count=$(find "$status_dir" -name '*.done' | wc -l | tr -d '[:space:]')

failed_tests=()
for marker in "$status_dir"/*.failed; do
    [ -e "$marker" ] || continue
    failed_tests+=("$(cat "$marker")")
done

for test_file in "${failed_tests[@]}"; do
    echo "--------------------------------------"
    echo -e "\033[31mLog of failed test: $test_file\033[0m"
    echo "--------------------------------------"
    cat "./$(slug_for "$test_file")_single_card.log"
done

echo "======================================"
echo -e "\033[34mTest files executed: $ran_count / $run_count\033[0m"

# The completion markers, not xargs' exit code, are the primary invariant: the
# code for "an invocation failed" is 123 on GNU xargs but 1 on BSD, and 1 is
# also GNU's code for "xargs itself failed" (an invalid -P, for one). A short
# count covers every one of those cases -- nothing launched, launched and gave
# up part way -- and cannot be confused with a green run.
if [ "$ran_count" -ne "$run_count" ]; then
    echo -e "::error:: \033[31mOnly $ran_count of $run_count test files ran (xargs exited $xargs_code)\033[0m"
    echo "======================================"
    exit 1
fi
if [ "$xargs_code" -ne 0 ] && [ ${#failed_tests[@]} -eq 0 ]; then
    echo -e "::error:: \033[31mxargs exited $xargs_code but no test file reported a failure\033[0m"
    echo "======================================"
    exit 1
fi

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
