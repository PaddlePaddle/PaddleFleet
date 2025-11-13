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

failed_tests=()

for test_file in tests/multi_card_tests/*.py; do
    echo "Running multi-card test: $test_file"
    uv run -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" "$test_file"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Test FAILED: $test_file"
        failed_tests+=("$test_file")
    else
        echo "Test PASSED: $test_file"
    fi
done

echo "======================================"
if [ ${#failed_tests[@]} -eq 0 ]; then
    echo "All multi-card tests passed!"
    echo "======================================"
else
    echo -e "::error:: Some multi-card tests failed:"
    for fail in "${failed_tests[@]}"; do
        echo -e "::error:: \033[31m- $fail\033[0m"
    done
    echo "======================================"
    exit 1
fi
