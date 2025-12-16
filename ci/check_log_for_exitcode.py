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

"""
ci.integration_test.check_log_for_exitcode
"""

import re
import sys


def check_unit_tests(log_path: str, check_string="OK") -> bool:
    """
    check_ci_logs_for_error - Docstring
    """
    pattern = r"Running.*?test:\s+(\S+\.py)"
    with open(log_path, "r", encoding="utf-8") as log_file:
        log_lines = log_file.readlines()
        current_test_name = None
        recode = {}
        for line in log_lines:
            m = re.search(pattern, line)
            if m:
                if current_test_name is not None:
                    print(f"Test {current_test_name} failed.")
                    recode[current_test_name] = False
                current_test_name = m.group(1)
            if check_string in line and current_test_name is not None:
                print(f"Test {current_test_name} passed.")
                current_test_name = None
                recode[current_test_name] = True
            if "Test PASSED" in line:
                split_line = line.split("Test PASSED:")
                test_name = split_line[1].strip()
                if current_test_name is not None:
                    assert test_name == current_test_name, (
                        "Mismatch in test names."
                    )
                    print(f"Test {current_test_name} passed.")
                    current_test_name = None
                    recode[test_name] = True
                else:
                    continue
        for test_name, status in recode.items():
            if not status:
                return False
    return True


def check_integration_tests(
    log_path: str, check_string="Training completed", need_loss_check=True
) -> bool:
    loss_check_string = "All loss checks passed"
    loss_check = False
    train_check = False
    with open(log_path, "r", encoding="utf-8") as log_file:
        log_lines = log_file.readlines()
        for line in log_lines:
            if check_string in line:
                print(f"Found '{check_string}' string in log file.'")
                print("Test passed.")
                train_check = True
            if loss_check_string in line:
                print(f"Found '{loss_check_string}' string in log file.'")
                print("Loss check passed.")
                loss_check = True
    print(need_loss_check)
    if need_loss_check:
        if train_check and loss_check:
            print("Both training and loss check passed.")
            return True
        else:
            print("Either training or loss check failed.")
            return False
    else:
        if train_check:
            print("Training check passed.")
            return True
        else:
            print("Training check failed.")
            return False


if __name__ == "__main__":
    log_path = sys.argv[1]
    type = sys.argv[2]
    need_loss_check_str = ""
    need_loss_check = True
    if len(sys.argv) == 4:
        need_loss_check_str = sys.argv[3]
        if need_loss_check_str.lower() == "false":
            need_loss_check = False
        elif need_loss_check_str.lower() == "true":
            need_loss_check = True
        else:
            raise ValueError("need_loss_check must be 'true' or 'false'")

    if type == "unit":
        result = check_unit_tests(log_path)
    elif type == "integration":
        result = check_integration_tests(
            log_path, need_loss_check=need_loss_check
        )
    else:
        raise ValueError(f"Unknown type: {type}")
    if result:
        sys.exit(0)
    else:
        sys.exit(1)
