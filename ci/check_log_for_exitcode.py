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

import os
import sys
import re

def check_unit_tests(log_path: str, check_string="OK") -> bool:
    """
    check_ci_logs_for_error - Docstring
    """
    pattern = r"Running.*?test:\s+.*?/([^/\s]+\.py)"

    with open(log_path, "r", encoding="utf-8") as log_file:
        log_lines = log_file.readlines()
        current_test_name = None
        recode = {}
        for line in log_lines:
            m = re.search(pattern, line)
            if m:
                if current_test_name is not None:
                    print("Test {} failed.".format(current_test_name))
                    recode[current_test_name] = False
                current_test_name = m.group(1)
            if check_string in line and current_test_name is not None:
                print("Test {} passed.".format(current_test_name))
                current_test_name = None
                recode[current_test_name] = True
            if "Test PASSED" in line:
                split_line = line.split(":")
                test_name = split_line[1].strip()
                assert test_name == current_test_name, "Mismatch in test names."
                print("Test {} passed.".format(current_test_name))
                current_test_name = None
                recode[test_name] = True
        for test_name, status in recode.items():
            if not status:
                return False
    return True

def check_integration_tests(log_path: str, check_string="Training completed") -> bool:
    with open(log_path, "r", encoding="utf-8") as log_file:
        log_lines = log_file.readlines()
        for line in log_lines:
            if check_string in line:
                return True
    return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_log_for_exitcode.py <log_path> <type>")
        exit(1)
    log_path = sys.argv[1]
    type = sys.argv[2]
    if type == "unit":
        result = check_unit_tests(log_path)
    elif type == "integration":
        result = check_integration_tests(log_path)
    else:
        raise ValueError("Unknown type: {}".format(type))
    if result:
        exit(0)
    else:
        exit(1)