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

import sys


def check_tests(log_path: str, check_string="Training completed") -> bool:
    with open(log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            if check_string in line:
                print(f"Found '{check_string}' string in log file.'")
                print("Test passed.")
                return True
    print(f"Did not find '{check_string}' string in log file.'")
    print("Test failed.")
    return False


def check_pod_status(log_path: str) -> bool:
    find_c_trace = False
    find_trace = False
    c_error = "FatalError: `Process abort signal` is detected by the operating system."
    c_error_status = False
    with open(log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            if "C++ Traceback (most recent call last):" in line:
                find_c_trace = True
            if find_c_trace and c_error in line:
                c_error_status = True
            if "Traceback (most recent call last):" in line:
                find_trace = True
    if c_error_status:
        print("Detected C++ FatalError in log file.")
        return True
    elif find_trace:
        print("Detected Python Traceback in log file.")
        return False
    else:
        print("No errors detected in log file.")
        return False


if __name__ == "__main__":
    log_path = sys.argv[1]
    check_str = sys.argv[2] if len(sys.argv) > 2 else "Training completed"
    result = check_tests(log_path, check_string=check_str)
    if result:
        sys.exit(0)
    else:
        if "_multi_card.log" in log_path:
            print("Since this is a multi-card log, check pod status instead.")
            result = check_pod_status(log_path)
            if result:
                sys.exit(0)
        sys.exit(1)