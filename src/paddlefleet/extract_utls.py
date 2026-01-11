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

import re

PATTERN = re.compile(r"Running .*? test: .*?/([^/]+\.py)\b")


def extract_test_name(log_line: str) -> str | None:
    """
    从一行日志中提取测试文件名
    如果不是 Running test 行，返回 None
    """
    m = PATTERN.search(log_line)
    if not m:
        return None
    return m.group(1)
