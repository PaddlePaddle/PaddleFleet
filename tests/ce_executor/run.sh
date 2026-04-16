#!/bin/bash

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

# CE 执行器启动脚本

cd "$(dirname "$0")"

# 默认参数
UPLOAD=""
CASE=""
DRY_RUN=""
ALLURE=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --upload|-u)
            UPLOAD="--upload"
            shift
            ;;
        --case|-k)
            CASE="--case $2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --allure)
            ALLURE="--allure"
            shift
            ;;
        *)
            echo "用法: $0 [--upload] [--case <case名>] [--dry-run] [--allure]"
            exit 1
            ;;
    esac
done

echo "启动 CE 执行器..."
python run_ce.py $UPLOAD $CASE $DRY_RUN $ALLURE
