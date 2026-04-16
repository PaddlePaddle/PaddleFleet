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

"""Insert data to database via HTTP POST"""

import json
import os
import subprocess
from datetime import datetime

# 配置
DB_INSERT_URL = os.getenv("DB_INSERT_URL", "http://127.0.0.1:8000/insert")


def insert_execution_log(data: dict) -> bool:
    """插入执行记录到数据库"""
    print(f"[INFO] 插入执行记录: {data.get('execution_id', 'N/A')}")

    try:
        # 使用 curl POST 请求
        cmd = [
            "curl",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(data),
            DB_INSERT_URL,
            "-s",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("[OK] 插入成功")
            return True
        else:
            print(f"[ERROR] 插入失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"[ERROR] 插入异常: {e}")
        return False


def insert_case_result(
    case_name: str,
    status: str,
    duration: float,
    log_files: list[str],
    execution_id: str,
) -> bool:
    """插入 case 结果到数据库"""
    data = {
        "execution_id": execution_id,
        "case_name": case_name,
        "status": status,
        "duration": f"{duration:.1f}s",
        "log_files": log_files,
        "timestamp": datetime.now().isoformat(),
    }
    return insert_execution_log(data)


def insert_precision_data(
    case_name: str, passed: bool, max_diff: float, execution_id: str
) -> bool:
    """插入精度数据到数据库"""
    data = {
        "execution_id": execution_id,
        "case_name": case_name,
        "type": "precision",
        "passed": passed,
        "max_diff": f"{max_diff:.6f}",
        "timestamp": datetime.now().isoformat(),
    }
    return insert_execution_log(data)


def insert_metrics_data(
    case_name: str, metrics: dict, execution_id: str
) -> bool:
    """插入性能指标到数据库"""
    data = {
        "execution_id": execution_id,
        "case_name": case_name,
        "type": "metrics",
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    return insert_execution_log(data)


if __name__ == "__main__":
    # 测试接口
    test_data = {
        "test": True,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"[INFO] 测试数据库接口: {DB_INSERT_URL}")
    if insert_execution_log(test_data):
        print("[OK] 数据库接口正常")
    else:
        print("[ERROR] 数据库接口异常")
