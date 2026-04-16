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

"""Upload files to BOS using bos_tools.py"""

import os
import subprocess
import sys
from pathlib import Path

# 配置
BOS_BASE_PATH = "paddle-github-action/PaddleFleet/ce"
BOS_TOOLS_URL = "https://paddle-qa.bj.bcebos.com/CodeSync/develop/PaddlePaddle/PaddleTest/tools/bos_tools.py"


def ensure_bos_tools() -> str:
    """确保 bos_tools.py 存在，返回路径"""
    cache_dir = Path(".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    bos_tools_path = cache_dir / "bos_tools.py"

    if not bos_tools_path.exists():
        print("[INFO] 下载 bos_tools.py...")
        cmd = [
            "wget",
            "-q",
            "--no-proxy",
            "--no-check-certificate",
            "-O",
            str(bos_tools_path),
            BOS_TOOLS_URL,
        ]
        try:
            subprocess.run(cmd, check=True, timeout=60)
            print("[OK] bos_tools.py 下载成功")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 下载 bos_tools.py 失败: {e.stderr}")
            raise

    return str(bos_tools_path.absolute())


def upload_file(file_path: str, bos_path: str) -> bool:
    """上传单个文件到 BOS"""
    if not Path(file_path).exists():
        print(f"[ERROR] 文件不存在: {file_path}")
        return False

    filename = os.path.basename(file_path)
    full_bos_path = f"{BOS_BASE_PATH}/{bos_path}"

    # 确保工具存在
    bos_tools = ensure_bos_tools()

    # 使用 bos_tools.py 上传
    cmd = ["python", bos_tools, file_path, full_bos_path]

    try:
        print(f"[INFO] 上传中: {filename} -> {full_bos_path}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )

        if result.returncode == 0:
            print(f"[OK] 上传成功: {filename}")
            return True
        else:
            print(f"[ERROR] 上传失败: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[ERROR] 上传超时: {filename}")
        return False
    except Exception as e:
        print(f"[ERROR] 上传异常: {e}")
        return False


def upload_directory(dir_path: str, bos_prefix: str) -> int:
    """上传目录下所有文件"""
    count = 0
    for file in Path(dir_path).glob("*"):
        if file.is_file():
            if upload_file(str(file), f"{bos_prefix}/{file.name}"):
                count += 1
    return count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python upload_bos.py <文件或目录> <BOS路径>")
        print("示例:")
        print("  python upload_bos.py logs/20240415/case.log cases/20240415/")
        print("  python upload_bos.py coverage.xml merged_coverage.xml")
        sys.exit(1)

    local_path = sys.argv[1]
    bos_path = sys.argv[2]

    if Path(local_path).is_file():
        upload_file(local_path, bos_path)
    else:
        upload_directory(local_path, bos_path)
