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
CE 执行器 - 主脚本

功能：
1. 执行所有 CE cases（通过 Docker）
2. 收集日志
3. 上传日志到 BOS
4. 插入结果到数据库

用法：
    python run_ce.py                    # 执行所有 cases
    python run_ce.py --case glm45_single_card  # 执行指定 case
    python run_ce.py --upload           # 启用 BOS 上传
    python run_ce.py --dry-run          # 只打印不执行
"""

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

# ==================== 配置 ====================

# Docker 镜像
DOCKER_IMAGES = {
    "cu130": "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:ubuntu24-cuda130-py312-dev",
    "cu129": "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:ubuntu24-cuda129-py312-dev",
}

# 环境组合（CUDA + Python）
ENVIRONMENTS = [
    {"cuda": "13.0", "python": "3.10", "image_key": "cu130"},
    {"cuda": "12.9", "python": "3.11", "image_key": "cu129"},
    {"cuda": "12.9", "python": "3.12", "image_key": "cu129"},
]

# CE cases
CASES = [
    {
        "name": "glm45_single_card",
        "script": "PaddleFormers/tests/integration_test/glm45_pt_single_card.sh",
        "timeout": 3600,
    },
    {
        "name": "qwen3_single_card",
        "script": "PaddleFormers/tests/integration_test/qwen3_single_card.sh",
        "timeout": 3600,
    },
    {
        "name": "qwen3vl_sft_single_card",
        "script": "PaddleFormers/tests/integration_test/qwen3vl_sft_single_card.sh",
        "args": "single",
        "timeout": 300,
    },
]

# Paddle 下载地址
PADDLE_URLS = {
    "cu130": "https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Linux-Gpu-Cuda130-Cudnn913-Trt1013-Mkl-Avx-Gcc11-SelfBuiltPypiUse/latest/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl",
    "cu129": "https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Linux-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-Gcc11-SelfBuiltPypiUse/latest/",
}


# ==================== 核心函数 ====================


def run_command(
    cmd: str | list, timeout: int | None = None, dry_run: bool = False
) -> tuple[int, str, str]:
    """执行命令，返回 (exit_code, stdout, stderr)"""
    if dry_run:
        print(f"[DRY-RUN] {cmd}")
        return 0, "", ""

    try:
        if isinstance(cmd, str):
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
        else:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)


def pull_docker_image(image: str, dry_run: bool = False) -> bool:
    """拉取 Docker 镜像"""
    print(f"[INFO] 拉取镜像: {image}")
    code, _, err = run_command(
        ["docker", "pull", image], timeout=600, dry_run=dry_run
    )
    return code == 0


def run_docker_container(
    image: str, work_dir: str, container_name: str, dry_run: bool = False
) -> str | None:
    """启动 Docker 容器"""
    print(f"[INFO] 启动容器: {container_name}")

    cmd = [
        "docker",
        "run",
        "-d",
        "-t",
        "--name",
        container_name,
        "--gpus",
        "all",
        "--shm-size",
        "32g",
        "-v",
        f"{work_dir}:/workspace",
        "-w",
        "/workspace",
        image,
    ]

    code, out, err = run_command(cmd, timeout=60, dry_run=dry_run)
    if code == 0:
        return out.strip()[:12]  # 返回容器 ID
    return None


def exec_in_container(
    container: str,
    command: str,
    timeout: int | None = None,
    dry_run: bool = False,
) -> tuple[int, str, str]:
    """在容器内执行命令"""
    cmd = ["docker", "exec", container, "/bin/bash", "-c", command]
    return run_command(cmd, timeout=timeout, dry_run=dry_run)


def setup_environment(container: str, env: dict, dry_run: bool = False) -> bool:
    """在容器内设置环境（安装依赖）"""
    print(f"[INFO] 设置环境: CUDA {env['cuda']}, Python {env['python']}")

    cuda_ver = f"cu{env['cuda'].replace('.', '')}"
    py_ver = env["python"]
    image_key = env["image_key"]

    setup_script = f"""
set -e

# 安装 Miniconda
source /root/proxy || true
wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda -u
source $HOME/miniconda/etc/profile.d/conda.sh
conda init bash

# 创建 Python 环境
conda create -n py{py_ver.replace(".", "")} python={py_ver} -y
conda activate py{py_ver.replace(".", "")}

# 安装 PaddleFleet
pip install --pre paddlefleet --index-url https://www.paddlepaddle.org.cn/packages/nightly/{cuda_ver}/ \\
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/{cuda_ver}/ \\
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 克隆 PaddleFormers
cd /workspace
git clone -b develop https://github.com/PaddlePaddle/PaddleFormers.git || true
cd PaddleFormers
pip install -e . --extra-index-url=https://www.paddlepaddle.org.cn/packages/nightly/{cuda_ver}/

# 安装其他依赖
pip install coverage==7.13.0 pytest parameterized

echo "环境设置完成"
"""

    code, out, err = exec_in_container(
        container, setup_script, timeout=600, dry_run=dry_run
    )
    if code != 0:
        print(f"[ERROR] 环境设置失败: {err}")
        return False
    return True


def run_case_in_container(
    container: str, case: dict, env: dict, dry_run: bool = False
) -> tuple[int, str]:
    """在容器内运行 case"""
    print(f"[INFO] 执行 case: {case['name']}")

    cuda_ver = f"cu{env['cuda'].replace('.', '')}"
    py_ver = env["python"]

    script_path = case["script"]
    script_args = case.get("args", "")
    timeout = case.get("timeout", 3600)

    run_script = f"""
source /root/proxy || true
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate py{py_ver.replace(".", "")}

cd /workspace
bash -x {script_path} {script_args}
"""

    code, out, err = exec_in_container(
        container, run_script, timeout=timeout, dry_run=dry_run
    )
    return code, out + "\n" + err


def collect_logs(
    container: str, case_name: str, output_dir: str, dry_run: bool = False
) -> list[str]:
    """从容器收集日志"""
    print(f"[INFO] 收集日志: {case_name}")

    log_dir = Path(output_dir) / case_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 从容器复制日志文件
    cmd = f"docker cp {container}:/workspace/. {log_dir.parent}/"
    run_command(cmd, timeout=60, dry_run=dry_run)

    # 查找日志文件
    log_files = []
    for pattern in ["*.log", "*.txt", "*.out"]:
        log_files.extend(log_dir.parent.glob(pattern))

    print(f"[INFO] 找到 {len(log_files)} 个日志文件")
    return [str(f) for f in log_files]


def cleanup_container(container: str, dry_run: bool = False):
    """清理容器"""
    print(f"[INFO] 清理容器: {container}")
    run_command(["docker", "stop", container], timeout=30, dry_run=dry_run)
    run_command(["docker", "rm", "-f", container], timeout=30, dry_run=dry_run)


def generate_allure_report(work_dir: str, dry_run: bool = False):
    """生成 Allure 报告"""
    print("[INFO] 生成 Allure 报告...")

    # 检查 allure 是否安装
    check_cmd = ["allure", "--version"]
    code, out, _ = run_command(check_cmd, timeout=10, dry_run=dry_run)

    if code != 0:
        print("[WARNING] Allure 未安装，跳过报告生成")
        print("[INFO]   安装命令: pip install allure-pytest")
        return

    # 收集 allure 结果目录
    allure_results_dir = work_dir / "results" / "allure-results"
    allure_report_dir = work_dir / "results" / "allure-report"

    # 如果没有 allure 结果，先创建一些测试结果
    if not allure_results_dir.exists():
        print("[WARNING] 没有 Allure 结果目录，跳过报告生成")
        return

    # 生成报告
    cmd = [
        "allure",
        "generate",
        str(allure_results_dir),
        "-o",
        str(allure_report_dir),
    ]

    code, out, err = run_command(cmd, timeout=60, dry_run=dry_run)

    if code == 0:
        print(f"[OK] Allure 报告生成成功: {allure_report_dir}")
        print(
            f"[INFO]   查看报告: file://{allure_report_dir.absolute()}/index.html"
        )
    else:
        print(f"[ERROR] Allure 报告生成失败: {err}")


# ==================== 主流程 ====================


def main():
    parser = argparse.ArgumentParser(description="CE 执行器")
    parser.add_argument("--case", "-k", help="指定 case 名称（逗号分隔）")
    parser.add_argument("--env", "-e", help="指定环境（如 cu130-py310）")
    parser.add_argument(
        "--upload", "-u", action="store_true", help="启用 BOS 上传"
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印不执行")
    parser.add_argument("--work-dir", default="./workspace", help="工作目录")
    parser.add_argument(
        "--allure", action="store_true", help="生成 Allure 报告"
    )
    args = parser.parse_args()

    # 初始化
    execution_id = str(uuid.uuid4())[:8]
    date_str = datetime.now().strftime("%Y%m%d")
    work_dir = Path(args.work_dir).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CE 执行器")
    print(f"执行 ID: {execution_id}")
    print(f"日期: {date_str}")
    print(f"工作目录: {work_dir}")
    print("=" * 60)

    # 过滤 cases
    if args.case:
        case_names = set(args.case.split(","))
        run_cases = [c for c in CASES if c["name"] in case_names]
    else:
        run_cases = CASES

    # 过滤环境
    if args.env:
        env_names = set(args.env.split(","))
        run_envs = [
            e
            for e in ENVIRONMENTS
            if f"cu{e['cuda'].replace('.', '')}-py{e['python'].replace('.', '')}"
            in env_names
        ]
    else:
        run_envs = ENVIRONMENTS

    print(f"\n执行 Cases: {[c['name'] for c in run_cases]}")
    print(f"执行环境: {[e['cuda'] + '/' + e['python'] for e in run_envs]}")

    # 结果统计
    results = []

    # 按环境分组执行
    for env in run_envs:
        env_name = f"{env['cuda']}-py{env['python']}"
        image = DOCKER_IMAGES[env["image_key"]]
        container_name = f"ce-{env_name.replace('.', '')}-{execution_id}"

        print(f"\n{'=' * 60}")
        print(f"环境: {env_name}")
        print(f"{'=' * 60}")

        # 拉镜像
        if not pull_docker_image(image, args.dry_run):
            print(f"[ERROR] 镜像拉取失败，跳过环境 {env_name}")
            continue

        # 启动容器
        container_id = run_docker_container(
            image, str(work_dir), container_name, args.dry_run
        )
        if not container_id:
            print(f"[ERROR] 容器启动失败，跳过环境 {env_name}")
            continue

        # 设置环境
        if not setup_environment(container_name, env, args.dry_run):
            print(f"[ERROR] 环境设置失败，跳过环境 {env_name}")
            cleanup_container(container_name, args.dry_run)
            continue

        # 执行每个 case
        for case in run_cases:
            case_start = datetime.now()

            # 运行 case
            exit_code, log_content = run_case_in_container(
                container_name, case, env, args.dry_run
            )

            case_end = datetime.now()
            duration = (case_end - case_start).total_seconds()
            status = "success" if exit_code == 0 else "failed"

            # 收集日志
            log_files = collect_logs(
                container_name,
                case["name"],
                str(work_dir / "results"),
                args.dry_run,
            )

            # 保存日志
            log_file = (
                work_dir / "results" / case["name"] / f"{case['name']}.log"
            )
            log_file.parent.mkdir(parents=True, exist_ok=True)
            if not args.dry_run:
                with open(log_file, "w") as f:
                    f.write(log_content)

            # 上传到 BOS
            if args.upload and log_files:
                bos_path = f"cases/{date_str}/{case['name']}"
                subprocess.run(
                    [
                        sys.executable,
                        "upload_bos.py",
                        str(log_file.parent),
                        bos_path,
                    ],
                    cwd=Path(__file__).parent,
                )

            # 插入数据库
            result_data = {
                "execution_id": execution_id,
                "case_name": case["name"],
                "env": env_name,
                "status": status,
                "duration": f"{duration:.1f}s",
                "exit_code": exit_code,
                "log_file": str(log_file),
                "timestamp": case_start.isoformat(),
            }

            subprocess.run(
                [sys.executable, "insert_db.py"],
                input=json.dumps(result_data),
                text=True,
                cwd=Path(__file__).parent,
            )

            results.append(result_data)
            print(f"[{status.upper()}] {case['name']}: {duration:.1f}s")

        # 清理容器
        cleanup_container(container_name, args.dry_run)

    # 打印结果
    print("\n" + "=" * 60)
    print("执行完成")
    print("=" * 60)

    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")

    print(f"总数: {len(results)}")
    print(f"成功: {success}")
    print(f"失败: {failed}")

    # 保存执行报告
    report_file = work_dir / "execution_report.json"
    with open(report_file, "w") as f:
        json.dump(
            {
                "execution_id": execution_id,
                "date": date_str,
                "total": len(results),
                "success": success,
                "failed": failed,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"报告: {report_file}")

    # 生成 Allure 报告
    if args.allure:
        generate_allure_report(str(work_dir), args.dry_run)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
