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

# 本地运行脚本 - CUDA 13.0 + Python 3.12
# 使用镜像中的 Python，所有依赖只安装一次

set -e

# 设置脚本开始时间（北京时间）
export RUN_DATE=$(TZ='Asia/Shanghai' date +%Y%m%d)_b

# 配置环境变量
export CUDA_VERSION="cu130"
export PYTHON_VERSION="3.12"
export PYTHONPATH=$(pwd)/PaddleFleet:$PYTHONPATH

# Paddle URL
export PADDLE_URL="https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-LinuxCentos-Gcc11-Cuda130-Cudnn913-Trt1013-Py312-Compile/bb09abe572684456a529e6838e05e857e1927e07/paddlepaddle_gpu-3.4.0.post20260415+bb09abe5726-cp312-cp312-linux_x86_64.whl"

# 测试选项 (默认全部为 false)
RUN_SINGLE_UNIT=false
RUN_SINGLE_SONIC=false
RUN_MULTI_UNIT=false
RUN_SINGLE_MODEL=false
RUN_MULTI_MODEL=false

# 默认模型测试
SINGLE_MODEL_TESTS=()
MULTI_MODEL_TESTS=()

# 是否仅安装依赖
INSTALL_ONLY=false

# 测试结果记录
RESULT_FILE="test_results_${RUN_DATE}.txt"
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --single-unit)
            RUN_SINGLE_UNIT=true
            shift
            ;;
        --single-sonic)
            RUN_SINGLE_SONIC=true
            shift
            ;;
        --multi-unit)
            RUN_MULTI_UNIT=true
            shift
            ;;
        --single-model)
            RUN_SINGLE_MODEL=true
            shift
            if [[ -n "$1" && ! "$1" =~ ^-- ]]; then
                IFS=',' read -ra SINGLE_MODEL_TESTS <<< "$1"
                shift
            fi
            ;;
        --multi-model)
            RUN_MULTI_MODEL=true
            shift
            if [[ -n "$1" && ! "$1" =~ ^-- ]]; then
                IFS=',' read -ra MULTI_MODEL_TESTS <<< "$1"
                shift
            fi
            ;;
        --all)
            RUN_SINGLE_UNIT=true
            RUN_SINGLE_SONIC=true
            RUN_MULTI_UNIT=true
            RUN_SINGLE_MODEL=true
            RUN_MULTI_MODEL=true
            shift
            ;;
        --install-only)
            INSTALL_ONLY=true
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --single-unit      运行单卡单元测试 (ci/single_card_test.sh)"
            echo "  --single-sonic     运行 Sonic MoE 单卡测试 (ci/single_card_sonic.sh)"
            echo "  --multi-unit       运行多卡单元测试 (ci/multi-card_test.sh)"
            echo "  --single-model     运行单卡模型测试"
            echo "                     可选指定模型: --single-model glm45,qwen3,qwen3vl"
            echo "  --multi-model      运行多卡模型测试"
            echo "                     可选指定模型: --multi-model glm45_pt,qwen3_pt,qwen3vl_sft"
            echo "  --all              运行所有测试"
            echo "  --install-only     仅安装依赖，不运行测试"
            echo "  --help             显示帮助信息"
            echo ""
            echo "依赖安装:"
            echo "  所有依赖（包括 PaddleFormers）只在第一次运行时安装一次"
            echo "  直接使用镜像中的 Python，无需虚拟环境"
            echo ""
            echo "示例:"
            echo "  $0 --single-unit --single-sonic"
            echo "  $0 --single-model glm45,qwen3"
            echo "  $0 --multi-model glm45_pt,qwen3vl_sft"
            echo "  $0 --all"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 如果没有指定任何选项，显示帮助
if [ "$INSTALL_ONLY" = false ] && \
   [ "$RUN_SINGLE_UNIT" = false ] && [ "$RUN_SINGLE_SONIC" = false ] && \
   [ "$RUN_MULTI_UNIT" = false ] && [ "$RUN_SINGLE_MODEL" = false ] && \
   [ "$RUN_MULTI_MODEL" = false ]; then
    echo "错误: 请指定至少一个测试选项"
    echo ""
    echo "使用 --help 查看帮助"
    exit 1
fi

# 显示将要运行的测试
echo "========================================"
echo "测试配置:"
echo "========================================"
[ "$RUN_SINGLE_UNIT" = true ] && echo "  ✓ 单卡单元测试"
[ "$RUN_SINGLE_SONIC" = true ] && echo "  ✓ Sonic MoE 单卡测试"
[ "$RUN_MULTI_UNIT" = true ] && echo "  ✓ 多卡单元测试"
[ "$RUN_SINGLE_MODEL" = true ] && echo "  ✓ 单卡模型测试: ${SINGLE_MODEL_TESTS[*]:-全部}"
[ "$RUN_MULTI_MODEL" = true ] && echo "  ✓ 多卡模型测试: ${MULTI_MODEL_TESTS[*]:-全部}"
echo "========================================"
echo ""

# 记录测试结果
record_result() {
    local test_name=$1
    local result=$2  # "PASS" 或 "FAIL"
    local message=$3

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if [ "$result" = "PASS" ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "[$(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')] [PASS] $test_name" >> $RESULT_FILE
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "[$(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S')] [FAIL] $test_name - $message" >> $RESULT_FILE
    fi
}

# 打印测试结果汇总
print_summary() {
    echo ""
    echo "========================================"
    echo "=== 测试结果汇总 ==="
    echo "========================================"
    echo "运行日期: ${RUN_DATE}"
    echo "总测试数: $TOTAL_TESTS"
    echo -e "通过: \033[32m$PASSED_TESTS\033[0m"
    echo -e "失败: \033[31m$FAILED_TESTS\033[0m"
    echo "结果文件: $RESULT_FILE"
    echo "========================================"

    if [ -f "$RESULT_FILE" ]; then
        echo ""
        echo "详细结果:"
        cat $RESULT_FILE
    fi
}

# 下载 bos_tools.py
download_bos_tools() {
    if [ ! -f "bos_tools.py" ]; then
        echo "下载 bos_tools.py..."
        wget -q --no-proxy --no-check-certificate \
            https://paddle-qa.bj.bcebos.com/CodeSync/develop/PaddlePaddle/PaddleTest/tools/bos_tools.py \
            -O bos_tools.py
    fi
}

# 上传日志到 BOS
upload_logs_to_bos() {
    local case_name=$1
    local base_name=$2
    local log_file="${case_name}.txt"
    local gt_file="${base_name}_${case_name}_gt.txt"

    if [ -f "$log_file" ]; then
        echo "上传测试日志到 BOS: $log_file"
        local target_path="PaddleFleet/ce/${RUN_DATE}/${case_name}"
        python bos_tools.py "$log_file" "$target_path" || echo "上传日志失败"
    fi

    if [ -f "$gt_file" ]; then
        echo "上传 ground truth 文件到 BOS: $gt_file"
        local target_path="PaddleFleet/ce/${RUN_DATE}/${case_name}"
        python bos_tools.py "$gt_file" "$target_path" || echo "上传 ground truth 失败"
    fi
}

# 安装所有依赖
install_dependencies() {
    echo ""
    echo "=== 安装依赖 ==="

    # 安装基础依赖
    echo "安装基础依赖..."
    pip install colorlog>=6.10.1

    pip uninstall paddlefleet -y || true
    pip uninstall paddlepaddle-gpu -y || true
    pip uninstall paddleformers -y || true

    # 安装 PaddleFleet
    echo "安装 PaddleFleet..."
    pip install --pre paddlefleet --index-url https://www.paddlepaddle.org.cn/packages/nightly/cu130/ --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/


    # 安装 Paddle（会覆盖 PaddleFleet 自带的 Paddle）
    echo "安装指定版本的 Paddle..."
    pip install ${PADDLE_URL} --index-url=https://www.paddlepaddle.org.cn/packages/nightly/${CUDA_VERSION}/

    # 安装测试依赖
    pip install bce-python-sdk==0.8.74 wrapt matplotlib pytest parameterized
    pip install uv coverage==7.13.0

    # 打印版本信息
    echo ""
    echo "=== 版本信息 ==="
    python -c "import paddle; print('paddle:', paddle.version.commit)" 2>/dev/null || echo "无法导入 paddle"
    python -c "import paddlefleet; print('paddlefleet:', paddlefleet.version.commit)" 2>/dev/null || echo "无法导入 paddlefleet"

    echo ""
    echo "✓ 依赖安装完成"
}

# 打印依赖版本信息
print_deps_version() {
    echo ""
    echo "=== 依赖版本信息 ==="
    python -c "import paddle; print('paddle:', paddle.version.commit)" 2>/dev/null || echo "无法导入 paddle"
    python -c "import paddlefleet; print('paddlefleet:', paddlefleet.version.commit)" 2>/dev/null || echo "无法导入 paddlefleet"
    python -c "import paddleformers; print('paddleformers:', paddleformers.version.commit)" 2>/dev/null || echo "无法导入 paddleformers"
    echo ""
}

# 如果是仅安装模式，安装依赖后退出
if [ "$INSTALL_ONLY" = true ]; then
    install_dependencies
    echo ""
    echo "========================================"
    echo "=== 依赖安装完成 ==="
    echo "========================================"
    echo "现在可以运行测试，依赖已准备好，不会再重新安装"
    echo "示例: $0 --single-unit"
    echo "========================================"
    exit 0
fi

# 运行单卡测试 (单元测试 + Sonic MoE)
if [ "$RUN_SINGLE_UNIT" = true ] || [ "$RUN_SINGLE_SONIC" = true ]; then
    print_deps_version

    # 运行单卡单元测试
    export work_dir=$(pwd)
    if [ "$RUN_SINGLE_UNIT" = true ]; then
        echo ""
        echo "=== 开始单卡单元测试 ==="
        if bash ci/single_card_test.sh; then
            record_result "单卡单元测试" "PASS"
            echo -e "\033[32m✓ 单卡单元测试完成\033[0m"
        else
            record_result "单卡单元测试" "FAIL" "测试执行失败"
            echo -e "\033[31m✗ 单卡单元测试失败\033[0m"
            exit 1
        fi
    fi

    # 运行 Sonic MoE 单卡测试
    if [ "$RUN_SINGLE_SONIC" = true ]; then
        echo ""
        echo "=== 开始 Sonic MoE 单卡测试 ==="
        if bash ci/single_card_sonic.sh; then
            record_result "Sonic MoE 单卡测试" "PASS"
            echo -e "\033[32m✓ Sonic MoE 单卡测试完成\033[0m"
        else
            record_result "Sonic MoE 单卡测试" "FAIL" "测试执行失败"
            echo -e "\033[31m✗ Sonic MoE 测试失败\033[0m"
            exit 1
        fi
    fi
fi

# 运行多卡单元测试
if [ "$RUN_MULTI_UNIT" = true ]; then
    print_deps_version

    echo ""
    echo "=== 开始多卡单元测试 ==="
    export work_dir=$(pwd)
    if [ -f "ci/multi-card_test.sh" ]; then
        if bash ci/multi-card_test.sh; then
            record_result "多卡单元测试" "PASS"
            echo -e "\033[32m✓ 多卡单元测试完成\033[0m"
        else
            record_result "多卡单元测试" "FAIL" "测试执行失败"
            echo -e "\033[31m✗ 多卡单元测试失败\033[0m"
            exit 1
        fi
    else
        echo "✗ 多卡测试脚本不存在，跳过"
    fi
fi

# 运行单卡模型测试
if [ "$RUN_SINGLE_MODEL" = true ]; then
    export CACHE_DIR=/root/paddlejob/workspace/env_run/fleet-model-cache
    BASE_NAME="${CUDA_VERSION}-${PYTHON_VERSION}-single"

    # 如果没有指定具体模型，运行所有单卡模型测试
    if [ ${#SINGLE_MODEL_TESTS[@]} -eq 0 ]; then
        SINGLE_MODEL_TESTS=("glm45" "qwen3" "qwen3vl")
    fi

    echo ""
    cd PaddleFormers
    pip install -e . --extra-index-url=https://www.paddlepaddle.org.cn/packages/nightly/${CUDA_VERSION}/
    cd ..
    print_deps_version
    echo "=== 开始单卡模型测试 ==="
    find PaddleFormers/tests/integration_test -type f -exec sed -i 's/--no-proxy//g' {} +

    for model in "${SINGLE_MODEL_TESTS[@]}"; do
        echo "  运行 $model 单卡测试..."
        case $model in
            glm45)
                case_name="glm45_pt_single_card"
                sed -i "s|/home/.cache|${CACHE_DIR}|g" PaddleFormers/tests/config/ci/glm45_single_pt-test.yaml
                bash PaddleFormers/tests/integration_test/glm45_pt_single_card.sh
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "glm45 单卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "glm45 单卡模型" "PASS"
                    fi
                else
                    record_result "glm45 单卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            qwen3)
                case_name="qwen3_single_card"
                bash PaddleFormers/tests/integration_test/qwen3_single_card.sh
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "qwen3 单卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "qwen3 单卡模型" "PASS"
                    fi
                else
                    record_result "qwen3 单卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            qwen3vl)
                case_name="qwen3vl_sft_single_card"
                timeout 5m bash PaddleFormers/tests/integration_test/qwen3vl_sft_single_card.sh single
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "qwen3vl 单卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "qwen3vl 单卡模型" "PASS"
                    fi
                else
                    record_result "qwen3vl 单卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            *)
                echo "  ✗ 未知模型: $model"
                ;;
        esac
    done
    echo "✓ 单卡模型测试完成"
fi

# 运行多卡模型测试
if [ "$RUN_MULTI_MODEL" = true ]; then
    export CACHE_DIR=/root/paddlejob/workspace/env_run/fleet-model-cache
    cd PaddleFormers
    pip install -e . --extra-index-url=https://www.paddlepaddle.org.cn/packages/nightly/${CUDA_VERSION}/
    cd ..
    print_deps_version

    BASE_NAME="${CUDA_VERSION}-${PYTHON_VERSION}-multi"

    # 如果没有指定具体模型，运行所有多卡模型测试
    if [ ${#MULTI_MODEL_TESTS[@]} -eq 0 ]; then
        MULTI_MODEL_TESTS=(
            "glm45_pt" "glm45_sft" "glm45_sft_cp" "glm45_lora" "glm45_dpo" "glm45_dpo_lora"
            "glm45_pt_ep4" "glm45_pt_fp8" "glm45_pt_grouped_gemm" "qwen_pt" "qwen_sft" "qwen_lora"
            "qwen3vl_sft" "qwen3vl_lora" "qwen3vl_moe"
        )
    fi

    echo ""
    echo "=== 开始多卡模型测试 ==="
    find PaddleFormers/tests/integration_test -type f -exec sed -i 's/--no-proxy//g' {} +


    for model in "${MULTI_MODEL_TESTS[@]}"; do
        echo "  运行 $model 多卡测试..."
        case $model in
            glm45_pt)
                case_name="glm45_pt"
                tests/integration_test/glm45_pt.sh
                bash PaddleFormers/tests/integration_test/glm45_pt.sh
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "glm45_pt 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "glm45_pt 多卡模型" "PASS"
                    fi
                else
                    record_result "glm45_pt 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            glm45_sft)
                case_name="glm45_sft"
                bash PaddleFormers/tests/integration_test/glm45_sft.sh
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "glm45_sft 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "glm45_sft 多卡模型" "PASS"
                    fi
                else
                    record_result "glm45_sft 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            glm45_sft_cp)
                case_name="glm45_sft_cp"
                bash PaddleFormers/tests/integration_test/glm45_sft_cp.sh
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "glm45_sft 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "glm45_sft 多卡模型" "PASS"
                    fi
                else
                    record_result "glm45_sft 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            glm45_lora)
                case_name="glm45_lora"
                bash PaddleFormers/tests/integration_test/glm45_lora.sh
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "glm45_lora 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "glm45_lora 多卡模型" "PASS"
                    fi
                else
                    record_result "glm45_lora 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            glm45_dpo)
                case_name="glm45_dpo"
                bash PaddleFormers/tests/integration_test/glm45_dpo.sh
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "glm45_dpo 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "glm45_dpo 多卡模型" "PASS"
                    fi
                else
                    record_result "glm45_dpo 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            glm45_dpo_lora)
                case_name="glm45_dpo_lora"
                timeout 5m bash PaddleFormers/tests/integration_test/glm45_dpo_lora.sh
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "glm45_dpo_lora 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "glm45_dpo_lora 多卡模型" "PASS"
                    fi
                else
                    record_result "glm45_dpo_lora 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            glm45_pt_ep4)
                case_name="glm45_pt_ep4"
                timeout 5m bash PaddleFormers/tests/integration_test/glm45_pt_ep4.sh
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "glm45_pt_ep4 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "glm45_pt_ep4 多卡模型" "PASS"
                    fi
                else
                    record_result "glm45_pt_ep4 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            glm45_pt_fp8)
                case_name="glm45_pt_fp8"
                timeout 5m bash PaddleFormers/tests/integration_test/glm45_pt_fp8.sh
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "glm45_pt_fp8 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "glm45_pt_fp8 多卡模型" "PASS"
                    fi
                else
                    record_result "glm45_pt_fp8 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            glm45_pt_grouped_gemm)
                case_name="glm45_pt_grouped_gemm"
                timeout 5m bash PaddleFormers/tests/integration_test/glm45_pt_grouped_gemm.sh
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "glm45_pt_grouped_gemm 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "glm45_pt_grouped_gemm 多卡模型" "PASS"
                    fi
                else
                    record_result "glm45_pt_grouped_gemm 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            qwen_pt)
                case_name="qwen_pt"
                bash PaddleFormers/tests/integration_test/qwen.sh pt
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "qwen_pt 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "qwen_pt 多卡模型" "PASS"
                    fi
                else
                    record_result "qwen_pt 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            qwen_sft)
                case_name="qwen_sft"
                bash PaddleFormers/tests/integration_test/qwen.sh sft
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "qwen_sft 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "qwen_sft 多卡模型" "PASS"
                    fi
                else
                    record_result "qwen_sft 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            qwen_lora)
                case_name="qwen_lora"
                bash PaddleFormers/tests/integration_test/qwen.sh lora
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "qwen_lora 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "qwen_lora 多卡模型" "PASS"
                    fi
                else
                    record_result "qwen_lora 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            qwen3vl_sft)
                case_name="qwen3vl_sft"
                timeout 5m bash PaddleFormers/tests/integration_test/qwen3vl_sft.sh tp8 h20
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "qwen3vl_sft 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "qwen3vl_sft 多卡模型" "PASS"
                    fi
                else
                    record_result "qwen3vl_sft 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            qwen3vl_lora)
                case_name="qwen3vl_lora"
                timeout 5m bash PaddleFormers/tests/integration_test/qwen3vl_lora.sh h20
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "qwen3vl_lora 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "qwen3vl_lora 多卡模型" "PASS"
                    fi
                else
                    record_result "qwen3vl_lora 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            qwen3vl_moe)
                case_name="qwen3vl_moe"
                timeout 10m bash PaddleFormers/tests/integration_test/qwen3vl_sft.sh moe h20
                exit_code=$?
                if [ "$exit_code" != "0" ]; then
                    bash ci/check_ce_precision.sh $case_name $BASE_NAME
                    precision_exit_code=$?
                    if [ "$precision_exit_code" != "0" ]; then
                        download_bos_tools
                        upload_logs_to_bos $case_name $BASE_NAME
                        record_result "qwen3vl_moe 多卡模型" "FAIL" "测试失败且精度检查失败"
                    else
                        record_result "qwen3vl_moe 多卡模型" "PASS"
                    fi
                else
                    record_result "qwen3vl_moe 多卡模型" "PASS"
                    echo -e "\033[32m✓ $model 测试成功\033[0m"
                fi
                ;;
            *)
                echo "  ✗ 未知模型: $model"
                ;;
        esac
    done
    echo "✓ 多卡模型测试完成"
fi

# 打印测试结果汇总
print_summary

echo ""
echo "========================================"
echo "=== 所有测试完成 ==="
echo "========================================"
