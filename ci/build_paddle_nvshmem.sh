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

# ============================================================================
# build_all_wheels.sh
# 一键从零编译 NVSHMEM wheel 包（paddle-nvidia-nvshmem-cu{CUDA_VER}）
#
# 用法:
#   bash build_all_wheels.sh
#
# 支持:
#   - CUDA 版本: 12, 13（通过 CUDA_CONFIGS 变量控制）
#   - GPU 架构: 80, 90, 100, 103, 120（通过 CUDA_ARCHITECTURES 变量控制）
#
# 产出:
#   output/
#   ├── paddle_nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
#   └── paddle_nvidia_nvshmem_cu13-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
# ============================================================================
set -e

# ============================================================================
# 配置区（按需修改）
# ============================================================================
NVSHMEM_VERSION="v3.4.5-0"
NVSHMEM_LIB_VERSION="3.4.5"
NVSHMEM_REPO="https://github.com/NVIDIA/nvshmem"

# CUDA 版本自动检测
# 通过 nvcc 获取当前环境的 CUDA 版本，自动找到 CUDA_HOME
detect_cuda() {
    local cuda_home=""
    local cuda_full_ver=""
    local cuda_major_ver=""

    # 优先从 nvcc 检测
    if command -v nvcc &>/dev/null; then
        cuda_full_ver=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        cuda_major_ver=$(echo "$cuda_full_ver" | cut -d. -f1)
    fi

    # 查找 CUDA_HOME 路径
    if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
        cuda_home="$CUDA_HOME"
    elif [ -n "$cuda_full_ver" ]; then
        # 优先匹配精确版本，如 /usr/local/cuda-12.9
        for candidate in "/usr/local/cuda-${cuda_full_ver}" "/usr/local/cuda-${cuda_major_ver}" "/usr/local/cuda"; do
            if [ -d "$candidate" ]; then
                cuda_home="$candidate"
                break
            fi
        done
    fi

    if [ -z "$cuda_home" ] || [ -z "$cuda_major_ver" ]; then
        err "无法自动检测 CUDA 版本。请安装 CUDA Toolkit 或手动设置 CUDA_HOME 环境变量"
    fi

    CUDA_CONFIGS=("${cuda_major_ver}:${cuda_home}")
    log "自动检测到 CUDA ${cuda_full_ver} (大版本: ${cuda_major_ver}), CUDA_HOME=${cuda_home}"
}

# GPU 架构
CUDA_ARCHITECTURES="80;90;100;103;120"

# 工作目录
WORK_DIR="$(pwd)/nvshmem_build"
OUTPUT_DIR="$(pwd)/output"

# pip 源（可选，加速下载）
export PIP_INDEX_URL=${PIP_INDEX_URL:-"https://pypi.tuna.tsinghua.edu.cn/simple"}
export PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-"pypi.tuna.tsinghua.edu.cn"}

# ============================================================================
# 辅助函数
# ============================================================================
log() { echo -e "\n\033[1;32m[$(date '+%H:%M:%S')] $1\033[0m"; }
err() { echo -e "\n\033[1;31m[ERROR] $1\033[0m"; exit 1; }

# 自动查找库文件路径
find_lib_path() {
    local lib_name="$1"
    local lib_path=""

    # 方法1: 使用 ldconfig -p 查找（最可靠）
    if command -v ldconfig &>/dev/null; then
        lib_path=$(ldconfig -p 2>/dev/null | grep -E "${lib_name}\.so" | head -1 | sed 's/.*=> \(.*\)/\1/')
        if [ -f "$lib_path" ]; then
            echo "$lib_path"
            return
        fi
    fi

    # 方法2: 常见系统路径尝试
    local common_paths=(
        "/usr/lib/x86_64-linux-gnu/${lib_name}.so"     # Ubuntu/Debian
        "/usr/lib64/${lib_name}.so"                    # CentOS/RHEL
        "/usr/lib/${lib_name}.so"                      # 其他系统
        "/usr/local/lib64/${lib_name}.so"              # 本地安装
        "/usr/local/lib/${lib_name}.so"                # 本地安装
    )

    for path in "${common_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "$path"
            return
        fi
    done

    # 如果都找不到，返回错误
    err "找不到库文件: ${lib_name}.so，请安装对应的 rdma/ibverbs 相关包"
}

# 执行 CUDA 自动检测
detect_cuda

# 检测系统架构，返回 wheel 平台标签使用的架构名
detect_arch() {
    local machine
    machine=$(uname -m)
    case "$machine" in
        x86_64)  echo "x86_64" ;;
        aarch64) echo "aarch64" ;;
        *)       err "不支持的架构: $machine" ;;
    esac
}

# 重新打包 wheel 的平台标签
retag_wheel_platform() {
    local whl="$1"
    local outdir="$2"
    local arch
    arch=$(detect_arch)
    local plat="manylinux_2_17_${arch}"
    local basename=$(basename "$whl")
    local new_name="${basename/none-any.whl/none-${plat}.whl}"

    python3 -c "
import zipfile, os, tempfile, shutil
plat = '${plat}'
tmpdir = tempfile.mkdtemp()
with zipfile.ZipFile('$whl', 'r') as z:
    z.extractall(tmpdir)
for root, dirs, files in os.walk(tmpdir):
    for f in files:
        if f == 'WHEEL':
            p = os.path.join(root, f)
            txt = open(p).read()
            txt = txt.replace('Tag: py3-none-any', f'Tag: py3-none-{plat}')
            open(p, 'w').write(txt)
with zipfile.ZipFile('$outdir/$new_name', 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(tmpdir):
        for f in files:
            fp = os.path.join(root, f)
            z.write(fp, os.path.relpath(fp, tmpdir))
shutil.rmtree(tmpdir)
"
    echo "$outdir/$new_name"
}

# ============================================================================
# 第一步: 克隆 NVSHMEM 源码
# ============================================================================
clone_nvshmem() {
    log "第一步: 克隆 NVSHMEM 源码 (${NVSHMEM_VERSION})"

    if [ -d "$WORK_DIR/nvshmem/.git" ]; then
        log "源码已存在，跳过克隆"
        return
    fi

    mkdir -p "$WORK_DIR"
    git clone -b "$NVSHMEM_VERSION" "$NVSHMEM_REPO" "$WORK_DIR/nvshmem"
}

# ============================================================================
# 第二步: 编译 NVSHMEM 核心库（每个 CUDA 版本编译一次）
# ============================================================================
build_core_lib() {
    local cuda_ver="$1"
    local cuda_home="$2"
    local build_dir="$WORK_DIR/nvshmem/build_cu${cuda_ver}"
    local install_dir="$build_dir/install"

    log "第二步: 编译 NVSHMEM 核心库 (CUDA ${cuda_ver}, CUDA_HOME=${cuda_home})"

    if [ ! -d "$cuda_home" ]; then
        err "CUDA_HOME 不存在: $cuda_home"
    fi

    # 如果已编译过，跳过
    if [ -f "$install_dir/lib/libnvshmem_host.so.3" ]; then
        log "核心库已编译，跳过 (${install_dir})"
        return
    fi

    export CUDA_HOME="$cuda_home"
    export NVSHMEM_IBGDA_SUPPORT=1
    export NVSHMEM_SHMEM_SUPPORT=0
    export NVSHMEM_UCX_SUPPORT=0
    export NVSHMEM_USE_NCCL=0
    export NVSHMEM_PMIX_SUPPORT=0
    export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
    export NVSHMEM_USE_GDRCOPY=0
    export NVSHMEM_IBRC_SUPPORT=0
    export NVSHMEM_BUILD_TESTS=0
    export NVSHMEM_BUILD_EXAMPLES=0
    export NVSHMEM_MPI_SUPPORT=0
    export NVSHMEM_BUILD_HYDRA_LAUNCHER=0
    export NVSHMEM_BUILD_TXZ_PACKAGE=0
    export NVSHMEM_BUILD_PYTHON_LIB=0
    export NVSHMEM_BUILD_PACKAGES=0
    export NVSHMEM_BUILD_BITCODE_LIBRARY=0

    cd "$WORK_DIR/nvshmem"

    # 自动检测库路径（兼容不同 Linux 发行版）
    local mlx5_lib=$(find_lib_path "libmlx5")
    local ibverbs_lib=$(find_lib_path "libibverbs")
    local include_path=$(dirname "$(dirname "$mlx5_lib")")/include
    [ ! -d "$include_path" ] && include_path="/usr/include"

    log "检测到库路径: MLX5=${mlx5_lib}, IBVERBS=${ibverbs_lib}"

    cmake -G Ninja -S . -B "$build_dir" \
        -DCMAKE_INSTALL_PREFIX="$install_dir" \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
        -DCMAKE_CUDA_FLAGS="-I${include_path}" \
        -DCMAKE_CXX_FLAGS="-I${include_path}" \
        -DMLX5_lib="$mlx5_lib" \
        -DIBVERBS_lib="$ibverbs_lib"

    cmake --build "$build_dir" --target install -j

    log "核心库编译完成: $install_dir"
}

# ============================================================================
# 第三步: 打包 paddle-nvidia-nvshmem-cu{VER} wheel（每个 CUDA 版本一个）
# ============================================================================
build_core_wheel() {
    local cuda_ver="$1"
    local install_dir="$WORK_DIR/nvshmem/build_cu${cuda_ver}/install"
    local pkg_dir="$WORK_DIR/core_wheel_cu${cuda_ver}"
    local pkg_name="paddle-nvidia-nvshmem-cu${cuda_ver}"

    log "第三步: 打包 ${pkg_name} wheel"

    rm -rf "$pkg_dir"
    mkdir -p "$pkg_dir/nvidia/nvshmem/include"
    mkdir -p "$pkg_dir/nvidia/nvshmem/lib"

    # 复制头文件
    cp -r "$install_dir/include/"* "$pkg_dir/nvidia/nvshmem/include/"

    # 复制 .so.3 文件
    for f in "$install_dir/lib/"*.so*; do
        [ -f "$f" ] && cp "$f" "$pkg_dir/nvidia/nvshmem/lib/"
    done

    # 复制静态库
    for f in "$install_dir/lib/"*.a; do
        [ -f "$f" ] && cp "$f" "$pkg_dir/nvidia/nvshmem/lib/"
    done

    # 复制 bitcode（如果存在）
    for f in "$install_dir/lib/"*.bc; do
        [ -f "$f" ] && cp "$f" "$pkg_dir/nvidia/nvshmem/lib/"
    done

    # 创建 __init__.py
    cat > "$pkg_dir/nvidia/__init__.py" << 'PYEOF'
from pathlib import Path as _Path
NVSHMEM_HOME = str(_Path(__file__).parent / "nvshmem")
PYEOF

    cat > "$pkg_dir/nvidia/nvshmem/__init__.py" << 'PYEOF'
from pathlib import Path as _Path
lib_path = str(_Path(__file__).parent / "lib")
include_path = str(_Path(__file__).parent / "include")
PYEOF

    # 复制 License
    [ -f "$WORK_DIR/nvshmem/License.txt" ] && cp "$WORK_DIR/nvshmem/License.txt" "$pkg_dir/"

    # 创建 pyproject.toml
    cat > "$pkg_dir/pyproject.toml" << TOMLEOF
[build-system]
requires = ["setuptools>=62", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "${pkg_name}"
version = "${NVSHMEM_LIB_VERSION}"
description = "NVSHMEM core library - custom build with extended GPU arch support"
requires-python = ">=3.9"

[tool.setuptools.packages.find]
include = ["nvidia*"]

[tool.setuptools.package-data]
"nvidia.nvshmem.include" = ["**/*.h", "**/*.cuh", "**/*.hpp"]
"nvidia.nvshmem.lib" = ["*.so*", "*.a", "*.bc"]
TOMLEOF

    cat > "$pkg_dir/MANIFEST.in" << 'MANIFESTEOF'
recursive-include nvidia *.h *.cuh *.hpp *.so* *.a *.bc *.py
MANIFESTEOF

    # 构建 wheel
    cd "$pkg_dir"
    python3 -m build --wheel --outdir "$pkg_dir/dist/" --no-isolation

    # 重打标签为 manylinux
    local src_whl=$(ls "$pkg_dir/dist/"*-none-any.whl 2>/dev/null | head -1)
    if [ -n "$src_whl" ]; then
        retag_wheel_platform "$src_whl" "$OUTPUT_DIR"
        log "${pkg_name} wheel 已生成"
    else
        # 如果已经不是 none-any，直接复制
        cp "$pkg_dir/dist/"*.whl "$OUTPUT_DIR/"
    fi
}

# ============================================================================
# 主流程
# ============================================================================
main() {
    log "===== NVSHMEM Wheel 全流程构建开始 ====="
    log "NVSHMEM 版本: ${NVSHMEM_VERSION}"
    log "CUDA 配置: ${CUDA_CONFIGS[*]}"
    log "GPU 架构: ${CUDA_ARCHITECTURES}"
    log "输出目录: ${OUTPUT_DIR}"

    # 检查构建依赖
    for cmd in cmake ninja git; do
        command -v "$cmd" &>/dev/null || err "缺少依赖: ${cmd}"
    done
    python3 -c "import build, wheel, setuptools" 2>/dev/null || {
        log "安装 Python 构建依赖..."
        pip3 install build wheel "setuptools>=62"
    }

    mkdir -p "$OUTPUT_DIR"

    # 第一步: 克隆源码
    clone_nvshmem

    # 遍历每个 CUDA 版本
    for cuda_config in "${CUDA_CONFIGS[@]}"; do
        local cuda_ver="${cuda_config%%:*}"
        local cuda_home="${cuda_config##*:}"

        # 第二步: 编译核心库
        build_core_lib "$cuda_ver" "$cuda_home"

        # 第三步: 打包核心库 wheel
        build_core_wheel "$cuda_ver"
    done

    # 汇总
    log "===== 构建完成！====="
    echo ""
    echo "所有 wheel 包:"
    ls -lh "$OUTPUT_DIR/"*.whl
    echo ""
    echo "安装方式:"
    echo "  pip uninstall nvidia-nvshmem-cu12 -y"
    echo "  pip install output/paddle_nvidia_nvshmem_cu12-*.whl"
}

main "$@"
