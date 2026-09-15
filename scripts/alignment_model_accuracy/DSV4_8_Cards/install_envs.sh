#!/usr/bin/env bash

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

# Merged from install_fleet_env.sh + install_megatron_env.sh, following the
# structure of PaddleFleet/scripts/alignment_model_accuracy/setup_venvs.sh.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ALIGN_ROOT="${SCRIPT_DIR}"

readonly PYTHON_VERSION="3.12"
readonly TORCH_VERSION="2.12.0"
readonly TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
readonly TE_VERSION="2.17.1"
readonly TRANSFORMERS_VERSION="4.57.6"
readonly FAST_HADAMARD_REV="f134af63deb2df17e1171a9ec1ea4a7d8604d5ca"
readonly DEEP_EP_REV="567632dd59810d77b3cc05553df953cc0f779799"
readonly DEFAULT_PROXY_URL="http://agent.baidu.com:8891"
readonly NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com,baidu-int.com,.baidu.com"
readonly MEGATRON_CORE_WHEEL="https://paddle-github-action.bj.bcebos.com/whl/megatron_core-0.19.0+21cfc08-cp312-cp312-linux_x86_64.whl"

PADDLE_ENV_DIR="${PADDLE_ENV_DIR:-${ALIGN_ROOT}/envs/paddle}"
TORCH_ENV_DIR="${TORCH_ENV_DIR:-${ALIGN_ROOT}/envs/torch}"
MEGATRON_REPO="${MEGATRON_REPO:-${ALIGN_ROOT}/Megatron-LM}"
PADDLE_INDEX_URL="${PADDLE_INDEX_URL:-https://www.paddlepaddle.org.cn/packages/stable/cu130/}"
FLEET_SOURCE_DIR="${FLEET_SOURCE_DIR:-${ALIGN_ROOT}/PaddleFleet}"
# FORMERS_SOURCE_DIR="${FORMERS_SOURCE_DIR:-${ALIGN_ROOT}/PaddleFormers}"


require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[install_envs] ERROR: required command not found: $1" >&2
        exit 2
    fi
}

setup_proxy() {
    local proxy_url="${PROXY_URL:-${DEFAULT_PROXY_URL}}"

    export http_proxy="${proxy_url}"
    export https_proxy="${proxy_url}"
    export no_proxy="${NO_PROXY_LIST}"
    export HTTP_PROXY="${http_proxy}"
    export HTTPS_PROXY="${https_proxy}"
    export NO_PROXY="${no_proxy}"
}

ensure_venv() {
    local venv_dir="$1"

    if [[ -d "${venv_dir}" ]]; then
        echo "[install_envs] reusing ${venv_dir}"
        return
    fi

    uv venv --relocatable --seed -p "${PYTHON_VERSION}" "${venv_dir}"
}

setup_torch_venv() {
    local torch_py="$1"
    export http_proxy="http://agent.baidu.com:8891"
    export https_proxy="http://agent.baidu.com:8891"

    echo "[install_envs] torch python : ${torch_py}"

    # Torch
    uv pip install --python "${torch_py}" \
        --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"

    uv pip install --python "${torch_py}" \
        "setuptools>=66.1.0" pip wheel packaging "ninja==1.11.1.1" \
        "pybind11[global]>=2.13,<3"

    # Transformer Engine
    uv pip install --python "${torch_py}" \
        "transformer-engine[core_cu13]==${TE_VERSION}"
    uv cache clean transformer-engine-torch
    NVTE_FRAMEWORK=pytorch NVTE_PYTORCH_FORCE_BUILD=TRUE \
        uv pip install --python "${torch_py}" --no-build-isolation \
        --no-binary transformer-engine-torch \
        --reinstall-package transformer-engine-torch \
        "transformer_engine_torch==${TE_VERSION}"

    # Build fast-hadamard-transform
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.3}" \
        uv pip install --python "${torch_py}" --no-cache --force-reinstall \
        --no-build-isolation --no-deps \
        "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@${FAST_HADAMARD_REV}"

    # Megatron runtime deps. Declared in Megatron-LM's [training]/[dev] extras,
    # but listed explicitly here: installing those extras would drag in
    # transformer-engine, flashinfer and mamba-ssm and clobber the builds above.
    uv pip install --python "${torch_py}" --index-strategy unsafe-best-match \
        --index-url https://pypi.org/simple/ \
        einops sentencepiece tiktoken wandb datasets omegaconf flask-restful \
        tensorboard tensorstore nvidia-ml-py nvidia-modelopt \
        "transformers==${TRANSFORMERS_VERSION}"

    # Build DeepEP from the immutable upstream revision.
    local site_packages
    site_packages="$("${torch_py}" -c 'import site; print(site.getsitepackages()[0])')"
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.3}" \
        DISABLE_AGGRESSIVE_PTX_INSTRS=1 \
        CPATH="${site_packages}/nvidia/cu13/include/cccl${CPATH:+:${CPATH}}" \
        uv pip install --python "${torch_py}" --no-build-isolation --no-deps \
        "git+https://github.com/deepseek-ai/DeepEP.git@${DEEP_EP_REV}"
    local -a deep_ep_binaries=()
    mapfile -t deep_ep_binaries < <(find "${site_packages}" -maxdepth 1 -type f \
        -name 'deep_ep_cpp*.so' -print)
    if [[ "${#deep_ep_binaries[@]}" -ne 1 ]]; then
        echo "[install_envs] ERROR: expected one DeepEP extension, found ${#deep_ep_binaries[@]}" >&2
        exit 1
    fi
    patchelf --set-rpath '$ORIGIN/nvidia/nvshmem/lib' "${deep_ep_binaries[0]}"

    # Megatron-LM
    # uv pip install --python "${torch_py}" --no-build-isolation --no-deps "${MEGATRON_REPO}"
    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python "${torch_py}" --index-strategy unsafe-best-match \
        --force-reinstall --no-deps "${MEGATRON_CORE_WHEEL}"
}

setup_paddle_venv() {
    local paddle_py="$1"
    echo "[install_envs] paddle python: ${paddle_py}"

    export http_proxy="http://agent.baidu.com:8891"
    export https_proxy="http://agent.baidu.com:8891"

    local -a paddle_index=(
        --no-config
        --index-url "${PADDLE_INDEX_URL}"
        --extra-index-url https://pypi.org/simple/
        --index-strategy unsafe-best-match
    )

    # Build-time deps
    uv pip install --python "${paddle_py}" "${paddle_index[@]}" \
        "setuptools>=66.1.0" pip wheel packaging "ninja==1.11.1.1" \
        "pybind11[global]>=2.13,<3" "paddle-nvidia-nvshmem-cu13>=3.3.9,<3.5" \
        "tensor-spec-worker"

    # PaddleFleet
    git -C "${FLEET_SOURCE_DIR}" submodule update --init --recursive
    uv pip install --python "${paddle_py}" "${paddle_index[@]}" \
        --inexact -v --no-build-isolation -e "${FLEET_SOURCE_DIR}" \
        --index "paddlepaddle-gpu=${PADDLE_INDEX_URL}"

    # paddlefleet-ops
    uv pip install --python "${paddle_py}" -v --no-build-isolation \
        -e "${FLEET_SOURCE_DIR}/packages/paddlefleet_ops"

    # # PaddleFormers
    # uv pip install --python "${paddle_py}" -v -e "${FORMERS_SOURCE_DIR}"
}

main() {
    setup_proxy
    require_command "uv"
    require_command "patchelf"
    require_command "git"

    export UV_NO_PROGRESS=1
    export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1200}"
    export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
    export PYTHONNOUSERSITE=1

    cd "${ALIGN_ROOT}"
    uv python install "${PYTHON_VERSION}"

    ensure_venv "${TORCH_ENV_DIR}"
    ensure_venv "${PADDLE_ENV_DIR}"

    setup_torch_venv "${TORCH_ENV_DIR}/bin/python"
    setup_paddle_venv "${PADDLE_ENV_DIR}/bin/python"

    echo "Paddle environment installed: ${PADDLE_ENV_DIR}"
    echo "Torch environment installed: ${TORCH_ENV_DIR}"
}

main "$@"
