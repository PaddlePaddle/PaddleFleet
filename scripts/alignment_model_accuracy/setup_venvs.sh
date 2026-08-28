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

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"

readonly PYTHON_VERSION="3.12"
readonly TORCH_VERSION="2.12.0+cu130"
readonly TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
readonly MCORE_BRIDGE_VERSION="1.4.3"
readonly TE_VERSION="2.17.1"
readonly PADDLE_INDEX_URL="https://www.paddlepaddle.org.cn/packages/stable/cu130/"
readonly NIGHTLY_WHL_BASE="https://paddle-whl.bj.bcebos.com/nightly/cu130"
# readonly PADDLE_VERSION="xx"
# PaddleFleet will install default paddle"
readonly PADDLEFLEET_WHEEL="${PADDLEFLEET_WHEEL_PATH:-${NIGHTLY_WHL_BASE}/paddlefleet/paddlefleet-0.4.0.dev20260807+d01517879a3-py3-none-any.whl}"
readonly PADDLEFLEET_OPS_WHEEL="${PADDLEFLEET_OPS_WHEEL_PATH:-${NIGHTLY_WHL_BASE}/paddlefleet-ops/paddlefleet_ops-0.4.0.dev20260807+d0151787-cp312-cp312-linux_x86_64.whl}"
readonly PADDLEFORMERS_WHEEL="${NIGHTLY_WHL_BASE}/paddleformers/paddleformers-0.0.0.dev-py3-none-any.whl"
readonly MEGATRON_CORE_WHEEL="${MEGATRON_CORE_WHEEL_PATH:-${NIGHTLY_WHL_BASE}/megatron_core-0.19.0+f2706b6f3-cp312-cp312-linux_x86_64.whl}"
readonly MS_SWIFT_WHEEL="${MS_SWIFT_WHEEL_PATH:-${NIGHTLY_WHL_BASE}/ms_swift-4.5.0.dev0-py3-none-any.whl}"
readonly NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com,baidu-int.com,.baidu.com,.bcebos.com"
# readonly PROXY_URL="set your proxy"
readonly UV_BIN_DIR="/home/.local/bin"
readonly UV_CACHE_DIR_PATH="/home/.cache/uv"

usage() {
    cat <<'EOF'
Usage: setup_venvs.sh

Create or reuse:
  - venv/torch   (torch + Megatron-LM + ms-swift)
  - venv/paddle  (paddlepaddle-gpu + PaddleFleet + PaddleFormers)

Both venvs are created next to this script, and the four sibling repositories
are installed editable from the same directory.
EOF
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[setup_venvs] missing required command: $1" >&2
        exit 1
    fi
}

setup_proxy() {
    if [[ -z "${PROXY_URL:-}" ]]; then
        echo "[setup_venvs] warning: PROXY_URL is not set, continuing without a proxy." >&2
        echo "  export PROXY_URL=http://<proxy-host>:<proxy-port> to use one." >&2
        return
    fi

    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    export no_proxy="${NO_PROXY_LIST}"
    export HTTP_PROXY="${http_proxy}"
    export HTTPS_PROXY="${https_proxy}"
    export NO_PROXY="${no_proxy}"
}

setup_cache() {
    export UV_CACHE_DIR="${UV_CACHE_DIR_PATH}"
    mkdir -p "${UV_CACHE_DIR}"
}

ensure_venv() {
    local venv_dir="$1"

    if [[ -d "${venv_dir}" ]]; then
        echo "[setup_venvs] reusing ${venv_dir}"
        return
    fi

    uv venv --relocatable --seed -p "${PYTHON_VERSION}" "${venv_dir}"
}

setup_torch_venv() {
    local torch_py="$1"

    echo "[setup_venvs] torch python : ${torch_py}"
    uv pip install --python "${torch_py}" --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}"

    uv pip install --python "${torch_py}" \
        "setuptools>=66.1.0" pip wheel packaging cmake "ninja==1.11.1.1" \
        "pybind11[global]>=2.13,<3" Pillow

    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python "${torch_py}" --index-strategy unsafe-best-match \
        --force-reinstall --no-deps \
        "${MEGATRON_CORE_WHEEL}" "${MS_SWIFT_WHEEL}"

    uv pip install --python "${torch_py}" --index-strategy unsafe-best-match \
        omegaconf tensor-spec-worker datasets transformers_stream_generator tensorboard json_repair matplotlib \
        "mcore-bridge==${MCORE_BRIDGE_VERSION}" \
        "transformer-engine[core_cu13]==${TE_VERSION}"

    # transformer_engine_torch
    uv cache clean transformer-engine-torch
    NVTE_FRAMEWORK=pytorch NVTE_PYTORCH_FORCE_BUILD=TRUE \
        uv pip install --python "${torch_py}" \
        --index-strategy unsafe-best-match --no-build-isolation \
        --no-binary transformer-engine-torch \
        --reinstall-package transformer-engine-torch \
        "transformer_engine_torch==${TE_VERSION}"
}

setup_paddle_venv() {
    local paddle_py="$1"

    echo "[setup_venvs] paddle python: ${paddle_py}"
    local -a paddle_index=(
        --index-url "${PADDLE_INDEX_URL}"
        --extra-index-url https://pypi.org/simple/
        --index-strategy unsafe-best-match
    )

    # uv pip install --python "${paddle_py}" "${paddle_index[@]}" \
    #     "${PADDLE_VERSION}" --force-reinstall

    # Build-time deps must live in the venv so `uv sync --no-build-isolation`
    # can compile paddlefleet-ops against the paddle installed above.
    uv pip install --python "${paddle_py}" \
        "setuptools>=66.1.0" pip wheel packaging "ninja==1.11.1.1" \
        "pybind11[global]>=2.13,<3" "paddle-nvidia-nvshmem-cu13>=3.3.9,<3.5" \
        "tensor-spec-worker"

    # PaddleFleet. --no-deps is intentionally dropped: the wheel's pinned
    # paddlepaddle-gpu dependency must be installed here, otherwise
    # venv/paddle/bin/paddleformers-cli fails to import paddle at runtime.
    uv pip install --python "${paddle_py}" "${paddle_index[@]}" \
        --force-reinstall \
        "${PADDLEFLEET_WHEEL}"
    # (
    #     cd ./PaddleFleet
    #     git submodule update --init --recursive
    #     VIRTUAL_ENV="${WORKSPACE_DIR}/venv/paddle" \
    #         uv sync -p "${PYTHON_VERSION}" --inexact --active --no-build-isolation -v
    # )

    # paddlefleet_ops
    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python "${paddle_py}" --force-reinstall \
        "${PADDLEFLEET_OPS_WHEEL}"
    # uv pip install --python "${paddle_py}" -v --no-build-isolation \
    #     -e ./PaddleFleet/packages/paddlefleet_ops

    # PaddleFormers
    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python "${paddle_py}" --force-reinstall \
        "${PADDLEFORMERS_WHEEL}"
    # uv pip install --python "${paddle_py}" -v -e ./PaddleFormers
}

main() {
    if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
        usage
        exit 0
    fi

    setup_proxy
    setup_cache
    export UV_NO_PROGRESS=1
    export PATH="${UV_BIN_DIR}:${PATH}"
    require_command "uv"

    cd "${WORKSPACE_DIR}"
    uv python install "${PYTHON_VERSION}"
    uv tool install tensor-spec

    ensure_venv "venv/torch"
    ensure_venv "venv/paddle"

    setup_torch_venv "${WORKSPACE_DIR}/venv/torch/bin/python"
    setup_paddle_venv "${WORKSPACE_DIR}/venv/paddle/bin/python"
}

main "$@"
