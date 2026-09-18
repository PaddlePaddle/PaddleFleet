#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# `uv` may only be reachable via the shared installer's UV_BIN_DIR
# (/home/.local/bin), which setup_venvs.sh exports inside its own process; GLM52
# can run before it. Make `uv` resolvable here without relying on that export.
if ! command -v uv >/dev/null 2>&1; then
    export PATH="${UV_BIN_DIR:-/home/.local/bin}:${PATH}"
fi
TORCH_VENV="${1:-${SCRIPT_DIR}/venv/torch}"
if [[ "$(realpath -m "${TORCH_VENV}")" == "$(realpath -m "${SCRIPT_DIR}/../venv/torch")" ]]; then
    echo "GLM52 reference setup must not modify the shared Torch environment" >&2
    exit 1
fi
if [[ ! -f "${TORCH_VENV}/pyvenv.cfg" ]]; then
    uv venv --relocatable --python 3.12 "${TORCH_VENV}"
fi
TORCH_PYTHON="${TORCH_VENV}/bin/python"
uv pip install --no-config --python "${TORCH_PYTHON}" \
    --index-url https://download.pytorch.org/whl/cu129 \
    "torch==2.12.1+cu129"
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --no-config --python "${TORCH_PYTHON}" \
    --no-deps --require-hashes --reinstall -r "${SCRIPT_DIR}/reference_wheels.txt"
uv pip install --no-config --python "${TORCH_PYTHON}" \
    "torch==2.12.1+cu129" "ms-swift[megatron]==4.5.0.dev0" "mcore-bridge==1.6.1" \
    "transformers==5.12.1" "pynvml==13.0.1" \
    "setuptools>=66.1.0" pip wheel packaging "ninja==1.11.1.1" \
    "pybind11[global]>=2.13,<3" tensorboard "transformer-engine[core_cu12]==2.17.1"
(
    # Compiler provisioning and build variables stay within this subprocess.
    GLM52_CUDA_ROOT="${TORCH_VENV}/cuda-12.9.1"
    source "${SCRIPT_DIR}/_prepare_cuda.sh"
    torch_cuda_headers=("${TORCH_VENV}/lib/"python*/site-packages/nvidia/*/include)
    cuda_include_path="${CUDA_HOME}/include/cccl"
    for include_dir in "${torch_cuda_headers[@]}"; do
        test -d "${include_dir}"
        cuda_include_path+=":${include_dir}"
    done
    if ! command -v cmake >/dev/null; then
        uv pip install --no-config --python "${TORCH_PYTHON}" "cmake==4.4.3"
        export PATH="${TORCH_VENV}/bin:${PATH}"
    fi
    # Bridge 1.6.1 imports TE even though GLM52 accuracy mode disables TE kernels.
    NVTE_FRAMEWORK=pytorch NVTE_PYTORCH_FORCE_BUILD=TRUE \
        CPATH="${cuda_include_path}${CPATH:+:${CPATH}}" MAX_JOBS="${MAX_JOBS:-2}" \
        uv pip install --no-config --python "${TORCH_PYTHON}" --no-build-isolation --no-cache \
        --no-binary transformer-engine-torch "transformer-engine-torch==2.17.1" \
        "torch==2.12.1+cu129"
    # Keep the full DeepEP/NVSHMEM interface; CUDA12 lacks only the optional
    # NVLink-utilization scheduling hint used by CUDA13 internode launches.
    deep_ep_source="$(mktemp -d "${TORCH_VENV}/deep-ep-build.XXXXXX")"
    git -C "${deep_ep_source}" init --quiet
    git -C "${deep_ep_source}" fetch --quiet --depth 1 \
        https://github.com/deepseek-ai/DeepEP.git \
        17cfb817bccec3a9c247013360cc550c2bac441e
    git -C "${deep_ep_source}" checkout --quiet --detach FETCH_HEAD
    git -C "${deep_ep_source}" apply --unidiff-zero "${SCRIPT_DIR}/deep_ep_cuda12.patch"
    FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
        TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS="${MAX_JOBS:-2}" \
        CPATH="${cuda_include_path}${CPATH:+:${CPATH}}" \
        uv pip install --no-config --python "${TORCH_PYTHON}" \
        --no-build-isolation --no-deps --no-cache \
        "${deep_ep_source}" \
        "fast_hadamard_transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@f134af63deb2df17e1171a9ec1ea4a7d8604d5ca"
)
# GLM52 bit-exact alignment requires BOTH frameworks to load the SAME cuBLAS
# build as the aligned reference (12.9.1.4); see setup_paddle.sh. torch==2.12.1+cu129
# already resolves nvidia-cublas-cu12 to 12.9.1.4, but pin it explicitly so a later
# wheel repack cannot silently drift the reference off the aligned math runtime.
uv pip install --no-config --python "${TORCH_PYTHON}" --no-deps \
    --index-url https://pypi.org/simple/ "nvidia-cublas-cu12==12.9.1.4"
uv pip check --python "${TORCH_PYTHON}"
"${TORCH_PYTHON}" -c 'import importlib.metadata as m; print({p: m.version(p) for p in ("torch", "mcore-bridge", "megatron-core", "ms-swift", "transformers")})'
