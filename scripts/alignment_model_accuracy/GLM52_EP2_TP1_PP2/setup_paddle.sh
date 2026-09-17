#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PADDLE_VENV="${1:-${SCRIPT_DIR}/venv/paddle}"
if [[ "$(realpath -m "${PADDLE_VENV}")" == "$(realpath -m "${SCRIPT_DIR}/../venv/paddle")" ]]; then
    echo "GLM52 setup must not modify the shared Paddle environment" >&2
    exit 1
fi
: "${PADDLEFLEET_WHEEL_PATH:?Set PADDLEFLEET_WHEEL_PATH to the current CI build}"
: "${PADDLEFLEET_OPS_WHEEL_PATH:?Set PADDLEFLEET_OPS_WHEEL_PATH to the matching CI build}"
if [[ ! -f "${PADDLE_VENV}/pyvenv.cfg" ]]; then
    uv venv --relocatable --python 3.12 "${PADDLE_VENV}"
fi
PADDLE_PYTHON="${PADDLE_VENV}/bin/python"
uv pip install --python "${PADDLE_PYTHON}" \
    "setuptools>=66.1.0" pip wheel packaging "ninja==1.11.1.1" \
    "pybind11[global]>=2.13,<3" tensor-spec-worker
uv pip install --python "${PADDLE_PYTHON}" --no-config \
    --index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/ \
    --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match \
    "paddlepaddle-gpu==3.4.0.post20260907" \
    "paddle-nvidia-nvshmem-cu12==3.4.5" "${PADDLEFLEET_WHEEL_PATH}"
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python "${PADDLE_PYTHON}" --no-config \
    --index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/ \
    --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match \
    "${PADDLEFLEET_OPS_WHEEL_PATH}"
uv pip check --python "${PADDLE_PYTHON}"

# Paddle preloads cuBLAS from its installed package, ignoring LD_LIBRARY_PATH.
# This case needs the 12.9.1.4 math runtime used by the aligned reference.
# Preserve and report Paddle's older exact-pin conflict; reject any other error.
uv pip install --no-config --python "${PADDLE_PYTHON}" --no-deps \
    --index-url https://pypi.org/simple/ "nvidia-cublas-cu12==12.9.1.4"
check_log="${PADDLE_VENV}/glm52-dependency-check.log"
check_status=0
uv pip check --python "${PADDLE_PYTHON}" >"${check_log}" 2>&1 || check_status=$?
cat "${check_log}"
if [[ "${check_status}" -ne 0 ]]; then
    check_output="$(cat "${check_log}")"
    if [[ "${check_status}" -ne 1 \
        || "${check_output}" != *"Found 1 incompatibility"* \
        || "${check_output}" != *'The package `paddlepaddle-gpu` requires `nvidia-cublas-cu12==12.9.0.13 '* \
        || "${check_output}" != *'but `12.9.1.4` is installed'* ]]; then
        exit "${check_status}"
    fi
    echo "GLM52 uses an explicit cuBLAS 12.9.1.4 override; the one declared Paddle dependency conflict is retained above."
fi
