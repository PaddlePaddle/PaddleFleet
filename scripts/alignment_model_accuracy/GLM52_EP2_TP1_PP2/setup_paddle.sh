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
# Let the paddlefleet wheel decide which paddlepaddle-gpu to pull, so the pin
# does not break every time develop bumps paddle. The cuBLAS override below is
# the real bit-exact invariant; paddlepaddle itself is a transitive dependency.
uv pip install --python "${PADDLE_PYTHON}" --no-config \
    --index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/ \
    --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match \
    "paddle-nvidia-nvshmem-cu12==3.4.5" "${PADDLEFLEET_WHEEL_PATH}"
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python "${PADDLE_PYTHON}" --no-config \
    --index-url https://www.paddlepaddle.org.cn/packages/nightly/cu129/ \
    --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match \
    "${PADDLEFLEET_OPS_WHEEL_PATH}"
uv pip check --python "${PADDLE_PYTHON}"

# ---------------------------------------------------------------------------
# cuBLAS override: bit-exact alignment requires both sides to run the same
# cuBLAS build. Paddle preloads the cuBLAS shipped in its own nvidia-cublas-cu12
# pin (ignoring LD_LIBRARY_PATH), so we force-install 12.9.1.4 and then assert
# that is what actually ended up in the venv.
# ---------------------------------------------------------------------------
readonly CUBLAS_REQUIRED="12.9.1.4"
uv pip install --no-config --python "${PADDLE_PYTHON}" --no-deps \
    --index-url https://pypi.org/simple/ "nvidia-cublas-cu12==${CUBLAS_REQUIRED}"
cublas_actual="$("${PADDLE_PYTHON}" -c \
    "from importlib.metadata import version; print(version('nvidia-cublas-cu12'))")"
if [[ "${cublas_actual}" != "${CUBLAS_REQUIRED}" ]]; then
    echo "FATAL: nvidia-cublas-cu12 expected ${CUBLAS_REQUIRED}, got ${cublas_actual}" >&2
    exit 1
fi
echo "GLM52 cuBLAS override verified: nvidia-cublas-cu12==${cublas_actual}"
# The override creates a single known pip-check incompatibility (paddle declares
# a different cuBLAS pin). Allow exactly that one; reject anything else.
check_log="${PADDLE_VENV}/glm52-dependency-check.log"
check_status=0
uv pip check --python "${PADDLE_PYTHON}" >"${check_log}" 2>&1 || check_status=$?
cat "${check_log}"
if [[ "${check_status}" -ne 0 ]]; then
    check_output="$(cat "${check_log}")"
    if [[ "${check_status}" -ne 1 \
        || "${check_output}" != *"Found 1 incompatibility"* \
        || "${check_output}" != *"nvidia-cublas-cu12"* \
        || "${check_output}" != *"${CUBLAS_REQUIRED}"*"is installed"* ]]; then
        echo "Unexpected dependency conflict beyond the cuBLAS override:" >&2
        exit "${check_status}"
    fi
    echo "One expected conflict: Paddle declared a different nvidia-cublas-cu12 pin; ${CUBLAS_REQUIRED} is installed."
fi
