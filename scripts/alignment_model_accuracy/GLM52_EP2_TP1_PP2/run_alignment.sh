#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ALIGNMENT_RUN_TAG="${ALIGNMENT_RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)-$$}"
RUN_DIR="${SCRIPT_DIR}/results/${ALIGNMENT_RUN_TAG}"
REPORT_PYTHON="${GLM52_VENV_ROOT:-${SCRIPT_DIR}/../venv}/paddle/bin/python"
report_on_exit() {
    local status=$?
    trap - EXIT
    if ! "${REPORT_PYTHON}" "${SCRIPT_DIR}/report_run.py" receipts --run-dir "${RUN_DIR}"; then
        echo "GLM52 native receipt reporting failed" >&2
    fi
    exit "${status}"
}
trap report_on_exit EXIT
# The shared runner executes legacy cases first with Transformers 4.57.1.
# GLM-5.2 needs glm_moe_dsa support; upgrade only when entering this final case.
# Explicitly supplied environments are provisioned by the caller.
if [[ -z "${GLM52_VENV_ROOT:-}" ]]; then
    # The shared latest wheels can omit GLM52 interfaces. Use existing,
    # checksum-pinned reference builds only after the legacy cases finish.
    # Their publisher adds the source SHA to filenames, not package metadata.
    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install \
        --python "${SCRIPT_DIR}/../venv/torch/bin/python" \
        --no-deps --require-hashes --reinstall \
        -r "${SCRIPT_DIR}/reference_wheels.txt"
    uv pip install --python "${SCRIPT_DIR}/../venv/torch/bin/python" \
        "transformers==5.12.1" "pynvml==13.0.1"
    # The flex dispatcher and DSA indexer need Torch DeepEP and Hadamard kernels.
    # Build against this environment's Torch; the Paddle extension cannot serve it.
    (
        # The shared image has CUDA 12.9; Torch's cu130 extension needs 13.0.
        # Limit the compiler environment to this build subprocess.
        source "${SCRIPT_DIR}/_prepare_cuda.sh"
        torch_cuda_headers=("${SCRIPT_DIR}/../venv/torch/lib/"python*/site-packages/nvidia/cu13/include)
        test "${#torch_cuda_headers[@]}" -eq 1
        test -f "${torch_cuda_headers[0]}/cusparse.h"
        FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
            TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS="${MAX_JOBS:-2}" \
            CPATH="${CUDA_HOME}/include/cccl:${torch_cuda_headers[0]}${CPATH:+:${CPATH}}" \
            uv pip install --python "${SCRIPT_DIR}/../venv/torch/bin/python" \
            --no-build-isolation --no-deps \
            "deep_ep @ git+https://github.com/deepseek-ai/DeepEP.git@17cfb817bccec3a9c247013360cc550c2bac441e" \
            "fast_hadamard_transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@f134af63deb2df17e1171a9ec1ea4a7d8604d5ca"
    )
fi
MODEL_DIR="${GLM52_MODEL_DIR:-/home/.cache/PaddleFormers/GLM-5.2-BF16-minimal}"
if ! "${REPORT_PYTHON}" "${SCRIPT_DIR}/report_run.py" inputs \
    --run-dir "${RUN_DIR}" --model-dir "${MODEL_DIR}" \
    --tokenizer-dir "${GLM52_TOKENIZER_DIR:-${MODEL_DIR}}" \
    --data-dir "${GLM52_DATA_DIR:-/home/.cache/PaddleFormers/MiniMax-V2.5-bf16_2EP}"; then
    echo "GLM52 input fingerprint reporting failed" >&2
fi
bash "${SCRIPT_DIR}/run_paddle_glm52.sh"
bash "${SCRIPT_DIR}/run_torch_glm52.sh"
python3 "${SCRIPT_DIR}/compare_loss.py" \
    "${RUN_DIR}/paddle/loss.json" "${RUN_DIR}/torch/loss.json" --required-steps 100
