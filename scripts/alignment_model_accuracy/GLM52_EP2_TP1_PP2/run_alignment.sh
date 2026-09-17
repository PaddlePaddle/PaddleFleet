#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ALIGNMENT_RUN_TAG="${ALIGNMENT_RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)-$$}"
RUN_DIR="${SCRIPT_DIR}/results/${ALIGNMENT_RUN_TAG}"
REPORT_PYTHON="${GLM52_VENV_ROOT:-${SCRIPT_DIR}/venv}/paddle/bin/python"
report_on_exit() {
    local status=$?
    trap - EXIT
    if ! "${REPORT_PYTHON}" "${SCRIPT_DIR}/report_run.py" receipts --run-dir "${RUN_DIR}"; then
        echo "GLM52 native receipt reporting failed" >&2
    fi
    exit "${status}"
}
trap report_on_exit EXIT
# GLM52 owns both environments; shared model environments are read-only.
# Explicitly supplied environments are provisioned by the caller.
if [[ -z "${GLM52_VENV_ROOT:-}" ]]; then
    bash "${SCRIPT_DIR}/setup_paddle.sh" "${SCRIPT_DIR}/venv/paddle"
    export GLM52_TORCH_VENV="${GLM52_TORCH_VENV:-${SCRIPT_DIR}/venv/torch}"
    bash "${SCRIPT_DIR}/setup_reference.sh" "${GLM52_TORCH_VENV}"
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
"${REPORT_PYTHON}" "${SCRIPT_DIR}/compare_loss.py" \
    "${RUN_DIR}/paddle/loss.json" "${RUN_DIR}/torch/loss.json" --required-steps 100
