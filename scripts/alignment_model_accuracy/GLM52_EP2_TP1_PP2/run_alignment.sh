#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ALIGNMENT_RUN_TAG="${ALIGNMENT_RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)-$$}"
RUN_DIR="${SCRIPT_DIR}/results/${ALIGNMENT_RUN_TAG}"
# The shared runner executes legacy cases first with Transformers 4.57.1.
# GLM-5.2 needs glm_moe_dsa support; upgrade only when entering this final case.
# Explicitly supplied environments are provisioned by the caller.
if [[ -z "${GLM52_VENV_ROOT:-}" ]]; then
    uv pip install --python "${SCRIPT_DIR}/../venv/torch/bin/python" \
        "transformers==5.12.1"
    # The GLM52 flex dispatcher requires Torch DeepEP on the H20 CI runner.
    # Build against this environment's Torch; the Paddle extension cannot serve it.
    TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS="${MAX_JOBS:-2}" \
        CPATH="${CUDA_HOME:-/usr/local/cuda}/include/cccl${CPATH:+:${CPATH}}" \
        uv pip install --python "${SCRIPT_DIR}/../venv/torch/bin/python" \
        --no-build-isolation --no-deps \
        "deep_ep @ git+https://github.com/deepseek-ai/DeepEP.git@17cfb817bccec3a9c247013360cc550c2bac441e"
fi
bash "${SCRIPT_DIR}/run_paddle_glm52.sh"
bash "${SCRIPT_DIR}/run_torch_glm52.sh"
python3 "${SCRIPT_DIR}/compare_loss.py" \
    "${RUN_DIR}/paddle/loss.json" "${RUN_DIR}/torch/loss.json" --required-steps 100
