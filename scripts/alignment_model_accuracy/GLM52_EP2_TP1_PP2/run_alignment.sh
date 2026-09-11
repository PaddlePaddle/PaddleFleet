#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ALIGNMENT_RUN_TAG="${ALIGNMENT_RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)-$$}"
RUN_DIR="${SCRIPT_DIR}/results/${ALIGNMENT_RUN_TAG}"
bash "${SCRIPT_DIR}/run_paddle_glm52.sh"
bash "${SCRIPT_DIR}/run_torch_glm52.sh"
python3 "${SCRIPT_DIR}/compare_loss.py" \
    "${RUN_DIR}/paddle/loss.json" "${RUN_DIR}/torch/loss.json" --required-steps 100
