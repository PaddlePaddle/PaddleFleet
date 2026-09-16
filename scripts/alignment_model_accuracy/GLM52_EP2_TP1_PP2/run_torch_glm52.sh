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
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_ROOT="${GLM52_VENV_ROOT:-${WORKSPACE_DIR}/venv}"
MODEL_DIR="${GLM52_MODEL_DIR:-/home/.cache/PaddleFormers/GLM-5.2-BF16-minimal}"
TOKENIZER_DIR="${GLM52_TOKENIZER_DIR:-${MODEL_DIR}}"
DATA_DIR="${GLM52_DATA_DIR:-/home/.cache/PaddleFormers/MiniMax-V2.5-bf16_2EP}"
RUN_TAG="${ALIGNMENT_RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)-$$}"
RUN_DIR="${SCRIPT_DIR}/results/${RUN_TAG}/torch"

for required in "${MODEL_DIR}/config.json" \
    "${TOKENIZER_DIR}/tokenizer.json" "${DATA_DIR}/alignment_torch.jsonl" \
    "${VENV_ROOT}/torch/bin/activate"; do
    [[ -f "${required}" ]] || { echo "missing GLM52 prerequisite: ${required}" >&2; exit 1; }
done
if [[ ! -f "${MODEL_DIR}/model.safetensors" && ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
    echo "missing GLM52 weights: expected model.safetensors or model.safetensors.index.json in ${MODEL_DIR}" >&2
    exit 1
fi
[[ ! -e "${RUN_DIR}" ]] || { echo "run directory already exists: ${RUN_DIR}" >&2; exit 1; }
mkdir -p "${RUN_DIR}"
# shellcheck disable=SC1091
source "${VENV_ROOT}/torch/bin/activate"
cd "${WORKSPACE_DIR}"

unset RANK LOCAL_RANK LOCAL_WORLD_SIZE WORLD_SIZE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_pick_master_port.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=Ring
export LD_LIBRARY_PATH="${VENV_ROOT}/torch/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MODEL_REPRO_DISABLE_LIVE_XY_DUMP=1
export MODEL_REPRO_RAW_LOSS_PATH="${RUN_DIR}/raw_loss.jsonl"
export MODEL_REPRO_LOSS_PATH="${RUN_DIR}/loss.json"
export MODEL_REPRO_ENV_PATH="${RUN_DIR}/env.json"
export MODEL_REPRO_CHECKPOINT_DIR="${RUN_DIR}/checkpoint"
export MODEL_REPRO_INPUT_RECEIPT_PATH="${RUN_DIR}/input_receipt.json"
export MODEL_REPRO_INPUT_DATASET_PATH="${DATA_DIR}/alignment_torch.jsonl"
export MODEL_REPRO_MODEL_CONFIG_PATH="${MODEL_DIR}/config.json"
export MODEL_REPRO_MODEL_ID=zai-org/GLM-5.2
export MODEL_REPRO_MODEL_REVISION=b4734de4facf877f85769a911abafc5283eab3d9
export MRK_INVOCATION_ID="${RUN_TAG}"

"${VENV_ROOT}/torch/bin/megatron" sft "${SCRIPT_DIR}/glm52_torch.yaml" \
    --model "${MODEL_DIR}" --tokenizer_name_or_path "${TOKENIZER_DIR}" \
    --dataset "${DATA_DIR}/alignment_torch.jsonl" --output_dir "${RUN_DIR}/trainer" \
    2>&1 | tee "${RUN_DIR}/run_torch.log"
[[ -s "${MODEL_REPRO_LOSS_PATH}" ]] || { echo "native Torch loss artifact missing" >&2; exit 1; }
