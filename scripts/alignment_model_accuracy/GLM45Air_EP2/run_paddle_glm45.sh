#!/usr/bin/env bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${SCRIPT_DIR}/GLM45Air_EP2.yaml"
MODEL_DIR="${GLM45_MODEL_DIR:-/home/.cache/PaddleFormers/GLM-4.5-Air-tiny-2L}"
PADDLE_DATA="${GLM45_PADDLE_DATA:-${MODEL_DIR}/alignment_paddle.jsonl}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
PADDLE_LOG_DIR="${WORKSPACE_DIR}/logs/paddle/${RUN_TS}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "missing model config: ${MODEL_DIR}/config.json" >&2
    exit 1
fi
if [[ ! -f "${PADDLE_DATA}" ]]; then
    echo "missing Paddle alignment data: ${PADDLE_DATA}" >&2
    exit 1
fi

source "${WORKSPACE_DIR}/venv/paddle/bin/activate"
cd "${WORKSPACE_DIR}"

# Avoid inheriting stale elastic-launch variables from a previous job.
unset PADDLE_ELASTIC_JOB_ID PADDLE_ELASTIC_TIMEOUT PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS PADDLE_CURRENT_ENDPOINT FLAGS_START_PORT
unset MASTER_ADDR MASTER_PORT NNODES RANK WORLD_SIZE NODE_RANK NPROC_PER_NODE
unset LOCAL_RANK LOCAL_WORLD_SIZE

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29503}"
export NNODES=1
export RANK=0
export NODE_RANK=0
export NPROC_PER_NODE=2
export WORLD_SIZE=2
export LOCAL_WORLD_SIZE=2
export FLAGS_use_accuracy_compatible_kernel=1
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NCCL_NVLS_ENABLE=0
export USE_ACCURACY_COMPATIBLE=1
export GLM_ALIGN_BIT_EXACT=1
export GLM_ALIGN_LOG=0
export ENABLE_SAVE_HOOK="${ENABLE_SAVE_HOOK:-0}"
export ENABLE_BACKWARD_HOOK="${ENABLE_BACKWARD_HOOK:-0}"
export SAVE_TENSOR_GRAD="${SAVE_TENSOR_GRAD:-0}"
export SAVE_TENSOR_SAVE_NPY="${SAVE_TENSOR_SAVE_NPY:-0}"
export MINIMAX_WORKSPACE="${WORKSPACE_DIR}"
export PADDLEFORMERS_DIST_LOG="${PADDLE_LOG_DIR}"
export PF_TENSOR_DEBUG_DIR="${WORKSPACE_DIR}/logs/pf"
mkdir -p "${PADDLE_LOG_DIR}" "${PF_TENSOR_DEBUG_DIR}"

# The checked-in YAML uses the same cache root by default. For an override,
# render a run-local copy so the Git-tracked config remains reproducible.
RUN_CONFIG="${PADDLE_LOG_DIR}/GLM45Air_EP2.yaml"
sed \
    -e "s#^model_name_or_path: .*#model_name_or_path: ${MODEL_DIR}#" \
    -e "s#^train_dataset_path: .*#train_dataset_path: ${PADDLE_DATA}#" \
    -e "s#^eval_dataset_path: .*#eval_dataset_path: ${PADDLE_DATA}#" \
    "${CONFIG}" > "${RUN_CONFIG}"

"${WORKSPACE_DIR}/venv/paddle/bin/paddleformers-cli" train "${RUN_CONFIG}" \
    2>&1 | tee "${PADDLE_LOG_DIR}/run_paddle.log"
