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
CONFIG="${SCRIPT_DIR}/MinimaxV2.5_EP2.yaml"

# ---- 环境 ----
# shellcheck disable=SC1091
source "${WORKSPACE_DIR}/venv/paddle/bin/activate"
cd "${WORKSPACE_DIR}"

# Match scripts/run_paddle.sh: clear stale distributed envs before launching.
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset PADDLE_CURRENT_ENDPOINT
unset PADDLE_TRAINERS_NUM
unset PADDLE_TRAINER_ID
unset PADDLE_RANK_IN_NODE
unset PADDLE_LOCAL_DEVICE_IDS
unset PADDLE_DISTRI_BACKEND
unset FLAGS_START_PORT
unset NVSHMEM_ENABLE_NIC_PE_MAPPING
unset MASTER_ADDR
unset MASTER_PORT
unset NNODES
unset RANK
unset WORLD_SIZE
unset NODE_RANK
unset NPROC_PER_NODE
unset LOCAL_RANK
unset LOCAL_WORLD_SIZE

export CUDA_VISIBLE_DEVICES=6,7
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="${MASTER_PORT:-29501}"

# EP2: 单机 2 卡
export NNODES="1"
export RANK="0"
export NODE_RANK="0"
export NPROC_PER_NODE="2"
export WORLD_SIZE="2"
export LOCAL_WORLD_SIZE="2"

# Match canonical Paddle alignment runtime knobs from scripts/run_paddle.sh.
export FLAGS_use_accuracy_compatible_kernel="1"
export FLAGS_set_to_1d="False"
export FLAGS_dataloader_use_file_descriptor="False"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# 本机 NVLS multicast 内存注册失败（CUDA error 401），NCCL init 会直接崩：
#   "Failed to bind NVLink SHARP (NVLS) Multicast memory ... Disable NVLS (NCCL_NVLS_ENABLE=0)"
export NCCL_NVLS_ENABLE=0

# ---- 精度对齐 Flag ----
export FLAGS_embedding_deterministic="1"
export FLAGS_cudnn_deterministic="1"

# ---- 精度对齐：逐层输出 md5 ----
export ENABLE_SAVE_HOOK=1
export ENABLE_BACKWARD_HOOK=0
export SAVE_TENSOR_GRAD=1
export SAVE_TENSOR_SAVE_NPY=0
export MINIMAX_WORKSPACE="${WORKSPACE_DIR}"
export SAVE_TENSOR_NAMES=output,grad,nzs,input

RUN_TS="$(date +%Y%m%d-%H%M%S)"
export PADDLEFORMERS_DIST_LOG="${WORKSPACE_DIR}/logs/paddle/${RUN_TS}"
export PF_TENSOR_DEBUG_DIR="${WORKSPACE_DIR}/logs/pf"
rm -rf "${PF_TENSOR_DEBUG_DIR}"
mkdir -p "${PADDLEFORMERS_DIST_LOG}" "${PF_TENSOR_DEBUG_DIR}"

exec "${WORKSPACE_DIR}/venv/paddle/bin/paddleformers-cli" train "${CONFIG}"
