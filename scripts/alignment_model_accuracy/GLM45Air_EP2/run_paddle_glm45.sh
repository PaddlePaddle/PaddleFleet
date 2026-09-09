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
CONFIG="${SCRIPT_DIR}/GLM45Air_EP2.yaml"

# ---- 环境 ----
# shellcheck disable=SC1091
source "${WORKSPACE_DIR}/venv/paddle/bin/activate"
cd "${WORKSPACE_DIR}"

# 清理上一个任务遗留的分布式环境变量
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset PADDLE_CURRENT_ENDPOINT
unset FLAGS_START_PORT
unset MASTER_ADDR
unset MASTER_PORT
unset NNODES
unset RANK
unset WORLD_SIZE
unset NODE_RANK
unset NPROC_PER_NODE
unset LOCAL_RANK
unset LOCAL_WORLD_SIZE

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="${MASTER_PORT:-29503}"

# EP2: 单机 2 卡
export NNODES="1"
export RANK="0"
export NODE_RANK="0"
export NPROC_PER_NODE="2"
export WORLD_SIZE="2"
export LOCAL_WORLD_SIZE="2"

export FLAGS_use_accuracy_compatible_kernel="1"
export USE_ACCURACY_COMPATIBLE="1"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# 本机 NVLS multicast 内存注册失败（CUDA error 401），NCCL init 会直接崩：
#   "Failed to bind NVLink SHARP (NVLS) Multicast memory ... Disable NVLS (NCCL_NVLS_ENABLE=0)"
export NCCL_NVLS_ENABLE=0

# ---- 精度对齐 Flag ----
export FLAGS_embedding_deterministic="1"
export FLAGS_cudnn_deterministic="1"

# ---- 精度对齐：逐层输出 md5（默认关闭）----
export ENABLE_SAVE_HOOK="${ENABLE_SAVE_HOOK:-0}"
export ENABLE_BACKWARD_HOOK="${ENABLE_BACKWARD_HOOK:-0}"
export SAVE_TENSOR_GRAD="${SAVE_TENSOR_GRAD:-0}"
export SAVE_TENSOR_SAVE_NPY="${SAVE_TENSOR_SAVE_NPY:-0}"
export SAVE_TENSOR_NAMES=output,grad,input

RUN_TS="$(date +%Y%m%d-%H%M%S)"
export PADDLEFORMERS_DIST_LOG="${WORKSPACE_DIR}/logs/paddle/${RUN_TS}"
export PF_TENSOR_DEBUG_DIR="${WORKSPACE_DIR}/logs/pf"
mkdir -p "${PADDLEFORMERS_DIST_LOG}" "${PF_TENSOR_DEBUG_DIR}"

exec "${WORKSPACE_DIR}/venv/paddle/bin/paddleformers-cli" train "${CONFIG}"
