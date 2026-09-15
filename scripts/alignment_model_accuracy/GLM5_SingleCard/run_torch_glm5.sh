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

# GLM-5 (ms-swift + Megatron-LM) 单机单卡精度对齐用例 —— torch 侧
# 对标 GLM45Air_EP2/run_torch_glm45.sh，并行度改为 TP=1/EP=1/PP=1（单卡）

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# 资产路径（可用环境变量覆盖；默认放在框架统一缓存目录）
GLM5_MODEL="${GLM5_MODEL:-/home/.cache/PaddleFormers/GLM-5-bf16_1Card}"
GLM5_DATASET="${GLM5_DATASET:-/home/.cache/PaddleFormers/GLM-5-bf16_1Card/alignment_torch.jsonl}"

# ---- 环境 ----
# shellcheck disable=SC1091
source "${WORKSPACE_DIR}/venv/torch/bin/activate"
cd "${WORKSPACE_DIR}"

export GLM5_MEGATRON_LM_PATH="${GLM5_MEGATRON_LM_PATH:-${WORKSPACE_DIR}/Megatron-LM}"
export MEGATRON_LM_PATH="${GLM5_MEGATRON_LM_PATH}"

# 单卡：TP=1 / EP=1 / PP=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE=1
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# 端口按用例错开：Minimax 29500 / GLM45Air 29502 / KimiK2 29504 / GLM5 29506
export MASTER_PORT="${MASTER_PORT:-29506}"

export FLAGS_use_accuracy_compatible_kernel=1
export USE_ACCURACY_COMPATIBLE=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# 本机 NVLS multicast 内存注册失败（CUDA error 401），NCCL init 会直接崩：
#   "Failed to bind NVLink SHARP (NVLS) Multicast memory ... Disable NVLS (NCCL_NVLS_ENABLE=0)"
export NCCL_NVLS_ENABLE=0
export TORCHDYNAMO_DISABLE=1
export PYTORCH_ALLOC_CONF='expandable_segments:True'
# TF32 会让 dense GEMM 与 paddle 侧对不上
export NVIDIA_TF32_OVERRIDE=0

# ---- 精度对齐：逐层输出 md5（默认关闭，排查时按需打开）----
export ENABLE_SAVE_HOOK="${ENABLE_SAVE_HOOK:-0}"
export ENABLE_BACKWARD_HOOK="${ENABLE_BACKWARD_HOOK:-0}"
export SAVE_TENSOR_GRAD="${SAVE_TENSOR_GRAD:-0}"
export SAVE_TENSOR_SAVE_NPY="${SAVE_TENSOR_SAVE_NPY:-0}"
export SAVE_TENSOR_NAMES=output,grad,input
export MG_TENSOR_DEBUG_DIR="${WORKSPACE_DIR}/logs/mg"

RUN_TS="$(date +%Y%m%d-%H%M%S)"
TORCH_LOG_DIR="${WORKSPACE_DIR}/logs/torch/${RUN_TS}"
mkdir -p "${TORCH_LOG_DIR}" "${MG_TENSOR_DEBUG_DIR}"

# ------- 训练参数（与 paddle 侧 GLM5.yaml 逐项对齐）-----
ARGS=(
    ### model
    --model "${GLM5_MODEL}"

    ### data
    --dataset "${GLM5_DATASET}"
    --max_length 8192
    --packing False
    --padding_free False
    --split_dataset_ratio 0
    --template dummy
    --template_backend swift
    --dataset_shuffle False
    --train_dataloader_shuffle False
    --dataloader_num_workers 4
    --dataloader_persistent_workers False

    ### finetuning
    --seed 42
    --finetune True
    --train_iters 1
    --logging_steps 1
    --eval_iters 0
    --output_dir "${TORCH_LOG_DIR}/trainer"

    ### parallel
    --tensor_model_parallel_size 1
    --pipeline_model_parallel_size 1
    --expert_model_parallel_size 1
    --expert_tensor_parallel_size 1
    --sequence_parallel False

    ### MoE
    --moe_router_load_balancing_type seq_aux_loss
    --moe_aux_loss_coeff 1e-4
    --moe_router_dtype fp32
    --moe_token_dispatcher_type alltoall
    --moe_grouped_gemm True
    --moe_permute_fusion True
    --moe_enable_deepep False

    ### DSA
    --dsa_indexer_loss_coeff 0.01

    ### overlap
    --overlap_grad_reduce False
    --overlap_param_gather False

    ### memory / recompute
    --recompute_granularity none

    ### compute
    --bf16 True
    --attention_backend flash
    --gradient_accumulation_fusion False
    --cross_entropy_loss_fusion False
    --calculate_per_token_loss False
    --micro_batch_size 1
    --global_batch_size 1

    ### optimizer
    --optimizer adam
    --lr 5e-5
    --min_lr 1e-5
    --lr_warmup_iters 0
    --lr_decay_style cosine
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0
    --clip_grad 0.0
    --optimizer_cpu_offload False
    --use_precision_aware_optimizer False
    --use_distributed_optimizer False
    --accumulate_allreduce_grads_in_fp32 True

    --use_accuracy_compatible True
)

# ------- 运行 -----
megatron sft "${ARGS[@]}" 2>&1 | tee "${TORCH_LOG_DIR}/run_torch.log"
