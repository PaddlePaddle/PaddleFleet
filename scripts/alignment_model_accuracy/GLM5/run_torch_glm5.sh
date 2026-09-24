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

# GLM-5 (ms-swift + Megatron-LM) 单机 2 卡精度对齐用例 —— torch 侧
# 对标 GLM45Air_EP2/run_torch_glm45.sh；对应 paddle 侧 sharding stage1 degree2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# ---- 环境 ----
# shellcheck disable=SC1091
source "${WORKSPACE_DIR}/venv/torch/bin/activate"
cd "${WORKSPACE_DIR}"

export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${WORKSPACE_DIR}/Megatron-LM}"

# sharding stage1 degree2 对侧: 单机 2 卡 DP2 + distributed optimizer
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29506}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# 本机 NVLS multicast 内存注册失败（CUDA error 401），NCCL init 会直接崩：
#   "Failed to bind NVLink SHARP (NVLS) Multicast memory ... Disable NVLS (NCCL_NVLS_ENABLE=0)"
export NCCL_NVLS_ENABLE=0
export TORCHDYNAMO_DISABLE=1
export PYTORCH_ALLOC_CONF='expandable_segments:True'
# TF32 会让 dense GEMM 与 paddle 侧对不上
export NVIDIA_TF32_OVERRIDE=0

# ---- 精度对齐 Flag ----
export FLAGS_use_accuracy_compatible_kernel=1
export USE_ACCURACY_COMPATIBLE=1

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
    --model /home/.cache/PaddleFormers/GLM-5-bf16_1Card

    ### data
    --dataset /home/.cache/PaddleFormers/GLM-5-bf16_1Card/alignment_torch.jsonl
    --max_length 8192
    --packing False
    --padding_free False
    --truncation_strategy right
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
    --train_iters 10
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
    --dsa_indexer_loss_coeff 0.0

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
    --global_batch_size 2

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
    --use_distributed_optimizer True
    --accumulate_allreduce_grads_in_fp32 True

    --use_accuracy_compatible True
)

# ------- 运行 -----
megatron sft "${ARGS[@]}" 2>&1 | tee "${TORCH_LOG_DIR}/run_torch.log"
