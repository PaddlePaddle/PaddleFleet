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

# ---- 环境 ----
# shellcheck disable=SC1091
source "${WORKSPACE_DIR}/venv/torch/bin/activate"
cd "${WORKSPACE_DIR}"

export MINIMAX_MEGATRON_LM_PATH="${MINIMAX_MEGATRON_LM_PATH:-${WORKSPACE_DIR}/Megatron-LM}"
export MEGATRON_LM_PATH="${MINIMAX_MEGATRON_LM_PATH}"

# EP2: 单机 2 卡
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# 本机 NVLS multicast 内存注册失败（CUDA error 401），NCCL init 会直接崩：
#   "Failed to bind NVLink SHARP (NVLS) Multicast memory ... Disable NVLS (NCCL_NVLS_ENABLE=0)"
export NCCL_NVLS_ENABLE=0
export TORCHDYNAMO_DISABLE=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_ALLOC_CONF='expandable_segments:True'

# ---- 精度对齐：逐层输出 md5 ----
export ENABLE_SAVE_HOOK=1
export ENABLE_BACKWARD_HOOK=0
export SAVE_TENSOR_GRAD=1
export SAVE_TENSOR_SAVE_NPY=0
export MINIMAX_WORKSPACE="${WORKSPACE_DIR}"
export SAVE_TENSOR_NAMES=output,grad,nzs,input
export MG_TENSOR_DEBUG_DIR="${WORKSPACE_DIR}/logs/mg"

RUN_TS="$(date +%Y%m%d-%H%M%S)"
TORCH_LOG_DIR="${WORKSPACE_DIR}/logs/torch/${RUN_TS}"
rm -rf "${MG_TENSOR_DEBUG_DIR}"
mkdir -p "${TORCH_LOG_DIR}" "${MG_TENSOR_DEBUG_DIR}"

# ------- 训练参数 -----
ARGS=(
    ### model
    --model /home/.cache/PaddleFormers/MiniMax-V2.5-bf16_2EP

    ### data
    --dataset /home/.cache/PaddleFormers/MiniMax-V2.5-bf16_2EP/alignment_torch.jsonl
    --max_length 128
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
    --seed 23
    --finetune True
    --train_iters 10
    --logging_steps 1
    --eval_iters 0
    --output_dir ./logs/torch/default/trainer

    ### parallel
    --tensor_model_parallel_size 1
    --pipeline_model_parallel_size 1
    --expert_model_parallel_size 2
    --sequence_parallel False

    ### MoE
    --moe_aux_loss_coeff 0.0001
    --moe_grouped_gemm False
    --moe_permute_fusion False
    --moe_enable_deepep False
    --moe_token_dispatcher_type alltoall

    ### overlap
    --overlap_grad_reduce False
    --overlap_param_gather False

    ### memory / recompute
    --recompute_granularity full
    --recompute_method uniform
    --recompute_num_layers 1

    ### compute
    --bf16 True
    --masked_softmax_fusion False
    --bias_dropout_fusion False
    --bias_activation_fusion True
    --gradient_accumulation_fusion False
    --cross_entropy_loss_fusion True
    --cross_entropy_fusion_impl native
    --attention_backend local
    --micro_batch_size 1
    --global_batch_size 2
    --calculate_per_token_loss False

    ### optimizer
    --optimizer adam
    --lr 1e-5
    --min_lr 1e-6
    --lr_warmup_iters 0
    --lr_decay_style cosine
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0.1
    --clip_grad 0.0
    --optimizer_cpu_offload False
    --optimizer_offload_fraction 0.0
    --use_precision_aware_optimizer False
    --use_distributed_optimizer False
    --accumulate_allreduce_grads_in_fp32 True

    --use_accuracy_compatible True
)

# ------- 运行 -----
megatron sft "${ARGS[@]}" 2>&1 | tee "${TORCH_LOG_DIR}/run_torch.log"
