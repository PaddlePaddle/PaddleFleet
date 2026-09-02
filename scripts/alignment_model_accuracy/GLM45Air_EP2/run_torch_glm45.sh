#!/usr/bin/env bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Model/data are intentionally outside the Git repository. Override these
# paths when running on a host with a different cache layout.
MODEL_DIR="${GLM45_MODEL_DIR:-/home/.cache/PaddleFormers/GLM-4.5-Air-tiny-2L}"
TORCH_DATA="${GLM45_TORCH_DATA:-${MODEL_DIR}/alignment_torch.jsonl}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
TORCH_LOG_DIR="${WORKSPACE_DIR}/logs/torch/${RUN_TS}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "missing model config: ${MODEL_DIR}/config.json" >&2
    exit 1
fi
if [[ ! -f "${TORCH_DATA}" ]]; then
    echo "missing Torch alignment data: ${TORCH_DATA}" >&2
    exit 1
fi

source "${WORKSPACE_DIR}/venv/torch/bin/activate"
cd "${WORKSPACE_DIR}"

export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${WORKSPACE_DIR}/Megatron-LM}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29502}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NCCL_NVLS_ENABLE=0
export TORCHDYNAMO_DISABLE=1
export PYTORCH_ALLOC_CONF='expandable_segments:True'

# Accuracy-compatible path and deterministic observation settings.
export FLAGS_use_accuracy_compatible_kernel=1
export USE_ACCURACY_COMPATIBLE=1
export GLM_ALIGN_BIT_EXACT=1
export GLM_ALIGN_LOG=0
export ENABLE_SAVE_HOOK="${ENABLE_SAVE_HOOK:-0}"
export ENABLE_BACKWARD_HOOK="${ENABLE_BACKWARD_HOOK:-0}"
export SAVE_TENSOR_GRAD="${SAVE_TENSOR_GRAD:-0}"
export SAVE_TENSOR_SAVE_NPY="${SAVE_TENSOR_SAVE_NPY:-0}"
export MINIMAX_WORKSPACE="${WORKSPACE_DIR}"
export MG_TENSOR_DEBUG_DIR="${WORKSPACE_DIR}/logs/mg"
mkdir -p "${TORCH_LOG_DIR}" "${MG_TENSOR_DEBUG_DIR}"

ARGS=(
    --model "${MODEL_DIR}"
    --dataset "${TORCH_DATA}"
    --max_length 8192
    --packing False
    --padding_free False
    --truncation_strategy right
    --split_dataset_ratio 0
    --template dummy
    --template_backend swift
    --dataset_shuffle False
    --train_dataloader_shuffle False
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}"
    --dataloader_persistent_workers False
    --seed 42
    --finetune True
    --train_iters 10
    --logging_steps 1
    --eval_iters 0
    --output_dir "${TORCH_LOG_DIR}/trainer"
    --tensor_model_parallel_size 1
    --pipeline_model_parallel_size 1
    --expert_model_parallel_size 2
    --sequence_parallel False
    --moe_aux_loss_coeff 0.0001
    --moe_grouped_gemm False
    --moe_permute_fusion False
    --moe_enable_deepep False
    --moe_token_dispatcher_type alltoall
    --overlap_grad_reduce False
    --overlap_param_gather False
    --recompute_granularity full
    --recompute_method uniform
    --recompute_num_layers 1
    --bf16 True
    --masked_softmax_fusion False
    --bias_dropout_fusion False
    --bias_activation_fusion True
    --gradient_accumulation_fusion False
    --cross_entropy_loss_fusion False
    --attention_backend unfused
    --micro_batch_size 1
    --global_batch_size 2
    --calculate_per_token_loss False
    --optimizer adam
    --lr 5e-5
    --min_lr 1e-5
    --lr_warmup_iters 0
    --lr_decay_style cosine
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0.1
    --clip_grad 1.0
    --use_distributed_optimizer False
    --accumulate_allreduce_grads_in_fp32 True
    --use_accuracy_compatible True
)

megatron sft "${ARGS[@]}" 2>&1 | tee "${TORCH_LOG_DIR}/run_torch.log"
