#!/usr/bin/env bash
set -euo pipefail

# Kimi-K2 (ms-swift + Megatron-LM) 单机 2 卡精度对齐用例 —— torch 侧
# 对标 MinimaxV2.5_EP2/run_torch_minimax.sh（TP=1/EP=2/PP=1）

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# 资产路径（可用环境变量覆盖；默认放在 torch 侧缓存目录）
KIMIK2_MODEL="${KIMIK2_MODEL:-/home/.cache/PaddleFormers/Kimi-K2-bf16_2EP}"
KIMIK2_DATASET="${KIMIK2_DATASET:-/home/.cache/PaddleFormers/Kimi-K2-bf16_2EP/alignment_torch.jsonl}"

# ---- 环境 ----
# shellcheck disable=SC1091
source "${WORKSPACE_DIR}/venv/torch/bin/activate"
cd "${WORKSPACE_DIR}"

export KIMIK2_MEGATRON_LM_PATH="${KIMIK2_MEGATRON_LM_PATH:-${WORKSPACE_DIR}/Megatron-LM}"
export MEGATRON_LM_PATH="${KIMIK2_MEGATRON_LM_PATH}"

# 单机 2 卡：TP=1 / EP=2 / PP=1
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export NCCL_NVLS_ENABLE=0
export TORCHDYNAMO_DISABLE=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_ALLOC_CONF='expandable_segments:True'

# unfused attention 路径（与 paddle 侧对齐）
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0

# ---- 精度对齐：逐层输出 md5 ----
export ENABLE_SAVE_HOOK=1
export ENABLE_BACKWARD_HOOK=0
export SAVE_TENSOR_GRAD=1
export SAVE_TENSOR_SAVE_NPY=0
export SAVE_TENSOR_NAMES=output,grad,nzs,input
export MG_TENSOR_DEBUG_DIR="${WORKSPACE_DIR}/logs/mg"

RUN_TS="$(date +%Y%m%d-%H%M%S)"
TORCH_LOG_DIR="${WORKSPACE_DIR}/logs/torch/${RUN_TS}"
rm -rf "${MG_TENSOR_DEBUG_DIR}"
mkdir -p "${TORCH_LOG_DIR}" "${MG_TENSOR_DEBUG_DIR}"

# ------- 训练参数（与 paddle 侧 KimiK2.yaml 对齐）-----
ARGS=(
    ### model
    --model_type deepseek_v3
    --model "${KIMIK2_MODEL}"

    ### data
    --dataset "${KIMIK2_DATASET}"
    --columns '{"src": "query", "tgt": "response"}'
    --max_length 8192
    --packing False
    --truncation_strategy delete
    --split_dataset_ratio 0
    --template dummy
    --dataset_shuffle False
    --train_dataloader_shuffle False
    --dataloader_num_workers 4
    --dataset_num_proc 24

    ### finetuning
    --seed 23
    --finetune True
    --train_type full
    --train_iters 1
    --logging_steps 1
    --eval_iters 0
    --eval_steps 200
    --save_steps 300
    --output_dir ./logs/torch/default/trainer

    ### parallel
    --tensor_model_parallel_size 1
    --pipeline_model_parallel_size 1
    --expert_model_parallel_size 2
    --sequence_parallel False

    ### MoE
    --moe_aux_loss_coeff 0
    --moe_router_dtype fp32
    --moe_grouped_gemm True
    --moe_permute_fusion True
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
    --fp16 False
    --torch_dtype float32
    --attention_backend unfused
    --bias_activation_fusion False
    --gradient_accumulation_fusion False
    --cross_entropy_loss_fusion False
    --calculate_per_token_loss False
    --micro_batch_size 1
    --global_batch_size 2

    ### optimizer
    --optimizer adam
    --lr 1e-5
    --use_precision_aware_optimizer True
    --optimizer_cpu_offload True
    --streaming False

    --use_accuracy_compatible True
)

# ------- 运行 -----
megatron sft "${ARGS[@]}" 2>&1 | tee "${TORCH_LOG_DIR}/run_torch.log"
