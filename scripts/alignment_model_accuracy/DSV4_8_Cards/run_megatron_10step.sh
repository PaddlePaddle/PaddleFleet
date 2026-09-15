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

ALIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MEGATRON_REPO_DIR:-${ALIGN_ROOT}/Megatron-LM}"
VENV_DIR="${MEGATRON_VENV_DIR:-${ALIGN_ROOT}/envs/torch}"
ASSET_ROOT="${DSV4_ASSET_ROOT:-${ALIGN_ROOT}/DSV4_Model_Accuracy}"
LOAD_CHECKPOINT="${MEGATRON_CHECKPOINT:-${ASSET_ROOT}/megatron_checkpoint}"
LOAD_FIXED_DATA_PATH="${DSV4_REALDATA_DUMP_DIR:-${ASSET_ROOT}/data/realdata_dump}"
# torch 侧与 paddle 侧共用 data/train.jsonl；bin/idx 是派生缓存，缺失或过期时重建
DATA_JSONL="${DSV4_TRAIN_JSONL:-${ASSET_ROOT}/data/train.jsonl}"
TOKENIZER_DIR="${ASSET_ROOT}/data/tokenizer"
DATA_PREFIX="${ASSET_ROOT}/data/train_text_document"

export TRAIN_ITERS="${TRAIN_ITERS:-10}"
export EXIT_INTERVAL="${EXIT_INTERVAL:-${TRAIN_ITERS}}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-10}"
export MEGATRON_CANONICAL_ENV_ONLY="${MEGATRON_CANONICAL_ENV_ONLY:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
MIN_LR="${MIN_LR:-1e-4}"
MBS="${MBS:-2}"
ACC="${ACC:-2}"
GBS="$((MBS * 8 * ACC))"
MASTER_PORT="${MASTER_PORT:-43720}"
RUN_ID="dpskv4_ep8_8layer_align"
OUTPUT_DIR="${ALIGN_ROOT}/logs/torch"
LOG_DIR="${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/megatron_10step.log"
export LOG_DIR

rm -rf ${OUTPUT_DIR}

for path in "${REPO}" "${VENV_DIR}/bin/python" "${LOAD_CHECKPOINT}" "${LOAD_FIXED_DATA_PATH}" "${DATA_JSONL}" "${TOKENIZER_DIR}"; do
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: required path does not exist: ${path}" >&2
        exit 1
    fi
done

# IndexedDataset 缓存：由 data/train.jsonl + data/tokenizer 派生，jsonl 更新后自动重建
if [[ ! -f "${DATA_PREFIX}.bin" || ! -f "${DATA_PREFIX}.idx" \
      || "${DATA_JSONL}" -nt "${DATA_PREFIX}.bin" ]]; then
    echo "Building IndexedDataset from ${DATA_JSONL}"
    "${VENV_DIR}/bin/python" "${REPO}/tools/preprocess_data.py" \
        --input "${DATA_JSONL}" --json-keys text \
        --tokenizer-type HuggingFaceTokenizer --tokenizer-model "${TOKENIZER_DIR}" \
        --append-eod --workers "${PREPROCESS_WORKERS:-32}" \
        --output-prefix "${DATA_PREFIX%_text_document}"
fi

mkdir -p "${OUTPUT_DIR}/tensorboard" "${OUTPUT_DIR}/data-cache" "${OUTPUT_DIR}/load-manifest" "${LOG_DIR}"

export PYTHONPATH="${REPO}"
export PATH="${VENV_DIR}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_CPU_OFFLOAD_V1=0
export NVTE_FUSED_ATTN=0
export CUDA_DEVICE_MAX_CONNECTIONS=32
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export MOE_PERMUTE_FUSION=0
export FLAGS_use_accuracy_compatible_kernel=1
if [[ "${MEGATRON_CANONICAL_ENV_ONLY:-0}" != "1" ]]; then
    # Retained only for the frozen historical snapshot runner. The delivery
    # candidate has no source consumers for these pre-canonical switches.
    export DSV4_MEGATRON_MOE_FP32_ACCUM=1
    export DSV4_EMBEDDING_INDEX_BACKWARD=1
    export DSV4_DISABLE_MEGATRON_JIT_FUSER=1
    export DSV4_DISABLE_TE_ROUTER_GEMM=1
    export DSV4_USE_TORCH_RMSNORM=1
    export DSV4_MEGATRON_DETERMINISTIC=1
fi
export LOAD_FIXED_DATA_PATH
export MEGATRON_LOAD_MANIFEST_DIR="${OUTPUT_DIR}/load-manifest"

cd "${REPO}"
HELPERS_EXT_SUFFIX="$("${VENV_DIR}/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
make -C megatron/core/datasets LIBEXT="${HELPERS_EXT_SUFFIX}"

CMD=(
    "${VENV_DIR}/bin/python" -m torch.distributed.run
    --nnodes 1 --nproc-per-node 8 --node-rank 0
    --master-addr localhost --master-port "${MASTER_PORT}"
    # 每个 rank 的 stdout/stderr 单独写 ${LOG_DIR}/workerlog.<rank>
    --no-python bash -c 'exec "$@" > "${LOG_DIR}/workerlog.${RANK}" 2>&1' bash
    "${VENV_DIR}/bin/python" pretrain_gpt.py
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --context-parallel-size 1
    --pipeline-model-parallel-layout 'Et*4mL'
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --use-mcore-models
    --transformer-impl transformer_engine
    --num-layers 4
    --hidden-size 4096
    --ffn-hidden-size 2048
    --num-attention-heads 64
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --disable-bias-linear
    --swiglu
    --activation-func-clamp-value 10.0
    --position-embedding-type rope
    --no-position-embedding
    --rotary-base 10000
    --rope-type yarn
    --rotary-scaling-factor 16
    --original-max-position-embeddings 65536
    --no-rope-fusion
    --seq-length 1024
    --max-position-embeddings 1024
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend unfused
    --qk-layernorm
    --multi-latent-attention
    --q-lora-rank 1024
    --v-head-dim 512
    --qk-pos-emb-head-dim 64
    --o-groups 8
    --o-lora-rank 1024
    --experimental-attention-variant dsv4_hybrid
    --csa-window-size 128
    --csa-compress-ratios '([0,0]+[4,128]+[0])'
    --csa-compress-rotary-base 160000
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 512
    --dsa-indexer-loss-coeff 0.01
    --dsa-indexer-use-sparse-loss
    --no-dsa-kernel-fusion
    --enable-hyper-connections
    --num-residual-streams 4
    --mhc-sinkhorn-iterations 20
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --num-experts 256
    --moe-layer-freq 1
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-topk 6
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
    --moe-router-dtype fp32
    --moe-router-score-function sqrtsoftplus
    --moe-router-topk-scaling-factor 1.5
    --moe-router-enable-expert-bias
    --moe-n-hash-layers 3
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --moe-grouped-gemm
    --data-path "${DATA_PREFIX}"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_DIR}"
    --trust-remote-code
    --make-vocab-size-divisible-by 128
    --data-cache-path "${OUTPUT_DIR}/data-cache"
    --split 949,50,1
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --num-workers 0
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --train-iters "${TRAIN_ITERS}"
    --lr-decay-iters "${LR_DECAY_ITERS}"
    --lr-warmup-iters 0
    --lr "${LEARNING_RATE}"
    --min-lr "${MIN_LR}"
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.999
    --weight-decay 0.1
    --clip-grad 0.0
    --seed 1234
    --bf16
    --attention-softmax-in-fp32
    --no-gradient-accumulation-fusion
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 10
    --empty-unused-memory-level 2
    --recompute-granularity selective
    --recompute-modules moe_act layernorm mla_up_proj mlp shared_experts
    --load "${LOAD_CHECKPOINT}"
    --ckpt-format torch_dist
    --finetune
    --no-load-optim
    --no-load-rng
    --log-interval 1
    --eval-interval 1000
    --eval-iters 0
    --log-throughput
    --log-memory-to-tensorboard
    --log-timers-to-tensorboard
    --tensorboard-dir "${OUTPUT_DIR}/tensorboard"
    --exit-interval "${EXIT_INTERVAL}"
    --deterministic-mode
)

{
    echo "Official dev: $(git rev-parse HEAD)"
    echo "Run ID: ${RUN_ID}"
    echo "Train: iters=${TRAIN_ITERS} MBS=${MBS} ACC=${ACC} GBS=${GBS}"
    echo "Checkpoint: ${LOAD_CHECKPOINT}"
    echo "Replay: ${LOAD_FIXED_DATA_PATH}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Per-rank log: ${LOG_DIR}/workerlog.<rank>"
    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
} | tee "${LOG_FILE}"

set +e
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
TRAIN_RC="${PIPESTATUS[0]}"
set -e
if [[ "${TRAIN_RC}" -ne 0 ]]; then
    exit "${TRAIN_RC}"
fi
