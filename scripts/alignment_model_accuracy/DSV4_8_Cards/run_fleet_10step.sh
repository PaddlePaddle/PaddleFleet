#!/usr/bin/env bash
set -euo pipefail

ALIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ALIGN_ROOT}"

yaml_path="dpskv4_ep8_4layer_1k_align.yaml"

export DSV4_ASSET_ROOT="${DSV4_ASSET_ROOT:-${ALIGN_ROOT}/DSV4_Model_Accuracy}"
export DSV4_REALDATA_DUMP_DIR="${DSV4_REALDATA_DUMP_DIR:-${DSV4_ASSET_ROOT}/data/realdata_dump}"
export LOAD_FIXED_DATA_PATH="${DSV4_REALDATA_DUMP_DIR}"
export RUN_STAMP="${RUN_STAMP:-legacy_0823_fleet_realdata20_$(date +%Y%m%d_%H%M%S)}"
export RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
export FLEET_OUTPUT_DIR="${ALIGN_ROOT}/logs/paddle"
export FLEET_ROOT="${FLEET_ROOT:-${ALIGN_ROOT}}"
export FLEET_SOURCE_DIR="${FLEET_SOURCE_DIR:-${ALIGN_ROOT}/PaddleFleet}"
export FLEET_ENV_DIR="${FLEET_ENV_DIR:-${ALIGN_ROOT}/envs/paddle}"
export MEGATRON_SITE_PACKAGES="${MEGATRON_SITE_OVERRIDE:-${ALIGN_ROOT}/envs/torch/lib/python3.12/site-packages}"
export NVSHMEM_LIB_DIR="${FLEET_ENV_DIR}/lib/python3.12/site-packages/nvidia/nvshmem/lib"
export LD_LIBRARY_PATH="${NVSHMEM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TRAININGJOB_REPLICA_NAME=skip
export FLAGS_use_accuracy_compatible_kernel=1
# New DSV4-only gate: the repo-native accuracy paths keep reading the flag above.
export FLAGS_use_dsv4_accuracy=1
export FLAGS_use_deterministic_algorithm="${FLAGS_use_deterministic_algorithm:-1}"
export FLAGS_cudnn_deterministic="${FLAGS_cudnn_deterministic:-1}"

unset DSV4_FLEET_FIXED_TOKENS

# 分布式拓扑：优先取 lshrun 注入的变量，否则按单机处理
nnodes=${LSHRUN_NNODES:-1}
rank=${LSHRUN_RANK:-0}
if [[ -n "${LSHRUN_MASTER:-}" ]]; then
  master=${LSHRUN_MASTER}
else
  master=$(head -n 1 /root/paddlejob/workspace/hostfile | awk '{print $1}')
fi
port=${MASTER_PORT:-43521}

# 清理平台注入的 PADDLE*/ENDPOINT* 变量，避免干扰 launcher 组网
while IFS= read -r name; do
  unset "${name}"
done < <(env | awk -F'=' '/PADDLE|ENDPOINT/ {print $1}')

source "${FLEET_ENV_DIR}/bin/activate"

export PYTHONPATH="${FLEET_SOURCE_DIR}/src:${PYTHONPATH:-}"
export PADDLEFLEET_DIST_LOG=${FLEET_OUTPUT_DIR}/output_${rank}

rm -rf "${FLEET_OUTPUT_DIR}"
mkdir -p "${FLEET_OUTPUT_DIR}"
if [[ "${rank}" == "0" ]]; then
  /usr/bin/cp -f "${yaml_path}" "${FLEET_OUTPUT_DIR}/"
fi

echo "rank: $rank, nnodes: $nnodes, master: $master"

export NNODES=${nnodes}
export MASTER_ADDR=${master}
export MASTER_PORT=${port}
export RANK=${rank}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

exec "${FLEET_ENV_DIR}/bin/paddlefleet-cli" train "${yaml_path}"
