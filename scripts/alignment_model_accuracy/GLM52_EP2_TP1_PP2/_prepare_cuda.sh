#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Sourced only by the GLM52 extension build subprocess.
set -euo pipefail
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]] && \
    "${CUDA_HOME}/bin/nvcc" --version | grep -q 'release 12.9,'; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    return
fi
if command -v nvcc >/dev/null && nvcc --version | grep -q 'release 12.9,'; then
    CUDA_HOME="$(dirname -- "$(dirname -- "$(readlink -f "$(command -v nvcc)")")")"
    export CUDA_HOME
    return
fi
: "${GLM52_CUDA_ROOT:?GLM52_CUDA_ROOT must be inside the case environment}"
if [[ -f "${GLM52_CUDA_ROOT}/.components-complete" ]]; then
    export CUDA_HOME="${GLM52_CUDA_ROOT}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    return
fi
mkdir -p "${GLM52_CUDA_ROOT}"
# NVIDIA CUDA 12.9.1 redistrib, linux-x86_64; no driver or system installation.
# https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json
while read -r component version checksum; do
    archive="${GLM52_CUDA_ROOT}/${component}.tar.xz"
    curl --fail --location --retry 3 --retry-all-errors \
        "https://developer.download.nvidia.com/compute/cuda/redist/${component}/linux-x86_64/${component}-linux-x86_64-${version}-archive.tar.xz" \
        --output "${archive}"
    printf '%s  %s\n' "${checksum}" "${archive}" | sha256sum --check -
    tar -xJf "${archive}" --strip-components=1 -C "${GLM52_CUDA_ROOT}"
done <<'COMPONENTS'
cuda_nvcc 12.9.86 7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35
cuda_cudart 12.9.79 1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99
cuda_cccl 12.9.27 8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa
cuda_profiler_api 12.9.79 8c50636bfb97e9420905aa795b9fa6e3ad0b30ec6a6c8b0b8db519beb9241ce6
cuda_nvtx 12.9.79 819bc39192955e6ba2067de39b85f30e157de462945e54b12bfdeda429d793fb
COMPONENTS
touch "${GLM52_CUDA_ROOT}/.components-complete"
export CUDA_HOME="${GLM52_CUDA_ROOT}"
export PATH="${CUDA_HOME}/bin:${PATH}"
"${CUDA_HOME}/bin/nvcc" --version
