#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Sourced only by the GLM52 case. Keep compiler provisioning inside its run.
set -euo pipefail
GLM52_CUDA_ROOT="${RUN_DIR}/cuda-13.0.2"
mkdir -p "${GLM52_CUDA_ROOT}"
# NVIDIA CUDA 13.0.2 redistrib manifest, linux-x86_64. Components are verified
# before extraction; this does not install a driver or change the system CUDA.
# https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json
while read -r component version checksum; do
    archive="${GLM52_CUDA_ROOT}/${component}.tar.xz"
    curl --fail --location --retry 3 \
        "https://developer.download.nvidia.com/compute/cuda/redist/${component}/linux-x86_64/${component}-linux-x86_64-${version}-archive.tar.xz" \
        --output "${archive}"
    printf '%s  %s\n' "${checksum}" "${archive}" | sha256sum --check -
    tar -xJf "${archive}" --strip-components=1 -C "${GLM52_CUDA_ROOT}"
done <<'COMPONENTS'
cuda_nvcc 13.0.88 48e35be3cfbf4b4fbc16828eaec8a7048ee789403049dc409f7b643d6259cf7a
cuda_crt 13.0.88 5a3279a049ffc1cdb951c44cb95206acfdde9e9ae5e87825fc18d7e4a6878bb0
cuda_cudart 13.0.96 25b8071951baba827be1580b841d363464f6ee6c39f48d33a81646f90cc95ed1
cuda_cccl 13.0.85 ed845eae8c1767706b6ee91e40c608a03f6f633551a849b63f7346d32d73ee60
cuda_profiler_api 13.0.85 dc233d88a5cafa095b197e6246b4c468a4581c128da8f951d67e063cdd6bca4c
cuda_nvtx 13.0.85 ed150e6fb1b50663ff068cccee3c5e2ca581c3b939b321656afbc9193671137d
libnvvm 13.0.88 17ef1665b63670887eeba7d908da5669fa8c66bb73b5b4c1367f49929c086353
COMPONENTS
export CUDA_HOME="${GLM52_CUDA_ROOT}"
export PATH="${CUDA_HOME}/bin:${PATH}"
"${CUDA_HOME}/bin/nvcc" --version
