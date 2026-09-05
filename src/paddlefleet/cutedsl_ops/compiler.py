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

"""Local CuTe DSL compiler options for PaddleFleet operators."""

from functools import cache

_ARCH_MAP = {
    (9, 0): "sm_90a",
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_100f",
}


@cache
def gpu_arch_flag() -> str:
    """Return the CuTe DSL architecture flag for the current CUDA device."""
    import paddle

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("CuTe DSL compilation requires a CUDA Paddle build")
    capability = paddle.device.cuda.get_device_capability()
    arch = _ARCH_MAP.get(tuple(capability))
    if arch is None:
        raise RuntimeError(
            f"unsupported CUDA compute capability {capability}; "
            f"please extend {__name__}._ARCH_MAP"
        )
    return arch


def compile_options(extra: str = "") -> str:
    """Build options for ``cute.compile`` without importing cuDNN Frontend."""
    parts = ["--enable-tvm-ffi", f"--gpu-arch {gpu_arch_flag()}"]
    if extra:
        parts.append(extra)
    return " ".join(parts)
