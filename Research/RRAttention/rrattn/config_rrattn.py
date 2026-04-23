from __future__ import annotations

from dataclasses import dataclass

import paddle


@dataclass(frozen=True)
class RRAttnConfig:
    # GEMM q-stride tile. Larger tiles improve reuse but increase register pressure.
    block_m: int = 128
    # K-stride tile. Larger tiles improve K throughput but can reduce occupancy.
    block_n: int = 32
    # Triton launch parameters for the qchunk GEMM kernel.
    num_warps: int = 4
    num_stages: int = 1
    # K-stride segment size used by the softmax/reduce kernel.
    segment_size: int = 128
TUNABLE_FIELDS = (
    "block_m",
    "block_n",
    "num_warps",
    "num_stages",
    "segment_size",
)


def gpu_info() -> tuple[str, int | None]:
    if not paddle.device.is_compiled_with_cuda():
        return "", None
    try:
        device_name = paddle.device.cuda.get_device_name().lower()
        major, _ = paddle.device.cuda.get_device_capability()
        return device_name, major
    except Exception:
        return "", None


GPU_NAME, GPU_MAJOR = gpu_info()


def get_rrattn_config(head_dim: int, gpu_name: str | None = None) -> RRAttnConfig:
    """Return the default estimate-kernel config.

    RRAttention currently fixes the public block size at 128, so block size is
    not a tuning dimension here. Tune the fields listed in TUNABLE_FIELDS per
    head_dim bucket.
    """
    gpu_name = (gpu_name or GPU_NAME or "").lower()

    if "h100" in gpu_name or "h800" in gpu_name:
        if head_dim <= 64:
            return RRAttnConfig(num_warps=4, num_stages=3)
        if head_dim <= 128:
            return RRAttnConfig(num_warps=8, num_stages=3)
        return RRAttnConfig(num_warps=8, num_stages=2)

    if head_dim <= 64:
        return RRAttnConfig(block_m=128, block_n=16, num_warps=4, num_stages=1, segment_size=256)
    if head_dim <= 128:
        return RRAttnConfig(block_m=128, block_n=32, num_warps=4, num_stages=2, segment_size=256)
    return RRAttnConfig(num_warps=8, num_stages=2)


__all__ = [
    "GPU_MAJOR",
    "GPU_NAME",
    "RRAttnConfig",
    "TUNABLE_FIELDS",
    "get_rrattn_config",
    "gpu_info",
]
