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

"""cuDNN DSA sparse attention backward API (aligned with Megatron).

Provides Paddle-compatible interface to NVIDIA cuDNN DeepSeek Sparse Attention
backward implementation using DLPack for zero-copy tensor conversion.

Aligned with Megatron-LM's dsa_kernels.py implementation:
- Uses cuDNN's unified sparse_attention_backward_wrapper API
- TopK padding aligns with GPU architecture: SM90=128, SM100=64
"""

from __future__ import annotations

import os

import paddle
from paddle import Tensor

# Global flag to control cuDNN usage (can be overridden by env var)
_USE_CUDNN_DSA = os.getenv("USE_CUDNN_DSA", "true").lower() == "true"


def _get_topk_alignment() -> int:
    """Get TopK alignment requirement based on GPU architecture.

    Aligned with Megatron-LM's _get_topk_alignment():
    * SM90 : dual-warpgroup loop steps by 2 blocks → 128
    * SM100: single-pipeline loop steps by 1 block → 64

    Returns:
        Alignment value (128 for SM90, 64 for SM100)
    """
    gpu_props = paddle.device.get_device_properties("gpu:0")
    major = gpu_props.major  # 10 for SM100, 9 for SM90
    if major >= 10:
        return 64
    return 128


def sparse_attention_backward(
    q: Tensor,
    kv: Tensor,
    o: Tensor,
    do: Tensor,
    lse: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    sm_scale: float | None = None,
    topk_length: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """DSv4 sparse attention backward (cuDNN DSA implementation).

    Aligned with Megatron-LM's SparseAttnFunc.backward(). Uses cuDNN's
    unified sparse_attention_backward_wrapper API which handles SM90/SM100
    internally.

    Args:
        q: [B, S, H, D] BF16 (batch-first)
        kv: [B, S_kv, D] BF16 (K=V, MQA)
        o: [B, S, H, D] BF16 (forward output)
        do: [B, S, H, D] BF16 (grad of output)
        lse: [B, S, H] FP32 (log-sum-exp from forward)
        attn_sink: [H] FP32
        topk_idxs: [B, S, topk] INT32
        sm_scale: float or None
        topk_length: Optional Tensor for compact mode (Megatron feature)

    Returns:
        dq: [B, S, H, D] BF16
        dkv: [B, S_kv, D] BF16
        d_attn_sink: [H] FP32

    Raises:
        ImportError: If cuDNN DSA is not installed.
        ValueError: If GPU architecture is not supported.
    """
    # Check if cuDNN DSA is available and enabled
    if not _USE_CUDNN_DSA:
        raise ImportError(
            "cuDNN DSA is disabled. Set USE_CUDNN_DSA=true to enable, "
            "or use TileLang implementation instead."
        )

    try:
        from cudnn import DSA
    except ImportError as e:
        raise ImportError(
            "cuDNN DSA not found. Install with: "
            "pip install 'nvidia-cudnn-frontend[cutedsl]'\n"
            "Or set USE_CUDNN_DSA=false to use TileLang fallback."
        ) from e

    # Import DLPack for zero-copy conversion
    try:
        from torch.utils import dlpack
    except ImportError as e:
        raise ImportError(
            "PyTorch and torch.utils.dlpack are required for cuDNN DSA integration. "
            f"Error: {e}"
        ) from e

    # 1. Data layout conversion: batch-first -> flat (cuDNN format)
    B, S, H, D = q.shape
    S_kv = kv.shape[1]

    q_flat = q.reshape([B * S, H, D])
    kv_flat = kv.reshape([B * S_kv, D])
    o_flat = o.reshape([B * S, H, D])
    do_flat = do.reshape([B * S, H, D])
    lse_flat = lse.reshape([B * S, H])
    topk_flat = topk_idxs.reshape([B * S, -1])

    # 2. TopK padding (aligned with Megatron's _get_topk_alignment)
    topk = topk_flat.shape[-1]
    topk_align = _get_topk_alignment()
    padded_topk = (topk + topk_align - 1) // topk_align * topk_align
    if padded_topk != topk:
        pad = paddle.full(
            [B * S, padded_topk - topk],
            -1,
            dtype=topk_flat.dtype,
        )
        topk_flat = paddle.concat([topk_flat, pad], axis=-1)
        topk = padded_topk

    # 3. Compute softmax scale (matches Megatron's default behavior)
    if sm_scale is None:
        sm_scale = D**-0.5

    # 4. Paddle -> PyTorch DLPack zero-copy conversion
    q_torch = dlpack.from_dlpack(paddle.utils.dlpack.to_dlpack(q_flat))
    kv_torch = dlpack.from_dlpack(paddle.utils.dlpack.to_dlpack(kv_flat))
    o_torch = dlpack.from_dlpack(paddle.utils.dlpack.to_dlpack(o_flat))
    do_torch = dlpack.from_dlpack(paddle.utils.dlpack.to_dlpack(do_flat))
    lse_torch = dlpack.from_dlpack(paddle.utils.dlpack.to_dlpack(lse_flat))
    attn_sink_torch = dlpack.from_dlpack(
        paddle.utils.dlpack.to_dlpack(attn_sink)
    )
    topk_torch = dlpack.from_dlpack(paddle.utils.dlpack.to_dlpack(topk_flat))

    # Convert topk_length if provided
    topk_length_torch = None
    if topk_length is not None:
        topk_length_flat = topk_length.reshape([B * S])
        topk_length_torch = dlpack.from_dlpack(
            paddle.utils.dlpack.to_dlpack(topk_length_flat)
        )

    # 5. Call cuDNN backward (unified API handles SM90/SM100 internally)
    # Aligned with Megatron's call: no block_tile parameter
    paddle.device.synchronize()
    result = DSA.sparse_attention_backward_wrapper(
        q=q_torch,
        kv=kv_torch,
        out=o_torch,
        dout=do_torch,
        lse=lse_torch,
        attn_sink=attn_sink_torch,
        topk_idxs=topk_torch,
        softmax_scale=sm_scale,
        topk_length=topk_length_torch,
    )

    # 6. Result is a dict-like TupleDict
    dq_torch = result["dq"]
    dkv_torch = result["dkv"]
    d_sink_torch = result["d_sink"]

    # 7. PyTorch -> Paddle DLPack conversion
    dq_flat = paddle.utils.dlpack.from_dlpack(dlpack.to_dlpack(dq_torch))
    dkv_flat = paddle.utils.dlpack.from_dlpack(dlpack.to_dlpack(dkv_torch))
    d_sink = paddle.utils.dlpack.from_dlpack(dlpack.to_dlpack(d_sink_torch))

    # 8. Convert back to batch-first
    dq = dq_flat.reshape([B, S, H, D])
    dkv = dkv_flat.reshape([B, S_kv, D])

    return dq, dkv, d_sink


def sparse_mqa_bwd_interface(
    q: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    o: Tensor,
    do: Tensor,
    topk_idxs: Tensor,
    lse: Tensor,
    sm_scale: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Backward interface for DSv4 sparse MQA attention (cuDNN implementation).

    Fully compatible with TileLang sparse_mqa_bwd_interface, can be used
    as a drop-in replacement.

    Args:
        q: [B, S, H, D] BF16
        kv: [B, S_kv, D] BF16 (K=V, MQA)
        attn_sink: [H] FP32
        o: [B, S, H, D] BF16 (forward output)
        do: [B, S, H, D] BF16 (grad of output)
        topk_idxs: [B, S, topk] INT32
        lse: [B, S, H] FP32 (log-sum-exp from forward)
        sm_scale: float or None

    Returns:
        dq: [B, S, H, D] BF16
        dkv: [B, S_kv, D] BF16
        d_attn_sink: [H] FP32
    """
    return sparse_attention_backward(
        q=q,
        kv=kv,
        o=o,
        do=do,
        lse=lse,
        attn_sink=attn_sink,
        topk_idxs=topk_idxs,
        sm_scale=sm_scale,
    )


def is_cudnn_dsa_available() -> bool:
    """Check if cuDNN DSA is available and enabled.

    Aligned with Megatron's availability check: requires SM90 or SM100.

    Returns:
        True if cuDNN DSA can be used, False otherwise.
    """
    if not _USE_CUDNN_DSA:
        return False

    try:
        pass
    except Exception:
        return False

    try:
        # GPU architecture check using Paddle (not PyTorch)
        gpu_props = paddle.device.get_device_properties("gpu:0")
        major = gpu_props.major
        return major >= 9  # SM90 or SM100
    except Exception:
        return False


def set_cudnn_dsa_enabled(enabled: bool) -> None:
    """Enable or disable cuDNN DSA.

    Args:
        enabled: If True, cuDNN DSA will be used when available.
                  If False, TileLang implementation will be used.
    """
    global _USE_CUDNN_DSA
    _USE_CUDNN_DSA = enabled
    os.environ["USE_CUDNN_DSA"] = "true" if enabled else "false"
