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

"""DSA TileLang indexer ops — facade with lazy imports to kernel files."""

import os
import threading
from collections import OrderedDict
from typing import TypeAlias

import paddle
from paddle import Tensor

try:
    paddle.enable_compat(scope={"tilelang"})
    import tilelang
    import tilelang.language as T

    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False

VarlenMetadata: TypeAlias = tuple[Tensor, Tensor]

# ===========================================================================
# Kernel Cache (LRU) — shared by fwd/bwd kernel files
# ===========================================================================

_TILELANG_KERNEL_CACHE_MAX = 512
_kernel_caches = {}
_kernel_cache_locks = {}


def _get_cache(name: str):
    if name not in _kernel_caches:
        _kernel_caches[name] = OrderedDict()
        _kernel_cache_locks[name] = threading.Lock()
    return _kernel_caches[name], _kernel_cache_locks[name]


def _cache_get_or_compile(cache_name: str, key, compile_fn):
    cache, lock = _get_cache(cache_name)
    with lock:
        kernel = cache.pop(key, None)
        if kernel is None:
            kernel = compile_fn()
        cache[key] = kernel
        cache.move_to_end(key)
        while len(cache) > _TILELANG_KERNEL_CACHE_MAX:
            cache.popitem(last=False)
        return kernel


# ===========================================================================
# Varlen metadata utilities
# ===========================================================================


def dsa_prepare_token_indices(cu_seqlens: Tensor) -> Tensor:
    """Convert cumulative sequence lengths to per-token (batch_id, position) pairs."""
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    parts = []
    for n in lens.numpy().tolist():
        parts.append(paddle.arange(int(n), dtype="int32"))
    pos_ids = paddle.concat(parts)
    seq_ids = paddle.cumsum(paddle.cast(pos_ids == 0, "int32")) - 1
    return paddle.stack([seq_ids.cast("int32"), pos_ids.cast("int32")], axis=1)


def dsa_prepare_valid_ranges(cu_seqlens: Tensor) -> Tensor:
    """Convert cumulative sequence lengths to per-token valid [start, end) ranges."""
    ranges = []
    cu_list = cu_seqlens.numpy().tolist()
    for seq_idx in range(len(cu_list) - 1):
        start = int(cu_list[seq_idx])
        end = int(cu_list[seq_idx + 1])
        length = end - start
        if length > 0:
            ranges.append(
                paddle.stack(
                    [
                        paddle.full([length], start, dtype="int32"),
                        paddle.full([length], end, dtype="int32"),
                    ],
                    axis=1,
                )
            )
    if not ranges:
        return paddle.empty([0, 2], dtype="int32")
    return paddle.concat(ranges, axis=0)


def dsa_prepare_token_indices_from_valid_ranges(valid_ranges: Tensor) -> Tensor:
    """Convert per-token valid [start, end) ranges to (batch_id, position) pairs."""
    if valid_ranges.shape[0] == 0:
        return paddle.empty([0, 2], dtype="int32")
    starts = valid_ranges[:, 0].cast("int32")
    positions = paddle.arange(valid_ranges.shape[0], dtype="int32") - starts
    changed = paddle.zeros_like(starts, dtype="int32")
    changed[0] = 1
    if valid_ranges.shape[0] > 1:
        changed[1:] = (starts[1:] != starts[:-1]).cast("int32")
    seq_ids = paddle.cumsum(changed) - 1
    return paddle.stack(
        [seq_ids.cast("int32"), positions.cast("int32")], axis=1
    )


def dsa_prepare_varlen_metadata(cu_seqlens: Tensor) -> VarlenMetadata:
    """Prepare varlen metadata for DSA kernels."""
    mode = os.getenv("DSA_VARLEN_METADATA_MODE", "offsets").lower()
    if mode in ("valid_range", "valid_ranges", "range"):
        valid_ranges = dsa_prepare_valid_ranges(cu_seqlens)
        return cu_seqlens, dsa_prepare_token_indices_from_valid_ranges(
            valid_ranges
        )
    if mode != "offsets":
        raise ValueError(
            f"Unsupported DSA_VARLEN_METADATA_MODE={mode!r}, expected 'offsets' or 'valid_range'"
        )
    return cu_seqlens, dsa_prepare_token_indices(cu_seqlens)


# ===========================================================================
# Format conversion utilities
# ===========================================================================


def dsa_sbhd_to_thd(tensor: Tensor) -> Tensor:
    """Convert [seq, batch, ...] to packed THD [seq*batch, ...] format."""
    s, b, *rest = tensor.shape
    return tensor.transpose([1, 0] + list(range(2, tensor.ndim))).reshape(
        [b * s] + rest
    )


def dsa_thd_to_sbhd(tensor: Tensor, s: int, b: int) -> Tensor:
    """Convert packed THD [seq*batch, ...] back to [seq, batch, ...] format."""
    rest = list(tensor.shape[1:])
    return tensor.reshape([b, s] + rest).transpose(
        [1, 0] + list(range(2, 2 + len(rest)))
    )


def dsa_bshd_to_thd(tensor: Tensor) -> Tensor:
    """Convert [batch, seq, ...] to packed THD [batch*seq, ...] format."""
    b, s, *rest = tensor.shape
    return tensor.reshape([b * s] + rest)


def dsa_thd_to_bshd(tensor: Tensor, b: int, s: int) -> Tensor:
    """Convert packed THD [batch*seq, ...] back to [batch, seq, ...] format."""
    rest = list(tensor.shape[1:])
    return tensor.reshape([b, s] + rest)


# ===========================================================================
# Lazy imports for kernel interfaces
# ===========================================================================


def _get_dsa_indexer_topk_fwd_interface():
    from .dsa_indexer_fwd import dsa_indexer_topk_reducesum_interface

    return dsa_indexer_topk_reducesum_interface


def _get_dsa_indexer_bwd_interface():
    from .dsa_indexer_bwd import dsa_indexer_bwd_interface

    return dsa_indexer_bwd_interface


__all__ = [
    "HAS_TILELANG",
    "VarlenMetadata",
    "_cache_get_or_compile",
    "dsa_bshd_to_thd",
    "dsa_sbhd_to_thd",
    "dsa_thd_to_bshd",
    "dsa_thd_to_sbhd",
    "dsa_indexer_bwd_interface",
    "dsa_indexer_topk_reducesum_interface",
    "dsa_prepare_token_indices",
    "dsa_prepare_token_indices_from_valid_ranges",
    "dsa_prepare_valid_ranges",
    "dsa_prepare_varlen_metadata",
]


# Facade exports — delegate to kernel files
def dsa_indexer_topk_reducesum_interface(*args, **kwargs):
    return _get_dsa_indexer_topk_fwd_interface()(*args, **kwargs)


def dsa_indexer_bwd_interface(*args, **kwargs):
    return _get_dsa_indexer_bwd_interface()(*args, **kwargs)
