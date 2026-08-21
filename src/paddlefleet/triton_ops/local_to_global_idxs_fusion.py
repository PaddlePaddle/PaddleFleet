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

"""
Fused Triton replacement for ``csa_sparse_attn_utils._local_to_global_flat``.

The eager reference spends five elementwise kernels on the full
``[b * sq, topk]`` index table -- ``full`` (the scalar in ``idxs >= 0`` is
materialised at full size), ``greater_equal``, ``add``, ``where``, ``cast``
-- to express a single pass. At sq=16384 / topk=640 (ernielite HCA layer,
cp=4) that is ~380 MB of traffic per call for 40 MB of useful output.

This module does the same thing in one kernel: one read, one write.

Not differentiable by design -- inputs and outputs are integer index tables.
All three call sites are inside ``paddle.autograd.PyLayer`` bodies
(``fusions/csa_sparse_attn.py`` forward+backward, ``fusions/mqa_sparse_attn.py``
backward, ``cudnn_ops/attn/csa_sparse_attn_fwd_cudnn.py``), which run with
grad tracking disabled and return ``None`` for the ``topk_idxs`` gradient.
"""

import paddle
from paddle import Tensor

from .utils import is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@triton.jit
def local_to_global_flat_kernel(
    Idxs_ptr,  # [n_rows, topk] local idxs, int32/int64, negative == invalid
    Out_ptr,  # [n_rows, topk] int32 global indices
    topk,
    sq,  # rows per batch; flat row == b * sq + s
    seqlen_kv,  # KV length of one batch entry
    BLOCK_K: tl.constexpr,  # power-of-2 block along the topk axis
):
    """One program per ``(row, topk-block)``.

    Computes ``out = idx + (row // sq) * seqlen_kv`` where ``idx >= 0`` and
    passes ``idx`` through untouched otherwise (the reference keeps the
    original negative value, it does not normalise it to -1).

    Arithmetic runs in int64 and is truncated on store. That matches the eager
    reference bit-for-bit for both input dtypes: the low 32 bits of a
    two's-complement sum are identical whether the reference accumulated in
    int32 (wrapping ``add``, then a no-op ``cast``) or in int64 (exact ``add``,
    then a truncating ``cast``).

    ``base`` is int64 so addressing stays correct past 2**31 elements. That is
    defensive only: the widest real table is ``65536 * 2176`` (~142M elements),
    so the unit tests cannot reach the 32-bit overflow point (it would need an
    ~8.6 GB index tensor) and a 32-bit ``base`` passes them unchanged.
    """
    row = tl.program_id(0)
    kblk = tl.program_id(1)

    batch_offset = (row // sq).to(tl.int64) * seqlen_kv

    offs = kblk * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < topk
    base = row.to(tl.int64) * topk

    idx = tl.load(Idxs_ptr + base + offs, mask=mask, other=0).to(tl.int64)
    result = tl.where(idx >= 0, idx + batch_offset, idx)

    tl.store(
        Out_ptr + base + offs,
        result.to(Out_ptr.dtype.element_ty),
        mask=mask,
    )


def local_to_global_flat_triton(
    local_idxs: Tensor,
    seqlen_kv: int,
    *,
    allow_alias: bool = False,
) -> Tensor:
    """Drop-in fused replacement for ``_local_to_global_flat``.

    Args:
        local_idxs: ``[b, sq, topk]`` int32/int64 indices into one batch
            entry's KV, negative == invalid slot.
        seqlen_kv: KV sequence length per batch entry.
        allow_alias: when ``b == 1`` the batch offset is 0 and the reference
            reduces to the identity, so the result can be returned as a
            reshaped *view* of ``local_idxs`` with no kernel at all. Off by
            default because the reference always returns fresh storage; only
            enable it where the caller treats the result as read-only.

    Returns:
        ``[b * sq, topk]`` int32, bit-identical to the eager reference.
    """
    assert local_idxs.ndim == 3, (
        f"local_idxs must be [b, sq, topk], got {local_idxs.shape}"
    )
    b, sq, topk = local_idxs.shape
    n_rows = b * sq

    if allow_alias and b == 1:
        flat = local_idxs.reshape([n_rows, topk])
        return flat if flat.dtype == paddle.int32 else flat.cast("int32")

    out = paddle.empty([n_rows, topk], dtype="int32")
    if n_rows == 0 or topk == 0:
        return out

    idxs = local_idxs.contiguous()
    BLOCK_K = min(triton.next_power_of_2(topk), 1024)
    grid = (n_rows, triton.cdiv(topk, BLOCK_K))
    local_to_global_flat_kernel[grid](
        idxs,
        out,
        topk,
        sq,
        int(seqlen_kv),
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out
