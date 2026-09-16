# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Expert row permutation for train_infer_consistent_inspect (training side).

The grouped GEMM consumes a buffer where each expert's rows are contiguous and
alignment-padded. Which row a given (dispatched token, local expert) pair lands
on is decided by each framework's own permute kernel, so the two sides hold the
same rows in a different order. `canonical_rows` maps a buffer into the
(token, local expert) order both sides agree on -- the only layout in which the
expert GEMM probes can be compared or loaded row for row -- and
`scatter_canonical_rows` is its inverse.

File layout: `pre_save_func` views first, then their `post_load_func` inverses,
then the entry points the network definition calls. The probe itself is always
plain `inspect_tensor(..., pre_save_func=canonical_rows, post_load_func=...)`;
the only entry point here is the row-map publisher the permute kernel calls.
"""

from __future__ import annotations

import paddle

from paddlefleet.train_infer_consistent_ops.inspect_util import inspect_enabled

# Row map published by the permute kernel: `[num_tokens, num_local_experts]`
# holding the buffer row of each pair, -1 where the pair is not routed. Both
# views below default to it.
_PERMUTE_INDEX = None


# ---------------------------------------------------------------------------
# pre_save_func views: expert-contiguous buffer -> canonical (token, expert)
# ---------------------------------------------------------------------------


def canonical_rows(buf, index=None):
    """Gather an expert-contiguous buffer into canonical (token, expert) order.

    Only rows the index points at are gathered, so the alignment padding the
    grouped GEMM never writes cannot reach a dump.

    Returns None when no index has been published, so callers dump nothing rather
    than dumping a layout-dependent tensor.
    """
    if index is None:
        index = _PERMUTE_INDEX
    if index is None or buf is None:
        return None
    flat = paddle.cast(index.reshape([-1]), "int64")
    safe = paddle.clip(flat, min=0)
    gathered = paddle.gather(buf, safe, axis=0)
    keep = paddle.cast(
        (flat >= 0).reshape([-1] + [1] * (len(buf.shape) - 1)), buf.dtype
    )
    return gathered * keep


# ---------------------------------------------------------------------------
# post_load_func inverses: canonical rows -> the live expert-contiguous buffer
# ---------------------------------------------------------------------------


def scatter_canonical_rows(buf, canon, index=None):
    """Inverse of `canonical_rows`: write canonical rows back into `buf`."""
    if index is None:
        index = _PERMUTE_INDEX
    if index is None or canon is None:
        return buf
    flat = paddle.cast(index.reshape([-1]), "int64")
    keep = paddle.nonzero(flat >= 0).reshape([-1])
    out = paddle.scatter(
        buf,
        paddle.gather(flat, keep, axis=0),
        paddle.gather(canon, keep, axis=0).astype(buf.dtype),
        overwrite=True,
    )
    # `paddle.scatter` can widen the result, and the fp8 quant kernels only accept
    # fp16/bf16, so pin the dtype back to the buffer's.
    return out.astype(buf.dtype)


# ---------------------------------------------------------------------------
# Probe entry points
# ---------------------------------------------------------------------------


def inspect_tensor_set_permute_index(index):
    """Publish `[num_tokens, num_local_experts]` rows (-1 = pair not routed).

    Self-gating, so the permute kernel can call it unconditionally.
    """
    global _PERMUTE_INDEX
    if not inspect_enabled():
        return
    _PERMUTE_INDEX = index
