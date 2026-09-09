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
"""Head-shard all-to-all helpers for linear-attention CP (``linear_cp_mode="headwise"``).

Everything in here is pure layout: a2a axis swaps and per-head slicing.  It is a
separate module from ``kimi_delta_attention.py`` for one reason -- the bitwise
test (``tests/multi_card_tests/transformer/test_kda_a2a_core_bitwise.py``) drives
the KDA core directly, below the layer, and must use *exactly* the slicing and
swap convention the layer uses.  A second copy of that convention in the test
would make the test agree with itself instead of with the product.

The one convention that everything hangs on, and that fails **silently** if
broken: ``UlyssesAlltoAll`` reshapes the head axis to ``[cp_size, h // cp_size]``
and exchanges over the first of those (``_ulysses_generate_layout_params``,
``context_parallel_utils.py:1620-1633``), so rank ``r`` receives the *contiguous*
head block ``[r * h/P, (r+1) * h/P)``.  Every parameter slice below uses the same
``head_range``.  Mismatch it and the heads get paired with the wrong ``A_log`` --
no error, just wrong numbers.

No gradient reduction happens here, on purpose.  CP is a sub-factor of the
sharding axis, so the sharding group already contains the CP group and the
trainer/optimizer reduces the parameter gradients over it; a second reduction
inside the layer would double-count.
"""

from __future__ import annotations

import paddle

from ..context_parallel_utils import UlyssesAlltoAll

__all__ = [
    "head_range",
    "head_to_seq",
    "seq_to_head",
    "seq_to_head_beta",
    "slice_channels_by_head",
    "slice_per_head_param",
    "split_qkv_seq_to_head",
]


def head_range(num_heads: int, cp_rank: int, cp_size: int) -> tuple[int, int]:
    """This rank's contiguous head block ``[h0, h1)``.

    Raises rather than truncating: a non-divisible head count would otherwise
    give overlapping or gapped blocks, and the a2a assert would fire later with
    a much less obvious message.
    """
    if num_heads % cp_size:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by cp_size ({cp_size}) "
            "for head-sharded all-to-all context parallel"
        )
    per_rank = num_heads // cp_size
    return cp_rank * per_rank, (cp_rank + 1) * per_rank


def seq_to_head(x, group):
    """``[b, s/P, h, d] -> [b, s, h/P, d]``: scatter heads, gather sequence."""
    return UlyssesAlltoAll.apply(x, 2, 1, 0, group)


def head_to_seq(x, group):
    """``[b, s, h/P, d] -> [b, s/P, h, d]``: scatter sequence, gather heads."""
    return UlyssesAlltoAll.apply(x, 1, 2, 0, group)


def seq_to_head_beta(beta, group):
    """``[b, s/P, h] -> [b, s, h/P]``.

    ``UlyssesAlltoAll`` only understands 4-D ``[b, s, h, d]``; a 3-D tensor would
    be silently read as ``[b, s, h]`` with the head axis in the wrong place.  So
    give ``beta`` a length-1 head_dim for the duration of the swap.
    """
    return seq_to_head(beta.unsqueeze(-1), group).squeeze(-1)


def split_qkv_seq_to_head(qkv, dims, heads, group):
    """Split the fused ``qkv`` channel block, then a2a each part to head shards.

    ``qkv`` is ``[b, s/P, sum(dims)]`` -- one contiguous channel block per q/k/v.
    Slicing its last axis by ``cp_size`` directly would cut across the q/k/v
    boundaries, so the split has to come first.  Under GVA the three blocks do
    not even have the same head count, which is why ``heads`` is per-block.

    Returns three 4-D tensors ``[b, s, h_block/P, d_block]``.
    """
    out = []
    parts = paddle.split(qkv, list(dims), axis=-1)
    for part, dim, num_heads in zip(parts, dims, heads):
        head_dim = dim // num_heads
        out.append(
            seq_to_head(
                part.reshape(
                    [part.shape[0], part.shape[1], num_heads, head_dim]
                ),
                group,
            )
        )
    return tuple(out)


def slice_channels_by_head(tensor, dims, heads, cp_rank, cp_size):
    """Slice a per-channel tensor laid out like ``qkv``: ``[sum(dims), ...]``.

    This is the depthwise conv weight ``[conv_dim, w]`` and its bias
    ``[conv_dim]``.  Each of the three blocks is sliced with *its own* head count,
    so it stays correct under GVA (``H != HV``) and ``K != V``.
    """
    out, offset = [], 0
    tail = list(tensor.shape[1:])
    for dim, num_heads in zip(dims, heads):
        h0, h1 = head_range(num_heads, cp_rank, cp_size)
        block = tensor[offset : offset + dim].reshape(
            [num_heads, dim // num_heads, *tail]
        )
        out.append(block[h0:h1].reshape([-1, *tail]))
        offset += dim
    return paddle.concat(out, axis=0)


def slice_per_head_param(param, num_heads, cp_rank, cp_size):
    """Slice a flat parameter whose leading axis folds into the head axis.

    Covers both ``A_log [HV]`` and the flat ``dt_bias [HV * K]``: the ``[HV, -1]``
    view degenerates to ``[HV, 1]`` for the former.

    The parameter itself is **not** sharded -- callers pass the full tensor and
    use the returned view.  Slicing is differentiable, so the gradient lands in a
    full-shape tensor that is exactly ``0.0`` outside this rank's heads; the CP
    sum is then ``x + 0 == x``, i.e. bitwise, and ``sharded_state_dict`` and the
    trainer-side reduction need no changes at all.
    """
    if len(param.shape) != 1:
        raise ValueError(
            f"expected a flat per-head parameter, got shape {param.shape}; "
            "use slice_channels_by_head for qkv-laid-out tensors"
        )
    h0, h1 = head_range(num_heads, cp_rank, cp_size)
    return param.reshape([num_heads, -1])[h0:h1].reshape([-1])
