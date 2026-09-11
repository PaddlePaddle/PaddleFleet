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

"""Triton core-attention for packed prefix-LM, a drop-in for `DotProductAttention`.

## What it does

This class occupies the `self_attention.submodules.core_attention` slot: the
outer `SelfAttention` still owns QKV projection, RoPE, and output projection;
only the softmax + masked matmul is replaced here.

## Tensor layout

The class works in batch-major layout:

| | q/k/v into core | tensor out of core |
|---|---|---|
| this class | batch-major `[B, S, N, D]` | `[B, S, N*D]` |

When sequence parallel (SP) is enabled, the surrounding attention module
transposes `[S, B, ...]` back to `[B, S, ...]` before calling core and
transposes it back afterwards, so this class only needs to handle
`[B, S, N, D]`. The kernel itself expects `[B, N, T, D]` (grid is
`(q_block, batch*head)`), so core permutes once before launching.

Note: under SP, `linear_qkv` is `ColumnParallelLinear(sequence_parallel=True)`,
which all-gathers `[S/tp, B, H]` back to the full `S` in the forward pass.
Therefore core sees the full `S` (equal to `layout.seq_total`), not `S/tp`,
and the plan does not need to be sliced per rank.

## Where the layout comes from

The segment layout is read from `packed_seq_params.prefix_lm_layout` (a dict).
It MUST be attached by the model code; if it is missing this class raises an
error rather than silently degrading to "no mask", which would drop the
prefix-LM three-region semantics while loss still appears to decrease normally.

The layout is passed through `packed_seq_params` rather than an instance
attribute or thread-local state: recompute reruns forward under `no_grad`,
where thread-local state has already been popped and instance attributes are
not recompute-safe.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.triton_ops.prefix_lm_attention import (
    SegmentLayout,
    layout_from_hyperbody_dict,
    triton_prefix_lm_attention,
)

if TYPE_CHECKING:
    from paddle import Tensor

__all__ = ["PrefixLMTritonCore", "PREFIX_LM_LAYOUT_ATTR"]

#: Field name attached to `PackedSeqParams`. There is deliberately no alias
#: fallback: a misspelled name should raise an error rather than silently
#: running with no mask.
PREFIX_LM_LAYOUT_ATTR = "prefix_lm_layout"

_LAYOUT_CACHE: dict[tuple, SegmentLayout] = {}


class PrefixLMTritonCore(FleetLayer):
    """Triton core-attention for packed prefix-LM.

    The constructor signature mirrors `DotProductAttention.__init__` so this
    class can directly occupy the `core_attention` slot; unused parameters are
    accepted and discarded.
    """

    def __init__(
        self,
        config,
        layer_number: int = 1,
        attn_mask_type=None,
        attention_type: str = "self",
        cp_comm_type: str | None = None,
        pg_collection=None,
        softmax_scale: float | None = None,
        **_unused,
    ) -> None:
        super().__init__(config=config)
        del attn_mask_type, attention_type, cp_comm_type, _unused

        self.config = config
        self.layer_number = layer_number
        # The surrounding attention module reads this attribute to decide
        # whether to slice flashmask row indices. This path does not use
        # flashmask, but the attribute must exist to avoid an AttributeError.
        self.context_parallel_size = 1

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim
        # Default softmax scale is 1/sqrt(head_dim); `softmax_scale` allows an
        # external override.
        self.softmax_scale = (
            float(softmax_scale)
            if softmax_scale is not None
            else 1.0 / math.sqrt(float(head_dim))
        )
        # Block sizes and launch tuning come from config fields (declared and
        # validated on the HyperEncoder config). Changing the block shape changes
        # both the plan and the softmax reduction tree, so it must be consistent
        # across a run.
        self.block_m = int(getattr(config, "hyperencoder_triton_block_m", 64))
        self.block_n = int(getattr(config, "hyperencoder_triton_block_n", 64))
        self.fwd_warps = int(
            getattr(config, "hyperencoder_triton_fwd_warps", 4)
        )
        self.fwd_stages = int(
            getattr(config, "hyperencoder_triton_fwd_stages", 2)
        )
        self.bwd_warps = int(
            getattr(config, "hyperencoder_triton_bwd_warps", 4)
        )
        self.bwd_stages = int(
            getattr(config, "hyperencoder_triton_bwd_stages", 2)
        )
        self.plan_cache_size = int(
            getattr(config, "hyperencoder_triton_plan_cache_size", 64)
        )

        if (
            pg_collection is not None
            and getattr(pg_collection, "cp", None) is not None
        ):
            cp = pg_collection.cp
            if getattr(cp, "nranks", 1) > 1:
                raise RuntimeError(
                    "PrefixLMTritonCore does not support context parallel"
                )

    def _resolve_layout(
        self, seq_total: int, packed_seq_params
    ) -> SegmentLayout:
        """Resolve the segment layout from `packed_seq_params.prefix_lm_layout`."""
        hint = (
            getattr(packed_seq_params, PREFIX_LM_LAYOUT_ATTR, None)
            if packed_seq_params is not None
            else None
        )
        if not isinstance(hint, dict):
            raise RuntimeError(
                f"PrefixLMTritonCore requires packed_seq_params.{PREFIX_LM_LAYOUT_ATTR} "
                f"(a dict), got {type(hint).__name__}. The packed prefix-LM segment "
                "layout must be supplied by the model code; without it the attention "
                "would silently degrade to no mask and lose the three-region semantics."
            )
        key = (
            tuple(int(x) for x in hint["segment_starts"]),
            tuple(int(x) for x in hint["n_contexts"]),
            tuple(int(x) for x in hint["n_queries"]),
            int(hint.get("pad_len", 0)),
            seq_total,
        )
        layout = _LAYOUT_CACHE.get(key)
        if layout is None:
            layout = layout_from_hyperbody_dict(hint, seq_total)
            _LAYOUT_CACHE[key] = layout
        return layout

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
        attn_mask_startend_row_indices: Tensor | None = None,
        *,
        attn_mask_type=None,
        attention_bias: Tensor | None = None,
        packed_seq_params=None,
        **_unused,
    ) -> Tensor:
        """`[B, S, N, D]` -> `[B, S, N*D]`."""
        del attn_mask_type, _unused
        if attention_bias is not None:
            raise ValueError(
                "PrefixLMTritonCore does not support attention_bias"
            )
        # Neither a dense mask nor flashmask row indices should be passed: the
        # mask semantics live entirely in the layout. Passing either means the
        # caller expected a different backend, so this is treated as an error
        # rather than silently ignored.
        if (
            attention_mask is not None
            or attn_mask_startend_row_indices is not None
        ):
            raise ValueError(
                "PrefixLMTritonCore derives its mask semantics entirely from "
                "prefix_lm_layout and does not accept attention_mask / "
                "attn_mask_startend_row_indices. Passing either indicates the "
                "wrong attention path was selected."
            )
        if query.dim() != 4:
            raise ValueError(
                f"PrefixLMTritonCore expects q/k/v as 4-D [B,S,N,D], got {tuple(query.shape)}"
            )

        b, s, n_heads, head_dim = query.shape
        layout = self._resolve_layout(s, packed_seq_params)

        # [B,S,N,D] -> contiguous [B,N,S,D] to feed the (batch*head) grid
        qt = query.transpose([0, 2, 1, 3]).contiguous()
        kt = key.transpose([0, 2, 1, 3]).contiguous()
        vt = value.transpose([0, 2, 1, 3]).contiguous()
        if kt.shape[1] != n_heads:
            raise ValueError(
                "PrefixLMTritonCore does not support GQA/MQA (the kernel assumes "
                f"equal head counts for q/k/v): q heads {n_heads}, k heads {kt.shape[1]}"
            )

        out = triton_prefix_lm_attention(
            qt,
            kt,
            vt,
            layout,
            scale=self.softmax_scale,
            block_m=self.block_m,
            block_n=self.block_n,
            fwd_warps=self.fwd_warps,
            fwd_stages=self.fwd_stages,
            bwd_warps=self.bwd_warps,
            bwd_stages=self.bwd_stages,
            plan_cache_size=self.plan_cache_size,
        )
        # [B,N,S,D] -> [B,S,N*D]
        return out.transpose([0, 2, 1, 3]).reshape([b, s, n_heads * head_dim])
