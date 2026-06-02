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

import paddle

from . import sparse_mqa_fwd


def _prepare_inputs(q, kv, attn_sink, topk_idxs):
    if len(q.shape) != 4:
        raise ValueError(f"q must have shape [B, S, H, D], got {q.shape}")
    if len(kv.shape) != 3:
        raise ValueError(f"kv must have shape [B, S_kv, D], got {kv.shape}")
    if len(topk_idxs.shape) != 3:
        raise ValueError(
            f"topk_idxs must have shape [B, S, topk], got {topk_idxs.shape}"
        )

    if topk_idxs.dtype != paddle.int32:
        topk_idxs = topk_idxs.cast("int32")

    if attn_sink.dtype != paddle.float32:
        attn_sink = attn_sink.cast("float32")
    return q, kv, attn_sink, topk_idxs


def sparse_attn(q, kv, attn_sink, topk_idxs, sm_scale=None):
    q, kv, attn_sink, topk_idxs = _prepare_inputs(q, kv, attn_sink, topk_idxs)
    out, lse = sparse_mqa_fwd.sparse_mqa_fwd_interface(
        q, kv, attn_sink, topk_idxs, sm_scale=sm_scale
    )
    if not isinstance(out, paddle.Tensor) or not isinstance(lse, paddle.Tensor):
        raise RuntimeError(
            f"TileLang must return Paddle tensors, got output={type(out)!r}, lse={type(lse)!r}. "
            "Ensure paddle.enable_compat(scope={'tilelang'}) runs before import tilelang."
        )
    return out, lse
