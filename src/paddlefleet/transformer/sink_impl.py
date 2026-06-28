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


import paddle

paddle.enable_compat(scope={"tilelang"})
import tilelang
import tilelang.language as T
from paddle.autograd.py_layer import PyLayer
from paddlefleet_ops.flash_mask.cute.flashmask_utils import (
    FlashMaskInfoPaddle,
)
from paddlefleet_ops.flash_mask.cute.interface import (
    _flash_attn_bwd,
    _flash_attn_fwd,
)


@tilelang.jit(out_idx=-1)
def flashattn_bwd_dsink(
    batch, heads, seq_len, block=128, dtype: T.dtype = T.float16
):
    accum_dtype = T.float32
    shape = [batch, heads, seq_len]

    @T.prim_func
    def flash_bwd_dsink(
        sink: T.Tensor([heads], dtype),  # type: ignore
        delta: T.Tensor(shape, accum_dtype),  # type: ignore
        lse: T.Tensor(shape, accum_dtype),  # type: ignore
        dsink: T.Tensor(shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block), batch, threads=128) as (
            bx,
            by,
            bz,
        ):
            lse_fragment = T.alloc_fragment([block], accum_dtype)
            delta_fragment = T.alloc_fragment([block], accum_dtype)
            dsink_fragment = T.alloc_fragment([block], accum_dtype)

            sink_value = sink[bx]
            T.copy(
                lse[bz, bx, by * block : (by + 1) * block],
                lse_fragment,
                disable_tma=True,
            )
            T.copy(
                delta[bz, bx, by * block : (by + 1) * block],
                delta_fragment,
                disable_tma=True,
            )
            for i in T.Parallel(block):
                dsink_fragment[i] = (
                    -T.exp2(sink_value * 1.44269504 - lse_fragment[i])
                    * delta_fragment[i]
                )
            T.copy(
                dsink_fragment,
                dsink[bz, bx, by * block : (by + 1) * block],
                disable_tma=True,
            )

    return flash_bwd_dsink


def _sink_attention_grad_sink(query, sink, output, lse, grad_output):
    delta = paddle.sum(
        output.astype("float32") * grad_output.astype("float32"), axis=-1
    )
    delta = delta.transpose(perm=[0, 2, 1]).contiguous()
    lse_log2 = (lse * 1.44269504).contiguous()
    batch_size, seq_len, num_heads, _ = query.shape
    dtype = (
        T.float32
        if sink.dtype == paddle.float32
        else T.float16
        if sink.dtype == paddle.float16
        else T.bfloat16
    )
    kernel_dsink = flashattn_bwd_dsink(
        batch_size, num_heads, seq_len, dtype=dtype
    )
    return kernel_dsink(sink.contiguous(), delta, lse_log2).sum(0).sum(1)


class FlashMaskSinkPyLayer(PyLayer):
    """FlashMask V4 attention with Sink mechanism."""

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        sink,
        startend_row_indices,
        attention_mask: paddle.Tensor | None = None,
        dropout=0.0,
        causal=False,
        return_softmax=False,
        *,
        fixed_seed_offset=None,
        rng_name="",
        training=True,
        name=None,
        softmax_scale=None,
    ):
        assert not return_softmax, "return_softmax must be false"
        assert attention_mask is None, (
            "FA4 do not support dense mask(attention_mask)"
        )
        assert query.ndim == 4, f"Query must be 4D tensor, got {query.ndim}D"
        assert key.ndim == 4, f"Key must be 4D tensor, got {key.ndim}D"
        assert value.ndim == 4, f"Value must be 4D tensor, got {value.ndim}D"
        assert sink.ndim == 1, f"Sink must be 1D tensor, got {sink.ndim}D"

        batch_q, seq_q, num_q_heads, head_dim_q = query.shape
        batch_k, seq_k, num_kv_heads, head_dim_k = key.shape
        batch_v, seq_v, num_kv_heads_v, head_dim_v = value.shape

        assert batch_q == batch_k == batch_v, (
            f"Batch sizes must match: query={batch_q}, key={batch_k}, value={batch_v}"
        )
        assert head_dim_q == head_dim_k, (
            f"Head dimensions must match: query={head_dim_q}, key={head_dim_k}"
        )
        assert num_kv_heads == num_kv_heads_v, (
            f"Key and value must have same number of heads: key={num_kv_heads}, value={num_kv_heads_v}"
        )
        assert num_q_heads % num_kv_heads == 0, (
            f"Query heads ({num_q_heads}) must be divisible by key/value heads ({num_kv_heads})"
        )
        assert sink.shape[0] == num_q_heads, (
            f"Sink parameter size ({sink.shape[0]}) must match number of query heads ({num_q_heads})"
        )

        if startend_row_indices is None:
            assert seq_q == seq_k == seq_v, (
                f"FlashAttention requires equal sequence lengths: seq_q={seq_q}, seq_k={seq_k}, seq_v={seq_v}"
            )
        else:
            assert seq_k == seq_v, (
                f"Key and value sequence lengths must match: seq_k={seq_k}, seq_v={seq_v}"
            )

        output, lse = _flash_attn_fwd(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            causal=causal,
            return_lse=True,
            startend_row_indices=startend_row_indices,
            pack_gqa=False,
            learnable_sink=sink,
        )

        ctx.save_for_backward(
            query,
            key,
            value,
            sink,
            attention_mask,
            output,
            lse,
            startend_row_indices,
        )
        ctx.dropout = dropout
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.fixed_seed_offset = fixed_seed_offset
        ctx.rng_name = rng_name
        ctx.training = training
        ctx.name = name

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            query,
            key,
            value,
            sink,
            attention_mask,
            output,
            lse,
            startend_row_indices,
        ) = ctx.saved_tensor()

        causal = ctx.causal
        softmax_scale = ctx.softmax_scale
        grad_output = grad_output.to(query.dtype)

        if startend_row_indices is not None:
            flashmask_info = FlashMaskInfoPaddle(
                startend_row_indices=startend_row_indices,
                is_causal=causal,
            )
        else:
            flashmask_info = None

        deterministic = bool(
            paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                "FLAGS_cudnn_deterministic"
            ]
        )
        grad_q, grad_k, grad_v = _flash_attn_bwd(
            query,
            key,
            value,
            output,
            grad_output,
            lse,
            flashmask_info,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )

        if sink.stop_gradient:
            grad_sink = None
        else:
            grad_sink = _sink_attention_grad_sink(
                query, sink, output, lse, grad_output
            )

        if query.dtype != grad_q.dtype:
            grad_q = grad_q.cast(query.dtype)
        if key.dtype != grad_k.dtype:
            grad_k = grad_k.cast(key.dtype)
        if value.dtype != grad_v.dtype:
            grad_v = grad_v.cast(value.dtype)

        if startend_row_indices is None:
            return grad_q, grad_k, grad_v, grad_sink
        else:
            return grad_q, grad_k, grad_v, grad_sink, None


def sink_attention(
    q,
    k,
    v,
    sink: paddle.Tensor,
    attention_mask: paddle.Tensor | None = None,
    startend_row_indices: paddle.Tensor | None = None,
    dropout=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    """Unified FlashMask V4 attention forward with Sink mechanism.

    Shapes follow the ``[B, S, H, D]`` layout (same as paddlefleet's
    ``DotProductAttention``).

    Args:
        q: Query tensor ``[B, S, H_q, D]``
        k: Key tensor ``[B, S, H_kv, D]``
        v: Value tensor ``[B, S, H_kv, D]``
        sink: Sink parameter tensor ``[H_q]``
        attention_mask: Dense mask. FA4 does not support it; must be None.
        startend_row_indices: Optional FlashMask row indices.
        dropout: Dropout probability.
        softmax_scale: Custom softmax scaling factor.
        causal: Whether to apply causal masking.
    """
    return FlashMaskSinkPyLayer.apply(
        q,
        k,
        v,
        sink,
        startend_row_indices,
        attention_mask=attention_mask,
        dropout=dropout,
        causal=causal,
        return_softmax=False,
        softmax_scale=softmax_scale,
        **kwargs,
    )
