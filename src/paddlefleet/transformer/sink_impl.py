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

_C_ops = paddle._C_ops

if paddle.cuda.get_device_capability()[0] == 10:
    from paddlefleet_ops.flash_mask.cute.flashmask_utils import (
        FlashMaskInfoPaddle,
    )
    from paddlefleet_ops.flash_mask.cute.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
    )


def _get_fa_version(hdim: int) -> int:
    """Pick the FlashAttention version for the given query head dim.

    Dispatch rules:
      * XPU device -> FA2.
      * Otherwise, respect ``FLAGS_flash_attn_version`` by default.
      * If ``fa_version == 3`` and deterministic is required, FA3 only supports
        ``hdim <= 128``. For ``hdim > 128``, fall back to FA2.

    Args:
        hdim: Query head dim, used to gate the FA3 deterministic fallback.

    Returns:
        The FlashAttention version to use (2, 3 or 4).
    """
    if "xpu" in paddle.get_device():
        return 2

    fa_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
        "FLAGS_flash_attn_version"
    ]

    deterministic = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
        "FLAGS_cudnn_deterministic"
    ]
    if fa_version == 3 and deterministic and hdim > 128:
        return 2

    return fa_version


def _flash_attention_forward_dispatch(
    query,
    key,
    value,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    attention_mask: paddle.Tensor | None = None,
    *,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
    softmax_scale=None,
    sink=None,
):
    """Dispatch FlashAttention forward based on version. seq_k must equal seq_v."""
    assert not return_softmax, "return_softmax must be false"

    seq_k, seq_v = key.shape[1], value.shape[1]
    assert seq_k == seq_v, (
        f"FlashAttention requires equal sequence lengths: seq_k={seq_k}, seq_v={seq_v}"
    )

    fa_version = _get_fa_version(query.shape[-1])

    if fa_version == 2:
        assert sink is None, "currently FA2 not support learnable sink"
        softmax_scale = softmax_scale or 1.0 / (query.shape[-1] ** 0.5)
        if hasattr(paddle.base.libpaddle.pir.ops, "flash_attn"):
            out, _, lse, _ = _C_ops.flash_attn(
                query,
                key,
                value,
                fixed_seed_offset,
                attention_mask,
                dropout,
                causal,
                False,
                not training,
                rng_name,
            )
        else:
            raise AssertionError(
                "flash_attn_v2 is not supported, may be due to paddle version"
            )
        lse = lse[:, :, : query.shape[1]]
    elif fa_version == 3:
        assert sink is None, "currently FA3 not support learnable sink"
        softmax_scale = softmax_scale or 1.0 / (query.shape[-1] ** 0.5)
        if hasattr(paddle.base.libpaddle.pir.ops, "flash_attn_v3"):
            out, lse = _C_ops.flash_attn_v3(
                query,
                key,
                value,
                None,
                None,
                None,
                None,
                softmax_scale,
                causal,
                -1,
                -1,
                0.0,
                1,
                False,
                False,
                0,
            )
        else:
            raise AssertionError(
                "flash_attn_v3 is not supported, may be due to paddle version"
            )

        assert attention_mask is None, (
            "FA3 do not support dense mask(attention_mask)"
        )
    elif fa_version == 4:
        out, lse = _flash_attn_fwd(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            causal=causal,
            return_lse=True,
            startend_row_indices=None,
            pack_gqa=False,
            learnable_sink=sink,
        )
        assert attention_mask is None, (
            "FA4 do not support dense mask(attention_mask)"
        )
    else:
        raise ValueError(f"Unsupported FlashAttention version: {fa_version}")

    return out, lse


def _flash_attention_backward_dispatch(
    grad_output,
    query,
    key,
    value,
    output,
    lse,
    dropout=0.0,
    attention_mask: paddle.Tensor | None = None,
    causal=False,
    softmax_scale=None,
):
    """Dispatch FlashAttention backward based on version."""
    fa_version = _get_fa_version(query.shape[-1])

    if fa_version == 2:
        assert softmax_scale is None, (
            "flashmask do not support softmax_scale when using FA2 as backend"
        )
        seed_offset = paddle.zeros(shape=[2], dtype="int64")
        if hasattr(paddle.base.libpaddle.pir.ops, "flash_attn_grad"):
            grad_q, grad_k, grad_v = _C_ops.flash_attn_grad(
                query,
                key,
                value,
                output,
                lse,
                seed_offset,
                attention_mask,
                grad_output,
                dropout,
                causal,
            )
        else:
            raise AssertionError(
                "flash_attn_v2_grad is not supported, may be due to paddle version"
            )
    elif fa_version == 3:
        softmax_scale = softmax_scale or 1.0 / (query.shape[-1] ** 0.5)
        if hasattr(paddle.base.libpaddle.pir.ops, "flash_attn_v3_grad"):
            grad_q, grad_k, grad_v = _C_ops.flash_attn_v3_grad(
                query,
                key,
                value,
                output,
                lse,
                grad_output,
                softmax_scale,
                causal,
                -1,
                -1,
                0.0,
                0,
            )
        else:
            raise AssertionError(
                "flash_attn_v3_grad is not supported, may be due to paddle version"
            )
        assert attention_mask is None, (
            "FA3 do not support dense mask(attention_mask)"
        )
    elif fa_version == 4:
        grad_q, grad_k, grad_v = _flash_attn_bwd(
            query,
            key,
            value,
            output,
            grad_output,
            lse,
            None,  # flashmask_info, startend_row_indices is not None
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=bool(
                paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                    "FLAGS_cudnn_deterministic"
                ]
            ),
        )
        assert attention_mask is None, (
            "FA4 do not support dense mask(attention_mask)"
        )
    else:
        raise ValueError(f"Unsupported FlashAttention version: {fa_version}")

    return grad_q, grad_k, grad_v


def _flashmask_attention_forward_dispatch(
    query,
    key,
    value,
    startend_row_indices,
    dropout=0.0,
    causal=False,
    training=True,
    softmax_scale=None,
    sink=None,
):
    """Dispatch FlashMask attention forward. Only FlashMask v1 doesn't support
    custom softmax_scale.
    """
    fa_version = _get_fa_version(query.shape[-1])

    if fa_version == 2:
        assert softmax_scale is None, (
            "flashmask do not support softmax_scale when using FA2 as backend"
        )
        assert sink is None, (
            "currently FlashMask doesn't support learnable sink when using FA2 as backend"
        )
        output, log_sum_exp = paddle.nn.functional.flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=startend_row_indices,
            causal=causal,
            dropout=dropout,
            return_softmax_lse=True,
            training=training,
        )
    elif fa_version == 3:
        assert sink is None, (
            "currently FlashMask doesn't support learnable sink when using FA3 as backend"
        )
        output, log_sum_exp = paddle.nn.functional.flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=startend_row_indices,
            causal=causal,
            dropout=dropout,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            training=training,
        )
    elif fa_version == 4:
        output, log_sum_exp = _flash_attn_fwd(
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
    else:
        raise ValueError(f"Unsupported FlashAttention version: {fa_version}")

    return output, log_sum_exp


def _flashmask_attention_backward_dispatch(
    grad_output,
    query,
    key,
    value,
    output,
    lse,
    startend_row_indices,
    dropout=0.0,
    causal=False,
    softmax_scale=None,
):
    """Dispatch FlashMask attention backward based on version."""
    fa_version = _get_fa_version(query.shape[-1])
    if fa_version == 2:
        seed_offset = paddle.zeros(shape=[2], dtype="int64")
        assert softmax_scale is None, (
            "flashmask do not support softmax_scale when using FA2 as backend"
        )
        if hasattr(paddle.base.libpaddle.pir.ops, "flashmask_attention_grad"):
            grad_q, grad_k, grad_v = _C_ops.flashmask_attention_grad(
                query,
                key,
                value,
                startend_row_indices,
                output,
                lse,
                seed_offset,
                grad_output,
                dropout,
                causal,
            )
        else:
            raise AssertionError(
                "flashmask_attention_grad is not supported, may be due to paddle version"
            )
    elif fa_version == 3:
        softmax_scale = softmax_scale or 1.0 / (query.shape[-1] ** 0.5)
        if hasattr(
            paddle.base.libpaddle.pir.ops, "flashmask_attention_v2_grad"
        ):
            block_mask = None
            grad_q, grad_k, grad_v = _C_ops.flashmask_attention_v2_grad(
                query,
                key,
                value,
                output,
                lse,
                startend_row_indices,
                block_mask,
                grad_output,
                softmax_scale,
                causal,
                0,  # rank
                1,  # nranks
            )
        else:
            raise AssertionError(
                "flashmask_attention_v2_grad is not supported, may be due to paddle version"
            )
    elif fa_version == 4:
        if startend_row_indices is not None:
            flashmask_info = FlashMaskInfoPaddle(
                startend_row_indices=startend_row_indices,
                is_causal=causal,
            )
        else:
            flashmask_info = None
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
            deterministic=bool(
                paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                    "FLAGS_cudnn_deterministic"
                ]
            ),
        )
    else:
        raise ValueError(f"Unsupported FlashAttention version: {fa_version}")

    return grad_q, grad_k, grad_v


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
    """FlashAttention/FlashMask with Sink mechanism."""

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
            assert attention_mask is None, (
                "Flashmask do not support dense mask(attention_mask)"
            )

        num_attention_heads = query.shape[2]
        num_key_value_heads = key.shape[2]
        num_key_value_groups = num_attention_heads // num_key_value_heads

        key_states = key
        value_states = value

        if startend_row_indices is None:
            output, lse = _flash_attention_forward_dispatch(
                query,
                key,
                value,
                dropout,
                causal,
                attention_mask=attention_mask,
                fixed_seed_offset=fixed_seed_offset,
                rng_name=rng_name,
                training=training,
                name=name,
                softmax_scale=softmax_scale,
                sink=sink,
            )
        else:
            output, lse = _flashmask_attention_forward_dispatch(
                query,
                key,
                value,
                startend_row_indices,
                dropout,
                causal,
                training=training,
                softmax_scale=softmax_scale,
                sink=sink,
            )

        origin_dtype = output.dtype
        batch_size, seq_len, num_heads, _ = query.shape

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
        ctx.num_key_value_groups = num_key_value_groups

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

        num_key_value_groups = ctx.num_key_value_groups

        dropout, causal, scale = ctx.dropout, ctx.causal, ctx.softmax_scale
        grad_output = grad_output.to(query.dtype)

        if startend_row_indices is None:
            grad_q, grad_k, grad_v = _flash_attention_backward_dispatch(
                grad_output,
                query,
                key,
                value,
                output,
                lse,
                dropout=dropout,
                attention_mask=attention_mask,
                causal=causal,
                softmax_scale=scale,
            )
        else:
            grad_q, grad_k, grad_v = _flashmask_attention_backward_dispatch(
                grad_output,
                query,
                key,
                value,
                output,
                lse,
                startend_row_indices,
                dropout,
                causal,
                scale,
            )

        grad_sink = _sink_attention_grad_sink(
            query, sink, output, lse, grad_output
        )
        if query.dtype != grad_q.dtype:
            grad_q = grad_q.cast(query.dtype)
        if key.dtype != grad_k.dtype:
            grad_k = grad_k.cast(key.dtype)
        if value.dtype != grad_v.dtype:
            grad_v = grad_v.cast(value.dtype)
        if sink.stop_gradient:
            if startend_row_indices is None:
                return grad_q, grad_k, grad_v, None
            else:
                return grad_q, grad_k, grad_v, None, None
        else:
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
    """Unified attention forward with Sink mechanism.

    Automatically chooses between FlashAttention and FlashMask based on
    ``startend_row_indices``. Shapes follow the ``[B, S, H, D]`` layout
    (same as paddlefleet's ``DotProductAttention``).

    Args:
        q: Query tensor ``[B, S, H_q, D]``
        k: Key tensor ``[B, S, H_kv, D]``
        v: Value tensor ``[B, S, H_kv, D]``
        sink: Sink parameter tensor ``[H_q]``
        attention_mask: Dense mask, only supported for FA2 path.
        startend_row_indices: If given, route to FlashMask.
        dropout_p: Dropout probability.
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
