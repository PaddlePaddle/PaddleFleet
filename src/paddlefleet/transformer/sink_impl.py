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


import numpy as np
import paddle
from paddle.autograd.py_layer import PyLayer

_C_ops = paddle._C_ops


def gen_dense_mask_from_startend_row_indices(
    attn_mask_startend_row_indices: paddle.Tensor,
    dtype: paddle.dtype = paddle.bfloat16,
    is_causal: bool | None = None,
):
    """Recover a 4-D dense attention mask from FlashMask's
    ``startend_row_indices`` representation.
    """
    if (
        attn_mask_startend_row_indices is not None
        and attn_mask_startend_row_indices.ndim == 3
    ):
        attn_mask_startend_row_indices = (
            attn_mask_startend_row_indices.unsqueeze(-1)
        )
    if (
        attn_mask_startend_row_indices is not None
        and attn_mask_startend_row_indices.shape[-1] == 1
    ):
        is_causal = True
    if (
        attn_mask_startend_row_indices is not None
        and attn_mask_startend_row_indices.shape[-1] == 4
    ):
        is_causal = False

    if is_causal is None:
        raise ValueError(
            "The `is_causal` argument must be specified when recovering the "
            "dense attention mask from the column-wise sparse attention mask "
            "row indices."
        )

    batch_size, num_head, seq_len, bound_num = (
        attn_mask_startend_row_indices.shape
    )
    has_end = (is_causal and bound_num == 2) or (
        (not is_causal) and bound_num == 4
    )

    attention_mask = paddle.ones([seq_len, seq_len], dtype="bool").expand(
        [batch_size, num_head, seq_len, seq_len]
    )
    if is_causal:
        attention_mask = paddle.tril(attention_mask)

    base = (
        paddle.arange(seq_len, dtype="int32")
        .unsqueeze(1)
        .expand([batch_size, num_head, -1, seq_len])
    )

    mask_indices = attn_mask_startend_row_indices.transpose([0, 1, 3, 2])

    downstart_mask_indices = mask_indices[:, :, 0:1, :]
    downstart_mask_indices = downstart_mask_indices.expand(
        [batch_size, num_head, seq_len, -1]
    )
    lower_tri = base < downstart_mask_indices
    if has_end:
        downend_mask_indices = mask_indices[:, :, 1:2, :]
        downend_mask_indices = downend_mask_indices.expand(
            [batch_size, num_head, seq_len, -1]
        )
        lower_tri = paddle.logical_or(lower_tri, base >= downend_mask_indices)

    attention_mask = paddle.logical_and(attention_mask, lower_tri)

    if not is_causal:
        if has_end:
            upstart_mask_indices = mask_indices[:, :, 2:3, :]
            upstart_mask_indices = upstart_mask_indices.expand(
                [batch_size, num_head, seq_len, -1]
            )
            upend_mask_indices = mask_indices[:, :, 3:4, :]
            upend_mask_indices = upend_mask_indices.expand(
                [batch_size, num_head, seq_len, -1]
            )
            upper_tri = base >= upend_mask_indices
            upper_tri = paddle.logical_or(
                upper_tri, base < upstart_mask_indices
            )
        else:
            upend_mask_indices = mask_indices[:, :, 1:2, :]
            upend_mask_indices = upend_mask_indices.expand(
                [batch_size, num_head, seq_len, -1]
            )
            upper_tri = base >= upend_mask_indices

        attention_mask = paddle.logical_and(attention_mask, upper_tri)

    attention_mask = paddle.scale(
        x=attention_mask.astype(dtype),
        scale=1000000.0,
        bias=-1.0,
        bias_after_scale=False,
    )
    return attention_mask


def _repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """Equivalent of ``paddle.repeat_interleave(hidden_states, n_rep, axis=2)``
    for tensors with layout ``[batch, seq_len, num_kv_heads, head_dim]``.
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states.unsqueeze(-2).tile([1, 1, 1, n_rep, 1])
    return hidden_states.reshape(
        [batch, slen, num_key_value_heads * n_rep, head_dim]
    )


def _get_fa_version():
    """Get the FlashAttention version based on environment flags."""
    if paddle.get_flags(["FLAGS_cudnn_deterministic"])[
        "FLAGS_cudnn_deterministic"
    ]:
        return 2
    return paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
        "FLAGS_flash_attn_version"
    ]


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
):
    """Dispatch FlashAttention forward based on version. seq_k must equal seq_v."""
    assert not return_softmax, "return_softmax must be false"

    seq_k, seq_v = key.shape[1], value.shape[1]
    assert seq_k == seq_v, (
        f"FlashAttention requires equal sequence lengths: seq_k={seq_k}, seq_v={seq_v}"
    )

    fa_version = _get_fa_version()

    if fa_version == 2:
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
    fa_version = _get_fa_version()

    if fa_version == 2:
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
):
    """Dispatch FlashMask attention forward. Only FlashMask v1 doesn't support
    custom softmax_scale.
    """
    fa_version = _get_fa_version()

    if fa_version == 2:
        if softmax_scale is not None and softmax_scale != 1.0 / (
            query.shape[-1] ** 0.5
        ):
            print(
                f"Warning: FlashMask v1 doesn't support custom softmax_scale, "
                f"ignoring provided value: {softmax_scale}"
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
    else:
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
    fa_version = _get_fa_version()
    if fa_version == 2:
        seed_offset = paddle.zeros(shape=[2], dtype="int64")
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
    else:
        raise ValueError(f"Unsupported FlashAttention version: {fa_version}")

    return grad_q, grad_k, grad_v


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
        assert head_dim_q == head_dim_k == head_dim_v, (
            f"Head dimensions must match: query={head_dim_q}, key={head_dim_k}, value={head_dim_v}"
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
        if startend_row_indices is None:
            key_states = _repeat_kv(key, num_key_value_groups)
            value_states = _repeat_kv(value, num_key_value_groups)
        else:
            key_states = key
            value_states = value

        if startend_row_indices is None:
            raw_output, lse_original = _flash_attention_forward_dispatch(
                query,
                key_states,
                value_states,
                dropout,
                causal,
                attention_mask=attention_mask,
                fixed_seed_offset=fixed_seed_offset,
                rng_name=rng_name,
                training=training,
                name=name,
                softmax_scale=softmax_scale,
            )
        else:
            raw_output, lse_original = _flashmask_attention_forward_dispatch(
                query,
                key_states,
                value_states,
                startend_row_indices,
                dropout,
                causal,
                training=training,
                softmax_scale=softmax_scale,
            )

        origin_dtype = raw_output.dtype
        scale = softmax_scale or 1.0 / (query.shape[-1] ** 0.5)
        batch_size, seq_len, num_heads, _ = query.shape

        # Compat with old LSE shape (seqlen_q_rounded)
        if lse_original.shape[-1] != seq_len:
            new_shape = (lse_original.shape[0], lse_original.shape[1], seq_len)
            num = np.prod(lse_original.shape[:2]) * seq_len
            lse_original = lse_original.flatten()[:num].reshape(new_shape)

        lse_transposed = lse_original.transpose(perm=[0, 2, 1]).unsqueeze(-1)
        sink_reshaped = sink.reshape(shape=[1, 1, -1, 1])
        sink_expanded = sink_reshaped.expand(
            [batch_size, seq_len, num_heads, 1]
        )

        # 1 / (exp(sink - lse) + 1)
        multiplier = 1 / (paddle.exp(sink_expanded - lse_transposed) + 1)
        final_out = (raw_output * multiplier).to(origin_dtype)

        ctx.save_for_backward(
            query,
            key,
            value,
            sink,
            attention_mask,
            raw_output,
            lse_original,
            multiplier,
            startend_row_indices,
        )
        ctx.dropout = dropout
        ctx.causal = causal
        ctx.softmax_scale = scale
        ctx.fixed_seed_offset = fixed_seed_offset
        ctx.rng_name = rng_name
        ctx.training = training
        ctx.name = name
        ctx.num_key_value_groups = num_key_value_groups

        return final_out

    @staticmethod
    def backward(ctx, grad_output):
        (
            query,
            key,
            value,
            sink,
            attention_mask,
            raw_output,
            lse_original,
            multiplier,
            startend_row_indices,
        ) = ctx.saved_tensor()

        num_key_value_groups = ctx.num_key_value_groups
        if startend_row_indices is None:
            key_states = _repeat_kv(key, num_key_value_groups)
            value_states = _repeat_kv(value, num_key_value_groups)
        else:
            key_states = key
            value_states = value

        dropout, causal, scale = ctx.dropout, ctx.causal, ctx.softmax_scale
        fixed_seed_offset, rng_name = ctx.fixed_seed_offset, ctx.rng_name
        training, name = ctx.training, ctx.name

        grad_raw_output = (grad_output * multiplier).to(query.dtype)

        if startend_row_indices is None:
            grad_q_main, grad_k_repeated, grad_v_repeated = (
                _flash_attention_backward_dispatch(
                    grad_raw_output,
                    query,
                    key_states,
                    value_states,
                    raw_output,
                    lse_original,
                    dropout=dropout,
                    attention_mask=attention_mask,
                    causal=causal,
                    softmax_scale=scale,
                )
            )
        else:
            grad_q_main, grad_k_repeated, grad_v_repeated = (
                _flashmask_attention_backward_dispatch(
                    grad_raw_output,
                    query,
                    key_states,
                    value_states,
                    raw_output,
                    lse_original,
                    startend_row_indices,
                    dropout,
                    causal,
                    scale,
                )
            )

        if (
            num_key_value_groups > 1
            and grad_k_repeated.shape[2] == key.shape[2] * num_key_value_groups
        ):
            batch, seq_len, num_kv_heads, head_dim = key.shape
            grad_k_main = grad_k_repeated.reshape(
                [batch, seq_len, num_kv_heads, num_key_value_groups, head_dim]
            ).sum(axis=3)
            grad_v = grad_v_repeated.reshape(
                [batch, seq_len, num_kv_heads, num_key_value_groups, head_dim]
            ).sum(axis=3)
        else:
            grad_k_main = grad_k_repeated
            grad_v = grad_v_repeated

        g_r = paddle.sum(grad_output * raw_output, axis=-1)
        multiplier_for_grad = multiplier.squeeze(-1)
        g_ell = g_r * multiplier_for_grad * (1 - multiplier_for_grad)

        grad_sink_temp = -paddle.sum(g_ell, axis=1)
        grad_sink = grad_sink_temp.sum(axis=0)

        if startend_row_indices is None:
            mu_k, lse_k = _flash_attention_forward_dispatch(
                query,
                key_states,
                key_states,
                dropout,
                causal,
                attention_mask=attention_mask,
                fixed_seed_offset=fixed_seed_offset,
                rng_name=rng_name,
                training=training,
                name=name,
                softmax_scale=scale,
            )
            x = (g_ell.unsqueeze(-1) * query).to(query.dtype)
            _, grad_k_extra_repeated, _ = _flash_attention_backward_dispatch(
                x,
                query,
                key_states,
                key_states,
                mu_k,
                lse_k,
                dropout=dropout,
                attention_mask=attention_mask,
                causal=causal,
                softmax_scale=scale,
            )
        else:
            mu_k, lse_k = _flashmask_attention_forward_dispatch(
                query,
                key_states,
                key_states,
                startend_row_indices,
                dropout,
                causal,
                training=training,
                softmax_scale=scale,
            )
            x = (g_ell.unsqueeze(-1) * query).to(query.dtype)
            _, grad_k_extra_repeated, _ = (
                _flashmask_attention_backward_dispatch(
                    x,
                    query,
                    key_states,
                    key_states,
                    mu_k,
                    lse_k,
                    startend_row_indices,
                    dropout,
                    causal,
                    scale,
                )
            )

        grad_q_extra = scale * g_ell.unsqueeze(-1) * mu_k

        if (
            num_key_value_groups > 1
            and grad_k_extra_repeated.shape[2]
            == key.shape[2] * num_key_value_groups
        ):
            batch, seq_len, num_kv_heads, head_dim = key.shape
            grad_k_extra_repeated = grad_k_extra_repeated.reshape(
                [batch, seq_len, num_kv_heads, num_key_value_groups, head_dim]
            )
            grad_k_extra = scale * grad_k_extra_repeated.sum(axis=3)
        else:
            grad_k_extra = scale * grad_k_extra_repeated

        grad_q = grad_q_main + grad_q_extra
        grad_k = grad_k_main + grad_k_extra
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


def sink_attention_forward(
    q,
    k,
    v,
    sink: paddle.Tensor,
    attention_mask: paddle.Tensor | None = None,
    startend_row_indices: paddle.Tensor | None = None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
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
        dropout=dropout_p,
        causal=causal,
        return_softmax=False,
        softmax_scale=softmax_scale,
    )
