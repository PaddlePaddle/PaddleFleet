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

import inspect

import paddle
from paddle.autograd.py_layer import PyLayer

_C_ops = paddle._C_ops
_FlashMaskInfoPaddle = None
_flash_attn_bwd = None
_flash_attn_fwd = None

try:
    if (
        paddle.cuda.is_available()
        and paddle.cuda.get_device_capability()[0] == 10
    ):
        from paddlefleet_ops.flash_mask.cute.flashmask_utils import (
            FlashMaskInfoPaddle as _FlashMaskInfoPaddle,
        )
        from paddlefleet_ops.flash_mask.cute.interface import (
            _flash_attn_bwd,
            _flash_attn_fwd,
        )
except (ImportError, AttributeError):
    pass


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


def _get_fa_version(head_dim: int = 0):
    """Get the FlashAttention version based on environment flags."""
    if "xpu" in paddle.get_device():
        return 2
    fa_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
        "FLAGS_flash_attn_version"
    ]
    if (
        fa_version == 3
        and paddle.base.framework.get_flags(["FLAGS_cudnn_deterministic"])[
            "FLAGS_cudnn_deterministic"
        ]
        and head_dim > 128
    ):
        return 2
    return fa_version


def _full_startend_row_indices(batch_size: int, seq_len: int):
    start = paddle.full(
        [batch_size, 1, seq_len, 1], seq_len, dtype=paddle.int32
    )
    end = paddle.zeros([batch_size, 1, seq_len, 1], dtype=paddle.int32)
    return paddle.concat([start, end], axis=-1)


def _dense_mask_to_startend_row_indices(attention_mask, query, key, is_causal):
    if attention_mask is None:
        return None

    bsz, q_len, _, _ = query.shape
    kv_len = key.shape[1]
    if q_len != kv_len:
        raise NotImplementedError(
            "FA3 sink attention dense mask conversion requires q_len == kv_len, "
            f"got q_len={q_len}, kv_len={kv_len}."
        )
    if len(attention_mask.shape) != 4:
        raise NotImplementedError(
            "FA3 sink attention only supports 4-D dense masks that are "
            "equivalent to causal or full attention."
        )

    if attention_mask.dtype == paddle.bool:
        masked = attention_mask
    elif bool(paddle.any(attention_mask < 0).item()):
        masked = attention_mask < 0
    else:
        masked = attention_mask < 0.5

    if masked.shape[0] not in (1, bsz) or masked.shape[1] != 1:
        raise NotImplementedError(
            "FA3 sink attention dense mask conversion only supports masks "
            "with shape [1|batch, 1, seq, seq]."
        )

    mask_sample = masked[0, 0]
    if is_causal:
        expected = paddle.triu(
            paddle.ones([q_len, kv_len], dtype="bool"), diagonal=1
        )
        startend = paddle.full([bsz, 1, kv_len, 1], kv_len, dtype=paddle.int32)
    else:
        expected = paddle.zeros([q_len, kv_len], dtype="bool")
        startend = _full_startend_row_indices(bsz, kv_len)

    if not bool(paddle.all(mask_sample == expected).item()):
        raise NotImplementedError(
            "FA3 sink attention only supports dense masks equivalent to "
            "causal or full attention in phase one."
        )
    if masked.shape[0] == bsz:
        for batch_idx in range(1, bsz):
            if not bool(paddle.all(masked[batch_idx, 0] == mask_sample).item()):
                raise NotImplementedError(
                    "FA3 sink attention does not support per-sample dense "
                    "mask differences in phase one."
                )

    return startend


def prepare_fa3_sink_attention(
    query,
    key,
    value,
    sink: paddle.Tensor | None,
    attention_mask=None,
    startend_row_indices=None,
    causal=False,
    context_parallel_size: int = 1,
    use_rr_flash_attention: bool = False,
    flashmask_use_varlen: bool = False,
):
    q_head_dim = query.shape[-1]
    v_head_dim = value.shape[-1]
    if sink is None or _get_fa_version(q_head_dim) != 3:
        return False, startend_row_indices
    if context_parallel_size > 1:
        raise NotImplementedError(
            "FA3 sink attention is only supported for non-context-parallel paths."
        )
    if use_rr_flash_attention:
        raise NotImplementedError(
            "FA3 sink attention does not support refined recompute yet."
        )
    if flashmask_use_varlen:
        raise NotImplementedError(
            "FA3 sink attention does not support flashmask_use_varlen yet."
        )
    # FA3 dense sink only supports equal q/k/v sequence lengths. Raise here
    # instead of falling through to avoid silently changing KV-cache decode behavior.
    if startend_row_indices is None and (
        query.shape[1] != key.shape[1] or key.shape[1] != value.shape[1]
    ):
        raise NotImplementedError(
            "FA3 sink attention does not support KV-cache decode with unequal q/k/v sequence lengths yet, "
            f"got q_len={query.shape[1]}, k_len={key.shape[1]}, v_len={value.shape[1]}."
        )
    if startend_row_indices is None:
        startend_row_indices = _dense_mask_to_startend_row_indices(
            attention_mask, query, key, causal
        )
    return True, startend_row_indices


def _stable_logaddexp(lhs, rhs):
    max_value = paddle.maximum(lhs, rhs)
    return max_value + paddle.log(
        paddle.exp(lhs - max_value) + paddle.exp(rhs - max_value)
    )


def _merge_kv_grad(grad_repeated, original, num_key_value_groups):
    if (
        num_key_value_groups > 1
        and grad_repeated.shape[2] == original.shape[2] * num_key_value_groups
    ):
        batch, seq_len, num_kv_heads, head_dim = original.shape
        return grad_repeated.reshape(
            [batch, seq_len, num_kv_heads, num_key_value_groups, head_dim]
        ).sum(axis=3)
    return grad_repeated


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
    q_head_dim = query.shape[-1]
    v_head_dim = value.shape[-1]
    assert seq_k == seq_v, (
        f"FlashAttention requires equal sequence lengths: seq_k={seq_k}, seq_v={seq_v}"
    )

    if q_head_dim != v_head_dim:
        raise NotImplementedError(
            "FA3 sink attention requires query/key and value head_dim to match, "
            f"got q/k={q_head_dim}, v={v_head_dim}."
        )

    fa_version = _get_fa_version(query.shape[-1])

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
        if dropout > 0.0 and training:
            raise NotImplementedError(
                "FA3 sink attention does not support attention_dropout > 0 yet."
            )
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
        if _flash_attn_fwd is None:
            raise AssertionError(
                "FA4 flash attention is not available, may be due to paddlefleet_ops version"
            )
        out, lse = _flash_attn_fwd(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            causal=causal,
            return_lse=True,
            startend_row_indices=None,
            pack_gqa=False,
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
        if _flash_attn_bwd is None:
            raise AssertionError(
                "FA4 flash attention backward is not available, may be due to paddlefleet_ops version"
            )
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
):
    """Dispatch FlashMask attention forward. Only FlashMask v1 doesn't support
    custom softmax_scale.
    """
    fa_version = _get_fa_version(query.shape[-1])

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
    elif fa_version == 3:
        if dropout > 0.0 and training:
            raise NotImplementedError(
                "FA3 sink attention with startend_row_indices does not support attention_dropout > 0 yet."
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
        if _flash_attn_fwd is None:
            raise AssertionError(
                "FA4 flashmask attention is not available, may be due to paddlefleet_ops version"
            )
        output, log_sum_exp = _flash_attn_fwd(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            causal=causal,
            return_lse=True,
            startend_row_indices=startend_row_indices,
            pack_gqa=False,
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
            sig_params = inspect.signature(
                paddle.nn.functional.flashmask_attention
            ).parameters
            if "group" in sig_params:
                grad_q, grad_k, grad_v = _C_ops.flashmask_attention_v2_grad(
                    query,
                    key,
                    value,
                    output,
                    lse,
                    startend_row_indices,
                    None,  # block_mask
                    grad_output,
                    softmax_scale,
                    causal,
                    0,  # rank
                    1,  # nranks
                )
            elif "block_mask" in sig_params:
                grad_q, grad_k, grad_v = _C_ops.flashmask_attention_v2_grad(
                    query,
                    key,
                    value,
                    output,
                    lse,
                    startend_row_indices,
                    None,  # block_mask
                    grad_output,
                    softmax_scale,
                    causal,
                )
            else:
                grad_q, grad_k, grad_v = _C_ops.flashmask_attention_v2_grad(
                    query,
                    key,
                    value,
                    output,
                    lse,
                    startend_row_indices,
                    grad_output,
                    softmax_scale,
                    causal,
                )
        else:
            raise AssertionError(
                "flashmask_attention_v2_grad is not supported, may be due to paddle version"
            )
    elif fa_version == 4:
        if _flash_attn_bwd is None:
            raise AssertionError(
                "FA4 flash attention backward is not available, may be due to paddlefleet_ops version"
            )
        if startend_row_indices is not None:
            flashmask_info = _FlashMaskInfoPaddle(
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


class FlashMaskSinkPyLayer(PyLayer):
    """FlashAttention/FlashMask with 1F1B sink correction.

    The underlying FA/FlashMask kernel remains unchanged. Forward first runs the
    no-sink kernel, then folds the sink logit into the returned output and LSE.
    Backward reuses the original kernel backward by passing the corrected output
    and corrected LSE, so q/k/v need only one attention backward.
    """

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
        if head_dim_q != head_dim_v:
            raise ValueError(
                "attention sink requires value head_dim to match query/key "
                f"head_dim, got q/k={head_dim_q}, v={head_dim_v}"
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
            raw_output, lse_raw = _flash_attention_forward_dispatch(
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
            raw_output, lse_raw = _flashmask_attention_forward_dispatch(
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

        # Compat with old LSE shape (seqlen_q_rounded).
        if lse_raw.shape[-1] != seq_len:
            lse_raw = lse_raw[:, :, :seq_len]

        lse_raw_fp32 = lse_raw.astype("float32")
        sink_bhs = sink.astype("float32").reshape([1, num_heads, 1])
        sink_lse = _stable_logaddexp(lse_raw_fp32, sink_bhs)
        multiplier = paddle.exp(lse_raw_fp32 - sink_lse)
        final_output = raw_output * multiplier.transpose([0, 2, 1]).unsqueeze(
            -1
        )
        final_output = final_output.to(origin_dtype)

        ctx.save_for_backward(
            query,
            key,
            value,
            sink,
            attention_mask,
            final_output,
            sink_lse,
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

        return final_output

    @staticmethod
    def backward(ctx, grad_output):
        (
            query,
            key,
            value,
            sink,
            attention_mask,
            final_output,
            sink_lse,
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
        grad_output_for_kernel = grad_output.to(query.dtype)

        if startend_row_indices is None:
            grad_q, grad_k_repeated, grad_v_repeated = (
                _flash_attention_backward_dispatch(
                    grad_output_for_kernel,
                    query,
                    key_states,
                    value_states,
                    final_output,
                    sink_lse,
                    dropout=dropout,
                    attention_mask=attention_mask,
                    causal=causal,
                    softmax_scale=scale,
                )
            )
        else:
            grad_q, grad_k_repeated, grad_v_repeated = (
                _flashmask_attention_backward_dispatch(
                    grad_output_for_kernel,
                    query,
                    key_states,
                    value_states,
                    final_output,
                    sink_lse,
                    startend_row_indices,
                    dropout,
                    causal,
                    scale,
                )
            )

        grad_k = _merge_kv_grad(grad_k_repeated, key, num_key_value_groups)
        grad_v = _merge_kv_grad(grad_v_repeated, value, num_key_value_groups)

        if sink.stop_gradient:
            grad_sink = None
        else:
            delta = paddle.sum(
                final_output.astype("float32") * grad_output.astype("float32"),
                axis=-1,
            )
            delta = delta.transpose([0, 2, 1])
            sink_prob = paddle.exp(
                sink.astype("float32").reshape([1, -1, 1])
                - sink_lse.astype("float32")
            )
            grad_sink = -(sink_prob * delta).sum(axis=0).sum(axis=1)

        if query.dtype != grad_q.dtype:
            grad_q = grad_q.cast(query.dtype)
        if key.dtype != grad_k.dtype:
            grad_k = grad_k.cast(key.dtype)
        if value.dtype != grad_v.dtype:
            grad_v = grad_v.cast(value.dtype)
        if grad_sink is not None and grad_sink.dtype != sink.dtype:
            grad_sink = grad_sink.cast(sink.dtype)

        if startend_row_indices is None:
            return grad_q, grad_k, grad_v, grad_sink
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
    training=True,
):
    """Unified attention forward with Sink mechanism.

    Automatically chooses between FlashAttention and FlashMask based on
    ``startend_row_indices``. Shapes follow the ``[B, S, H, D]`` layout
    (same as paddlefleet's ``DotProductAttention``).

    Args:
        q: Query tensor ``[B, S, H_q, D]``
        k: Key tensor ``[B, S, H_kv, D]``
        v: Value tensor ``[B, S, H_kv, D_v]``
        sink: Sink parameter tensor ``[H_q]``
        attention_mask: Dense mask, only supported for FA2 path.
        startend_row_indices: If given, route to FlashMask.
        dropout_p: Dropout probability.
        softmax_scale: Custom softmax scaling factor.
        causal: Whether to apply causal masking.
        training: Whether to apply training-only dropout behavior.
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
        training=training,
        softmax_scale=softmax_scale,
    )
