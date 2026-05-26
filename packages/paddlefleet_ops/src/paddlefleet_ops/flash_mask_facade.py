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

import inspect
from functools import partial

import paddle
from paddle import _C_ops

from . import is_flash_mask_available

if is_flash_mask_available():
    try:
        from .flash_mask import (
            flash_attention as _flash_attention,
            flashmask_attention as _flashmask_attention,
        )
    except (ImportError, ModuleNotFoundError):
        from .flash_mask.cute.interface import (
            flash_attention as _flash_attention,
            flashmask_attention as _flashmask_attention,
        )
    from .flash_mask.cute.flashmask_utils import FlashMaskInfoPaddle
    from .flash_mask.cute.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
    )
else:
    from paddle.nn.functional.flash_attention import (
        flash_attention as _flash_attention,
        flashmask_attention as _flashmask_attention,
    )

    _flash_attn_fwd = None
    _flash_attn_bwd = None
    FlashMaskInfoPaddle = None


def get_fa_version(
    head_dim: int,
    head_dim_v: int | None = None,
    startend_row_indices: paddle.Tensor | None = None,
) -> int:
    """Pick the FlashAttention version for the given head dims.

    Dispatch rules:
      * XPU device -> FA2.
      * Otherwise, respect ``FLAGS_flash_attn_version`` by default.
      * If ``fa_version == 3`` and deterministic is required, FA3 only supports
        ``head_dim <= 128``. For ``head_dim > 128``, fall back to FA2.
      * FA4 is only used when both ``hdim_ok`` and ``mask_ok`` hold:

        - ``hdim_ok``: one of
          * ``head_dim <= 128`` and ``head_dim_v <= 128``
          * ``head_dim == 192`` and ``head_dim_v == 128``
          * ``head_dim == 256`` and ``head_dim_v == 256``
        - ``mask_ok``: ``startend_row_indices is None`` or
          ``startend_row_indices.shape[-1] != 4``

        When ``startend_row_indices`` is not provided (``None``), ``mask_ok``
        is treated as ``True`` -- this covers the ``flash_attention`` path
        which has no mask tensor. Aligned with flash-attention ``interface.py``.

    Args:
        head_dim: Query/Key head dim (always equal).
        head_dim_v: Value head dim. Defaults to ``head_dim`` when not provided.
        startend_row_indices: FlashMask indices tensor. Pass ``None`` (default)
            for the plain ``flash_attention`` path where no mask check is needed.

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
    if fa_version == 3 and deterministic and head_dim > 128:
        return 2

    if fa_version == 4:
        _head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        fa4_hdim_ok = (
            (head_dim <= 128 and _head_dim_v <= 128)
            or (head_dim == 192 and _head_dim_v == 128)
            or (head_dim == 256 and _head_dim_v == 256)
        )
        fa4_mask_ok = (
            startend_row_indices is None
            or startend_row_indices.shape[-1] != 4
        )
        if not (fa4_hdim_ok and fa4_mask_ok):
            return 2

    return fa_version


def _need_value_padding(fa_version, q_head_dim, v_head_dim):
    """Determine if value needs padding to match query head dim."""
    if q_head_dim == v_head_dim:
        return False
    # FA4 natively supports q_head_dim=192, v_head_dim=128
    if fa_version == 4 and q_head_dim == 192 and v_head_dim == 128:
        return False
    return True


def _pad_value(value, q_head_dim):
    """Pad value tensor to match query head dim."""
    v_head_dim = value.shape[-1]
    bsz, q_len = value.shape[0], value.shape[1]
    value_padding = paddle.zeros(
        [bsz, q_len, value.shape[2], q_head_dim - v_head_dim],
        dtype=value.dtype,
    )
    return paddle.concat([value, value_padding], axis=-1)


def _get_flashmask_v2_sig_variant():
    """Detect which signature variant of flashmask_attention_v2 is available.

    Returns one of: "group", "block_mask", or "basic".
    """
    from paddle.nn.functional.flash_attention import flashmask_attention as _fm

    sig_params = inspect.signature(_fm).parameters
    if "group" in sig_params:
        return "group"
    elif "block_mask" in sig_params:
        return "block_mask"
    else:
        return "basic"


# ---------------------------------------------------------------------------
# Unified dispatch: forward
# ---------------------------------------------------------------------------


def flash_attn_dispatch_fwd(
    q,
    k,
    v,
    startend_row_indices=None,
    causal=False,
    dropout=0.0,
    training=True,
    return_softmax=False,
    softmax_scale=None,
):
    """Unified forward dispatch for FlashAttention across all versions.

    Handles:
      - FA version selection via get_fa_version
      - Value padding when q_head_dim != v_head_dim (except FA4 192/128)
      - FA2/3/4 kernel dispatch

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_heads_kv, head_dim]
        v: Value tensor [batch, seq_len, num_heads_kv, head_dim_v]
        startend_row_indices: FlashMask indices or None for plain attention
        causal: Whether to use causal masking
        dropout: Dropout probability (FA2 only)
        training: Whether in training mode
        return_softmax: Whether to return softmax result (FA2 only)
        softmax_scale: Custom softmax scale (default 1/sqrt(head_dim))

    Returns:
        dict with keys:
          - "output": attention output tensor
          - "softmax_lse": log-sum-exp tensor
          - "seed_offset": seed offset tensor (FA2 only, None otherwise)
          - "result_softmax": softmax result (FA2 + return_softmax only)
          - "fa_version": int, the version actually used
          - "causal": bool
          - "dropout": float
          - "need_value_padding": bool
          - "v_head_dim": original v head dim before padding
    """
    q_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]

    fa_version = get_fa_version(q_head_dim, v_head_dim, startend_row_indices)

    need_pad = _need_value_padding(fa_version, q_head_dim, v_head_dim)
    if need_pad:
        v = _pad_value(v, q_head_dim)

    result = {
        "fa_version": fa_version,
        "causal": causal,
        "dropout": dropout,
        "need_value_padding": need_pad,
        "v_head_dim": v_head_dim,
        "seed_offset": None,
        "result_softmax": None,
    }

    if fa_version == 2:
        if startend_row_indices is not None:
            # flashmask path
            (output, result_softmax, softmax_lse, seed_offset) = (
                _C_ops.flashmask_attention(
                    q, k, v,
                    startend_row_indices,
                    None,
                    dropout,
                    causal,
                    return_softmax,
                    not training,
                    "",
                )
            )
        else:
            # plain flash attention path
            (output, result_softmax, softmax_lse, seed_offset) = (
                _C_ops.flash_attn(
                    q, k, v,
                    None,
                    None,
                    dropout,
                    causal,
                    return_softmax,
                    not training,
                    "",
                )
            )
        result["output"] = output
        result["softmax_lse"] = softmax_lse
        result["seed_offset"] = seed_offset
        result["result_softmax"] = result_softmax

    elif fa_version == 3:
        _scale = softmax_scale if softmax_scale is not None else q_head_dim ** (-0.5)

        if startend_row_indices is not None:
            # flashmask v2 path
            variant = _get_flashmask_v2_sig_variant()
            if variant == "group":
                (output, softmax_lse) = _C_ops.flashmask_attention_v2(
                    q, k, v,
                    startend_row_indices,
                    None,  # block_mask
                    None,  # nvshmem unique id
                    _scale,
                    causal,
                    0,  # rank
                    1,  # nranks
                )
            elif variant == "block_mask":
                (output, softmax_lse) = _C_ops.flashmask_attention_v2(
                    q, k, v,
                    startend_row_indices,
                    None,  # block_mask
                    _scale,
                    causal,
                )
            else:
                (output, softmax_lse) = _C_ops.flashmask_attention_v2(
                    q, k, v,
                    startend_row_indices,
                    _scale,
                    causal,
                )
        else:
            # plain flash attention v3 path
            (output, softmax_lse) = _C_ops.flash_attn_v3(
                q, k, v,
                None,
                None,
                None,
                None,
                _scale,
                causal,
                -1,
                -1,
                0.0,
                1,
                False,
                False,
                0,
            )
        result["output"] = output
        result["softmax_lse"] = softmax_lse

    elif fa_version == 4:
        (output, softmax_lse) = _flash_attn_fwd(
            q, k, v,
            causal=causal,
            return_lse=True,
            startend_row_indices=startend_row_indices,
            pack_gqa=False,
            softmax_scale=softmax_scale,
        )
        result["output"] = output
        result["softmax_lse"] = softmax_lse

    else:
        raise ValueError(f"Invalid flash attention version: {fa_version}")

    return result


# ---------------------------------------------------------------------------
# Unified dispatch: backward
# ---------------------------------------------------------------------------


def flash_attn_dispatch_bwd(
    q,
    k,
    v,
    output,
    grad,
    softmax_lse,
    fa_version,
    startend_row_indices=None,
    seed_offset=None,
    dropout=0.0,
    causal=False,
):
    """Unified backward dispatch for FlashAttention across all versions.

    Args:
        q: Query tensor (detached)
        k: Key tensor (detached)
        v: Value tensor (detached)
        output: Forward pass output (result_attention)
        grad: Gradient of output
        softmax_lse: Log-sum-exp from forward pass
        fa_version: FlashAttention version (2, 3, or 4) from forward pass
        startend_row_indices: FlashMask indices or None
        seed_offset: Seed offset tensor (FA2 only)
        dropout: Dropout probability (FA2 only)
        causal: Whether causal masking was used

    Returns:
        tuple: (q_grad, k_grad, v_grad)
    """
    if fa_version == 2:
        if startend_row_indices is not None:
            q_grad, k_grad, v_grad = _C_ops.flashmask_attention_grad(
                q, k, v,
                startend_row_indices,
                output,
                softmax_lse,
                seed_offset,
                grad,
                dropout,
                causal,
            )
        else:
            q_grad, k_grad, v_grad = _C_ops.flash_attn_grad(
                q, k, v,
                output,
                softmax_lse,
                seed_offset,
                None,  # attn_mask (dense mask)
                grad,
                dropout,
                causal,
            )

    elif fa_version == 3:
        _scale = q.shape[-1] ** (-0.5)

        if startend_row_indices is not None:
            variant = _get_flashmask_v2_sig_variant()
            if variant == "group":
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q, k, v,
                    output,
                    softmax_lse,
                    startend_row_indices,
                    None,  # block_mask
                    grad,
                    _scale,
                    causal,
                    0,  # rank
                    1,  # nranks
                )
            elif variant == "block_mask":
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q, k, v,
                    output,
                    softmax_lse,
                    startend_row_indices,
                    None,  # block_mask
                    grad,
                    _scale,
                    causal,
                )
            else:
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q, k, v,
                    output,
                    softmax_lse,
                    startend_row_indices,
                    grad,
                    _scale,
                    causal,
                )
        else:
            q_grad, k_grad, v_grad = _C_ops.flash_attn_v3_grad(
                q, k, v,
                output,
                softmax_lse,
                grad,
                _scale,
                causal,
                -1,  # window_size_left
                -1,  # window_size_right
                0.0,  # softcap
                0,  # sm_margin
            )

    elif fa_version == 4:
        if startend_row_indices is not None:
            flashmask_info = FlashMaskInfoPaddle(
                startend_row_indices=startend_row_indices,
                is_causal=causal,
            )
        else:
            flashmask_info = None
        q_grad, k_grad, v_grad = _flash_attn_bwd(
            q, k, v,
            output,
            grad,
            softmax_lse,
            flashmask_info,
            causal=causal,
            deterministic=bool(
                paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                    "FLAGS_cudnn_deterministic"
                ]
            ),
        )

    else:
        raise ValueError(f"Invalid flash attention version: {fa_version}")

    return q_grad, k_grad, v_grad


# ---------------------------------------------------------------------------
# Public API wrappers (preserved for backward compatibility)
# ---------------------------------------------------------------------------


def flashmask_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    startend_row_indices: paddle.Tensor | None = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: int | tuple | None = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: paddle.Tensor | None = None,
    rng_name: str = "",
    training: bool = True,
    name: str | None = None,
    softmax_scale: float | None = None,
    block_mask: paddle.Tensor | None = None,
    use_varlen: bool = False,
):
    if use_varlen:
        assert (
            "use_varlen" in inspect.signature(_flashmask_attention).parameters
        ), "The flash_mask installed does not support use_varlen"

    fa_version = get_fa_version(query.shape[-1], value.shape[-1], startend_row_indices)

    bsz, q_len, num_heads, q_head_dim = query.shape
    v_head_dim = value.shape[-1]

    need_value_padding = _need_value_padding(fa_version, q_head_dim, v_head_dim)

    if need_value_padding:
        value = _pad_value(value, q_head_dim)

    if use_varlen:
        flashmask_attention_func = partial(
            _flashmask_attention, use_varlen=True
        )
    else:
        flashmask_attention_func = _flashmask_attention

    outs = flashmask_attention_func(
        query=query,
        key=key,
        value=value,
        startend_row_indices=startend_row_indices.clone(),
        dropout=dropout,
        causal=causal,
        window_size=window_size,
        return_softmax_lse=return_softmax_lse,
        return_seed_offset=return_seed_offset,
        fixed_seed_offset=fixed_seed_offset,
        rng_name=rng_name,
        training=training,
        name=name,
        softmax_scale=softmax_scale,
        block_mask=block_mask,
    )

    if return_softmax_lse:
        attn_out, lse = outs
        lse = lse.reshape([bsz, q_len])
    else:
        attn_out = outs

    if need_value_padding:
        attn_out = attn_out[..., :v_head_dim]

    attn_out = attn_out.reshape([bsz, q_len, num_heads, v_head_dim])

    if return_softmax_lse:
        return [attn_out, lse]
    else:
        return attn_out


def flash_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
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
    bsz, q_len, num_heads, q_head_dim = query.shape
    v_head_dim = value.shape[-1]
    need_value_padding = q_head_dim != v_head_dim

    if need_value_padding:
        value = _pad_value(value, q_head_dim)

    attn_output, softmax_result = _flash_attention(
        query=query,
        key=key,
        value=value,
        dropout=dropout,
        causal=causal,
        return_softmax=return_softmax,
        fixed_seed_offset=fixed_seed_offset,
        rng_name=rng_name,
        training=training,
        name=name,
        softmax_scale=softmax_scale,
    )

    if need_value_padding:
        attn_output = attn_output[..., :v_head_dim]

    attn_output = attn_output.reshape([bsz, q_len, num_heads, v_head_dim])

    return attn_output, softmax_result


__all__ = [
    "flashmask_attention",
    "flash_attention",
    "get_fa_version",
    "flash_attn_dispatch_fwd",
    "flash_attn_dispatch_bwd",
    "_need_value_padding",
    "_pad_value",
]
