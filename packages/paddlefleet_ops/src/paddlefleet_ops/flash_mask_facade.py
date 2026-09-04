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

import paddle

from . import is_flash_mask_available

FA3_BACKEND_CHOICES = ("cutedsl", "cpp")

# Which kernel FA3 FlashMask runs on. Not an environment variable: the value
# comes from ``TransformerConfig.flash_attn_fa3_backend`` and is handed over by
# :func:`set_fa3_backend` in ``__post_init__``. An env switch would be fixed
# before ``from_config`` runs, so no config could override it and the choice
# would neither be validated nor stored alongside the config.
#
# Resolved once per process. Version dispatch, backend pick
# (``uses_cutedsl_backend``) and value padding (``needs_value_padding``) all read
# this module global, and ``PyLayer.forward``/``backward`` read it independently,
# so a mid-process flip would pair two different kernels -- and a second config
# would reroute a model that is already running. ``set_fa3_backend`` therefore
# accepts the same choice any number of times and rejects a different one;
# ``_FA3_BACKEND`` is what it has resolved so far (``None`` before the first
# call).
_FA3_BACKEND = None
FLASHMASK_FA3_USE_CUTEDSL = True


def set_fa3_backend(backend: str) -> None:
    """Point FA3 FlashMask at ``backend``; called from config, not by user code.

    Resolved once per process: repeating the same choice is a no-op, asking for a
    different one raises instead of silently rerouting kernels behind models that
    are already running.

    Args:
        backend: One of ``FA3_BACKEND_CHOICES``. ``"cutedsl"`` selects the CuTe
            DSL kernels, ``"cpp"`` the C++ FA3 kernels behind Paddle's built-in
            FlashMask op.

    Raises:
        ValueError: If ``backend`` is not a known choice -- the string is parsed
            here only, so every consumer reads a plain bool -- or if it differs
            from the choice already resolved in this process.
    """
    global FLASHMASK_FA3_USE_CUTEDSL, _FA3_BACKEND
    if backend not in FA3_BACKEND_CHOICES:
        raise ValueError(
            f"unknown FA3 backend {backend!r}, expected one of "
            f"{list(FA3_BACKEND_CHOICES)}"
        )
    if _FA3_BACKEND is not None and _FA3_BACKEND != backend:
        raise ValueError(
            "flash_attn_fa3_backend cannot change within a process: already "
            f"{_FA3_BACKEND!r}, got {backend!r} -- a forward and its backward "
            "read the choice independently, so a mid-process flip would pair "
            "two different kernels. Give every TransformerConfig in this "
            "process the same value."
        )
    _FA3_BACKEND = backend
    FLASHMASK_FA3_USE_CUTEDSL = backend == "cutedsl"


def _reset_fa3_backend() -> None:
    """Drop the resolved choice so the next ``set_fa3_backend`` wins. Tests only.

    Not part of the public surface: production code resolves the backend exactly
    once, from ``TransformerConfig.flash_attn_fa3_backend``.
    """
    global FLASHMASK_FA3_USE_CUTEDSL, _FA3_BACKEND
    _FA3_BACKEND = None
    FLASHMASK_FA3_USE_CUTEDSL = True


def _dispatch_fa_version(
    head_dim: int,
    head_dim_v: int | None = None,
    startend_row_indices: paddle.Tensor | None = None,
) -> int:
    """Pick the FlashAttention version for the given head dims.

    Prefer the public ``get_fa_version``, which validates what this returns.

    Dispatch rules:
      * XPU device -> FA2.
      * Otherwise, respect ``FLAGS_flash_attn_version`` by default.
      * If ``fa_version == 3`` and ``FLASHMASK_FA3_USE_CUTEDSL`` is off, FA3 runs
        on the Paddle kernel, which only supports ``head_dim <= 128`` when
        deterministic is required. For ``head_dim > 128``, fall back to FA2. On
        the cutedsl backend FA3 needs no such degrade -- its backward has an
        ordered-accumulation variant, so deterministic holds for every head dim
        in the cutedsl whitelist below.
      * FA4 is only used when both ``hdim_ok`` and ``mask_ok`` hold:

        - ``hdim_ok``: one of
          * ``head_dim <= 128`` and ``head_dim_v <= 128``
          * ``head_dim == 192`` and ``head_dim_v == 128``
          * ``head_dim == 256`` and ``head_dim_v == 256``
          * ``head_dim == 512`` and ``head_dim_v == 512``, unless deterministic
            is required
          * ``head_dim == 576`` and ``head_dim_v == 512``, unless deterministic
            is required -- both of these pairs take FA4's big-head-dim backward,
            which has no ordered-accumulation variant.
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

    if fa_version == 4 or (fa_version == 3 and FLASHMASK_FA3_USE_CUTEDSL):
        if not is_flash_mask_available():
            return 2

    if FLASHMASK_FA3_USE_CUTEDSL:
        if fa_version in (3, 4):
            _head_dim_v = head_dim_v if head_dim_v is not None else head_dim
            cutedsl_hdim_ok = (
                (head_dim <= 128 and _head_dim_v <= 128)
                or (head_dim == 192 and _head_dim_v == 128)
                or (head_dim == 256 and _head_dim_v == 256)
            )
            # FA4 additionally supports larger head dims (non-deterministic only)
            if fa_version == 4 and not deterministic:
                cutedsl_hdim_ok = cutedsl_hdim_ok or (
                    (head_dim == 512 and _head_dim_v == 512)
                    or (head_dim == 576 and _head_dim_v == 512)
                )
            if not cutedsl_hdim_ok:
                return 2
            # Only FA4 is restricted here: the FA4 kernel does not serve a
            # 4-column flashmask, while FA3 on cutedsl accepts it. Not a missing
            # FA3 branch.
            fa4_mask_ok = (
                startend_row_indices is None
                or startend_row_indices.shape[-1] != 4
            )
            if fa_version == 4:
                if not fa4_mask_ok:
                    return 2
    else:
        if fa_version == 3:
            if deterministic and head_dim > 128:
                return 2

        if fa_version == 4:
            _head_dim_v = head_dim_v if head_dim_v is not None else head_dim
            fa4_hdim_ok = (
                (head_dim <= 128 and _head_dim_v <= 128)
                or (head_dim == 192 and _head_dim_v == 128)
                or (head_dim == 256 and _head_dim_v == 256)
                # Both of these exceed 256 and so take FA4's big-head-dim backward,
                # which asserts ``not deterministic`` (``flash_mask/cute/
                # interface.py``: ``is_bigd_bwd`` -> "deterministic reduction is not
                # supported by big-headdim bwd"). Degrade instead of aborting, the
                # same way FA3 degrades above.
                or (
                    head_dim == 512 and _head_dim_v == 512 and not deterministic
                )
                or (
                    head_dim == 576 and _head_dim_v == 512 and not deterministic
                )
            )
            fa4_mask_ok = (
                startend_row_indices is None
                or startend_row_indices.shape[-1] != 4
            )
            if not (fa4_hdim_ok and fa4_mask_ok):
                return 2

    return fa_version


def get_fa_version(
    head_dim: int,
    head_dim_v: int | None = None,
    startend_row_indices: paddle.Tensor | None = None,
) -> int:
    """Pick the FlashAttention version, rejecting versions we cannot dispatch.

    Thin validating wrapper over ``_dispatch_fa_version``; see that function for
    the dispatch rules and argument meanings.

    Returns:
        The FlashAttention version to use (2, 3 or 4).

    Raises:
        ValueError: If the resolved version is not 2, 3 or 4 -- in practice this
            means ``FLAGS_flash_attn_version`` was set to something unsupported,
            since every degrade path returns 2.
    """
    fa_version = _dispatch_fa_version(
        head_dim, head_dim_v, startend_row_indices
    )
    if fa_version not in (2, 3, 4):
        raise ValueError(f"Invalid flash attention version: {fa_version}")
    return fa_version


def uses_cutedsl_backend(fa_version: int) -> bool:
    """Whether ``fa_version`` runs on the cutedsl backend.

    FA4 is cutedsl-only, FA3 depends on ``FLASHMASK_FA3_USE_CUTEDSL``, FA2 never
    uses it.

    Args:
        fa_version: The version returned by :func:`get_fa_version`.

    Returns:
        ``True`` when the call site must use the cutedsl kernels.
    """
    if fa_version == 3:
        return FLASHMASK_FA3_USE_CUTEDSL
    elif fa_version == 4:
        return True
    else:
        return False


def needs_value_padding(
    fa_version: int,
    head_dim: int,
    head_dim_v: int,
) -> bool:
    """Whether ``value`` must be zero-padded from ``head_dim_v`` to ``head_dim``.

    Args:
        fa_version: The version returned by :func:`get_fa_version`.
        head_dim: Query/Key head dim.
        head_dim_v: Value head dim.

    Returns:
        ``True`` when ``value`` needs padding.
    """
    fa3_native_pair = (
        FLASHMASK_FA3_USE_CUTEDSL
        and (fa_version == 3)
        and (head_dim == 192 and head_dim_v == 128)
    )
    fa4_native_pair = (fa_version == 4) and (
        (head_dim == 192 and head_dim_v == 128)
        or (head_dim == 576 and head_dim_v == 512)
    )
    return head_dim != head_dim_v and not (fa3_native_pair or fa4_native_pair)


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
    learnable_sink: paddle.Tensor | None = None,
):
    bsz, q_len, num_heads, q_head_dim = query.shape
    v_head_dim = value.shape[-1]
    fa_version = get_fa_version(q_head_dim, v_head_dim, startend_row_indices)

    if uses_cutedsl_backend(fa_version):
        try:
            from .flash_mask import (
                flashmask_attention as _flashmask_attention,
            )
        except (ImportError, ModuleNotFoundError):
            from .flash_mask.cute.interface import (
                flashmask_attention as _flashmask_attention,
            )
    else:
        from paddle.nn.functional.flash_attention import (
            flashmask_attention as _flashmask_attention,
        )

    if use_varlen:
        assert (
            "use_varlen" in inspect.signature(_flashmask_attention).parameters
        ), "The flash_mask installed does not support use_varlen"

    if learnable_sink is not None:
        if (
            "learnable_sink"
            not in inspect.signature(_flashmask_attention).parameters
        ):
            raise NotImplementedError(
                "learnable_sink (softmax sink) requires FA4 (cute backend); the "
                "installed flash_mask / current device (e.g. H-card fa2/fa3) does "
                "not support it. Disable the attention sink or run on a "
                "FA4-capable device."
            )

    need_value_padding = needs_value_padding(fa_version, q_head_dim, v_head_dim)

    if need_value_padding:
        value_padding = paddle.zeros(
            [*value.shape[:-1], q_head_dim - v_head_dim],
            dtype=value.dtype,
        )
        value = paddle.concat([value, value_padding], axis=-1)

    extra_kwargs = {}
    if use_varlen:
        # use_varlen is no longer used and will be removed soon.
        extra_kwargs["use_varlen"] = True
    if learnable_sink is not None:
        extra_kwargs["learnable_sink"] = learnable_sink

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
        **extra_kwargs,
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

    # startend_row_indices is None
    fa_version = get_fa_version(q_head_dim, v_head_dim)

    if uses_cutedsl_backend(fa_version):
        try:
            from .flash_mask import (
                flash_attention as _flash_attention,
            )
        except (ImportError, ModuleNotFoundError):
            from .flash_mask.cute.interface import (
                flash_attention as _flash_attention,
            )
    else:
        from paddle.nn.functional.flash_attention import (
            flash_attention as _flash_attention,
        )

    need_value_padding = needs_value_padding(fa_version, q_head_dim, v_head_dim)

    if need_value_padding:
        value_padding = paddle.zeros(
            [*value.shape[:-1], q_head_dim - v_head_dim],
            dtype=value.dtype,
        )
        value = paddle.concat([value, value_padding], axis=-1)

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
    "FA3_BACKEND_CHOICES",
    "flashmask_attention",
    "flash_attention",
    "get_fa_version",
    "needs_value_padding",
    "set_fa3_backend",
    "uses_cutedsl_backend",
]
