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

import functools

import paddle


def get_quant_func(
    fp8_recipe,
    input_trans=False,
    out_scale_trans=False,
    pow2_scale=False,
    use_ue8m0=False,
):
    """Build ``inp_quant_func`` / ``weight_quant_func`` for a given recipe.

    Both quant callables accept a raw tensor. ``weight_quant_func`` returns
    a 4-tuple ``(fp8_bwd, scale_bwd, fp8_fwd, scale_fwd)`` where the fwd
    orientation is the one ``fp8_gemm_nt`` expects as its B operand and the
    bwd orientation is stored for reuse in dgrad. Both orientations are
    produced by a single kernel launch via ``input_transpose=True``.

    When ``use_ue8m0=True`` scales are int32-packed pow2 (UE8M0), which
    lets DeepGEMM's SM100 dispatch take the INT / (1, 1, 128) branch. The
    UE8M0 scale layout is not invariant under ``.T``, so both orientations
    must be emitted at quant time rather than derived by transpose.

    ``_mn_major`` applies a stride-only ``.T`` view (no memcpy) to satisfy
    DeepGEMM's ``check_sf_layout`` requirement (``stride(-2) == 1``).
    """
    if fp8_recipe != "blockwise":
        raise ValueError(
            f"fp8_recipe {fp8_recipe} is not supported. Supported recipes are blockwise."
        )

    def _mn_major(scale):
        return scale.T if scale is not None else None

    def _cached_weight_result(x):
        """Return a cached 4-tuple if ``x`` was pre-quantized, else None.

        The cache stores the forward-orientation fp8 tensor plus both fwd
        and bwd scales; the backward-orientation fp8 is derived on demand
        via ``.T.contiguous()`` in the caller.
        """
        fp8_fwd = getattr(x, "fp8_weight_fwd", None)
        if fp8_fwd is None:
            return None
        scale_fwd = getattr(x, "fp8_scale_fwd", None)
        scale_bwd = getattr(x, "fp8_scale_bwd", None)
        if scale_fwd is None or scale_bwd is None:
            return None
        return None, scale_bwd, fp8_fwd, scale_fwd

    _quant = paddle.incubate.nn.functional.fp8_quant_blockwise

    if use_ue8m0:
        if input_trans:

            def inp_quant_func(x):
                fp8, scale, fp8_t, scale_t = _quant(
                    x,
                    output_scale_transpose=True,
                    quant_method="1x128",
                    input_transpose=True,
                    using_pow2_scale=True,
                    using_ue8m0_scale=True,
                )
                return fp8, _mn_major(scale), fp8_t, _mn_major(scale_t)
        else:

            def inp_quant_func(x):
                fp8, scale = _quant(
                    x,
                    output_scale_transpose=True,
                    quant_method="1x128",
                    input_transpose=False,
                    using_pow2_scale=True,
                    using_ue8m0_scale=True,
                )[:2]
                return fp8, _mn_major(scale)

        def weight_quant_func(x):
            cached = _cached_weight_result(x)
            if cached is not None:
                return cached
            fp8_bwd, scale_bwd, fp8_fwd, scale_fwd = _quant(
                x,
                output_scale_transpose=True,
                quant_method="128x128",
                input_transpose=True,
                using_pow2_scale=True,
                using_ue8m0_scale=True,
            )
            return (
                fp8_bwd,
                _mn_major(scale_bwd),
                fp8_fwd,
                _mn_major(scale_fwd),
            )
    else:
        inp_quant_func = functools.partial(
            _quant,
            output_scale_transpose=out_scale_trans,
            quant_method="1x128",
            input_transpose=input_trans,
            using_pow2_scale=pow2_scale,
        )

        def weight_quant_func(x):
            cached = _cached_weight_result(x)
            if cached is not None:
                return cached
            fp8_bwd, scale_bwd, fp8_fwd, scale_fwd = _quant(
                x,
                output_scale_transpose=out_scale_trans,
                quant_method="128x128",
                input_transpose=True,
                using_pow2_scale=pow2_scale,
            )
            if out_scale_trans:
                scale_bwd = _mn_major(scale_bwd)
                scale_fwd = _mn_major(scale_fwd)
            return fp8_bwd, scale_bwd, fp8_fwd, scale_fwd

    return inp_quant_func, weight_quant_func
