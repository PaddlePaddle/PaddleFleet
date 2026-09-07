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

"""TileLang mHC proj_rms forward mirroring the sglang inference kernel.

The cuTile forward (``fused_mhc_kernels._ct_proj_rms_fwd_kernel``) accumulates
sum_sq with an F32x2 pairwise tree over 128-wide k tiles; sglang's ``mhc_pre``
(``mhc_pre_gemm_sqrsum_tilelang``) uses four stride-4 chain accumulators over
256-wide blocks and reduces 4 -> 1 only at the end. Equal math, different bits.
This module reproduces the sglang order so the two can be bit-compared. Only
the accumulation order is aligned: the ``r``/``norm`` formulas stay cuTile's,
which is what the inference ``mhc_align_fleet`` gate already produces.

Forward only; the backward keeps the cuTile kernel, which is order-invariant.
Used when ``ABLATION_INSPECT_TENSOR=1``; the hot path is unchanged.
"""

import functools
import math

import tilelang
from tilelang import language as T

# Must match sglang's mhc_pre_gemm_sqrsum_* so ptxas makes the same choices.
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
}


@functools.cache
@tilelang.jit(
    out_idx=[-3, -2, -1],
    pass_configs=_PASS_CONFIGS,
)
def mhc_align_proj_rms_fwd(
    hc_mult3: int,
    hc_hidden_size: int,
    eps: float,
    token_block: int = 32,
    hidden_block: int = 256,
    threads: int = 128,
):
    """Build the training-side proj_rms forward that mirrors sglang ``mhc_pre``.

    Args:
        hc_mult3: output width (24 for EB5; must be <= 32).
        hc_hidden_size: k dimension (16384 for EB5).
        eps: RMS epsilon, placed as in cuTile:
            ``r = 1 / (sqrt(ss) / sqrt(hc_hidden_size) + eps)``.
        token_block / hidden_block / threads: tile shape, must match inference.

    Returns:
        A TileLang JIT function
        ``kernel(x[M, K] bf16, fn[32, K] fp32) -> (proj, norm, r)``; the three
        outputs are allocated by the kernel via ``out_idx``.
    """
    assert hc_mult3 <= 32, f"hc_mult3 must be <= 32, got {hc_mult3}"
    assert hc_hidden_size % hidden_block == 0, (
        f"hc_hidden_size={hc_hidden_size} must be a multiple of "
        f"hidden_block={hidden_block}"
    )
    # Static constant, matching cuTile's ``norm_tile / ct.sqrt(K) + eps``.
    sqrt_k = float(math.sqrt(hc_hidden_size))

    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def _kernel(
        x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16),
        fn: T.Tensor((32, hc_hidden_size), T.float32),
        proj: T.Tensor((num_tokens, hc_mult3), T.float32),
        norm: T.Tensor((num_tokens, 1), T.float32),
        r: T.Tensor((num_tokens, 1), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(num_tokens, token_block), threads=threads
        ) as px:
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sq_part4 = T.alloc_fragment((token_block, 4), T.float32)
            T.clear(out_frag)
            T.clear(sq_part4)

            # Single ascending chain over all of K, no split-k.
            for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=2):
                x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
                fn_smem = T.alloc_shared((32, hidden_block), T.float32)

                T.annotate_layout(
                    {x_smem: tilelang.layout.make_swizzled_layout(x_smem)}
                )

                T.copy(x[px * token_block, pz * hidden_block], x_smem)
                T.copy(fn[0, pz * hidden_block], fn_smem)

                x_f16 = T.alloc_fragment(
                    (token_block, hidden_block), T.bfloat16
                )
                T.copy(x_smem, x_f16)
                x_f = T.alloc_fragment((token_block, hidden_block), T.float32)
                T.copy(x_f16, x_f)

                for jj in T.serial(hidden_block // 4):
                    for i, j in T.Parallel(token_block, 4):
                        v = x_f[i, jj * 4 + j]
                        sq_part4[i, j] += v * v

                T.gemm(
                    x_f,
                    fn_smem,
                    out_frag,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=False,
                )

            sq_l = T.alloc_fragment((token_block,), T.float32)
            T.reduce_sum(sq_part4, sq_l)

            for i, j in T.Parallel(token_block, 32):
                t = px * token_block + i
                if t < num_tokens and j < hc_mult3:
                    proj[t, j] = out_frag[i, j]
            for i in T.Parallel(token_block):
                t = px * token_block + i
                if t < num_tokens:
                    # cuTile formula verbatim, eps outside the sqrt.
                    r[t, 0] = 1.0 / (T.sqrt(sq_l[i]) / sqrt_k + eps)
                    norm[t, 0] = T.sqrt(sq_l[i])

    return _kernel
