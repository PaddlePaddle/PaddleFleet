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

The fused cuTile proj_rms forward (``fused_mhc_kernels._ct_proj_rms_fwd_kernel``)
accumulates sum_sq with an F32x2 pairwise tree over 128-wide k tiles and keeps a
single chained ``ct.mma`` accumulator across all of K, while sglang's
``mhc_pre`` single-chain path (``mhc_pre_gemm_sqrsum_tilelang``) accumulates
with four stride-4 chain accumulators across the WHOLE of K (one ``Pipelined``
pass over ``K / hidden_block`` 256-wide blocks) and reduces 4 -> 1 with a tree
only once, at the very end. Both are mathematically equal but not bit-identical.

This module re-implements the training-side proj_rms forward in TileLang with
the SAME accumulation order as the sglang inference single-chain kernel used by
the train/infer alignment probe when the ``mhc_align_fleet`` flag is on
(``mhc_pre``: ``num_tokens <= 2048`` path OR ``mhc_align_fleet=True``):
``mhc_pre_gemm_sqrsum_tilelang`` accumulates one 256-wide block at a time with
four stride-4 accumulators and folds the 4->1 tree at the end of the chain.
Only the forward is re-implemented; the backward stays on the cuTile kernel,
which is mathematically equivalent for any of these orders.

Note the alignment is one-directional and only concerns accumulation order:
``eps`` and the sinkhorn ``exp`` implementation are NOT changed here -- the
inference side already carries training's forms on ``align-train-infer-base``
(``mhc_align_fleet`` -> ``1/(sqrt(ss)/sqrt(K) + eps)`` for ernie, i.e. the
cuTile formula; ``exp2``, i.e. the cuTile sinkhorn), so the training-side
``r``/``norm`` formulas in this kernel are the cuTile ones verbatim and only
the accumulation order differs from ``_ct_proj_rms_fwd_kernel``.

Only used when ``ABLATION_INSPECT_TENSOR=1`` (train/infer bit-compare run); the
training hot path keeps the cuTile kernel.
"""

import functools
import math

import tilelang
from tilelang import language as T

# Same pass configuration as sglang's mhc_pre_gemm_sqrsum_* kernels so the
# compiler choices (warp specialization, TMA, ptxas) match the inference build.
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
        eps: RMS stability epsilon, same placement as the cuTile formula:
            ``r = 1 / (sqrt(ss) / sqrt(hc_hidden_size) + eps)`` (eps outside
            the sqrt). The inference side's ``mhc_align_fleet`` gate (ernie)
            produces the identical form, so no eps change is needed here.
        token_block / hidden_block / threads: tile shape, must match inference.

    Returns:
        A TileLang JIT function with signature
        ``kernel(x[M, K] bf16, fn[32, K] fp32) -> (proj, norm, r)`` where
        ``proj[n, hc_mult3]`` / ``norm[n, 1]`` / ``r[n, 1]`` are allocated by
        the kernel (``out_idx``) and returned, like the other PaddleFleet
        tilelang ops (e.g. ``sparse_mqa_fwd``).

    Accumulation order per token, identical to sglang inference
    (``mhc_pre_gemm_sqrsum_tilelang``, single chain -- no split-k)::

        out_frag = 0;  sq_part4[*, 0:4] = 0
        for pz in 0..hc_hidden_size/hidden_block-1:   # one ascending chain
            x_f = fp32(bf16(x[pz*hidden_block : ...]))
            for jj in 0..hidden_block/4-1:            # serial, ascending
                for i, j in Parallel(token_block, 4):  # stride-4 accumulators
                    v = x_f[i, jj*4 + j]
                    sq_part4[i, j] += v * v
            T.gemm(x_f, fn[*, k], out_frag, transpose_B=True, clear_accum=False)
        sq_l = reduce_sum(sq_part4, dim=1)            # 4 -> 1 tree, once at end
        r = 1 / (sqrt(sq_l) / sqrt(hc_hidden_size) + eps)  # cuTile formula
        norm = sqrt(sq_l)                                      # for the cuTile backward
    """
    assert hc_mult3 <= 32, f"hc_mult3 must be <= 32, got {hc_mult3}"
    assert hc_hidden_size % hidden_block == 0, (
        f"hc_hidden_size={hc_hidden_size} must be a multiple of "
        f"hidden_block={hidden_block}"
    )
    # Static constant (K is a perfect square in practice: 16384 -> 128); matches
    # the cuTile ``v = norm_tile / ct.sqrt(K) + eps`` expression exactly.
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

            # Single ascending chain over all of K (no split-k): this is the
            # sglang ``mhc_pre_gemm_sqrsum_tilelang`` accumulation order.
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
                    # r uses the cuTile formula verbatim
                    # (1 / (sqrt(ss)/sqrt(K) + eps), eps outside the sqrt --
                    # the inference big_fuse's ``mhc_align_fleet`` gate for
                    # ernie produces the identical form, so the training side
                    # changes NOTHING about eps; only the accumulation order
                    # above differs from _ct_proj_rms_fwd_kernel).
                    r[t, 0] = 1.0 / (T.sqrt(sq_l[i]) / sqrt_k + eps)
                    norm[t, 0] = T.sqrt(sq_l[i])

    return _kernel
