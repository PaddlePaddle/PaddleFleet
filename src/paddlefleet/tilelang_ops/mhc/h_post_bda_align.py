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

"""TileLang mHC h_post_bda forward mirroring the sglang inference kernel.

The post site computes ``out = comb^T @ res + post (outer) x`` in an fp32
accumulator with a single bf16 store::

    out[i][h] = post[i] * x[h] + sum_{j=0..3} comb[j][i] * res[j][h]

The cuTile kernel (``fused_mhc_kernels._ct_hpb_fwd_kernel``) and sglang's
``mhc_post_tilelang`` compile to the same instruction budget yet differ in bits:
``post*x + comb[0]*res[0]`` has two multiplies but one FFMA slot, so one of them
must round as a standalone FMUL. cuTile rounds ``post*x``, ptxas rounds
``comb[0]*res[0]``. Both are legal ``-fmad=true`` contractions, ~1 fp32 ULP
apart, visible in bf16 only on rounding ties. So this module is the sglang
kernel body verbatim, inheriting its contraction choice -- do not "clean it up",
any edit risks changing what ptxas contracts and silently losing the alignment.

Forward only; the backward keeps the cuTile kernel, which is bilinear and hence
invariant to both the order and the contraction choice.
Used when ``ABLATION_INSPECT_TENSOR=1``; the hot path is unchanged.
"""

import math

import tilelang
from tilelang import language as T

# Must match sglang's mhc_post_tilelang so ptxas makes the same choices.
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
}

_ENABLE_PDL = False


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_post_tilelang(
    a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
):
    """sglang ``mhc_post_tilelang`` verbatim (PDL intrinsics excepted).

    Follows the sglang convention rather than PaddleFleet's: tensors are
    annotated inline and the output ``x`` is caller-allocated, not ``out_idx``.

    Args:
        a: [num_tokens, hc, hc] fp32 ``comb`` (training ``h_res``), indexed
            ``a[j, i]`` -- j input stream, i output stream.
        b: [num_tokens, hc, hidden] bf16 ``res`` (``original_residual``); the
            cuTile and TileLang index conventions already agree, no transpose.
        c: [num_tokens, hc] fp32 ``post`` (``h_post``), trailing 1 squeezed.
        d: [num_tokens, hidden] bf16 ``x``, the attn/mlp output.
        x: [num_tokens, hc, hidden] bf16 OUTPUT, allocated by the caller.
        hc: number of residual streams (4 for EB5).
        hidden: channels per stream (4096 for EB5).
        n_thr / h_blk: tile shape, must match inference.
    """
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)
    b: T.Tensor((n, hc, h), T.bfloat16)
    c: T.Tensor((n, hc), T.float32)
    d: T.Tensor((n, h), T.bfloat16)
    x: T.Tensor((n, hc, h), T.bfloat16)

    ENABLE_PDL = _ENABLE_PDL
    with T.Kernel(n, threads=n_thr) as i_n:
        if ENABLE_PDL:
            T.pdl_sync()

        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            # fmt: off
            # Left byte-identical to sglang's mhc_post_tilelang: the expression
            # order below is what the FMA contraction alignment depends on.
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.serial(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
            # fmt: on
            T.copy(x_local, x_shared)

            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])

        if ENABLE_PDL:
            T.pdl_trigger()
