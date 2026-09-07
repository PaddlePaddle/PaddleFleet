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

The mHC post site computes, for every token, output stream ``i`` and channel
``h``::

    out[i][h] = post[i] * x[h] + sum_{j=0..3} comb[j][i] * res[j][h]

i.e. ``out = comb^T @ res + post (outer) x``, in an fp32 accumulator with a
single bf16 round-to-nearest-even store at the end.

The training cuTile kernel (``fused_mhc_kernels._ct_hpb_fwd_kernel``) and the
sglang inference TileLang kernel (``mhc_post_tilelang``) implement the SAME
expression with the SAME rounding budget -- both compile to one multiply plus
four fused multiply-adds per output, zero plain adds, and the identical
``F2FP.BF16.F32.PACK_AB`` store -- yet they are not bit-identical. The reason is
narrow: ``post*x + comb[0]*res[0]`` contains two multiplies but only one FFMA
slot, so exactly one of them must be issued as a standalone (rounding) FMUL and
the other is absorbed exactly into the FFMA. cuTile's codegen absorbs
``comb[0]*res[0]`` and rounds ``post*x``; TileLang -> nvcc -> ptxas does the
opposite. Both are legal ``-fmad=true`` contractions -- IEEE-754 addition is
commutative and bit-exact, so this is a contraction choice, not a reassociation,
and neither side is "more correct". The gap is ~1 fp32 ULP and is invisible in
bf16 except for elements sitting on a bf16 rounding tie.

Brute-forcing all 120 orderings of the five product terms against the actual
run's dumps gives a unique consistent solution across both the attn and the ffn
segment: inference evaluates ``comb[0]*res[0], post*x, comb[1], comb[2],
comb[3]`` and training evaluates ``post*x, comb[0], comb[1], comb[2],
comb[3]``.

This module therefore does not re-derive anything: it is the sglang kernel body
verbatim, so that the training side goes through the same TileLang -> TVM TIR ->
CUDA C -> nvcc -> ptxas chain and inherits the same contraction choice. Do not
"clean up" the kernel below -- any edit risks changing what ptxas contracts and
silently undoing the alignment.

Only the forward is re-implemented; the backward stays on the cuTile kernel.
The post-site forward is bilinear in ``(comb, res, post, x)``, so every gradient
is a plain sum and is mathematically invariant to the accumulation order and to
the FMA contraction choice -- only the forward output is bit-compared.

Only used when ``ABLATION_INSPECT_TENSOR=1`` (train/infer bit-compare run); the
training hot path keeps the cuTile kernel.
"""

import math

import tilelang
from tilelang import language as T

# Same pass configuration as sglang's mhc_post_tilelang so the compiler choices
# (warp specialization, TMA, ptxas register budget) match the inference build.
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

    Unlike ``proj_rms_align.mhc_align_proj_rms_fwd`` this is NOT a kernel
    factory: it follows the sglang convention where the tensors are the
    function's own parameters (annotated inline) and the output buffer ``x`` is
    passed in by the caller rather than allocated via ``out_idx``. Keeping that
    convention is part of the alignment -- it is what the generated TIR is
    compared against.

    Args:
        a: [num_tokens, hc, hc] fp32 -- ``comb``, the residual mixing matrix
            (training ``h_res``). Indexed ``a[j, i]``: j is the input stream,
            i the output stream.
        b: [num_tokens, hc, hidden] bf16 -- ``res``, the n-stream residual
            (training ``original_residual``). No transpose is needed: the cuTile
            and TileLang index conventions already agree.
        c: [num_tokens, hc] fp32 -- ``post``, the expansion weights (training
            ``h_post``), squeezed of its trailing 1.
        d: [num_tokens, hidden] bf16 -- ``x``, the layer (attn/mlp) output.
        x: [num_tokens, hc, hidden] bf16 -- OUTPUT, allocated by the caller.
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
            # Kept textually identical to sglang mhc.py's mhc_post_tilelang so
            # the two can be diffed line-for-line. Wrapping would not change
            # the emitted CUDA, but the expression order below (post*x first,
            # then the comb chain) is what the FMA contraction alignment
            # depends on, so this block is left exactly as inference has it.
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.serial(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
            # fmt: on
            T.copy(x_local, x_shared)

            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])

        if ENABLE_PDL:
            T.pdl_trigger()
