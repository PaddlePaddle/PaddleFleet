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

"""Fused attention-sink gradient for the sparse-attention backward epilogue.

Replaces the eager epilogue in ``paddlefleet.fusions.mqa_sparse_attn`` that
materialises three ``[b, s, h, d_v]`` fp32 temporaries (two ``astype`` copies of
``out``/``do`` plus their product) just to produce a ``[h]`` gradient. At
``b=1, s=8192, h=64, d_v=512`` that is ~3.0 GiB of transient allocation; this
kernel reads ``out``/``do`` once in their native dtype and writes a
``[num_blocks, h]`` fp32 buffer of 128 KB.

Measured on that shape (interleaved runs, min of 7): **2.664 ms -> 0.423 ms
(6.3x), transient 3.002 GiB -> ~0**. The kernel is entirely memory-bound -- it
reads 1 GiB at ~2.6 TB/s -- so the sink weight's transcendentals are free (3.5%
of runtime).

Math. For a virtual sink logit ``s_h`` competing in the softmax denominator, with
``Delta[n, h] = sum_dv(out * do)``::

    d_sink[h] = -sum_n( p_sink[n, h] * Delta[n, h] )

The forward LSE is KV-only (it excludes the sink), so the full log-denominator is
``logaddexp(lse, s_h)`` and ``p_sink = exp(s_h - logaddexp(lse, s_h))``.

The per-block partial sums are reduced by a separate ``paddle.sum`` rather than
``tl.atomic_add`` on purpose: atomic fp32 accumulation is not run-to-run
reproducible, and this gradient feeds a training run compared against a reference
loss curve. The extra pass costs nothing at 128 KB.

Accuracy vs the eager path: 1.9e-7 relative to the gradient vector's own scale
(~1.6 fp32 ulp), and *closer* to an fp64 reference than eager is. It is not
bitwise identical, and the gap is entirely attributable to the fp32 summation
order of ``Delta``: every other step here -- the casts, the multiply,
``logaddexp``, ``exp`` -- was verified to reproduce paddle bit for bit, and
leaving only ``sum`` to paddle reproduces eager exactly. Paddle's reduction tree
cannot be replicated in triton (17 candidate orderings were probed; the best
matched 4/8 rows and the winner changed with the seed), so this difference is
structural, not a defect.
"""

import paddle

from .utils import enable_compat_on_triton_kernel, is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@enable_compat_on_triton_kernel
@triton.jit
def _sink_grad_partial_kernel(  # pragma: no cover - triton kernel body compiles to PTX, not python-instrumentable
    out_ptr,  # [n_rows, h_pad, d_v], row-major
    do_ptr,  # same layout as out
    lse_ptr,  # [n_rows, h_pad] fp32, KV-only LSE
    sink_ptr,  # [h_pad] fp32
    partial_ptr,  # [n_blocks, num_heads] fp32, out
    n_rows,
    d_v,
    stride_row,  # h_pad * d_v
    stride_head,  # d_v
    stride_lse,  # h_pad
    stride_partial,  # num_heads
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (row block, head): ``-sum_n(p_sink * Delta)``."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_rows
    base = offs_n[:, None] * stride_row + pid_h * stride_head

    # Delta[n] = sum_dv(out * do), accumulated in fp32 while out/do stay in
    # their native dtype -- this is what removes the two fp32 casts.
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d0 in range(0, d_v, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = mask_n[:, None] & (offs_d[None, :] < d_v)
        o = tl.load(out_ptr + base + offs_d[None, :], mask=mask, other=0.0)
        g = tl.load(do_ptr + base + offs_d[None, :], mask=mask, other=0.0)
        acc += tl.sum(o.to(tl.float32) * g.to(tl.float32), axis=1)

    lse = tl.load(lse_ptr + offs_n * stride_lse + pid_h, mask=mask_n, other=0.0)
    sink = tl.load(sink_ptr + pid_h)
    # p_sink = exp(sink - logaddexp(lse, sink)), spelled to match paddle bit for
    # bit. Both spellings below are load-bearing:
    #
    # ``log(1.0 + u)`` and NOT ``log1p(u)``: paddle's log1p is literally
    # log(1 + x) (verified bitwise over 2^20 samples), while libdevice.log1p runs
    # a real log1p algorithm and disagrees on 24% of inputs. libdevice.log and
    # libdevice.exp do match paddle exactly.
    #
    # ``u`` must come from exp, never from a multiply: on a product the compiler
    # folds ``1.0 + u`` into an FMA, and that single rounding stops matching
    # paddle's two.
    #
    # sigmoid(sink - lse) is the same function analytically and one op cheaper,
    # but lands 1.4e-7 away from the eager form.
    u = libdevice.exp(-tl.abs(lse - sink))
    lse_full = libdevice.log(1.0 + u) + tl.maximum(lse, sink)
    p_sink = libdevice.exp(sink - lse_full)

    contrib = tl.where(mask_n, acc * p_sink, 0.0)
    tl.store(partial_ptr + pid_n * stride_partial + pid_h, -tl.sum(contrib))


def fused_sink_grad(out, do, lse, sink, num_heads, block_n=16, block_d=128):
    """Attention-sink gradient ``[num_heads]`` fp32 from the saved fwd tensors.

    Args:
        out: ``[b, s, h_pad, d_v]`` forward output, contiguous, fp16/bf16/fp32.
        do:  ``[b, s, h_pad, d_v]`` output grad, same layout and dtype as ``out``.
        lse: ``[b, s, h_pad]`` fp32 KV-only log-sum-exp from the forward.
        sink: ``[h_pad]`` fp32 per-head sink logit.
        num_heads: number of real heads; ``[num_heads:]`` are zero-padded heads
            that contribute no gradient and are simply not launched.

    Returns:
        ``[num_heads]`` fp32.
    """
    if out.shape != do.shape:
        raise ValueError(f"out/do shape mismatch: {out.shape} vs {do.shape}")
    if out.dtype != do.dtype:
        raise ValueError(f"out/do dtype mismatch: {out.dtype} vs {do.dtype}")
    if lse.dtype != paddle.float32 or sink.dtype != paddle.float32:
        raise ValueError(
            f"lse/sink must be fp32, got {lse.dtype} / {sink.dtype}"
        )

    b, s, h_pad, d_v = out.shape
    if num_heads > h_pad:
        raise ValueError(f"num_heads={num_heads} exceeds h_pad={h_pad}")
    if list(lse.shape) != [b, s, h_pad]:
        raise ValueError(f"lse must be [{b}, {s}, {h_pad}], got {lse.shape}")
    if list(sink.shape) != [h_pad]:
        raise ValueError(f"sink must be [{h_pad}], got {sink.shape}")

    # Flat pointer arithmetic assumes dense row-major.
    out = out.contiguous()
    do = do.contiguous()
    lse = lse.contiguous()

    n_rows = b * s
    n_blocks = triton.cdiv(n_rows, block_n)
    partial = paddle.empty([n_blocks, num_heads], dtype="float32")

    _sink_grad_partial_kernel[(n_blocks, num_heads)](
        out,
        do,
        lse,
        sink,
        partial,
        n_rows,
        d_v,
        h_pad * d_v,
        d_v,
        h_pad,
        num_heads,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return partial.sum(axis=0)
