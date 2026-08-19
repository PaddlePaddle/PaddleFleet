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

"""Deterministic TileLang backward for the absorbed-MQA latent (``Dk != Dv``).

Drop-in replacement for the ``csa_sparse_attn_bwd_cudnn`` half of
:mod:`paddlefleet.fusions.mqa_sparse_attn`. The FlashMLA sparse forward is left
alone; only the backward changes, because the cuDNN DSA backward accumulates
``dkv`` with atomics and is therefore not run-to-run reproducible (bounded by
``test_block_sparse_dsa_gradcheck.py::TestDeterminism``), which is the one
remaining source of step-to-step aadiff in the non-absorbed MQA layers once the
38 CSA/HCA layers are on ``csa_sparse_attn_backend=tilelang``.

Two host-side tricks let this reuse the already-CI-covered *symmetric* kernels in
:mod:`paddlefleet.tilelang_ops.attn.sparse_mqa_bwd` verbatim, with no new kernel:

1. **Zero-pad ``dO`` from ``Dv`` to ``Dk``.** The absorbed latent's value is the
   leading ``Dv`` slice of the ``Dk``-wide key, so padding the cotangent with
   ``Dk - Dv`` zero columns turns the asymmetric problem into the symmetric one
   the kernel already solves, *exactly*:

   - ``dP = P * (dO_pad . KV^T - Delta)`` -- the padded columns multiply
     ``KV[:, Dv:]`` and contribute 0, so this equals ``P * (dO . V^T - Delta)``.
   - ``dKV = dP^T . Q + P^T . dO_pad`` -- the first term is the full ``Dk``-wide
     key gradient, the second lands only in the leading ``Dv`` columns. That is
     precisely the shared-latent gradient combination the cuDNN path produces.
   - ``dQ = dP . KV`` is ``Dk``-wide either way.
   - ``Delta = sum_Dv(O * dO)`` is computed on the *unpadded* tensors, so ``O``
     never needs a padded copy (and the kernel never reads ``O`` at all).

   The only cost is running two of the four GEMMs at ``Dk`` instead of ``Dv``
   (~12% extra FLOPs at 576/512). ``Dk`` need not be a power of two: that assert
   lives in the *forward* kernel (``sparse_mqa_fwd.py:38``), not the backward.

2. **Tile heads and query rows on the host** instead of touching the kernel.
   ``bwd_det`` stores ``dKV_buf[b, s, slot, :]`` with a plain (non-atomic) write,
   which is only race-free while its head-group grid dim is 1, i.e. while
   ``block_H == padded_H``. At ``Dk = 576`` the three ``[64, Dk]`` bf16 shared
   buffers (Q, dO, dQ) alone are 216 KB and do not fit SM100's 227 KB budget, so
   heads *must* be tiled -- done here by calling the kernel once per
   ``block_h``-wide head group with ``H = block_h``, which keeps its internal
   ``NH == 1``. Query rows are chunked the same way to bound the
   ``[B, Sc, L, Dk]`` fp32 ``dKV_buf``: unchunked that is 11.25 GB at
   ``S=8192, L=640, Dk=576`` (the 38 HCA layers already pay 3.0 GB for it at
   ``L=192, D=512``).

**Determinism.** ``dq`` is a private per-(row, head-group) accumulator. ``dkv``
goes through ``bwd_det``'s atomic-free per-slot buffer plus the stable-argsort
CSR reduction in ``dkv_reduce``; the cross-chunk / cross-head-group combination
is a fixed-order Python loop. Chunking changes the summation order, so results
are *not* bitwise equal to a hypothetical unchunked run -- but they are a
deterministic function of the input shapes, which is what aadiff needs. None of
this is conditional on ``FLAGS_cudnn_deterministic``: ``bwd_det`` is called
unconditionally here, unlike the symmetric ``sparse_mqa_bwd``, which reads that
flag to choose between an atomic and an atomic-free kernel.

**LSE convention bridge.** FlashMLA returns a natural-log, **sink-exclusive**
LSE, while this backward wants base-2 and **sink-inclusive**: it evaluates
``P = exp2(score * sm_scale * log2e - Lse)`` and
``d_sink = -Delta * exp2(sink * log2e - Lse)``. Both are satisfied by
``lse_tl = logaddexp(lse_kv, sink) * log2(e)`` with ``attn_sink`` passed through
unscaled. Sinkless callers pass ``sink = -1e30``, for which
``logaddexp(lse, -1e30) == lse`` and ``exp2(-1e30 * log2e - lse) == 0``, so the
bridge degenerates correctly and ``d_sink`` comes back as zeros.

**Cost.** This is materially slower than the cuDNN DSA backward it replaces:
measured on B30Z (SM100) at ``B=1, S=8192, S_kv=8192, H=64, Dk=576, Dv=512,
L=640``, one full fwd+bwd is 6.2 ms with ``backend="cudnn"`` and 59 ms with this
one, i.e. ~14x on the backward alone. Three reasons, in order:

1. The ``[B, S, L, Dk]`` fp32 per-slot buffer is written and then read back --
   24 GB of DRAM traffic where the atomic version keeps its accumulation in L2.
2. ``block_h`` is capped at 32 by shared memory, so the scores are recomputed
   once per head group (2 groups at ``H=64``). 64 compiles but spills and
   measured 3x *worse* (178 ms); 16 is also worse (73 ms).
3. There is no ``topk_length`` early-stop, so every query row scans the full
   index width even where it is mostly ``-1``.

Accept that trade only where reproducibility is the point.
"""

import paddle

from paddlefleet.tilelang_ops.attn.sparse_mqa_bwd import (
    _build_csr_index,
    bwd_det,
    dkv_reduce,
    postprocess,
    preprocess,
)

_LOG2E = 1.4426950408889634

# Sink logit for pad heads: ``exp2(sink * log2e - lse)`` underflows to 0, so a
# padded head can contribute nothing to ``d_sink``. Same value the sinkless
# contract uses in ``fusions/mqa_sparse_attn.py``.
_NEG_SINK = -1e30

# ``bwd_det`` requires ``topk % _BLOCK_SIZE == 0`` (sparse_mqa_bwd.py:336) and
# ``preprocess`` tiles S by 32, so every chunk length must be a multiple of this.
_BLOCK_SIZE = 32

# Default byte budget for the ``[B, Sc, L, Dk]`` fp32 per-slot gradient buffer.
# 12 GiB is deliberately above the 11.25 GiB the online shape
# (B=1, S=8192, L=640, Dk=576) needs, i.e. **no chunking by default**, because
# chunking is the slower option: measured on B30Z (SM100, 275 GB) at that shape,
# 1 chunk = 59 ms, 8 chunks of 1024 rows = 73 ms, for 15.1 GB vs 4.3 GB of peak
# allocation. Lower this only under real memory pressure. The budget is the
# whole buffer including the batch axis, so the same number holds at B > 1 (it
# chunks there instead of silently allocating B times as much).
_DEFAULT_DKV_BUF_BYTES = 12 << 30


def _pick_chunk(b, s, topk, dk, budget_bytes):
    """Query-row chunk length for the ``[b, Sc, topk, Dk]`` fp32 buffer.

    Prefers an exact divisor of ``s`` so every chunk has the same length: that
    keeps a single ``@tilelang.jit`` variant and lets one ``dKV_buf`` allocation
    be reused across all chunks (``bwd_det`` overwrites every slot, so the buffer
    needs no clearing between launches).

    ``b`` is part of the row cost because the buffer carries the batch axis:
    budgeting per ``[topk, Dk]`` row instead would keep the whole sequence in
    one chunk at ``B=2`` and allocate twice the budget.
    """
    row_bytes = b * topk * dk * 4
    sc_max = max(int(budget_bytes // row_bytes), _BLOCK_SIZE)
    sc_max = min(sc_max // _BLOCK_SIZE * _BLOCK_SIZE, s)
    sc_max = max(sc_max, _BLOCK_SIZE)
    for cand in range(sc_max, 0, -_BLOCK_SIZE):
        if s % cand == 0:
            return cand
    # ``s`` is not a multiple of _BLOCK_SIZE; fall back to a padded tail.
    return sc_max


def _pick_block_h(h):
    """Head-group width for ``bwd_det``: 32 when it divides ``h``, else 16.

    ``bwd_det`` derives its own tile from the ``H`` it is given --
    ``padded_H = max(next_power_of_2(H), 16)`` and ``block_H = min(64,
    padded_H)`` (``sparse_mqa_bwd.py:352``) -- and then copies
    ``Q[b, s, 0:block_H, :]`` / stores ``dQ[b, s, 0:block_H, :]``. So a group
    width below 16 does **not** make the kernel read fewer heads: at
    ``block_h = 8`` it still moves 16 heads, i.e. it reads and *writes* 8 heads
    past the end of an 8-head tensor. Only 16 and 32 satisfy
    ``block_h == padded_H``; head counts that are not a multiple of the chosen
    width are zero-padded by the caller instead (see ``_pad_heads``), which is
    how ``h = 8`` -- the per-rank count the hybrid-MLA fixtures and any TP > 4
    split produce -- stays supported.

    32 is preferred where it fits because it halves the number of launches and
    the score recomputation; the docstring's timings were measured with it.
    """
    return 32 if h % 32 == 0 else 16


def _pad_heads(x, h_pad, fill=0):
    """``fill``-pad ``x`` along its head axis (the last or second-to-last).

    ``[b, s, h, d]`` and ``[b, s, h]`` both pad on ``h``; ``[h]`` (the sink)
    pads on its only axis.
    """
    axis = 2 if x.ndim == 4 else (x.ndim - 1)
    pad_shape = list(x.shape)
    pad_shape[axis] = h_pad - x.shape[axis]
    if pad_shape[axis] == 0:
        return x
    return paddle.concat(
        [x, paddle.full(pad_shape, fill, dtype=x.dtype)], axis=axis
    )


def _reduce_threads(dk):
    """Thread count for ``dkv_reduce``, which maps ``T.Parallel(Dk)`` onto threads.

    Its layout inference fails outright ("no available layout found") unless the
    block width divides ``Dk``, so the default 128 is unusable at ``Dk = 576``.
    """
    for cand in (256, 192, 128, 96, 64, 32):
        if dk % cand == 0:
            return cand
    return 32


def _pad_rows(x, s_pad, fill=0):
    """``fill``-pad ``x`` along axis 1 up to ``s_pad`` rows."""
    pad_shape = list(x.shape)
    pad_shape[1] = s_pad - x.shape[1]
    return paddle.concat(
        [x, paddle.full(pad_shape, fill, dtype=x.dtype)], axis=1
    )


def mqa_latent_sparse_bwd(
    query,
    kv,
    out,
    grad_out,
    lse,
    attn_sink,
    token_indices,
    sm_scale,
    block_h=None,
    dkv_buf_bytes=_DEFAULT_DKV_BUF_BYTES,
):
    """Deterministic backward for absorbed-MQA sparse attention.

    Args:
        query:         ``[B, S, H, Dk]`` bf16.
        kv:            ``[B, S_kv, Dk]`` bf16 shared latent; the value is its
                       leading ``Dv`` slice.
        out:           ``[B, S, H, Dv]`` bf16 normalized forward output.
        grad_out:      ``[B, S, H, Dv]`` bf16 cotangent.
        lse:           ``[B, S, H]`` natural-log, **sink-exclusive** LSE, i.e.
                       exactly what ``flash_mla_sparse_attn`` returns.
        attn_sink:     ``[H]`` fp32 pre-scaled sink logit in natural-log space.
                       Sinkless callers pass ``-1e30``.
        token_indices: ``[B, S, L]`` int32 per-batch-local key columns, ``-1``
                       for invalid. Width is padded to a multiple of 32 here.
        sm_scale:      softmax scale applied to the raw ``q.k`` logits.
        block_h:       query heads per kernel launch, 16 or 32 -- the two widths
                       for which ``bwd_det``'s own
                       ``padded_H = max(next_power_of_2(H), 16)`` equals what it
                       is handed (32 is the upper bound: three
                       ``[block_h, Dk]`` bf16 buffers plus ``[32, Dk]`` for the
                       gathered latent must fit shared memory, which at
                       ``Dk=576`` rules out 64). ``H`` need not be a multiple of
                       it -- the head axis is zero-padded here. ``None``
                       (default) picks 32 when it divides ``H``, else 16.
        dkv_buf_bytes: byte budget for the per-slot fp32 gradient buffer, which
                       sets the query-row chunk length.

    Returns:
        ``(dq [B, S, H, Dk] bf16, dkv [B, S_kv, Dk] bf16, d_attn_sink [H] fp32)``
    """
    b, s, h, dk = query.shape
    _, s_kv, kv_dk = kv.shape
    dv = grad_out.shape[-1]
    if block_h is None:
        block_h = _pick_block_h(h)

    # Explicit raises rather than ``assert``: these guard a TileLang JIT
    # specialisation and a multi-GiB allocation, and ``python -O`` strips
    # asserts, which would turn an illegal Dk/Dv, head count or ``block_h``
    # into an opaque kernel failure (or an out-of-bounds write) much later.
    if kv_dk != dk:
        raise ValueError(f"kv width {kv_dk} != query width {dk}")
    if dv > dk:
        raise ValueError(f"Dv ({dv}) must be <= Dk ({dk})")
    if dk % 16 or dv % 16:
        raise ValueError(f"Dk/Dv must be a multiple of 16: {dk}/{dv}")
    if list(out.shape) != [b, s, h, dv]:
        raise ValueError(f"out {out.shape} != {[b, s, h, dv]}")
    if list(grad_out.shape) != [b, s, h, dv]:
        raise ValueError(f"grad_out {grad_out.shape} != {[b, s, h, dv]}")
    if list(lse.shape) != [b, s, h]:
        raise ValueError(f"lse {lse.shape} != {[b, s, h]}")
    if list(attn_sink.shape) != [h]:
        raise ValueError(f"attn_sink {attn_sink.shape} != {[h]}")
    if list(token_indices.shape[:2]) != [b, s]:
        raise ValueError(
            f"token_indices {token_indices.shape} != [{b}, {s}, L]"
        )
    if block_h not in (16, 32):
        raise ValueError(
            f"block_h must be 16 or 32, got {block_h}: ``bwd_det`` rounds its "
            "tile up to max(next_power_of_2(H), 16), so any other width makes "
            "the kernel move more heads than the tensors hold"
        )
    # The reused kernels are JIT-specialised on bf16 and assert it internally
    # (``sparse_mqa_bwd.py:33,86,129,337``), so a caller in an fp16 run would
    # otherwise land on a bf16 kernel reading fp16 memory -- silently wrong, or
    # an opaque TileLang AssertionError. Name the offending tensor instead.
    for name, tensor in (
        ("query", query),
        ("kv", kv),
        ("out", out),
        ("grad_out", grad_out),
    ):
        if tensor.dtype != paddle.bfloat16:
            raise ValueError(
                f"{name} must be bfloat16 (the TileLang kernels are built for "
                f"it); got {tensor.dtype}"
            )

    # ``Delta[b, s, h] = sum_Dv(O * dO)`` -- computed at the *value* width, so
    # ``out`` is never padded (and the backward kernel never reads it).
    delta = preprocess(b, s, h, dv)(out.contiguous(), grad_out.contiguous())

    # LSE convention bridge: natural-log sink-exclusive -> base-2 sink-inclusive.
    sink32 = attn_sink.astype("float32").contiguous()
    lse_tl = (
        paddle.logaddexp(lse.astype("float32"), sink32.reshape([1, 1, h]))
        * _LOG2E
    )

    # Zero-pad the cotangent to the key width; see the module docstring for why
    # this makes the Dk != Dv problem an exact instance of the symmetric one.
    if dv < dk:
        grad_out = paddle.concat(
            [
                grad_out,
                paddle.zeros([b, s, h, dk - dv], dtype=grad_out.dtype),
            ],
            axis=-1,
        )
    grad_out = grad_out.contiguous()

    topk = token_indices.shape[-1]
    idx = token_indices.astype("int32")
    if topk % _BLOCK_SIZE:
        padded = (topk + _BLOCK_SIZE - 1) // _BLOCK_SIZE * _BLOCK_SIZE
        idx = paddle.concat(
            [idx, paddle.full([b, s, padded - topk], -1, dtype="int32")],
            axis=-1,
        )
        topk = padded

    sc = _pick_chunk(b, s, topk, dk, dkv_buf_bytes)
    n_chunks = (s + sc - 1) // sc
    s_pad = n_chunks * sc
    # Pad the head axis to a whole number of groups. ``bwd_det`` moves
    # ``block_h`` heads per launch whatever the tensors hold (see
    # ``_pick_block_h``), so a head count below the group width -- ``h = 8`` on
    # the fixtures, anything from a TP split -- must be padded here rather than
    # left for the kernel to read past. The pad heads are inert: q and dO are
    # zero, so ``dP`` and both ``dKV`` terms vanish, and ``Delta`` is zero, so
    # ``d_sink`` is too; only their ``dq`` rows are garbage and those are sliced
    # off before returning.
    h_pad = (h + block_h - 1) // block_h * block_h
    n_hg = h_pad // block_h
    if h_pad != h:
        query = _pad_heads(query, h_pad)
        grad_out = _pad_heads(grad_out, h_pad)
        delta = _pad_heads(delta, h_pad)
        lse_tl = _pad_heads(lse_tl, h_pad)
        sink32 = _pad_heads(sink32, h_pad, fill=_NEG_SINK)

    # Pad the query axis up to a whole number of chunks once, so the loop body is
    # uniform (one JIT variant, one reusable buffer) and the tail needs no
    # special case. Padded rows are inert: q/dO are zero and their whole index
    # row is -1, so every score masks to -inf, P underflows to 0, and the row
    # adds nothing to dq / dkv / d_sink (Delta is 0 there too).
    if s_pad != s:
        query = _pad_rows(query, s_pad)
        grad_out = _pad_rows(grad_out, s_pad)
        lse_tl = _pad_rows(lse_tl, s_pad)
        delta = _pad_rows(delta, s_pad)
        idx = _pad_rows(idx, s_pad, fill=-1)
    idx = idx.contiguous()
    lse_tl = lse_tl.contiguous()

    # One batched CSR build for *all* chunks instead of one per chunk. Paddle's
    # stable ``argsort`` is launch-bound at these sizes (measured ~6 ms whether
    # the input is 0.6M or 5.2M elements), so folding the chunk axis into the
    # batch axis turns n_chunks sorts into one and was worth ~55% of this
    # backward's wall time at S=8192 / L=640 / 8 chunks.
    sort_perm, seg_offsets = _build_csr_index(
        idx.reshape([b * n_chunks, sc, topk]), s_kv
    )
    sort_perm = sort_perm.reshape([b, n_chunks, sc * topk])
    seg_offsets = seg_offsets.reshape([b, n_chunks, s_kv + 1])

    bwd_kernel = bwd_det(b, sc, s_kv, block_h, dk, topk, float(sm_scale))
    reduce_kernel = dkv_reduce(
        b, sc, s_kv, topk, dk, threads=_reduce_threads(dk)
    )
    cast_kernel = postprocess(b, s_kv, dk)

    # Reused across every launch: ``bwd_det`` writes every slot unconditionally.
    dkv_buf = paddle.empty([b, sc, topk, dk], dtype="float32")
    dsink_buf = paddle.empty([sc, b, block_h], dtype="float32")

    dq = paddle.empty([b, s, h_pad, dk], dtype=query.dtype)
    dkv_f32 = paddle.zeros([b, s_kv, dk], dtype="float32")
    dsink_parts = [None] * n_hg

    kv = kv.contiguous()
    for c in range(n_chunks):
        s0 = c * sc
        n = min(sc, s - s0)  # real (unpadded) rows in this chunk
        idx_c = idx[:, s0 : s0 + sc].contiguous()
        sort_perm_c = sort_perm[:, c].contiguous()
        seg_offsets_c = seg_offsets[:, c].contiguous()
        for g in range(n_hg):
            h0, h1 = g * block_h, (g + 1) * block_h
            dq_c = bwd_kernel(
                query[:, s0 : s0 + sc, h0:h1, :].contiguous(),
                kv,
                grad_out[:, s0 : s0 + sc, h0:h1, :].contiguous(),
                sink32[h0:h1].contiguous(),
                idx_c,
                lse_tl[:, s0 : s0 + sc, h0:h1].contiguous(),
                delta[:, s0 : s0 + sc, h0:h1].contiguous(),
                dkv_buf,
                dsink_buf,
            )
            dq[:, s0 : s0 + n, h0:h1, :] = dq_c[:, :n]
            dkv_f32 += reduce_kernel(dkv_buf, sort_perm_c, seg_offsets_c)
            part = dsink_buf[:n].sum(axis=[0, 1])
            dsink_parts[g] = (
                part if dsink_parts[g] is None else dsink_parts[g] + part
            )

    # Slice the pad heads back off: their ``dq`` rows are zeros the caller must
    # not see, and ``d_sink`` carries one entry per real head.
    d_sink = paddle.concat(dsink_parts)
    if h_pad != h:
        dq = dq[:, :, :h, :]
        d_sink = d_sink[:h]
    return dq.contiguous(), cast_kernel(dkv_f32), d_sink.contiguous()
