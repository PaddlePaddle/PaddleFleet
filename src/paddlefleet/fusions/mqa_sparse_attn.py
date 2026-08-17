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

"""Sparse attention over the **absorbed** MQA latent (``d_qk=576`` / ``d_v=512``).

Always uses FlashMLA sparse forward.  The backward is selected by
``mqa_sparse_attn_backward_backend``:

  - ``"cudnn"`` (default): ``paddlefleet.cudnn_ops.csa_sparse_attn_bwd_cudnn``
    (cuDNN DSA).  Fast but ``dkv`` is not run-to-run reproducible (atomics);
    ``test_block_sparse_dsa_gradcheck.py::TestDeterminism`` bounds the drift.
  - ``"tilelang"``: ``paddlefleet.tilelang_ops.attn.mqa_latent_sparse_bwd``
    (deterministic).  ~14x slower on SM100, bitwise stable across runs for
    identical inputs.  That does not depend on ``FLAGS_cudnn_deterministic``:
    it always runs the atomic-free kernel, unlike the symmetric
    ``sparse_mqa_bwd``, which reads that flag to pick one.

The forward cannot be tilelang because ``sparse_mqa_fwd`` asserts
``dim == next_power_of_2(dim)`` and ``d_qk = 576`` is not a power of two.

It is kept as a separate ``PyLayer`` instead of a fourth ``backend`` branch of
``CSASparseAttention`` because the absorbed layout needs three things the 38
production CSA/HCA layers never do, none of which may leak onto that path:

1. **Asymmetric ``d_v``** -- query/key are 576 wide while the value is only the
   leading 512; CSA is symmetric.
2. **Optional sink** -- ``attn_sink=None`` means plain sinkless softmax,
   emulated with a per-head ``-1e30``; CSA always carries a learnable sink and
   ``csa_sparse_attn_utils.prepare_inputs`` unconditionally casts it.
3. **Query-head padding up to the DSA-fixed ``h_q == 64``**, plus (only on the
   ``d_qk != d_v`` kernel branch) a finite-sink LSE correction and an analytic
   ``d_sink``, because the SM100 backward returns an all-zero ``d_sink`` there.

Consumers: the hybrid-MLA non-absorbed MQA layers
(``paddlefleet.transformer.mqa_latent_attention``) and the HySparse block-sparse
gather branch (``paddlefleet.cudnn_ops.block_sparse_mqa_dsa``), which expands
its selected block ids into token columns first.
"""

import paddle

# FlashMLA's sparse prefill fixes the query-head count on SM100.
_DSA_HEADS = 64
# Sink logit that makes ``exp(sink - m) -> 0``, i.e. plain softmax.
_NEG_SINK = -1e30


class _MQASparseAttention(paddle.autograd.PyLayer):
    """FlashMLA sparse forward + selectable backward for absorbed MQA.

    forward inputs:
        query:            ``[b, s, h, d_qk]`` (``d_qk`` = 576), ``h <= 64``.
        kv:               ``[b, s_kv, d_qk]`` single shared K/V latent; the value is
                          its leading ``d_v`` slice.
        token_indices:    ``[b, s, L]`` int32 per-batch-local key columns (``-1``
                          invalid), already causal/doc masked.
        sm_scale, d_v:    softmax scale and value width.
        attn_sink:        ``[h]`` learnable per-head sink logit, or ``None`` for a
                          sinkless softmax.
        backward_backend: ``"cudnn"`` (default, fast, non-deterministic dkv) or
                          ``"tilelang"`` (deterministic, ~14x slower on SM100).

    output: ``[b, s, h * d_v]``, differentiable in ``query``, ``kv`` and
    ``attn_sink``.
    """

    # Side channel for ``lse_indexer``: a PyLayer's returned tensors all become
    # autograd outputs needing a matching backward grad, and this LSE is a
    # by-product consumed under no_grad. ``mqa_sparse_attn`` pops it right after
    # ``apply`` so it never outlives one call.
    _lse_indexer = None

    @staticmethod
    def forward(
        ctx,
        query,
        kv,
        token_indices,
        sm_scale,
        d_v,
        attn_sink=None,
        indexer_topk=0,
        sink_grad_fusion=False,
        backward_backend="cudnn",
    ):
        from paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
            flash_mla_sparse_attn,
        )
        from paddlefleet.fusions.csa_sparse_attn import _csa_compute_topk_length

        b, s, h, dk = query.shape
        ctx.num_heads = h
        ctx.d_v = d_v
        ctx.sm_scale = float(sm_scale)
        ctx.query_dtype = query.dtype
        ctx.kv_dtype = kv.dtype
        if backward_backend not in ("cudnn", "tilelang"):
            raise ValueError(
                f"mqa_sparse_attn backward_backend must be 'cudnn' or "
                f"'tilelang', got {backward_backend!r}"
            )
        ctx.backward_backend = backward_backend

        # Pad query heads up to the DSA-supported h_q == 64. The FlashMLA
        # sparse backend fixes h_q at _DSA_HEADS (sink is [_DSA_HEADS]); it can
        # only handle h <= _DSA_HEADS by zero-padding the head dim. h > 64 is
        # unsupported and must be rejected here rather than failing deep in the
        # CUDA op with an opaque shape error.
        if h > _DSA_HEADS:
            raise ValueError(
                f"DSA sparse attention supports at most {_DSA_HEADS} query "
                f"heads per rank, but got {h}. Reduce num_attention_heads / "
                f"swa_num_attention_heads (per-rank after TP) to "
                f"<= {_DSA_HEADS}."
            )
        if h < _DSA_HEADS:
            pad = paddle.zeros([b, s, _DSA_HEADS - h, dk], dtype=query.dtype)
            q_pad = paddle.concat([query, pad], axis=2)
        else:
            q_pad = query

        # Attention sink over the DSA-fixed 64 heads. When ``attn_sink`` is None
        # the layer is sinkless: a per-head ``-1e30`` makes ``exp(sink - m) -> 0``,
        # recovering plain softmax and discarding the (unused) sink gradient.
        # When a learnable ``attn_sink [h]`` is supplied, its logits fill the
        # real heads and the padded heads keep ``-1e30`` (they contribute no
        # output / gradient). The sink gradient is routed back to the parameter.
        ctx.learnable_sink = attn_sink is not None
        ctx.sink_grad_fusion = sink_grad_fusion
        if attn_sink is None:
            sink = paddle.full([_DSA_HEADS], _NEG_SINK, dtype="float32")
        else:
            assert list(attn_sink.shape) == [h], (
                f"attn_sink must be [num_heads={h}]; got {attn_sink.shape}"
            )
            sink_real = attn_sink.cast("float32")
            if h < _DSA_HEADS:
                sink_pad = paddle.full(
                    [_DSA_HEADS - h], _NEG_SINK, dtype="float32"
                )
                sink = paddle.concat([sink_real, sink_pad], axis=0)
            else:
                sink = sink_real
            sink = sink.contiguous()

        # Per-query early-stop bound shared by both kernels (``mTopkLength``).
        # ``token_indices`` is dense-width (window+topk, or the full causal
        # width) and right-padded with ``-1``, so most rows only use a short
        # prefix; without this the kernels scan the full width for every query.
        # ``_csa_compute_topk_length`` returns the TRAILING bound (last valid
        # column + 1), which stays correct when doc masking leaves interleaved
        # ``-1`` holes, and clamps to >=1 so the backward writes every dq row
        # (``dq`` is allocated uninitialized in the SM100 backend).
        topk_len_flat = _csa_compute_topk_length(
            token_indices.reshape([b * s, -1])
        )

        out, lse, lse_indexer = flash_mla_sparse_attn(
            q_pad,
            kv,
            sink,
            token_indices,
            sm_scale=ctx.sm_scale,
            d_v=d_v,
            topk_length=topk_len_flat.reshape([b, s]),
            indexer_topk=int(indexer_topk),
        )  # out [b, s, 64, d_v], lse [b, s, 64]
        _MQASparseAttention._lse_indexer = lse_indexer

        # ``token_indices`` is saved rather than recomputed so the backward always
        # differentiates the exact support the forward used. Under full recompute
        # the forward runs twice: the indexer's top-k *order* is not reproducible
        # once a document is longer than the top-k budget (measured 0% churn at
        # doc_len<=512, ~83% at 8192), but the selected *set* is -- 0% set churn
        # over Gaussian, low-rank and heavy-tailed activations; it only churns
        # under exact score ties, which continuous q.k does not produce.
        ctx.save_for_backward(
            q_pad, kv, out, lse, token_indices, sink, topk_len_flat
        )
        ctx.needs_grad = (
            not query.stop_gradient,
            not kv.stop_gradient,
            ctx.learnable_sink and not attn_sink.stop_gradient,
        )
        out_h = out[:, :, :h, :].contiguous()  # drop padded heads
        return out_h.reshape([b, s, h * d_v])

    @staticmethod
    def backward(ctx, grad_output):
        (
            q_pad,
            kv,
            out,
            lse,
            token_indices,
            sink,
            topk_len_flat,
        ) = ctx.saved_tensor()

        b, s, hpad, dk = q_pad.shape
        d_v = ctx.d_v
        h = ctx.num_heads
        _, skv, _ = kv.shape

        # Re-pad the incoming grad back to hpad heads (padded heads get 0 grad,
        # so they contribute nothing to dq / dkv).
        grad_output = grad_output.reshape([b, s, h, d_v])
        if h < hpad:
            gpad = paddle.zeros([b, s, hpad - h, d_v], dtype=grad_output.dtype)
            do = paddle.concat([grad_output, gpad], axis=2)
        else:
            do = grad_output
        do = do.contiguous()

        gq, gk, gsink = ctx.needs_grad

        if ctx.backward_backend == "tilelang":
            from paddlefleet.tilelang_ops.attn.mqa_latent_sparse_bwd import (
                mqa_latent_sparse_bwd,
            )

            # The tilelang backward takes the un-padded q/do/lse at the real
            # head count ``h``, and the sink as [h] (sinkless = -1e30 [hpad]
            # was built in the forward for the FlashMLA kernel, slice back).
            sink_h = sink[:h].contiguous()
            dq_full, dkv_full, d_sink_h = mqa_latent_sparse_bwd(
                q_pad[:, :, :h, :].contiguous(),
                kv,
                out[:, :, :h, :].contiguous(),
                do[:, :, :h, :].contiguous(),
                lse[:, :, :h].contiguous(),
                sink_h,
                token_indices,
                ctx.sm_scale,
            )
            dq = dq_full if gq else None
            dkv = dkv_full.cast(ctx.kv_dtype) if gk else None
            # ``d_sink`` comes back fp32 from the kernel's own reduction, which
            # is the dtype the cuDNN branch below returns as well.
            d_attn_sink = d_sink_h if gsink else None
        else:
            from paddlefleet.cudnn_ops import csa_sparse_attn_bwd_cudnn
            from paddlefleet.fusions.csa_sparse_attn_utils import (
                _local_to_global_flat,
            )

            q_flat = q_pad.reshape([b * s, hpad, dk])
            o_flat = out.reshape([b * s, hpad, d_v])
            do_flat = do.reshape([b * s, hpad, d_v])
            kv_flat = kv.reshape([b * skv, dk])
            gidx_flat = _local_to_global_flat(token_indices, skv)

            # dq/dkv softmax normalization for the finite-sink absorbed-MQA path.
            #
            # The forward output was formed with the sink competing in the softmax
            # denominator: p_k = exp(l_k - lse_full), lse_full = logaddexp(lse_kv,
            # sink). But the forward kernel returns a KV-only ``lse`` (the sink is
            # excluded), and the cuDNN DSA backward's ``d_qk != d_v`` branch (the
            # absorbed-MQA Dk=576 / Dv=512 layout used here) consumes the passed LSE
            # verbatim -- it does NOT fold the sink into the denominator itself.
            # Feeding it the KV-only LSE therefore overestimates every p_k for a
            # finite sink and corrupts dq (confirmed: packed finite-sink dQ cos
            # 0.976 vs the dense reference). Fix: for a finite (learnable) sink on
            # this Dk!=Dv path, pass a sink-inclusive LSE and neutralize the sink
            # argument (a -1e30 sink can no longer double-count in the kernel), so
            # p_k matches the forward exactly.
            #
            # Sinkless keeps the KV-only ``lse`` and the -1e30 ``sink`` untouched
            # (logaddexp(lse, -1e30) == lse), so that path is bit-for-bit unchanged.
            # The analytic d_sink below intentionally keeps using the original
            # KV-only ``lse`` (it re-derives lse_full from it).
            lse_bwd = lse
            sink_bwd = sink
            # This correction is load-bearing and silent: dropping it leaves the
            # forward output bit-identical (it only touches the backward), while
            # ``dq`` gets ~75x and ``dkv`` ~120x worse against an autograd
            # reference. Do not use forward agreement to conclude the backward is
            # fine. See ``tests/.../test_block_sparse_dsa_gradcheck.py::
            # test_finite_sink_lse_fix_matters``, which re-drives the raw kernels
            # with the uncorrected KV-only LSE to pin the gap.
            if ctx.learnable_sink and dk != d_v:
                lse_bwd = paddle.logaddexp(
                    lse.astype("float32"),
                    sink.astype("float32").reshape([1, 1, hpad]),
                ).astype(lse.dtype)
                sink_bwd = paddle.full([hpad], _NEG_SINK, dtype="float32")
            lse_flat = lse_bwd.reshape([b * s, hpad])

            # DSA passes topk_length=None (the guarded, full-width backward path)
            # rather than compacting. Its ``[top-k | window]`` layout carries
            # interior -1 holes, and the compact KV-load path is unguarded against
            # them (would gather mKV[-1] -> NaN dq); None takes the guarded path,
            # which handles -1. Compaction is deliberately NOT used here: the
            # window sits last so ``_csa_compute_topk_length`` is already ~full
            # width, i.e. there is almost no early-stop to recover, so a per-step
            # sort would be pure overhead. (HCA *does* compact -- it has real
            # leading holes and a much shorter valid count; see
            # ``_csa_compact_topk_idxs`` in csa_sparse_attn.)
            dq_flat, dkv_flat, _d_sink_unused = csa_sparse_attn_bwd_cudnn(
                q_flat,
                kv_flat,
                o_flat,
                do_flat,
                lse_flat,
                sink_bwd,
                gidx_flat,
                softmax_scale=ctx.sm_scale,
                topk_length=None,
            )
            # ``dkv`` is not run-to-run reproducible: this kernel accumulates the KV
            # gradient with atomics, so two calls on identical inputs differ --
            # for the MQA latent layout and for the symmetric Dk=Dv layout the
            # plain CSA/HCA layers use, i.e. this is a pre-existing property of
            # the shared kernel, not of the non-absorbed MQA path. The magnitude
            # is bounded by
            # ``test_block_sparse_dsa_gradcheck.py::TestDeterminism`` rather
            # than restated here, so one measurement lives in one place.
            # ``dq_flat`` is bit-stable. Use ``backward_backend="tilelang"`` for
            # deterministic dkv.
            #
            # The cuDNN backward allocates ``dq``/``dkv`` with ``empty_like(q)``
            # / ``dtype=kv.dtype`` (``_interface_sm100.py:85,91``), so both
            # already match the dtypes recorded in the forward and these casts
            # are no-ops. Paddle's ``cast`` copies even when the dtype is
            # unchanged (measured: different ``data_ptr``), which for dq is a
            # pointless ``[b, s, h, d_qk]`` round trip -- 1.2 GB of traffic at
            # b=1/s=8192/h=64/d_qk=576. Guard on the dtype instead of dropping
            # the cast, so a backend that starts returning fp32 still converts.
            dq = None
            if gq:
                dq = dq_flat.reshape([b, s, hpad, dk])[:, :, :h, :].contiguous()
                if dq.dtype != ctx.query_dtype:
                    dq = dq.cast(ctx.query_dtype)
            dkv = None
            if gk:
                dkv = dkv_flat.reshape([b, skv, dk])
                if dkv.dtype != ctx.kv_dtype:
                    dkv = dkv.cast(ctx.kv_dtype)
            d_attn_sink = None
            if gsink:
                # The cuDNN DSA backward (SM100) allocates ``d_sink`` but its kernel
                # never populates it -- it always returns zeros. So compute the sink
                # gradient analytically here from the saved forward tensors.
                #
                # For a virtual sink logit ``s_h`` competing in the softmax denom
                # ``Z = sum_k exp(logit_k) + exp(s_h)``, weight ``p_k = exp(l_k)/Z``
                # and sink mass ``p_sink = exp(s_h)/Z``. Since
                # ``d p_k / d s_h = -p_k * p_sink`` and ``out = sum_k p_k v_k``:
                #   d out / d s_h = -p_sink * out
                #   d_sink[h] = sum_{b,s}( dO . (d out / d s_h) )
                #             = -sum_{b,s}( p_sink * (dO . out) )
                #             = -sum_{b,s}( p_sink * Delta )
                # with ``Delta[b,s,h] = sum_dv(out * dO)``. The forward LSE is
                # KV-only (excludes the sink), so the full log-denominator is
                # ``logaddexp(lse_kv, s_h)`` and ``p_sink = exp(s_h - lse_full)``.
                #
                # ``dsa_sink_grad_fusion`` runs that formula as a single Triton
                # kernel, reading out/do once in their native dtype instead of
                # materialising three ``[b, s, h, d_v]`` fp32 temporaries (~3.0
                # GiB at b=1/s=8192/h=64/d_v=512). Both paths are kept so the
                # switch can be flipped mid-run to isolate a regression.
                if ctx.sink_grad_fusion:
                    from paddlefleet.triton_ops.fused_sink_grad import (
                        fused_sink_grad,
                    )

                    d_attn_sink = fused_sink_grad(out, do, lse, sink, h)
                else:
                    out_h = out[:, :, :h, :].astype("float32")
                    do_h = do[:, :, :h, :].astype("float32")
                    delta = (out_h * do_h).sum(axis=-1)  # [b, s, h]
                    sink_real = sink[:h].astype("float32").reshape([1, 1, h])
                    lse_h = lse[:, :, :h].astype("float32")
                    lse_full = paddle.logaddexp(lse_h, sink_real)
                    p_sink = paddle.exp(sink_real - lse_full)  # [b, s, h]
                    d_attn_sink = (
                        -(delta * p_sink).sum(axis=[0, 1])
                    ).contiguous()
                    d_attn_sink = d_attn_sink.cast("float32")

        # One returned grad per **tensor** input, in order. Non-tensor inputs
        # (sm_scale, d_v, backward_backend) occupy no slot. ``attn_sink``
        # occupies a slot only when it was passed as a tensor (sinkless -> None
        # -> no slot), so the returned count is 3 (sinkless) or 4 (learnable sink).
        grads = [dq, dkv, None]  # query, kv, token_indices
        if ctx.learnable_sink:
            grads.append(d_attn_sink)
        return tuple(grads)


def mqa_sparse_attn(
    query,
    kv,
    token_indices,
    sm_scale,
    d_v,
    attn_sink=None,
    indexer_topk=0,
    sink_grad_fusion=False,
    backward_backend="cudnn",
):
    """Absorbed-MQA sparse attention (FlashMLA sparse fwd + selectable bwd).

    Args:
        query:         ``[b, s, h, d_qk]``, ``h <= 64``.
        kv:            ``[b, s_kv, d_qk]`` shared K/V latent; value = leading
                       ``d_v`` slice.
        token_indices: ``[b, s, L]`` int, per-batch-local key columns already
                       causal/doc masked (``-1`` = invalid). Must not carry an
                       autograd graph.
        sm_scale:      softmax scale.
        d_v:           value width (the kernel requires ``d_v == 512`` and
                       ``d_qk in {512, 576}``).
        attn_sink:     ``[h]`` learnable per-head sink logit, or ``None`` for a
                       sinkless softmax.
        indexer_topk:  when > 0, also return the per-head LSE restricted to the
                       **first** ``indexer_topk`` columns of ``token_indices``.
                       That is the normalizer the indexer-loss target needs, so
                       the caller must put the indexer-selected columns first.
                       ``0`` keeps the single-output signature.
        sink_grad_fusion: use the fused Triton sink-gradient epilogue instead of
                       the eager one. No effect when ``attn_sink`` is ``None``
                       (a sinkless layer has no sink gradient to compute).
                       Only ``MQALatentAttention._sparse_attn`` wires this up,
                       from the ``dsa_sink_grad_fusion`` config field; HySparse's
                       ``block_sparse_mqa_attention_dsa`` leaves it at the default
                       and keeps the eager epilogue by design.
        backward_backend: ``"cudnn"`` (default, fast, non-deterministic dkv) or
                       ``"tilelang"`` (deterministic, ~14x slower on SM100).

    Returns:
        ``[b, s, h * d_v]``, or ``(output, lse_indexer [b, s, 64] fp32)`` when
        ``indexer_topk > 0``.
    """
    output = _MQASparseAttention.apply(
        query,
        kv,
        token_indices,
        float(sm_scale),
        int(d_v),
        attn_sink,
        int(indexer_topk),
        sink_grad_fusion,
        str(backward_backend),
    )
    lse_indexer = _MQASparseAttention._lse_indexer
    _MQASparseAttention._lse_indexer = None
    if indexer_topk > 0:
        return output, lse_indexer
    return output
