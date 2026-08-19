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

"""Determinism and numerics for the TileLang absorbed-MQA backward.

``mqa_sparse_attn_backward_backend="tilelang"`` swaps the cuDNN DSA backward of
:mod:`paddlefleet.fusions.mqa_sparse_attn` for
:func:`paddlefleet.tilelang_ops.attn.mqa_latent_sparse_bwd`. The FlashMLA
sparse forward is untouched by the switch. The reason the switch exists is
reproducibility: cuDNN accumulates ``dkv`` with atomics, so two backward passes
over identical inputs differ (rel ~8e-6, bounded in
``test_block_sparse_dsa_gradcheck.TestDeterminism``), which is the last source
of step-to-step aadiff in the latent-MQA layers. The tilelang path is
atomic-free and must be *bitwise* stable.

What is checked, and against what:

* **Numerics** -- ``dq`` / ``dkv`` / ``d_sink`` against the fp32 differentiable
  reference forward of ``test_block_sparse_dsa_gradcheck``
  (``_analytic_ref_grads``), i.e. an independent oracle, never the cuDNN
  backward alone. The floor is bf16-limited (measured ~3.8e-3 dq, ~2.8e-3 dkv,
  ~2.5e-3 d_sink -- the same order that suite measures for cuDNN).
* **Determinism** -- repeated identical runs must be bitwise equal.
* **Host-side tiling** in ``mqa_latent_sparse_bwd``: head groups (``block_h``),
  query-row chunking (``dkv_buf_bytes``), the ``topk`` padding up to a multiple
  of 32 and the padded query tail. A tiling bug shows up as a wrong *sum*, not
  as a crash, so each variant is compared against the same reference.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.fusions.mqa_sparse_attn import (
    _DSA_HEADS,
    _NEG_SINK,
    mqa_sparse_attn,
)
from paddlefleet.tilelang_ops.attn.mqa_latent_sparse_bwd import (
    _BLOCK_SIZE,
    _pad_rows,
    _pick_block_h,
    _pick_chunk,
    _reduce_threads,
    mqa_latent_sparse_bwd,
)

from .hybrid_mla_utils import _GPU
from .test_block_sparse_dsa_gradcheck import (
    DK,
    DV,
    SM,
    _analytic_ref_grads,
    _causal_indices,
    _rand,
    _relerr,
    _to_bf16,
)

# ``mqa_latent_sparse_bwd`` launches one kernel per head group; the group width
# defaults to the largest power of two dividing ``H`` (capped at 32 by shared
# memory), so every head count the forward accepts is reachable.
H_SMALL = 32
H_FULL = 64  # production head count -> two head groups

# bf16 round-off floor, measured on B30Z (SM100): dq 3.8e-3, dkv 2.8e-3,
# d_sink 2.5e-3 against the fp32 reference. 8e-3 leaves ~2x headroom.
REF_TOL = 8e-3
# Reordering-only differences (chunked vs unchunked summation, tilelang vs
# cuDNN accumulation order). Measured 2.7e-6 and 6.2e-4 respectively.
REORDER_TOL = 1e-3


# The tilelang backward needs CUDA and a tilelang JIT, and nothing else: it is
# deliberately *not* behind ``_GPU`` (``is_dsa_available()``), which refuses
# anything below SM100 on behalf of the FlashMLA forward + cuDNN DSA backward.
# Only the end-to-end dispatch tests, which do run those two, keep that gate.
_TILELANG = unittest.skipUnless(
    paddle.is_compiled_with_cuda(),
    "tilelang absorbed-MQA backward requires CUDA (but not SM100)",
)


def _inputs(h, s, topk, seed):
    """``(q [1,s,h,DK], kv [1,s,DK], ti [1,s,topk], dO [1,s,h*DV])``."""
    return (
        _rand([1, s, h, DK], 0.3, seed),
        _rand([1, s, DK], 0.3, seed + 1),
        _causal_indices(s, topk, seed=seed + 2),
        _rand([1, s, h * DV], 1.0, seed + 3),
    )


def _layer_run(q, kv, ti, dO, backend, sink_mag=None):
    """One fwd+bwd through the PyLayer. Returns fp32 numpy copies of
    ``(out, dq, dkv, d_sink)``; ``d_sink`` is ``None`` when sinkless."""
    qb, kvb = _to_bf16(q), _to_bf16(kv)
    qb.stop_gradient = False
    kvb.stop_gradient = False
    sink_t = None
    if sink_mag is not None:
        sink_t = paddle.full([q.shape[2]], float(sink_mag), dtype="float32")
        sink_t.stop_gradient = False
    out = mqa_sparse_attn(
        qb,
        kvb,
        paddle.to_tensor(ti),
        float(SM),
        DV,
        sink_t,
        backward_backend=backend,
    )
    dO_t = paddle.to_tensor(dO.reshape(out.shape))
    (out.cast("float32") * dO_t).sum().backward()
    return (
        out.cast("float32").numpy().copy(),
        qb.grad.cast("float32").numpy()[0].copy(),
        kvb.grad.cast("float32").numpy()[0].copy(),
        None if sink_t is None else sink_t.grad.numpy().copy(),
    )


def _eager_forward(q, kv, ti, sink_mag):
    """The backward's inputs built in fp32 paddle -- no FlashMLA, no SM100.

    ``mqa_latent_sparse_bwd`` needs ``out`` and ``lse``, and both are pure
    definitions over the selected columns: ``lse`` is the natural-log,
    **sink-exclusive** logsumexp of the masked logits and ``out`` is the
    softmax-weighted value sum *with* the sink in the denominator, exactly what
    ``flash_mla_sparse_attn`` returns. Building them here instead of calling
    that kernel does two things: it takes the tilelang backward off the
    ``is_dsa_available()`` SM100 gate (the tilelang kernels have no such
    requirement -- only the FlashMLA forward does), and it makes the backward
    tests independent of the forward kernel, so a forward regression cannot
    mask a backward bug.

    ``sink_mag=None`` is the sinkless contract: ``-1e30``, for which
    ``logaddexp(lse, -1e30) == lse`` and ``d_sink`` comes back as zeros.

    Returns ``(q_bf16, kv_bf16, out, lse, sink[h], token_indices)``.
    """
    b, s, h, _ = q.shape
    topk = ti.shape[-1]
    qb, kvb = _to_bf16(q), _to_bf16(kv)
    q32 = paddle.to_tensor(q[0])
    kv32 = paddle.to_tensor(kv[0])
    cols = paddle.to_tensor(ti.reshape([s, topk])).cast("int64")
    valid = cols >= 0
    safe = paddle.where(valid, cols, paddle.zeros_like(cols))
    ksel = paddle.gather(kv32, safe.flatten(), axis=0).reshape([s, topk, DK])
    logit = paddle.einsum("shd,sld->shl", q32, ksel) * float(SM)
    logit = paddle.where(
        valid.unsqueeze(1), logit, paddle.full_like(logit, _NEG_SINK)
    )
    lse = paddle.logsumexp(logit, axis=-1)  # [s, h], sink-exclusive
    real = _NEG_SINK if sink_mag is None else float(sink_mag)
    sink = paddle.full([h], real, dtype="float32")
    lse_full = paddle.logaddexp(lse, sink.reshape([1, h]))
    p = paddle.exp(logit - lse_full.unsqueeze(-1))
    out = paddle.einsum("shl,sld->shd", p, ksel[:, :, :DV])
    return (
        qb,
        kvb,
        out.unsqueeze(0).cast("bfloat16").contiguous(),
        lse.unsqueeze(0).cast("float32").contiguous(),
        sink,
        paddle.to_tensor(ti),
    )


def _kernel_run(q, kv, ti, dO, sink_mag=None, **kwargs):
    """``mqa_latent_sparse_bwd`` on eager-built inputs. fp32 numpy copies."""
    b, s, h, _ = q.shape
    qb, kvb, out, lse, sink, ti_t = _eager_forward(q, kv, ti, sink_mag)
    do = paddle.to_tensor(dO.reshape([b, s, h, DV])).cast(out.dtype)
    dq, dkv, dsink = mqa_latent_sparse_bwd(
        qb, kvb, out, do, lse, sink, ti_t, float(SM), **kwargs
    )
    return (
        dq.cast("float32").numpy()[0].copy(),
        dkv.cast("float32").numpy()[0].copy(),
        dsink.numpy().copy(),
    )


class _Base(unittest.TestCase):
    """GPU device only -- deliberately no ``FLAGS_cudnn_deterministic`` pin.

    ``mqa_latent_sparse_bwd`` calls ``bwd_det`` unconditionally, unlike the
    symmetric ``sparse_mqa_bwd`` (``sparse_mqa_bwd.py:613``) which reads that
    flag to choose between an atomic and an atomic-free kernel. So the
    determinism asserted below is a property of the code path, not of a process
    flag, and pinning one here would both weaken the test and leak: the flag is
    process-global and feeds ``flash_mask_facade.get_fa_version``, which pushes
    the other hybrid-MLA suites off FA4 when pytest shares a process.
    """

    @classmethod
    def setUpClass(cls):
        paddle.set_device("gpu")

    def _assert_matches_reference(self, got, q, kv, ti, dO, sink_mag=None):
        """dq / dkv (/ d_sink) within the bf16 floor of the fp32 oracle."""
        dq, dkv, dsink = got
        ref_q, ref_kv, ref_sink = _analytic_ref_grads(q, kv, ti, dO, sink_mag)
        self.assertLess(_relerr(dq, ref_q), REF_TOL, "dq")
        self.assertLess(_relerr(dkv, ref_kv), REF_TOL, "dkv")
        if sink_mag is not None:
            self.assertLess(_relerr(dsink, ref_sink), REF_TOL, "d_sink")


# ===========================================================================
# 1. Backend dispatch through ``mqa_sparse_attn``
# ===========================================================================
@_GPU
class TestBackendDispatch(_Base):
    """The switch selects a backward and changes nothing else."""

    def test_tilelang_backward_matches_fp32_reference(self):
        for sink_mag in (None, 0.5):
            with self.subTest(sink=sink_mag):
                q, kv, ti, dO = _inputs(H_SMALL, 64, 64, 300)
                _, dq, dkv, dsink = _layer_run(
                    q, kv, ti, dO, "tilelang", sink_mag
                )
                self._assert_matches_reference(
                    (dq, dkv, dsink), q, kv, ti, dO, sink_mag
                )

    def test_forward_is_identical_across_backends(self):
        # The forward is FlashMLA either way, so it must be bit-identical; only
        # the gradients may differ, and then only by accumulation order.
        q, kv, ti, dO = _inputs(H_SMALL, 64, 64, 310)
        o_c, dq_c, dkv_c, ds_c = _layer_run(q, kv, ti, dO, "cudnn", 0.25)
        o_t, dq_t, dkv_t, ds_t = _layer_run(q, kv, ti, dO, "tilelang", 0.25)
        self.assertTrue(
            np.array_equal(o_c, o_t), "backward_backend changed the forward"
        )
        self.assertLess(_relerr(dq_t, dq_c), REORDER_TOL * 4, "dq")
        self.assertLess(_relerr(dkv_t, dkv_c), REORDER_TOL, "dkv")
        # The two d_sink values come from different code entirely -- the eager
        # analytic epilogue for cuDNN (whose kernel returns zeros) versus the
        # tilelang kernel's own reduction -- so agreement here is a real
        # cross-check of both, not a tautology.
        self.assertLess(_relerr(ds_t, ds_c), REORDER_TOL, "d_sink")

    def test_default_backend_is_cudnn(self):
        # Omitting the argument must keep the fast path, i.e. the switch is
        # opt-in and cannot silently cost 14x.
        q, kv, ti, dO = _inputs(H_SMALL, 32, 32, 320)
        qb, kvb = _to_bf16(q), _to_bf16(kv)
        qb.stop_gradient = False
        out = mqa_sparse_attn(qb, kvb, paddle.to_tensor(ti), float(SM), DV)
        dO_t = paddle.to_tensor(dO.reshape(out.shape))
        (out.cast("float32") * dO_t).sum().backward()
        default = _layer_run(q, kv, ti, dO, "cudnn")[1]
        self.assertTrue(
            np.array_equal(qb.grad.cast("float32").numpy()[0], default)
        )

    def test_fixture_head_count_works_end_to_end(self):
        # The hybrid-MLA fixtures run 8 heads per rank, which used to be a hard
        # error on this backend. The rest of the head-count range, and the
        # numerics, are covered ungated in ``TestTilelangBackward``; this only
        # pins that the *layer* path survives it.
        q, kv, ti, dO = _inputs(8, 64, 64, 340)
        _, dq_t, dkv_t, _ = _layer_run(q, kv, ti, dO, "tilelang", 0.25)
        _, dq_c, dkv_c, _ = _layer_run(q, kv, ti, dO, "cudnn", 0.25)
        self.assertLess(_relerr(dq_t, dq_c), REORDER_TOL * 4, "dq")
        self.assertLess(_relerr(dkv_t, dkv_c), REORDER_TOL, "dkv")

    def test_cudnn_dkv_is_the_baseline_this_replaces(self):
        # Pins *why* the tilelang backward exists: same inputs, same flags,
        # different dkv. Only bounded (not asserted unequal) because the drift
        # is a race and a lucky run can come out equal; the bitwise assertion
        # lives in ``TestTilelangBackward``.
        q, kv, ti, dO = _inputs(H_SMALL, 64, 64, 410)
        _, dq1, dkv1, _ = _layer_run(q, kv, ti, dO, "cudnn")
        _, dq2, dkv2, _ = _layer_run(q, kv, ti, dO, "cudnn")
        self.assertTrue(np.array_equal(dq1, dq2), "cuDNN dq should be stable")
        self.assertLess(_relerr(dkv1, dkv2), 1e-4)


# ===========================================================================
# 2. The tilelang backward itself: numerics + determinism, no FlashMLA
# ===========================================================================
@_TILELANG
class TestTilelangBackward(_Base):
    """Driven with eager-built ``out``/``lse``, so no SM100 gate applies.

    This is where the backend's actual promise is checked, and it is checked
    against the fp32 oracle rather than against the cuDNN backward, so the two
    implementations never validate each other.
    """

    def test_matches_the_fp32_reference(self):
        for sink_mag in (None, 0.5):
            with self.subTest(sink=sink_mag):
                q, kv, ti, dO = _inputs(H_SMALL, 64, 64, 400)
                self._assert_matches_reference(
                    _kernel_run(q, kv, ti, dO, sink_mag),
                    q,
                    kv,
                    ti,
                    dO,
                    sink_mag,
                )

    def test_all_three_gradients_are_bitwise_reproducible(self):
        q, kv, ti, dO = _inputs(H_SMALL, 64, 64, 405)
        first = _kernel_run(q, kv, ti, dO, 0.25)
        second = _kernel_run(q, kv, ti, dO, 0.25)
        for name, a, b in zip(("dq", "dkv", "d_sink"), first, second):
            self.assertTrue(
                np.array_equal(a, b), f"{name} is not bitwise deterministic"
            )

    def test_sinkless_gives_a_zero_sink_gradient(self):
        # The sinkless contract is a ``-1e30`` sink, for which the kernel's
        # ``exp2(sink * log2e - lse)`` underflows to 0 rather than to a NaN.
        q, kv, ti, dO = _inputs(H_SMALL, 64, 64, 410)
        _, _, dsink = _kernel_run(q, kv, ti, dO, None)
        self.assertTrue(np.all(dsink == 0.0), f"d_sink not zero: {dsink[:4]}")

    def test_unknown_backend_is_rejected(self):
        # The PyLayer validates the name before it touches a kernel, so this
        # runs anywhere -- which is the point: a typo in a YAML must fail on the
        # box that reads the YAML, not only on an SM100 one.
        q, kv, ti, _ = _inputs(H_SMALL, 32, 32, 430)
        with self.assertRaisesRegex(
            ValueError, "backward_backend must be 'cudnn' or 'tilelang'"
        ):
            mqa_sparse_attn(
                _to_bf16(q),
                _to_bf16(kv),
                paddle.to_tensor(ti),
                float(SM),
                DV,
                backward_backend="paddle",
            )

    def test_head_counts_the_forward_accepts(self):
        # Any ``h <= 64`` is legal for the forward and for the cuDNN backward,
        # including the 8 the hybrid-MLA fixtures use, so this backward must
        # cover the same range: the head-group width adapts to ``h`` instead of
        # demanding a multiple of 32. H=64 additionally exercises two head
        # groups, whose dkv parts are summed on the host.
        for h in (8, 16, H_SMALL, H_FULL):
            with self.subTest(h=h):
                q, kv, ti, dO = _inputs(h, 64, 64, 420 + h)
                got = _kernel_run(q, kv, ti, dO, 0.25)
                self._assert_matches_reference(got, q, kv, ti, dO, 0.25)
                again = _kernel_run(q, kv, ti, dO, 0.25)
                for name, a, b in zip(("dq", "dkv", "d_sink"), got, again):
                    self.assertTrue(np.array_equal(a, b), name)


# ===========================================================================
# 3. Host-side tiling in ``mqa_latent_sparse_bwd``
# ===========================================================================
@_TILELANG
class TestKernelTiling(_Base):
    """``block_h`` / ``dkv_buf_bytes`` / padding paths the PyLayer cannot reach.

    These arguments are host-side only: the PyLayer always takes the defaults,
    so they are driven directly against the same fp32 oracle.
    """

    # A budget of exactly ``_BLOCK_SIZE`` query rows forces chunking.
    def _tight_budget(self, topk):
        return topk * DK * 4 * _BLOCK_SIZE

    def test_chunked_equals_unchunked(self):
        q, kv, ti, dO = _inputs(H_SMALL, 128, 64, 500)
        whole = _kernel_run(q, kv, ti, dO, 0.25)
        chunked = _kernel_run(
            q, kv, ti, dO, 0.25, dkv_buf_bytes=self._tight_budget(64)
        )
        self._assert_matches_reference(chunked, q, kv, ti, dO, 0.25)
        # dq is a private per-row accumulator, so chunking cannot move it at
        # all. dkv and d_sink are cross-chunk sums, so they may differ by
        # summation order -- deterministically, but not bitwise.
        self.assertTrue(np.array_equal(chunked[0], whole[0]), "dq")
        self.assertLess(_relerr(chunked[1], whole[1]), REORDER_TOL, "dkv")
        self.assertLess(_relerr(chunked[2], whole[2]), REORDER_TOL, "d_sink")
        again = _kernel_run(
            q, kv, ti, dO, 0.25, dkv_buf_bytes=self._tight_budget(64)
        )
        for name, a, b in zip(("dq", "dkv", "d_sink"), chunked, again):
            self.assertTrue(np.array_equal(a, b), f"chunked {name} unstable")

    def test_topk_width_not_a_multiple_of_the_block(self):
        # 48 columns are padded up to 64 with -1 inside the kernel wrapper;
        # the pad slots must contribute nothing.
        q, kv, ti, dO = _inputs(H_SMALL, 64, 48, 510)
        self._assert_matches_reference(
            _kernel_run(q, kv, ti, dO, 0.25), q, kv, ti, dO, 0.25
        )

    def test_query_rows_not_a_multiple_of_the_chunk(self):
        # s=100 with a 32-row chunk pads the query axis to 128. The pad rows
        # are inert (q/dO zero, index row all -1), so the result must equal the
        # reference over the 100 real rows.
        q, kv, ti, dO = _inputs(H_SMALL, 100, 64, 520)
        got = _kernel_run(
            q, kv, ti, dO, 0.25, dkv_buf_bytes=self._tight_budget(64)
        )
        self.assertEqual(got[0].shape, (100, H_SMALL, DK))
        self._assert_matches_reference(got, q, kv, ti, dO, 0.25)

    def test_non_bf16_inputs_are_rejected(self):
        # The reused kernels are JIT-built for bf16, so an fp16 tensor would
        # otherwise be read by a bf16 kernel: wrong numbers, or an opaque
        # TileLang assert. The guard must name the tensor.
        b, s, h = 1, _BLOCK_SIZE, H_SMALL
        q, kv, ti, dO = _inputs(h, s, 32, 550)
        qb, kvb, out, lse, sink, ti_t = _eager_forward(q, kv, ti, 0.25)
        do = paddle.to_tensor(dO.reshape([b, s, h, DV])).cast(out.dtype)
        cases = {
            "query": (qb.cast("float16"), kvb, out, do),
            "kv": (qb, kvb.cast("float16"), out, do),
            "out": (qb, kvb, out.cast("float16"), do),
            "grad_out": (qb, kvb, out, do.cast("float16")),
        }
        for name, (q_t, kv_t, o_t, do_t) in cases.items():
            with (
                self.subTest(tensor=name),
                self.assertRaisesRegex(ValueError, f"{name} must be bfloat16"),
            ):
                mqa_latent_sparse_bwd(
                    q_t, kv_t, o_t, do_t, lse, sink, ti_t, float(SM)
                )

    def test_block_h_must_divide_the_head_count(self):
        q, kv, ti, dO = _inputs(H_SMALL, 32, 32, 530)
        with self.assertRaisesRegex(ValueError, "divisible by block_h"):
            _kernel_run(q, kv, ti, dO, 0.25, block_h=48)

    def test_shape_guards_raise_valueerror_not_assert(self):
        # ``python -O`` strips ``assert``, and these guard a TileLang JIT
        # specialisation plus a multi-GiB allocation, so they must be real
        # raises. ``AssertionError`` is not a subclass of ``ValueError``, so
        # each case here also pins that the conversion happened.
        b, s, h = 1, 32, H_SMALL
        q, kv, ti, dO = _inputs(h, s, 32, 540)
        qb, kvb, out, lse, sink, ti_t = _eager_forward(q, kv, ti, 0.25)
        do = paddle.to_tensor(dO.reshape([b, s, h, DV])).cast(out.dtype)
        good = (qb, kvb, out, do, lse, sink, ti_t, float(SM))
        cases = {
            "kv width": (
                qb,
                kvb[:, :, : DK - 16].contiguous(),
                out,
                do,
                lse,
                sink,
                ti_t,
                float(SM),
            ),
            # ``dv`` is read off ``grad_out``, so this one perturbs the head
            # axis instead: a truncated last axis would make ``out`` the first
            # guard to fire.
            "grad_out": (qb, kvb, out, do[:, :, :1, :], lse, sink, ti_t, SM),
            "out": (qb, kvb, out[:, :, :, :16], do, lse, sink, ti_t, SM),
            "lse": (qb, kvb, out, do, lse[:, :, :1], sink, ti_t, SM),
            "attn_sink": (qb, kvb, out, do, lse, sink[:1], ti_t, SM),
            "token_indices": (
                qb,
                kvb,
                out,
                do,
                lse,
                sink,
                ti_t[:, :1],
                float(SM),
            ),
        }
        for name, args in cases.items():
            with (
                self.subTest(guard=name),
                self.assertRaisesRegex(ValueError, name),
            ):
                mqa_latent_sparse_bwd(*args)
        # The same call with nothing perturbed must still go through, so a
        # guard cannot pass this test by rejecting everything.
        self.assertEqual(mqa_latent_sparse_bwd(*good)[0].shape, [b, s, h, DK])

    def test_width_guards_raise_before_any_kernel(self):
        # ``Dv > Dk`` and a non-multiple-of-16 width are rejected on the widths
        # alone, before the JIT and before ``dkv_buf`` is allocated, so these
        # need no real forward -- which is the point: an illegal width must not
        # reach TileLang at all.
        def call(dk, dv):
            b, s, h = 1, _BLOCK_SIZE, H_SMALL
            zeros = paddle.zeros
            return mqa_latent_sparse_bwd(
                zeros([b, s, h, dk], dtype="bfloat16"),
                zeros([b, s, dk], dtype="bfloat16"),
                zeros([b, s, h, dv], dtype="bfloat16"),
                zeros([b, s, h, dv], dtype="bfloat16"),
                zeros([b, s, h], dtype="float32"),
                zeros([h], dtype="float32"),
                paddle.full([b, s, _BLOCK_SIZE], -1, dtype="int32"),
                float(SM),
            )

        with self.assertRaisesRegex(ValueError, r"must be <= Dk"):
            call(DK, DK + 64)
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            call(24, 24)


# ===========================================================================
# 4. Host helpers -- pure Python, no kernels
# ===========================================================================
class TestHostHelpers(unittest.TestCase):
    """The three sizing helpers, which decide launch geometry.

    Getting these wrong is either a wrong answer (a chunk length the kernel's
    ``topk % 32`` / S-tiling contract rejects) or a layout-inference failure in
    ``dkv_reduce``, so they are checked without a GPU.
    """

    def test_pick_chunk_prefers_an_exact_divisor(self):
        # Budget large enough for all of s: one chunk, no padding.
        self.assertEqual(_pick_chunk(1, 64, 64, DK, 12 << 30), 64)
        # Budget for 32 rows: 32 divides 128 exactly.
        self.assertEqual(_pick_chunk(1, 128, 64, DK, 64 * DK * 4 * 32), 32)

    def test_pick_chunk_budgets_the_batch_axis(self):
        # ``dkv_buf`` is [b, sc, topk, dk]: at the same budget, doubling the
        # batch must halve the chunk, not silently double the allocation.
        budget = 64 * DK * 4 * 64
        self.assertEqual(_pick_chunk(1, 128, 64, DK, budget), 64)
        self.assertEqual(_pick_chunk(2, 128, 64, DK, budget), 32)
        for b in (1, 2, 4, 8):
            sc = _pick_chunk(b, 8192, 640, DK, 12 << 30)
            with self.subTest(b=b):
                self.assertLessEqual(b * sc * 640 * DK * 4, 12 << 30)

    def test_pick_chunk_is_always_block_aligned(self):
        for s in (32, 64, 96, 100, 128, 8192):
            for rows in (1, 8, 32, 4096):
                sc = _pick_chunk(1, s, 64, DK, 64 * DK * 4 * rows)
                with self.subTest(s=s, rows=rows):
                    self.assertGreaterEqual(sc, _BLOCK_SIZE)
                    self.assertLessEqual(sc, max(s, _BLOCK_SIZE))
                    # Multi-chunk lengths must be block aligned; the one
                    # exception is the single chunk that covers all of ``s``,
                    # where the kernel sees the sequence length itself.
                    self.assertTrue(
                        sc % _BLOCK_SIZE == 0 or sc == s, f"sc={sc}"
                    )

    def test_pick_chunk_falls_back_to_a_padded_tail(self):
        # No multiple of 32 divides 100, so the tail must be padded instead.
        sc = _pick_chunk(1, 100, 64, DK, 64 * DK * 4 * 32)
        self.assertEqual(sc, 32)
        self.assertNotEqual(100 % sc, 0)

    def test_pick_block_h_divides_every_legal_head_count(self):
        # The cuDNN backward takes any ``h <= 64``; the group width must too,
        # and stay within the 32 the shared-memory budget allows.
        for h in range(1, _DSA_HEADS + 1):
            bh = _pick_block_h(h)
            with self.subTest(h=h):
                self.assertEqual(h % bh, 0)
                self.assertLessEqual(bh, 32)
                self.assertGreaterEqual(bh, 1)
        self.assertEqual(_pick_block_h(64), 32)
        self.assertEqual(_pick_block_h(48), 16)
        self.assertEqual(_pick_block_h(24), 8)
        self.assertEqual(_pick_block_h(8), 8)
        # Odd head counts fall back to one head per launch rather than raising.
        self.assertEqual(_pick_block_h(7), 1)

    def test_reduce_threads_divides_dk(self):
        # ``dkv_reduce`` maps ``T.Parallel(Dk)`` onto threads and its layout
        # inference fails outright unless the block width divides Dk -- the
        # reason the library default of 128 is unusable at the absorbed 576.
        for dk in (DK, DV, 192, 64):
            with self.subTest(dk=dk):
                self.assertEqual(dk % _reduce_threads(dk), 0)
        self.assertEqual(_reduce_threads(DK), 192)
        self.assertEqual(_reduce_threads(DV), 256)
        # Nothing in the candidate list divides 48: fall back to 32.
        self.assertEqual(_reduce_threads(48), 32)

    def test_pad_rows_extends_axis_one_with_the_fill(self):
        x = paddle.ones([2, 3, 4], dtype="int32")
        padded = _pad_rows(x, 5, fill=-1)
        self.assertEqual(padded.shape, [2, 5, 4])
        got = padded.numpy()
        self.assertTrue(np.all(got[:, :3] == 1))
        self.assertTrue(np.all(got[:, 3:] == -1))


if __name__ == "__main__":
    unittest.main()
