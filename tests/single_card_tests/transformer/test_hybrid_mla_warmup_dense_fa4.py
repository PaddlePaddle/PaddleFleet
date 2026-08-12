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

"""The warmup phase's attention half runs as dense FA4, not as an s^2 table.

Phase 2 (``hybrid_mla_attention="mqa_dsa"`` +
``dsa_indexer_use_sparse_loss=False``) attends over the *whole* per-document
causal span. Expressing that span as an explicit ``[b, s, s]`` column table for
the FlashMLA sparse kernel costs ``O(s^2)`` memory for no benefit: measured peak
for the table alone is 2.1 GiB at s=8192, 27.3 GiB at s=32768 and 100.0 GiB at
s=65536, and past s=46336 the sparse kernel's own ``(b*s-1)*topk_padded`` index
arithmetic overflows int32 -- which does *not* reliably crash: s=46337 returns
finite but wrong numbers (see ``_assert_index_table_addressable``). So
``MQALatentAttention._dense_attn`` hands the same softmax to FA4's dense
flashmask instead, keeping the document structure in the caller's own
``startend_row_indices``.

What is pinned here:

* ``TestDensePathSelection`` -- exactly which configurations take the dense
  path. Phase 1 (``mqa_full_causal``) attends over the same whole span and shares
  ``_forward_full_causal``, so it goes dense too; only the phase-3 sparse forward,
  which genuinely selects 640 columns, still reaches ``_sparse_attn``.
* ``TestDenseWarmupPrecision`` -- the numerical case. Both backends are compared
  against the same fp32 eager reference, and the sparse path is used as the
  *yardstick*: replacing it is only legitimate if dense is at least as close to
  fp32 as sparse is, forward and backward. No hand-picked tolerance decides the
  verdict.
* ``TestDenseWarmupPadRows`` -- a pad row is a query row with *every* column
  masked, which is a softmax-over-nothing hazard the sparse path avoids
  structurally (all ``-1`` columns). FA4 returns exact zeros there instead of
  NaN; that is a kernel property, so it needs a regression test.
* ``TestFrozenSink`` -- ``train_indexer_only`` freezes ``softmax_offset``, and a
  ``PyLayer`` whose input has ``stop_gradient=True`` must get ``None`` back at
  that position, which ``flashmask_attention`` does not do. ``_dense_sink_arg``
  works around it; the workaround must be forward-neutral.

``FLAGS_flash_attn_version`` is process-global and a bare pytest process never
constructs ``TrainingArguments``, so it keeps the image default 2. Production on
this SM100 box gets 4 from ``training_args.py:1764-1780``; the tests pin it
explicitly with ``hybrid_mla_utils._flash_attn_version``.

Run:
    R=<erniebot checkout>
    PYTHONPATH=$R/third_party/PaddleFleet/src \\
        CUDA_VISIBLE_DEVICES=0 FLAGS_selected_gpus=0 \\
        python -m pytest <this file> -q
"""

import contextlib
import unittest

import paddle

from .hybrid_mla_utils import (
    _GPU,
    H,
    MQALatentAttention,
    _build_module,
    _create_mqa_config,
    _cudnn_deterministic,
    _dense_reference,
    _flash_attn_version,
    _make_inputs,
    _pad_row_end,
    _production_fa_version,
    _rel,
    _row_end,
)

# Per-head sink logits, ``H == 8``. Mixed signs and magnitudes so the sink is
# not a no-op on any head and its gradient is not symmetric.
_SINK = [0.5, -0.25, 0.0, 1.0, -1.5, 0.75, 0.25, -0.5]

# ``(doc_lens, seqlen)``. seqlen=512 is the discriminating shape: the phase-3
# sparse budget (WINDOW + INDEX_TOPK == 256) covers at most half of a row's
# causal span there, so "full causal" is distinguishable from "top-k".
_LAYOUTS = [
    ([512], 512),  # one document spanning the buffer
    ([200, 312], 512),  # two documents
    ([100, 50, 106], 256),  # three, none a multiple of the window
]

# Real pad rows: the trailing gap stays outside every document.
_PAD_LAYOUT = ([100, 50, 60], 256)


def _fa4_is_the_production_kernel():
    """``_production_fa_version()`` needs a CUDA device to answer.

    Evaluated at import time by the ``skipUnless`` below, so it must not raise on
    a CPU-only box -- an exception here would abort collection instead of
    skipping (and upstream CI only accepts pytest exit 0).
    """
    try:
        return _production_fa_version() == 4
    except Exception:
        return False


_FA4 = unittest.skipUnless(
    _fa4_is_the_production_kernel(),
    "the dense warmup path needs the FA4 (cute) kernel, which production only "
    "selects on SM100",
)


def _module(mode="mqa_dsa", sparse_loss=False, sink=None, loss_coeff=0.0):
    """A hybrid-MLA latent module with the phase fixed from construction.

    ``loss_coeff=0.0`` is deliberate for the numerical tests: it makes
    ``_needs_indexer_loss()`` false, so the forward is the attention half alone
    and a backward reaches only the attention inputs. Path-selection tests do not
    care either way.
    """
    config = _create_mqa_config(mode, loss_coeff=loss_coeff)
    config.dsa_indexer_use_sparse_loss = sparse_loss
    config.pad_token_id = 0
    module = _build_module(config, bf16=True, sink=sink)
    return module


@contextlib.contextmanager
def _backend_spy(module):
    """Record ``"dense"`` / ``"sparse"`` per attention call, in order."""
    calls = []
    real_dense, real_sparse = module._dense_attn, module._sparse_attn

    def dense(*args, **kwargs):
        calls.append("dense")
        return real_dense(*args, **kwargs)

    def sparse(*args, **kwargs):
        calls.append("sparse")
        return real_sparse(*args, **kwargs)

    module._dense_attn, module._sparse_attn = dense, sparse
    try:
        yield calls
    finally:
        module._dense_attn, module._sparse_attn = real_dense, real_sparse


def _inputs(seqlen, seed=1):
    """``((query, key, x, qr), w_v)`` as differentiable leaves."""
    query, key, w_v, x, qr = _make_inputs(seqlen, seed=seed, with_hidden=True)
    tensors = [query, key, x, qr]
    for tensor in tensors:
        tensor.stop_gradient = False
    w_v.stop_gradient = False
    return tensors, w_v


def _forward(module, tensors, row_end, w_v, training=True):
    module.train() if training else module.eval()
    query, key, x, qr = tensors
    return module(
        query,
        key,
        None,
        None,
        row_end,
        v_b_proj_weight=w_v,
        x=x,
        qr=qr,
    )


@_GPU
@_FA4
class TestDensePathSelection(unittest.TestCase):
    """Which configurations reach ``_dense_attn`` and which keep the table."""

    def _calls(self, module, fa_version, seqlen=256):
        row_end = _row_end([seqlen], seqlen)
        tensors, w_v = _inputs(seqlen)
        with _flash_attn_version(fa_version), _backend_spy(module) as calls:
            _forward(module, tensors, row_end, w_v)
        return calls

    def test_fa4_takes_the_dense_path(self):
        module = _module()
        self.assertEqual(self._calls(module, 4), ["dense"])

    def test_fa2_falls_back_to_the_sparse_table(self):
        """The fallback is what keeps a non-FA4 box working, so pin it too."""
        module = _module()
        self.assertEqual(self._calls(module, 2), ["sparse"])

    def test_cp_falls_back_to_the_sparse_table(self):
        """flashmask's causal diagonal assumes query row i == key column i.

        Under CP this rank owns a row *slice* against an all-gathered key, so the
        row indices in ``row_end`` no longer address its rows. Only the predicate
        is exercised here: building a real CP group is
        ``tests/multi_card_tests/transformer/test_mqa_dsa_cp.py``'s job.
        """
        module = _module()
        row_end = _row_end([256], 256)
        with _flash_attn_version(4):
            self.assertTrue(
                module._dense_can_serve_full_causal(576, 512, row_end)
            )
            module.cp_size = 2
            self.assertFalse(
                module._dense_can_serve_full_causal(576, 512, row_end)
            )

    def test_unsupported_head_dims_fall_back(self):
        """The head-dim pair comes from the layer, not from a constant."""
        module = _module()
        row_end = _row_end([256], 256)
        with _flash_attn_version(4):
            self.assertFalse(
                module._dense_can_serve_full_causal(576, 511, row_end)
            )

    def test_deterministic_falls_back_to_the_sparse_table(self):
        """(576, 512) has no deterministic FA4 backward.

        FA4 solves it with the big-head-dim kernel, which asserts
        ``not deterministic``. The accuracy-diff harnesses do set
        ``FLAGS_cudnn_deterministic=1``, so without the fallback in
        ``get_fa_version`` they would abort in the first backward rather than
        merely lose bit-reproducibility -- and this path used to be reachable for
        them, because before the head-dim pair was whitelisted it took FA2.
        """
        module = _module()
        row_end = _row_end([256], 256)
        with _flash_attn_version(4), _cudnn_deterministic(1):
            self.assertFalse(
                module._dense_can_serve_full_causal(576, 512, row_end)
            )
        # Nothing else about the layer changed: the same call goes dense again
        # once determinism is off.
        with _flash_attn_version(4), _cudnn_deterministic(0):
            self.assertTrue(
                module._dense_can_serve_full_causal(576, 512, row_end)
            )

    def test_deterministic_backward_runs_on_the_fallback(self):
        """The point of the fallback: a deterministic run must survive backward."""
        module = _module()
        seqlen = 256
        row_end = _row_end([seqlen], seqlen)
        tensors, w_v = _inputs(seqlen)
        with (
            _flash_attn_version(4),
            _cudnn_deterministic(1),
            _backend_spy(module) as calls,
        ):
            out = _forward(module, tensors, row_end, w_v)
            paddle.autograd.backward(out.cast("float32").sum())
        self.assertEqual(calls, ["sparse"])
        self.assertTrue(paddle.isfinite(out.cast("float32")).all())
        self.assertIsNotNone(tensors[0].grad)

    def test_full_causal_phase_also_takes_the_dense_path(self):
        """Phase 1 attends over the same whole causal span, so same backend.

        It shares ``_forward_full_causal`` with the warmup's attention half --
        the point of routing the choice there rather than into the warmup is that
        there is exactly one place where a column table can be built.
        """
        module = _module(mode="mqa")
        self.assertEqual(self._calls(module, 4), ["dense"])

    def test_full_causal_phase_keeps_the_fallback(self):
        module = _module(mode="mqa")
        self.assertEqual(self._calls(module, 2), ["sparse"])

    def test_sparse_phase_is_untouched(self):
        """Phase 3 selects 640 columns, not s -- 544 MiB at s=65536."""
        module = _module(sparse_loss=True)
        self.assertEqual(self._calls(module, 4), ["sparse"])


class TestSparseTableAddressability(unittest.TestCase):
    """The fallback's own ceiling, checked before the table is allocated.

    Pure index arithmetic, so no device is needed. The boundary values come from
    measurements on this box, quoted in
    ``MQALatentAttention._assert_index_table_addressable``: at ``topk == s`` the
    last clean length is 46336, and 46337 does *not* crash -- it returns wrong
    numbers for its last rows. That silent-corruption window is the reason the
    guard raises instead of trusting the kernel to fail.
    """

    _guard = staticmethod(MQALatentAttention._assert_index_table_addressable)

    def test_the_production_shape_is_far_from_the_limit(self):
        """s=8192 x b=8 is two orders of magnitude below the wrap."""
        self._guard(8, 8192, 8192)

    def test_last_addressable_length_is_accepted(self):
        self._guard(1, 46336, 46336)

    def test_first_overflowing_length_is_rejected(self):
        """The one that returns finite-but-wrong numbers unguarded."""
        with self.assertRaisesRegex(ValueError, "does not fit int32"):
            self._guard(1, 46337, 46337)

    def test_the_bound_is_on_b_times_rows_not_on_rows(self):
        """Batch multiplies the flat row index, so it moves the ceiling.

        The last addressable length at b=1 must be rejected at b=2 -- pinning
        this keeps the check from degrading into a plain sequence-length limit.
        Measured on the raw kernel the same way round: b=64, s=1024 crashes at a
        column count that b=1, s=65536 survives.
        """
        self._guard(1, 46336, 46336)
        with self.assertRaisesRegex(ValueError, "does not fit int32"):
            self._guard(2, 46336, 46336)

    def test_the_phase_3_budget_never_trips_it(self):
        """Only the full-causal fallback can reach the limit.

        Phase 3 passes ``topk = index_topk <= 2048``, which needs s > 1e6 -- so
        the guard must not become a surprise ceiling for the sparse phase.
        """
        self._guard(8, 65536, 2048)

    def test_the_column_count_is_padded_before_the_check(self):
        """The kernel pads ``topk`` up to 64, so the check has to as well.

        A budget of 2049 costs 2112 columns per row, not 2049; ignoring that
        would let a table through whose real stride overflows.
        """
        rows_ok = (2**31 - 1) // 2112 + 1
        self._guard(1, rows_ok, 2049)
        with self.assertRaisesRegex(ValueError, r"\* 2112 = "):
            self._guard(1, rows_ok + 1, 2049)

    def test_the_guard_runs_before_the_table_is_built(self):
        """Reached through the real builder, not only as a static method.

        ``_build_full_causal_indices`` would otherwise try to materialise
        ``[1, 46337, 46337]`` int32 (8.6 TiB) before anything checked it.
        """
        with self.assertRaisesRegex(ValueError, "does not fit int32"):
            MQALatentAttention._build_full_causal_indices(1, 46337, None, None)


def _loss_weight(seqlen, seed=7):
    """A fixed non-uniform output weighting, so ``dq`` is not the ``sum``
    special case (every row weighted identically) that hides row-dependent
    errors."""
    paddle.seed(seed)
    return paddle.randn([1, seqlen, H * 64], dtype="float32")


@_GPU
@_FA4
class TestDenseWarmupPrecision(unittest.TestCase):
    """Dense must be at least as close to fp32 as the sparse path it replaces.

    Both backends run on bitwise-identical inputs against one fp32 eager
    reference (``hybrid_mla_utils._dense_reference``, which *is* the mathematical
    dense-MHA path -- absorption is exactly score preserving). The sparse path's
    own error is the yardstick, so the verdict does not rest on a hand-picked
    tolerance; the absolute floors below only stop the assertion from tightening
    to zero when the yardstick happens to be unusually good on a shape.

    Measured relative Frobenius error against fp32, over the three layouts with
    and without a sink (dense / sparse):

        out    2.54-2.78e-3 / 2.53-2.78e-3   dense within 0.1% of sparse
        dq     3.54-3.86e-3 / 3.58-4.03e-3   dense *better* on every shape
        dkv    4.05-4.16e-3 / 3.56-3.69e-3   dense worse, worst ratio 1.13
        dw_v   3.36-3.58e-3 / 3.34-3.58e-3   within 0.3%

    So the only quantity where dense loses is ``dkv``, by 13% of an error that is
    itself bf16 rounding; the floors below are set just above the measured band.
    """

    # Both backends sit in the 3-4e-3 band against fp32 and neither is a
    # "reference" -- the fp32 eager path is. The floors are the measured worst
    # case plus ~40%, so a real regression cannot hide behind them, while
    # ``_SLACK`` absorbs which of two equally-wrong accumulation orders lands
    # closer on a given shape.
    _FORWARD_FLOOR = 3e-3
    _GRAD_FLOOR = 6e-3
    _SLACK = 1.5

    def _measure(self, row_end, seqlen, sink):
        """``(expected, got)``: fp32 autograd reference vs both backends."""
        module = _module(sink=sink)
        weight = _loss_weight(seqlen)

        tensors, w_v = _inputs(seqlen)
        ref = _dense_reference(
            tensors[0], tensors[1], w_v, row_end, module.softmax_scale, sink
        )
        paddle.autograd.backward((ref * weight).sum())
        expected = (ref, tensors[0].grad, tensors[1].grad, w_v.grad)

        got = {}
        for tag, version in (("dense", 4), ("sparse", 2)):
            module.clear_gradients()
            tensors, w_v = _inputs(seqlen)
            with _flash_attn_version(version), _backend_spy(module) as calls:
                out = _forward(module, tensors, row_end, w_v)
            self.assertEqual(calls, [tag])
            paddle.autograd.backward((out.cast("float32") * weight).sum())
            sink_grad = (
                None
                if module.softmax_offset is None
                else module.softmax_offset.grad.clone()
            )
            got[tag] = (
                out,
                tensors[0].grad,
                tensors[1].grad,
                w_v.grad,
                sink_grad,
            )
        return expected, got

    def _assert_no_worse(self, doc_lens, seqlen, sink):
        row_end = _row_end(doc_lens, seqlen)
        expected, got = self._measure(row_end, seqlen, sink)
        for i, name in enumerate(("out", "dq", "dkv", "dw_v")):
            floor = self._FORWARD_FLOOR if i == 0 else self._GRAD_FLOOR
            dense = _rel(got["dense"][i], expected[i])
            sparse = _rel(got["sparse"][i], expected[i])
            with self.subTest(
                quantity=name, docs=doc_lens, sink=sink is not None
            ):
                self.assertTrue(
                    paddle.isfinite(got["dense"][i].cast("float32")).all()
                )
                self.assertLessEqual(
                    dense,
                    max(sparse * self._SLACK, floor),
                    f"{name}: dense rel {dense:.3e} vs fp32 is worse than the "
                    f"sparse path's {sparse:.3e} (docs={doc_lens}, s={seqlen}, "
                    f"sink={sink is not None})",
                )
        return got

    def test_forward_and_grads_no_worse_than_sparse_sinkless(self):
        for doc_lens, seqlen in _LAYOUTS:
            self._assert_no_worse(doc_lens, seqlen, None)

    def test_forward_and_grads_no_worse_than_sparse_with_sink(self):
        for doc_lens, seqlen in _LAYOUTS:
            self._assert_no_worse(doc_lens, seqlen, _SINK)

    def test_sink_gradient_cross_validates(self):
        """Two independent computations of ``d_sink`` must agree.

        FA4 differentiates the sink column natively. The sparse backend cannot:
        the SM100 kernel returns an all-zero ``d_sink`` on the ``d_qk != d_v``
        branch, so ``mqa_sparse_attn`` computes it analytically from the KV-only
        LSE. Neither is a reference for the other, which is exactly why their
        agreement is worth pinning -- a sign or scale error on either side would
        show up here and nowhere else.

        Measured: identical to the last bit on all three layouts (max abs diff
        0.0, with ``|d_sink|`` up to 4.4), though ``d_sink`` is bf16, so read that
        as agreement within bf16 resolution rather than as identical arithmetic.
        """
        for doc_lens, seqlen in _LAYOUTS:
            row_end = _row_end(doc_lens, seqlen)
            _, got = self._measure(row_end, seqlen, _SINK)
            dense, sparse = got["dense"][4], got["sparse"][4]
            with self.subTest(docs=doc_lens):
                self.assertIsNotNone(dense)
                self.assertTrue(paddle.isfinite(dense.cast("float32")).all())
                self.assertGreater(
                    float(dense.cast("float32").abs().max()), 0.0
                )
                self.assertLessEqual(_rel(dense, sparse), self._GRAD_FLOOR)


@_GPU
@_FA4
class TestDenseWarmupPadRows(unittest.TestCase):
    """A fully-masked query row must give zeros, not NaN.

    ``_row_end`` fills the trailing gap with ``seqlen``, which
    ``_derive_csa_doc_boundaries`` reads as one more document, so it produces no
    pad rows at all; ``_pad_row_end`` repeats the last document's end instead,
    which is the ``is_valid == False`` state a packed batch's padding actually
    takes. On the sparse path such a row gets an all-``-1`` column list and the
    kernel is structurally safe. On the dense path it is softmax over an
    all-masked row -- a kernel property of FA4, hence a regression test.
    """

    def _run(self, sink):
        doc_lens, seqlen = _PAD_LAYOUT
        row_end = _pad_row_end(doc_lens, seqlen)
        module = _module(sink=sink)
        tensors, w_v = _inputs(seqlen)
        with _flash_attn_version(4), _backend_spy(module) as calls:
            out = _forward(module, tensors, row_end, w_v)
        self.assertEqual(calls, ["dense"])
        paddle.autograd.backward(out.cast("float32").sum())
        return out.cast("float32"), tensors, w_v

    def _assert_pad_rows_vanish(self, sink):
        out, tensors, w_v = self._run(sink)
        body = sum(_PAD_LAYOUT[0])
        with self.subTest(sink=sink is not None):
            self.assertTrue(paddle.isfinite(out).all())
            self.assertEqual(float(out[0, body:].abs().max()), 0.0)
            self.assertGreater(float(out[0, :body].abs().max()), 0.0)
            for name, tensor in (
                ("dq", tensors[0].grad),
                ("dkv", tensors[1].grad),
                ("dw_v", w_v.grad),
            ):
                self.assertIsNotNone(tensor, name)
                self.assertTrue(
                    paddle.isfinite(tensor.cast("float32")).all(), name
                )

    def test_pad_rows_are_zero_sinkless(self):
        self._assert_pad_rows_vanish(None)

    def test_pad_rows_are_zero_with_sink(self):
        """With a sink the row is *not* empty -- the sink column is always there.

        So this is the stronger statement: the zeroing comes from the caller's
        ``is_valid`` masking downstream of the kernel, not from an accidental
        ``exp(-inf)/exp(-inf)``.
        """
        self._assert_pad_rows_vanish(_SINK)


@_GPU
@_FA4
class TestFrozenSink(unittest.TestCase):
    """``train_indexer_only`` freezes the sink; the dense path must survive it.

    A ``PyLayer`` whose input arrives with ``stop_gradient=True`` must get
    ``None`` back at that position, and ``flashmask_attention`` does not do that
    -- it aborts with "backward function should return None at N position".
    ``_dense_sink_arg`` hands the kernel a ``stop_gradient=False`` detached proxy
    instead. Neither ``.detach()`` alone nor ``* 1.0`` works, and the sparse
    backend needs none of this (``csa_sparse_attn.py:178-182`` records
    ``attn_sink_needs_grad`` itself), so the workaround is dense-only and its
    forward-neutrality is what has to be pinned.
    """

    def _forward_with(self, freeze):
        doc_lens, seqlen = _LAYOUTS[1]
        row_end = _row_end(doc_lens, seqlen)
        module = _module(sink=_SINK)
        module.softmax_offset.stop_gradient = freeze
        tensors, w_v = _inputs(seqlen)
        with _flash_attn_version(4), _backend_spy(module) as calls:
            out = _forward(module, tensors, row_end, w_v)
        self.assertEqual(calls, ["dense"])
        paddle.autograd.backward(out.cast("float32").sum())
        return out, module.softmax_offset, tensors

    def test_frozen_sink_backward_runs_and_gets_no_gradient(self):
        out, sink, tensors = self._forward_with(freeze=True)
        self.assertTrue(paddle.isfinite(out.cast("float32")).all())
        self.assertIsNone(sink.grad)
        self.assertIsNotNone(tensors[0].grad)
        self.assertTrue(paddle.isfinite(tensors[0].grad.cast("float32")).all())

    def test_freezing_the_sink_does_not_change_the_forward(self):
        """The proxy must be numerically the same tensor, bit for bit."""
        frozen, _, _ = self._forward_with(freeze=True)
        trainable, sink, _ = self._forward_with(freeze=False)
        self.assertIsNotNone(sink.grad)
        # ``equal_all`` has no bfloat16 kernel; the cast is exact.
        self.assertTrue(
            paddle.equal_all(frozen.cast("float32"), trainable.cast("float32"))
        )
