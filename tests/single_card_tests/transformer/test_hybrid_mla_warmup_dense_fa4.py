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

"""The full-causal phases run as dense FA4, and refuse to run as anything else.

Phase 1 (``hybrid_mla_attention="mqa_full_causal"``) and phase 2
(``"mqa_dsa"`` + ``dsa_indexer_use_sparse_loss=False``) both attend over the
*whole* per-document causal span. Expressing that span as an explicit
``[b, s, s]`` column table for the FlashMLA sparse kernel costs ``O(s^2)`` memory
for no benefit: measured net allocation for the table alone is 1.6 GiB at s=8192,
25.0 GiB at s=32768 and 100.0 GiB at s=65536 (against 3.3 / 13.0 / 26.0 GiB for a
dense forward *and* backward), and past s=46336 the sparse kernel's own
``(b*s-1)*topk_padded`` index arithmetic overflows int32 -- which does *not*
reliably crash: s=46337 returns finite but wrong numbers for its last 55 rows.
So ``MQALatentAttention._dense_attn`` hands the same softmax to FA4's dense
flashmask instead, keeping the document structure in the caller's own
``startend_row_indices`` -- and ``_assert_dense_fa4`` raises rather than
substituting the table when FA4 is not what the environment resolves to.

What is pinned here:

* ``TestBackendSelection`` -- the dense path is taken whenever it can be, and
  every way of not resolving to FA4 raises instead of degrading. Phase 1 shares
  ``_forward_full_causal`` with the warmup's attention half, so both are covered
  by one selection point; phase 3, which genuinely selects 640 columns, keeps
  ``_sparse_attn`` and is *not* subject to the assertion.
* ``TestCpRowBounds`` -- context parallelism keeps the dense path, which needs
  the causal diagonal expressed as an explicit flashmask bound in this rank's
  row space instead of the kernel's own bottom-right-aligned one.
* ``TestDenseWarmupPrecision`` -- the numerical case. Both backends are compared
  against the same fp32 eager reference, and the sparse path is used as the
  *yardstick*: replacing it is only legitimate if dense is at least as close to
  fp32 as sparse is, forward and backward. No hand-picked tolerance decides the
  verdict. Since production has no sparse full-causal path any more, the
  yardstick is built here, by feeding ``_sparse_attn`` the explicit table from
  ``hybrid_mla_utils._full_causal_indices``.
* ``TestDenseWarmupPadRows`` -- a pad row is a query row with *every* column
  masked, which is a softmax-over-nothing hazard the sparse path avoids
  structurally (all ``-1`` columns). FA4 returns exact zeros there instead of
  NaN; that is a kernel property, so it needs a regression test.
* ``TestFrozenInputs`` -- ``train_indexer_only`` freezes the backbone and can
  freeze ``softmax_offset``, and a ``PyLayer`` whose input has
  ``stop_gradient=True`` must get ``None`` back at that position, which
  ``flashmask_attention`` does not do. ``_dense_pylayer_inputs`` works around
  it, for the sink and for a frozen q/k/v alike; the workaround must be
  forward-neutral.

``FLAGS_flash_attn_version`` is process-global and a bare pytest process never
constructs ``TrainingArguments``, so it keeps the image default 2. Production on
this SM100 box gets 4 from ``training_args.py:1764-1780``. Every case that runs a
full-causal forward therefore has to pin it with
``hybrid_mla_utils._flash_attn_version(4)`` -- without that the forward now
raises rather than quietly taking a second backend.

Run:
    R=<erniebot checkout>
    PYTHONPATH=$R/third_party/PaddleFleet/src \\
        CUDA_VISIBLE_DEVICES=0 FLAGS_selected_gpus=0 \\
        python -m pytest <this file> -q
"""

import contextlib
import types
import unittest

import numpy as np
import paddle

from paddlefleet.transformer.csa_attention import _derive_csa_doc_boundaries

from .hybrid_mla_utils import (
    _GPU,
    DV,
    H,
    MQALatentAttention,
    _build_module,
    _create_mqa_config,
    _cudnn_deterministic,
    _dense_reference,
    _flash_attn_version,
    _full_causal_indices,
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
    "the dense warmup path needs the FA4 (cute) kernel, which production selects "
    "from the compute capability -- the cutedsl kernels are available from SM90 "
    "on, but only SM100 derives FLAGS_flash_attn_version=4",
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


def _sparse_full_causal(module, tensors, row_end, w_v):
    """The full-causal softmax through the *sparse* kernel, as a yardstick.

    Production no longer has this path: ``_forward_full_causal`` only knows
    ``_dense_attn``. But the sparse kernel is what the dense one replaced, and
    "at least as accurate as what it replaced" is the only tolerance-free way to
    judge the replacement, so the comparison is reconstructed here out of the
    same two production pieces the phase-3 forward uses -- ``_sparse_attn`` on an
    explicit column table, then ``_deabsorb``.

    Mirrors ``_forward_sparse``'s tail exactly (``mqa_latent_attention.py``
    ``:1147-1150``): no indexer, no ``indexer_topk``, and the table is the whole
    per-document causal span rather than window + top-k.
    """
    query, key = tensors[0], tensors[1]
    seqlen = int(query.shape[1])
    kv = key.squeeze(2).contiguous()
    with paddle.no_grad():
        doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(
            row_end, seqlen
        )
    table = _full_causal_indices(1, seqlen, doc_start, is_valid)
    core_out = module._sparse_attn(query, kv, table, module.softmax_scale, DV)
    return module._deabsorb(core_out, w_v, module.split_kv_b)


@_GPU
@_FA4
class TestBackendSelection(unittest.TestCase):
    """Dense whenever FA4 serves it, a hard error whenever it does not.

    The full-causal phases have one backend. Not resolving to FA4 used to mean
    falling back to an ``O(s^2)`` column table; it now means refusing to start,
    so each way of not resolving to FA4 has to be pinned as a raise -- otherwise
    a silent change of backend could reappear and only show up as a memory
    figure.
    """

    def _calls(self, module, fa_version, seqlen=256):
        row_end = _row_end([seqlen], seqlen)
        tensors, w_v = _inputs(seqlen)
        with _flash_attn_version(fa_version), _backend_spy(module) as calls:
            _forward(module, tensors, row_end, w_v)
        return calls

    def _assert_forward_refuses(self, module, fa_version=4, deterministic=0):
        """The forward must raise, and must do so before touching a backend."""
        seqlen = 256
        row_end = _row_end([seqlen], seqlen)
        tensors, w_v = _inputs(seqlen)
        with (
            _flash_attn_version(fa_version),
            _cudnn_deterministic(deterministic),
            _backend_spy(module) as calls,
            self.assertRaisesRegex(RuntimeError, "requires FA4"),
        ):
            _forward(module, tensors, row_end, w_v)
        self.assertEqual(calls, [])

    def test_fa4_takes_the_dense_path(self):
        module = _module()
        self.assertEqual(self._calls(module, 4), ["dense"])

    def test_fa2_raises(self):
        """No FA4, no full-causal attention -- and no s^2 table either."""
        self._assert_forward_refuses(_module(), fa_version=2)

    def test_the_message_names_all_three_inputs(self):
        """The head-dim pair alone rarely explains ``get_fa_version``'s answer.

        It reads a whitelist *and* two process-global flags, so a message that
        only quoted the head dims would send the reader looking at the layer
        when the cause is the environment.
        """
        module = _module()
        seqlen = 256
        row_end = _row_end([seqlen], seqlen)
        tensors, w_v = _inputs(seqlen)
        with _flash_attn_version(2), self.assertRaises(RuntimeError) as caught:
            _forward(module, tensors, row_end, w_v)
        message = str(caught.exception)
        self.assertIn("(576, 512)", message)
        self.assertIn("FLAGS_flash_attn_version=2", message)
        self.assertIn("FLAGS_cudnn_deterministic", message)

    def test_cp_is_not_a_rejection_condition(self):
        """CP keeps the dense path -- ``_cp_row_bounds`` handles it.

        The row indices in ``row_end`` address *global* rows while this rank owns
        a row slice, so they cannot be handed to the kernel as they are; but
        localising them is arithmetic, not an obstacle, and it is the same
        contract the rest of the model's CP attention uses. An earlier revision
        did refuse under CP, which is precisely the geometry where the ``s^2``
        table is least affordable -- its column count is ``s_global``. The
        localisation itself is checked by ``TestCpRowBounds`` and end to end by
        ``tests/multi_card_tests/transformer/test_mqa_dsa_warmup_cp.py``.
        """
        module = _module()
        row_end = _row_end([256], 256)
        with _flash_attn_version(4):
            module.cp_size, module.cp_rank = 2, 1
            self.assertIsNone(module._assert_dense_fa4(576, 512, row_end))

    def test_unsupported_head_dims_raise(self):
        """The head-dim pair comes from the layer, not from a constant."""
        module = _module()
        row_end = _row_end([256], 256)
        with (
            _flash_attn_version(4),
            self.assertRaisesRegex(RuntimeError, r"\(576, 511\)"),
        ):
            module._assert_dense_fa4(576, 511, row_end)

    def test_deterministic_raises(self):
        """(576, 512) has no deterministic FA4 backward.

        FA4 solves it with the big-head-dim kernel, which asserts
        ``not deterministic`` (``flash_mask/cute/interface.py:1238,1249``), so
        ``get_fa_version`` degrades the pair under ``FLAGS_cudnn_deterministic``.
        For these phases that degradation is not usable, and a Python error
        naming the flag is more actionable than the kernel's own assert firing in
        the middle of a backward.
        """
        module = _module()
        row_end = _row_end([256], 256)
        with (
            _flash_attn_version(4),
            _cudnn_deterministic(1),
            self.assertRaisesRegex(RuntimeError, "cudnn_deterministic"),
        ):
            module._assert_dense_fa4(576, 512, row_end)
        # Nothing else about the layer changed: the same call is fine again
        # once determinism is off.
        with _flash_attn_version(4), _cudnn_deterministic(0):
            self.assertIsNone(module._assert_dense_fa4(576, 512, row_end))

    def test_deterministic_forward_raises_before_the_backward(self):
        """The error has to arrive in the forward, not in the backward.

        Reaching the kernel's own ``assert not deterministic`` would mean a step
        that had already spent its forward, with a message that names neither
        the layer nor the flag.
        """
        self._assert_forward_refuses(_module(), deterministic=1)

    def test_full_causal_phase_also_takes_the_dense_path(self):
        """Phase 1 attends over the same whole causal span, so same backend.

        It shares ``_forward_full_causal`` with the warmup's attention half --
        the point of routing the check there rather than into the warmup is that
        there is exactly one selection point for both phases.
        """
        module = _module(mode="mqa")
        self.assertEqual(self._calls(module, 4), ["dense"])

    def test_full_causal_phase_raises_on_fa2(self):
        self._assert_forward_refuses(_module(mode="mqa"), fa_version=2)

    def test_sparse_phase_is_untouched(self):
        """Phase 3 selects 640 columns, not s -- 544 MiB at s=65536."""
        module = _module(sparse_loss=True)
        self.assertEqual(self._calls(module, 4), ["sparse"])

    def test_the_assertion_is_scoped_to_the_full_causal_phases(self):
        """Phase 3 does not go through FA4 at all, so it must not be gated.

        Its forward is the FlashMLA sparse kernel, whose availability has nothing
        to do with ``FLAGS_flash_attn_version``. Pinning this keeps the new hard
        error from spreading into the phase that legitimately selects columns.
        """
        module = _module(sparse_loss=True)
        self.assertEqual(self._calls(module, 2), ["sparse"])


class TestCpRowBounds(unittest.TestCase):
    """``_cp_row_bounds`` must reproduce the global mask on every rank.

    Under CP the dense path cannot use the kernel's own ``causal=True``: FA4
    bottom-right-aligns the diagonal at ``seqlen_k - seqlen_q``, which is this
    rank's offset only on the last rank, and ``row_end``'s values are global row
    ids compared against local rows. So the diagonal becomes an explicit second
    bound and both bounds are shifted into local row space -- the contract
    ``DotProductAttention`` and the HySparse MLA scorer already use.

    The assertion is the mask itself rather than the numbers in the bounds: with
    ``causal=False`` and two bounds the kernel masks ``row >= LTS or row < UTE``
    (``flash_mask/cute/mask.py:513-518``), i.e. column ``j`` is visible on rows
    ``[UTE_j, LTS_j)``. Decoding the returned pair through *that* rule and
    comparing against a brute-force per-document causal mask, sliced to this
    rank's rows, is what makes the test independent of how the shift is spelled.

    Pure integer arithmetic on 1-element-wide bounds, so no device is needed --
    ``preprocess_index`` is a subtract plus a clip.
    """

    @staticmethod
    def _localise(row_end, cp_rank, s_local):
        """The method under test, on a stand-in carrying only ``cp_rank``.

        Bound as a plain function so the case does not need a built layer (and
        therefore a GPU) to exercise index arithmetic.
        """
        return MQALatentAttention._cp_row_bounds(
            types.SimpleNamespace(cp_rank=cp_rank), row_end, s_local
        )

    @staticmethod
    def _global_mask(doc_lens, s_global):
        """``[s_global, s_global]`` bool: per-document causal, brute force."""
        mask = np.zeros([s_global, s_global], dtype=bool)
        pos = 0
        for length in doc_lens:
            end = min(pos + length, s_global)
            for row in range(pos, end):
                mask[row, pos : row + 1] = True
            pos = end
        return mask

    def _assert_rank_mask(self, doc_lens, s_global, cp_size):
        row_end = _row_end(doc_lens, s_global)
        s_local = s_global // cp_size
        expected = self._global_mask(doc_lens, s_global)
        for cp_rank in range(cp_size):
            bounds = self._localise(row_end, cp_rank, s_local).numpy()[0, 0]
            lts, ute = bounds[:, 0], bounds[:, 1]
            rows = np.arange(s_local).reshape([s_local, 1])
            got = ~((rows >= lts) | (rows < ute))
            base = cp_rank * s_local
            with self.subTest(cp_size=cp_size, cp_rank=cp_rank, docs=doc_lens):
                self.assertEqual(bounds.shape, (s_global, 2))
                np.testing.assert_array_equal(
                    got, expected[base : base + s_local]
                )

    def test_single_rank_reproduces_the_kernel_causal_mask(self):
        """cp_size=1 is the calibration: the pair must equal ``causal=True``.

        ``_dense_attn`` does not take this branch at cp_size=1, but if the two
        formulations disagreed here they would disagree everywhere, and this is
        the one shape where the kernel's own diagonal is known to be right.
        """
        for doc_lens in ([16], [7, 9], [5, 3, 8]):
            self._assert_rank_mask(doc_lens, 16, 1)

    def test_every_rank_sees_its_own_slice(self):
        for cp_size in (2, 4):
            for doc_lens in ([16], [8, 8], [7, 9], [5, 3, 8], [1, 14, 1]):
                self._assert_rank_mask(doc_lens, 16, cp_size)

    def test_documents_are_not_leaked_across_ranks(self):
        """The failure mode the localisation exists to prevent.

        Handing the kernel the *unshifted* global bounds with ``causal=True``
        leaves an earlier document fully visible to a later rank -- rank 1's
        local row 0 is global row 8, and every column of document A (0-7) sits
        below the bottom-right diagonal with ``LTS`` values it never reaches. So
        pin that document A contributes nothing on rank 1, and that rank 0 sees
        nothing of document B.
        """
        row_end = _row_end([8, 8], 16)
        first = self._localise(row_end, cp_rank=1, s_local=8).numpy()[0, 0]
        # Document A masked from local row 0 on == masked everywhere.
        np.testing.assert_array_equal(first[:8, 0], np.zeros(8, dtype="int32"))
        # Document B keeps its own diagonal, now in local rows.
        np.testing.assert_array_equal(first[8:, 1], np.arange(8, dtype="int32"))

        second = self._localise(row_end, cp_rank=0, s_local=8).numpy()[0, 0]
        # Document B is entirely in the future: visible from local row 8, which
        # this rank does not have.
        np.testing.assert_array_equal(second[8:, 1], np.full(8, 8, "int32"))
        np.testing.assert_array_equal(second[:8, 0], np.full(8, 8, "int32"))

    def test_pad_rows_stay_empty_on_every_rank(self):
        """A pad row must not acquire columns from the shift.

        ``_pad_row_end`` repeats the last document's end, so the tail rows are
        past every ``LTS``. Clipping must keep them that way rather than folding
        them to ``s_local``, which would make them fully *visible*.
        """
        doc_lens, s_global = [4, 4], 16
        row_end = _pad_row_end(doc_lens, s_global)
        for cp_size in (2, 4):
            s_local = s_global // cp_size
            for cp_rank in range(cp_size):
                bounds = self._localise(row_end, cp_rank, s_local).numpy()[0, 0]
                lts, ute = bounds[:, 0], bounds[:, 1]
                rows = np.arange(s_local).reshape([s_local, 1])
                got = ~((rows >= lts) | (rows < ute))
                base = cp_rank * s_local
                pad = max(0, min(s_local, base + s_local - sum(doc_lens)))
                with self.subTest(cp_size=cp_size, cp_rank=cp_rank):
                    self.assertFalse(got[s_local - pad :].any())


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

    The yardstick is built by ``_sparse_full_causal`` rather than by flipping a
    flag: production has one backend for these phases, so the sparse comparison
    exists only here, out of ``_sparse_attn`` plus an explicit column table.

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
        for tag in ("dense", "sparse"):
            module.clear_gradients()
            tensors, w_v = _inputs(seqlen)
            if tag == "dense":
                # The production path, through the module's own forward.
                with _flash_attn_version(4), _backend_spy(module) as calls:
                    out = _forward(module, tensors, row_end, w_v)
                self.assertEqual(calls, ["dense"])
            else:
                # The yardstick, assembled from the same two production pieces
                # phase 3 uses. Not reachable through the module's forward any
                # more, which is the point of the comparison.
                module.train()
                out = _sparse_full_causal(module, tensors, row_end, w_v)
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
class TestFrozenInputs(unittest.TestCase):
    """Frozen inputs on the dense path, which ``train_indexer_only`` produces.

    A ``PyLayer`` whose input arrives with ``stop_gradient=True`` must get
    ``None`` back at that position, and ``flashmask_attention`` does not do that
    -- it aborts with "backward function should return None at N position".
    ``_dense_pylayer_inputs`` hands the kernel a ``stop_gradient=False``
    detached proxy of every frozen input instead. Neither ``.detach()`` alone
    nor ``* 1.0`` works, and the sparse backend needs none of this
    (``csa_sparse_attn.py:178-182`` records ``attn_sink_needs_grad`` itself), so
    the workaround is dense-only and its forward-neutrality is what has to be
    pinned.

    Which subset arrives frozen is a property of the configuration, not of this
    layer: ``train_indexer_only`` freezes the backbone (so q/kv reach the kernel
    frozen) while the sink is a parameter of its own, so all four combinations
    below are reachable.
    """

    def _forward_with(self, freeze, freeze_qkv=False):
        doc_lens, seqlen = _LAYOUTS[1]
        row_end = _row_end(doc_lens, seqlen)
        module = _module(sink=_SINK)
        module.softmax_offset.stop_gradient = freeze
        tensors, w_v = _inputs(seqlen)
        if freeze_qkv:
            # ``query`` / ``key`` are what become the kernel's q/k/v; freezing
            # them while the sink stays trainable is the *partially* frozen case
            # that trips the contract at position 1 rather than 3.
            tensors[0].stop_gradient = True
            tensors[1].stop_gradient = True
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

    def test_frozen_qkv_backward_runs_and_leaves_them_alone(self):
        """Frozen q/kv with a trainable sink: only the sink may receive."""
        out, sink, tensors = self._forward_with(freeze=False, freeze_qkv=True)
        self.assertTrue(paddle.isfinite(out.cast("float32")).all())
        self.assertIsNotNone(sink.grad)
        self.assertTrue(paddle.isfinite(sink.grad.cast("float32")).all())
        self.assertIsNone(tensors[0].grad, "frozen query received a gradient")
        self.assertIsNone(tensors[1].grad, "frozen key received a gradient")

    def test_everything_frozen_builds_no_grad_node(self):
        """Nothing wants a gradient, so no proxy and no contract to satisfy."""
        out, sink, tensors = self._forward_with(freeze=True, freeze_qkv=True)
        self.assertTrue(paddle.isfinite(out.cast("float32")).all())
        self.assertIsNone(sink.grad)
        self.assertIsNone(tensors[0].grad)
        self.assertIsNone(tensors[1].grad)

    def test_freezing_does_not_change_the_forward(self):
        """The proxy must be numerically the same tensor, bit for bit."""
        trainable, sink, _ = self._forward_with(freeze=False)
        self.assertIsNotNone(sink.grad)
        reference = trainable.cast("float32")
        for label, kwargs in (
            ("frozen sink", {"freeze": True}),
            ("frozen q/kv", {"freeze": False, "freeze_qkv": True}),
            ("all frozen", {"freeze": True, "freeze_qkv": True}),
        ):
            with self.subTest(case=label):
                out, _, _ = self._forward_with(**kwargs)
                # ``equal_all`` has no bfloat16 kernel; the cast is exact.
                self.assertTrue(
                    paddle.equal_all(out.cast("float32"), reference),
                    f"{label} changed the forward",
                )
