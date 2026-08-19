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

"""The warmup-phase KL on the dense cuDNN indexer backend.

``csa_indexer_backend="cudnn"`` routes ``MQALatentAttention._forward_warmup``
onto ``_warmup_kl_dense_cudnn`` (``mqa_latent_attention.py:857``), which scores
the *same* full per-document causal candidate set as the tilelang default but
through the three dense cuDNN-frontend ops in
``paddlefleet.cudnn_ops.indexer.dense_indexer_kl_cudnn``. It exists because the
tilelang full-candidate indexer sizes two bitonic buffers at ``2 * topk`` and so
cannot be launched past ``_TILELANG_KL_MAX_WIDTH`` columns, which the production
64k/cp=8 geometry exceeds by 4.6x.

The dense path is not a reimplementation of the same arithmetic: it derives both
distributions from raw score matrices, keeps the weights in ``DSAIndexer``'s
pre-baked scaling with ``sm_scale=1.0``, computes the attention per-head LSE
itself (``_dense_kl_attn_lse``), and folds ``1 / total_q`` into the kernel's
``grad_scale``. Each of those is a place the two backends could silently
disagree, so what is pinned here is the *objective*, not the plumbing:

* ``test_logged_kl_matches_tilelang`` -- the forward scalar, over seven layouts
  (single document, packed, unaligned packed, real pad rows, ``input_ids`` loss
  mask, and both maskings at once). This is the tight assertion: it covers the
  THD descriptor, the document borders, the row gating, ``num_rows``/``coeff``
  and the target definition all at once.
* ``test_matches_fp32_reference`` -- both backends against the same objective
  written a third time in plain fp32 paddle with autograd. In warmup the indexer
  does not touch the attention output, so its parameter gradients are pure KL
  gradients. cuDNN dense lands 5-8x closer to fp32 than tilelang does, which is
  why the *inter-backend* gradient gap below is loose.
* ``test_logged_kl_is_reproducible`` -- two identical steps, needed because
  warmup runs under recompute in production.
* ``test_loss_coeff_is_linear`` -- ``dsa_indexer_loss_coeff`` scales the logged
  scalar and the gradients exactly.
* ``test_masked_rows_do_not_affect_the_loss`` -- pad rows are switched out
  through ``index_lse = +inf`` / ``attn_lse = +inf``, not through a clamp.
* ``test_narrow_index_heads_raise_from_the_forward`` -- the ``>= 64`` head guard
  fires at the call site, not from a recompute replay of the backward.

Shared fixtures come from ``hybrid_mla_utils``; its ``INDEX_HEADS = 64``
satisfies both backends' head-count constraints.

Run::

    R=<erniebot checkout>
    PYTHONPATH=$R/third_party/PaddleFleet/src:$R/third_party/PaddleFormers \\
        CUDA_VISIBLE_DEVICES=5 FLAGS_selected_gpus=0 \\
        python -m pytest <this file> -q -p no:randomly
"""

import traceback
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.csa_attention import _derive_csa_doc_boundaries
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper

from .hybrid_mla_utils import (
    _GPU,
    _build_module,
    _create_mqa_config,
    _fa4_module_hooks,
    _make_inputs,
    _pad_row_end,
    _row_end,
)

setUpModule, tearDownModule = _fa4_module_hooks()

_EPS = 1e-10  # mqa_latent_attention._EPS

_COEFF = 0.01

# ``(label, seqlen, row_end_fn, doc_lens, with_input_ids)``. ``_row_end`` folds a
# trailing gap into one more *valid* document, so genuine pad rows need
# ``_pad_row_end``; ``with_input_ids`` additionally exercises the ``loss_mask``
# row gating, which uses a different denominator (``valid_rows`` instead of
# ``s_local``) and drops the ``1 / cp_size`` factor.
_SCENARIOS = [
    ("single-doc s=128", 128, _row_end, [128], False),
    ("single-doc s=512", 512, _row_end, [512], False),
    ("packed s=512", 512, _row_end, [100, 200, 212], False),
    ("unaligned packed s=384", 384, _row_end, [1, 130, 253], False),
    ("pad rows s=512", 512, _pad_row_end, [200, 180], False),
    ("loss_mask s=512", 512, _row_end, [200, 312], True),
    ("pad+loss_mask s=512", 512, _pad_row_end, [200, 180], True),
]

# The forward scalar is the assertion that discriminates: measured 3.0e-06 ..
# 5.7e-05 across the seven layouts.
_LOSS_RTOL = 1e-3

# The inter-backend *gradient* gap is deliberately loose. At this fixture's
# geometry the KL is only ~9e-3 nat, i.e. ``predict ~= target``, so the score
# gradient ``predict - target`` is a difference of near-equal quantities and a
# ~3e-3 relative perturbation of the target is amplified ~14x. Measured worst
# case 5.8e-02, and ``test_matches_fp32_reference`` shows almost all of it is
# tilelang's own distance to fp32 (cuDNN stays at ~6e-03 there).
_CROSS_GRAD_RTOL = 1e-1

_FP32_LOSS_RTOL = 1e-4
_FP32_CUDNN_GRAD_RTOL = 2e-2
_FP32_TILELANG_GRAD_RTOL = 8e-2

# ``wk`` and ``k_norm`` sit downstream of ``index_k``, whose gradient the dense
# backward reduces through fp32 atomics, so they move from run to run. The
# absolute drift is tiny -- ``test_logged_kl_is_reproducible`` pins it at 1e-8 --
# but the KL here is only ~9e-3 nat, so the ``predict - target`` cancellation
# turns it into a large *relative* number on a ~1e-6-norm gradient: measured
# single-process floor (same module, same leaves, four backwards) 2.9e-2 on
# ``single-doc s=512`` and 6.5e-2 on the padded layout, against tilelang's
# 1.1e-3. A relative bound tighter than that measures the atomics rather than
# this backend, so these two get their own; ``wq_b`` / ``weights_proj``, which
# come back bitwise identical, keep the strict one.
_DK_PARAMS = ("wk.", "k_norm.")
_DK_GRAD_RTOL = 2e-1


def _grad_rtol(name, strict):
    """``strict``, loosened on the parameters fed by the atomic ``d_index_k``."""
    return _DK_GRAD_RTOL if name.startswith(_DK_PARAMS) else strict


def _module(loss_coeff=_COEFF, index_n_heads=None, seed=0):
    """A warmup-phase (``dsa_indexer_use_sparse_loss=False``) layer.

    ``_build_module`` does not seed and ``_make_inputs`` seeds only *after* the
    module exists, so an unseeded process draws different indexer weights every
    run. The metrics here are dominated by ``predict - target`` cancellation,
    which alone moves the reported gaps by ~3x, so seed before the build.
    """
    config = _create_mqa_config("mqa_dsa", loss_coeff=loss_coeff)
    config.dsa_indexer_use_sparse_loss = False
    config.pad_token_id = 0
    if index_n_heads is not None:
        config.dsa_index_n_heads = index_n_heads
    paddle.seed(seed)
    module = _build_module(config, bf16=True)
    assert module.indexer is not None
    return module


def _leaves(seqlen, seed=1):
    query, key, w_v, x, qr = _make_inputs(seqlen, seed=seed, with_hidden=True)
    tensors = [query, key, x, qr]
    for tensor in tensors:
        tensor.stop_gradient = False
    return tensors, w_v


def _pad_tail_ids(seqlen, n_pad):
    """``input_ids`` whose last ``n_pad`` rows are ``pad_token_id == 0``."""
    ids = np.ones([1, seqlen], dtype="int64")
    ids[0, seqlen - n_pad :] = 0
    return paddle.to_tensor(ids)


def _grads(module):
    return {
        name: param.grad.cast("float32").numpy().copy()
        for name, param in module.indexer.named_parameters()
        if param.grad is not None
    }


def _step(module, tensors, row_end, w_v, backend, input_ids=None):
    """One warmup step on ``backend``; ``(logged_kl, {param: grad})``."""
    module.config.csa_indexer_backend = backend
    module.train()
    module.clear_gradients()
    DSAIndexerLossLoggingHelper.tracker.clear()
    query, key, x, qr = tensors
    out = module(
        query,
        key,
        None,
        None,
        row_end,
        v_b_proj_weight=w_v,
        x=x,
        qr=qr,
        input_ids=input_ids,
    )
    out.cast("float32").sum().backward()
    logged = float(
        DSAIndexerLossLoggingHelper.tracker["values"].astype("float32").sum()
    )
    grads = _grads(module)
    module.clear_gradients()
    return logged, grads


def _rel(actual, expected):
    denom = float(np.linalg.norm(expected.ravel()))
    diff = float(np.linalg.norm((actual - expected).ravel()))
    return diff if denom == 0.0 else diff / denom


def _perturb_rows(tensors, rows):
    """A copy of ``tensors`` with the indexer's ``x`` / ``qr`` rewritten.

    Only the rows selected by the boolean ``rows`` mask move, as
    ``v -> 3 v + 1``; ``query`` / ``key`` are left alone so the attention side
    (and hence the KL target on the surviving rows) is untouched.
    """
    seqlen = int(tensors[0].shape[1])
    mask = paddle.to_tensor(
        np.asarray(rows, dtype="float32").reshape([1, seqlen, 1])
    )
    out = list(tensors)
    for slot in (2, 3):  # x, qr
        data = tensors[slot].detach().cast("float32")
        moved = (data * (1.0 + 2.0 * mask) + mask).cast(tensors[slot].dtype)
        moved.stop_gradient = False
        out[slot] = moved
    return out


def _fp32_reference(module, tensors, row_end):
    """The warmup objective in plain fp32 paddle; ``(loss, {param: grad})``.

    The candidate set is rebuilt the way ``_indexer_valid_range(window=0)``
    does: columns in ``[doc_start, min(row, doc_start + doc_len - 1)]``, with
    doc-invalid rows contributing a zero target instead of being removed from
    the denominator. ``weights`` already carries
    ``n_heads**-0.5 * head_dim**-0.5``, which is exactly the scaling a
    pure-paddle score evaluation wants, so nothing is un-baked here.
    """
    module.train()
    module.clear_gradients()
    query, key, x, qr = tensors
    s = int(query.shape[1])
    doc_start, doc_len, is_valid, _, _ = _derive_csa_doc_boundaries(row_end, s)

    index_q, index_k, weights = module._indexer_projections(
        x, qr, 0, grad_enabled=True
    )
    iq = index_q[0].cast("float32")
    ik = index_k[0].cast("float32")
    iw = weights[0].cast("float32")

    rows = paddle.arange(s, dtype="int64")
    cols = rows.unsqueeze(0)
    right = paddle.minimum(rows, doc_start + doc_len - 1).unsqueeze(1)
    keep = (cols >= doc_start.unsqueeze(1)) & (cols <= right)
    # An all-``-inf`` row would softmax to ``nan`` and poison the backward, so
    # every row keeps its diagonal and is removed from the loss through the
    # target's ``is_valid`` factor instead.
    keep = keep | (cols == rows.unsqueeze(1))
    neg = paddle.full([s, s], -1e30, dtype="float32")

    scores = paddle.einsum("qhd,td->qht", iq, ik)
    index_score = (F.relu(scores) * iw.unsqueeze(-1)).sum(axis=1)
    probs = F.softmax(paddle.where(keep, index_score, neg), axis=-1)

    with paddle.no_grad():
        logits = paddle.matmul(
            query[0].cast("float32"),
            key[0, :, 0].cast("float32"),
            transpose_y=True,
        ) * float(module.softmax_scale)
        logits = paddle.where(keep.unsqueeze(1), logits, neg.unsqueeze(1))
        target = F.softmax(logits, axis=-1).mean(axis=1)
        target = target * is_valid.astype("float32").unsqueeze(1)

    kl = (target * (paddle.log(target + _EPS) - paddle.log(probs + _EPS))).sum(
        axis=-1
    )
    loss = kl.mean() * (module.indexer_loss_coeff / module.cp_size)
    loss.backward()
    grads = _grads(module)
    module.clear_gradients()
    return float(loss), grads


@_GPU
class TestWarmupKLCudnnBackend(unittest.TestCase):
    """Dense cuDNN warmup KL against tilelang and against fp32."""

    def test_logged_kl_matches_tilelang(self):
        module = _module()
        for label, seqlen, row_end_fn, doc_lens, masked in _SCENARIOS:
            with self.subTest(label):
                tensors, w_v = _leaves(seqlen)
                row_end = row_end_fn(doc_lens, seqlen)
                ids = _pad_tail_ids(seqlen, seqlen // 4) if masked else None
                tl, _ = _step(module, tensors, row_end, w_v, "tilelang", ids)
                cu, _ = _step(module, tensors, row_end, w_v, "cudnn", ids)
                self.assertGreater(abs(tl), 0.0, "the KL collapsed to zero")
                self.assertLess(
                    abs(cu - tl) / abs(tl),
                    _LOSS_RTOL,
                    f"{label}: tilelang {tl:.8e} vs cudnn {cu:.8e}",
                )

    def test_indexer_grads_match_tilelang(self):
        module = _module()
        for label, seqlen, row_end_fn, doc_lens, masked in _SCENARIOS:
            with self.subTest(label):
                tensors, w_v = _leaves(seqlen)
                row_end = row_end_fn(doc_lens, seqlen)
                ids = _pad_tail_ids(seqlen, seqlen // 4) if masked else None
                _, g_tl = _step(module, tensors, row_end, w_v, "tilelang", ids)
                _, g_cu = _step(module, tensors, row_end, w_v, "cudnn", ids)
                self.assertEqual(
                    sorted(g_cu),
                    sorted(g_tl),
                    f"{label}: the two backends reached different parameters",
                )
                self.assertEqual(len(g_cu), 8, sorted(g_cu))
                for name in sorted(g_tl):
                    self.assertLess(
                        _rel(g_cu[name], g_tl[name]),
                        _grad_rtol(name, _CROSS_GRAD_RTOL),
                        f"{label}: {name}",
                    )

    def test_matches_fp32_reference(self):
        """Both backends against fp32 autograd; cuDNN must be the closer one.

        "Closer" is claimed over the reproducible parameters only: on ``wk`` /
        ``k_norm`` the dense backward's atomic ``d_index_k`` swamps its own bf16
        distance (see ``_DK_GRAD_RTOL``), so an ordering there would be a
        coin flip.
        """
        module = _module()
        for label, seqlen, row_end_fn, doc_lens, masked in _SCENARIOS:
            if masked:
                continue  # the reference has no ``input_ids`` gating
            with self.subTest(label):
                tensors, w_v = _leaves(seqlen)
                row_end = row_end_fn(doc_lens, seqlen)
                ref, g_ref = _fp32_reference(module, tensors, row_end)
                tl, g_tl = _step(module, tensors, row_end, w_v, "tilelang")
                cu, g_cu = _step(module, tensors, row_end, w_v, "cudnn")
                self.assertLess(abs(cu - ref) / abs(ref), _FP32_LOSS_RTOL)
                self.assertLess(abs(tl - ref) / abs(ref), _FP32_LOSS_RTOL)
                worst_cu = worst_tl = 0.0
                for name in sorted(g_ref):
                    r_cu = _rel(g_cu[name], g_ref[name])
                    r_tl = _rel(g_tl[name], g_ref[name])
                    self.assertLess(
                        r_cu,
                        _grad_rtol(name, _FP32_CUDNN_GRAD_RTOL),
                        f"{label}: {name}",
                    )
                    self.assertLess(
                        r_tl, _FP32_TILELANG_GRAD_RTOL, f"{label}: {name}"
                    )
                    if name.startswith(_DK_PARAMS):
                        continue
                    worst_cu = max(worst_cu, r_cu)
                    worst_tl = max(worst_tl, r_tl)
                self.assertLess(
                    worst_cu,
                    worst_tl,
                    f"{label}: cudnn {worst_cu:.3e} is no longer closer to "
                    f"fp32 than tilelang {worst_tl:.3e} on the reproducible "
                    f"parameters",
                )

    def test_logged_kl_is_reproducible(self):
        """Warmup runs under recompute, so a replay must repeat bitwise."""
        module = _module()
        for label, seqlen, row_end_fn, doc_lens, masked in _SCENARIOS:
            with self.subTest(label):
                tensors, w_v = _leaves(seqlen)
                row_end = row_end_fn(doc_lens, seqlen)
                ids = _pad_tail_ids(seqlen, seqlen // 4) if masked else None
                first, g_first = _step(
                    module, tensors, row_end, w_v, "cudnn", ids
                )
                second, g_second = _step(
                    module, tensors, row_end, w_v, "cudnn", ids
                )
                self.assertEqual(first, second, label)
                # ``d_index_k`` is reduced through fp32 ``cp.reduce.async.bulk``
                # atomics inside the kernel, so the two parameters it feeds
                # (``wk.*``, ``k_norm.*``) are not bitwise reproducible;
                # measured maxabs 4.7e-10. The tilelang backend shows the same
                # signature. The other two are exact.
                for name in sorted(g_first):
                    if name.startswith(("wk.", "k_norm.")):
                        np.testing.assert_allclose(
                            g_second[name],
                            g_first[name],
                            rtol=0,
                            atol=1e-8,
                            err_msg=f"{label}: {name}",
                        )
                    else:
                        np.testing.assert_array_equal(
                            g_second[name],
                            g_first[name],
                            err_msg=f"{label}: {name}",
                        )

    def test_loss_coeff_is_linear(self):
        """``grad_scale = loss_coeff * grad_loss / total_q`` inside the kernel."""
        tensors, w_v = _leaves(512)
        row_end = _row_end([100, 200, 212], 512)
        base = _module(loss_coeff=_COEFF)
        twice = _module(loss_coeff=2.0 * _COEFF)
        loss_a, g_a = _step(base, tensors, row_end, w_v, "cudnn")
        loss_b, g_b = _step(twice, tensors, row_end, w_v, "cudnn")
        self.assertAlmostEqual(loss_b / loss_a, 2.0, places=6)
        for name in sorted(g_a):
            norm_a = float(np.linalg.norm(g_a[name].ravel()))
            ratio = float(np.linalg.norm(g_b[name].ravel())) / norm_a
            # ``wk.*`` / ``k_norm.*`` ride the ``d_index_k`` atomics noise
            # floor; the other two are exact to fp32 rounding.
            places = 3 if name.startswith(("wk.", "k_norm.")) else 6
            self.assertAlmostEqual(ratio, 2.0, places=places, msg=name)

    def test_masked_rows_do_not_affect_the_loss(self):
        """Pad rows are gated by ``+inf`` LSEs, not by a clamp with a residue."""
        module = _module()
        seqlen = 512
        row_end = _pad_row_end([200, 180], seqlen)
        _, _, is_valid, _, _ = _derive_csa_doc_boundaries(row_end, seqlen)
        invalid = ~is_valid.numpy().astype(bool)
        self.assertTrue(invalid.any(), "the fixture produced no pad rows")

        tensors, w_v = _leaves(seqlen)
        clean, g_clean = _step(module, tensors, row_end, w_v, "cudnn")

        # Pad rows sit past the last document's end, so no valid row can reach
        # them as either a query or a candidate: perturbing them must be exactly
        # neutral. The mirror-image perturbation on the *valid* rows is measured
        # too, otherwise the assertion would also pass on a forward that ignored
        # ``x`` altogether.
        masked, _ = _step(
            module, _perturb_rows(tensors, invalid), row_end, w_v, "cudnn"
        )
        active, _ = _step(
            module, _perturb_rows(tensors, ~invalid), row_end, w_v, "cudnn"
        )
        self.assertEqual(clean, masked)
        self.assertGreater(abs(active - clean) / abs(clean), 1e-2)

        # The gradients are only allclose, not bitwise: ``d_index_k`` is reduced
        # through fp32 atomics whose order depends on the buffer contents, so
        # the ``wk.*`` / ``k_norm.*`` pair moves at the ~1e-10 floor measured in
        # ``test_logged_kl_is_reproducible``.
        _, g_masked = _step(
            module, _perturb_rows(tensors, invalid), row_end, w_v, "cudnn"
        )
        for name in sorted(g_clean):
            np.testing.assert_allclose(
                g_masked[name], g_clean[name], rtol=0, atol=1e-8, err_msg=name
            )

    def test_narrow_index_heads_raise_from_the_forward(self):
        """``>= 64`` heads, rejected at the call site and not from a replay.

        ``DenseIndexerBackward.check_support`` would raise from inside the
        backward, i.e. out of a recompute replay, where the message has no
        connection to the configuration that caused it.
        """
        module = _module(index_n_heads=32)
        tensors, w_v = _leaves(128)
        row_end = _row_end([128], 128)
        # Not ``assertRaises``: it hands back ``exc.with_traceback(None)``, and
        # the frame list is the whole point of this test.
        frames, message = [], ""
        try:
            _step(module, tensors, row_end, w_v, "cudnn")
        except ValueError as error:
            message = str(error)
            frames = [
                frame.name
                for frame in traceback.extract_tb(error.__traceback__)
            ]
        else:
            self.fail("index_n_heads=32 was accepted by the dense backend")
        self.assertIn("index_n_heads >= 64", message)
        self.assertIn("_check_cudnn_dense_indexer_support", frames)
        self.assertNotIn("backward", frames)


if __name__ == "__main__":
    unittest.main()
