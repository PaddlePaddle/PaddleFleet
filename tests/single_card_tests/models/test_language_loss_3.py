# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for paddlefleet.models.common.language_loss.language_loss.

Model-layer / training-objective tests designed from the production source.
The slice under test is the set of public names the language-loss module
exposes: the ``subbatch`` chunking helper and the ``LanguageLoss`` /
``MainLanguageLoss`` / ``MTPLanguageLoss`` forward objectives. Numeric
expectations are re-derived with independent NumPy cross-entropy (log-softmax +
gather) at float64 -- never by calling the production loss as its own oracle --
so a wrong reduction, a wrong MTP label shift, or a broken add_mtp_loss toggle
is rejected rather than passed.

The module imports paddle at load time. The environment used to author this
file has no paddle installed, so every test is gated behind an honest
``skipUnless(IMPORT_OK, ...)`` guard and skips rather than fakes a pass. Only a
genuine ImportError is treated as a missing dependency; runtime API errors are
allowed to surface instead of being swallowed as a skip.
"""

import os
import sys
import unittest

import numpy as np

# Make ``src/`` importable when tests are run from a source checkout.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

IMPORT_OK = True
IMPORT_ERR = ""
try:
    import paddle

    from paddlefleet.models.common.language_loss.language_loss import (
        LanguageLoss,
        MainLanguageLoss,
        MTPLanguageLoss,
        subbatch,
    )
except ImportError as exc:  # only genuine missing-dependency, not API errors
    IMPORT_OK = False
    IMPORT_ERR = f"paddle/paddlefleet not importable: {exc}"

IGNORE_INDEX = -100


def _log_softmax(logits):
    """Numerically stable log-softmax over the last axis, in float64."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - lse


def _ce_per_token(logits, labels, ignore_index=IGNORE_INDEX):
    """Independent per-token cross entropy, [B, S]; 0.0 at ignored positions.

    Re-derived from the definition (negative log-prob of the target class),
    not from paddle's CrossEntropyLoss.
    """
    logp = _log_softmax(logits)
    labels = np.asarray(labels)
    b_dim, s_dim = labels.shape
    ce = np.zeros((b_dim, s_dim), dtype=np.float64)
    for b in range(b_dim):
        for s in range(s_dim):
            lbl = int(labels[b, s])
            if lbl != ignore_index:
                ce[b, s] = -logp[b, s, lbl]
    return ce


def _token_avg_loss(logits, labels, ignore_index=IGNORE_INDEX):
    """sum(per_token_ce * mask) / sum(mask) -- the non-experimental reduction."""
    ce = _ce_per_token(logits, labels, ignore_index)
    mask = (np.asarray(labels) != ignore_index).astype(np.float64)
    denom = mask.sum()
    if denom == 0:
        return 0.0
    return float((ce * mask).sum() / denom)


def _line_wise_loss(logits, labels, ignore_index=IGNORE_INDEX):
    """Experimental per-sample-mean-then-average reduction.

    Mirrors the documented EC line-wise formula including the 1e-6 guards so an
    invalid (all-ignored) line contributes 0 and does not divide by zero.
    """
    ce = _ce_per_token(logits, labels, ignore_index)
    mask = (np.asarray(labels) != ignore_index).astype(np.float64)
    loss_2d = ce * mask
    count = mask.sum(axis=-1)
    invalid = (count == 0).astype(np.float64)
    per_line = loss_2d.sum(axis=-1) / (count + 1e-6 * invalid)
    per_line = per_line * (1.0 - invalid)
    return float(per_line.sum() / ((1.0 - invalid).sum() + 1e-6))


class _LossConfig:
    """Minimal stand-in for TransformerConfig exposing only the attributes the
    language-loss code paths under test actually read.

    Real explicit values (no MagicMock) so no attribute is accidentally truthy
    and every branch selection in the test is deliberate.
    """

    def __init__(self, **overrides):
        self.parallel_output = False
        self.loss_subbatch_sequence_length = 0
        self.gpt_model_use_experimental_version = False
        self.sequence_parallel = False
        self.use_accuracy_compatible = False
        self.cp_balance_mode = "contiguous_allgather"
        self.recompute_modules = None
        self.recompute_num_layers = None
        self.num_nextn_predict_layers = None
        self.mtp_load_weight_only = False
        self.mtp_distillation_loss = False
        self.train_mtp_only = False
        self.add_mtp_loss = True
        self.mtp_loss_scaling_factor = 1.0
        self.fused_linear_ce_loss_chunk = 0
        self.use_erndata = False
        self.experimental_dataflow = False
        for key, value in overrides.items():
            setattr(self, key, value)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR or "paddle not installed")
class TestSubbatch(unittest.TestCase):
    """Contract of the subbatch chunking wrapper.

    subbatch(f, arg_idx, axis, bs, out_idx) slices each subbatched arg along its
    axis into bs-sized chunks, applies f per chunk, and concatenates the outputs
    along out_idx. Below-bs inputs bypass chunking entirely.
    """

    def test_returns_unchanged_when_axis_width_below_bs(self):
        # width 4 < bs 8 -> f is applied once to the full (unsliced) args.
        x_np = np.arange(4 * 3, dtype=np.float32).reshape([4, 3])
        y_np = np.arange(4 * 3, dtype=np.float32).reshape([4, 3]) + 100.0
        sb = subbatch(
            lambda a, b: a + b, arg_idx=[0, 1], axis=[0, 0], bs=8, out_idx=0
        )
        out = sb(paddle.to_tensor(x_np), paddle.to_tensor(y_np))
        np.testing.assert_array_equal(out.numpy(), x_np + y_np)

    def test_chunks_cover_full_axis_with_distinguishable_content(self):
        # width 9, bs 3 -> exactly three chunks [0:3], [3:6], [6:9].
        x_np = np.arange(9 * 4, dtype=np.float32).reshape([9, 4])
        expected_chunks = [x_np[0:3], x_np[3:6], x_np[6:9]]
        seen = []

        def marker(block):
            idx = len(seen)
            # Verify this call received the correct slice, by full content.
            np.testing.assert_array_equal(block.numpy(), expected_chunks[idx])
            seen.append(idx)
            # Distinguishable per-chunk transform so a wrong reassembly order
            # or a dropped chunk changes the concatenated result.
            return block * (idx + 1) + 1000.0 * (idx + 1)

        sb = subbatch(marker, arg_idx=[0], axis=[0], bs=3, out_idx=0)
        out = sb(paddle.to_tensor(x_np))

        self.assertEqual(seen, [0, 1, 2])
        ref = np.concatenate(
            [
                chunk * (i + 1) + 1000.0 * (i + 1)
                for i, chunk in enumerate(expected_chunks)
            ],
            axis=0,
        )
        np.testing.assert_array_equal(out.numpy(), ref)

    def test_same_arg_idx_reuses_sliced_arg_and_ignores_duplicate(self):
        # same_arg_idx={1: 0} means position 1 must reuse the *sliced* arg 0,
        # never the tensor actually passed as arg 1.
        x_np = np.arange(6 * 2, dtype=np.float32).reshape([6, 2])
        decoy_np = np.full([6, 2], -777.0, dtype=np.float32)

        sb = subbatch(
            lambda a, b: a * 10.0 + b,
            arg_idx=[0, 1],
            axis=[0, 0],
            bs=3,
            out_idx=0,
            same_arg_idx={1: 0},
        )
        out = sb(paddle.to_tensor(x_np), paddle.to_tensor(decoy_np))
        # f(slice_x, slice_x) = 11 * slice_x; the decoy must not appear.
        np.testing.assert_array_equal(out.numpy(), 11.0 * x_np)

    def test_arg_idx_axis_length_mismatch_raises(self):
        sb = subbatch(
            lambda a, b: a + b, arg_idx=[0, 1], axis=[0], bs=3, out_idx=0
        )
        with self.assertRaises(AssertionError):
            sb(
                paddle.to_tensor(np.zeros([6, 2], np.float32)),
                paddle.to_tensor(np.zeros([6, 2], np.float32)),
            )

    def test_unequal_axis_width_raises(self):
        sb = subbatch(
            lambda a, b: a + b, arg_idx=[0, 1], axis=[0, 0], bs=3, out_idx=0
        )
        with self.assertRaises(AssertionError):
            sb(
                paddle.to_tensor(np.zeros([6, 2], np.float32)),
                paddle.to_tensor(np.zeros([4, 2], np.float32)),
            )


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR or "paddle not installed")
class TestLanguageLossForwardImpl(unittest.TestCase):
    """forward_impl reductions (single process, TP=1, CP=1, no subbatch)."""

    def _make(self, **cfg):
        return LanguageLoss(config=_LossConfig(**cfg))

    def test_token_average_matches_independent_reference(self):
        rng = np.random.RandomState(0)
        b, s, v = 2, 4, 5
        logits = rng.randn(b, s, v).astype(np.float32)
        labels = np.array([[1, 4, 0, 2], [3, 3, 1, 0]], dtype=np.int64)

        loss_fn = self._make()
        out = loss_fn.forward_impl(
            paddle.to_tensor(logits), paddle.to_tensor(labels)
        )
        ref = _token_avg_loss(logits, labels)
        self.assertAlmostEqual(float(out.numpy()), ref, places=4)

    def test_ignored_tokens_excluded_from_average(self):
        # A wrong denominator (dividing by B*S instead of valid count) or a
        # non-zero contribution from the ignored slot would miss this.
        rng = np.random.RandomState(1)
        b, s, v = 2, 4, 5
        logits = rng.randn(b, s, v).astype(np.float32)
        labels = np.array([[1, -100, 0, 2], [-100, 3, 1, -100]], dtype=np.int64)

        loss_fn = self._make()
        out = loss_fn.forward_impl(
            paddle.to_tensor(logits), paddle.to_tensor(labels)
        )
        ref = _token_avg_loss(logits, labels)
        self.assertAlmostEqual(float(out.numpy()), ref, places=4)

    def test_all_ignored_returns_exact_zero(self):
        rng = np.random.RandomState(2)
        logits = rng.randn(2, 4, 5).astype(np.float32)
        labels = np.full([2, 4], -100, dtype=np.int64)

        loss_fn = self._make()
        out = loss_fn.forward_impl(
            paddle.to_tensor(logits), paddle.to_tensor(labels)
        )
        self.assertEqual(float(out.numpy()), 0.0)

    def test_experimental_line_wise_matches_reference(self):
        # Line-wise: per-sample mean, then average across valid samples. Row 1
        # is fully ignored to exercise the invalid-line 1e-6 guard, which the
        # reference reproduces exactly.
        rng = np.random.RandomState(3)
        b, s, v = 3, 4, 5
        logits = rng.randn(b, s, v).astype(np.float32)
        labels = np.array(
            [[1, 4, 0, 2], [-100, -100, -100, -100], [2, 1, -100, 3]],
            dtype=np.int64,
        )

        loss_fn = self._make(gpt_model_use_experimental_version=True)
        out = loss_fn.forward_impl(
            paddle.to_tensor(logits), paddle.to_tensor(labels)
        )
        ref = _line_wise_loss(logits, labels)
        self.assertAlmostEqual(float(out.numpy()), ref, places=4)
        # Line-wise must differ from the flat token average on this fixture,
        # otherwise the experimental branch would be indistinguishable.
        self.assertNotAlmostEqual(
            ref, _token_avg_loss(logits, labels), places=4
        )


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR or "paddle not installed")
class TestLanguageLossMTPFusedForward(unittest.TestCase):
    """LanguageLoss.forward with a list of logits (fused main+MTP path)."""

    def _fixture(self):
        rng = np.random.RandomState(7)
        b, s, v, k = 2, 3, 5, 2
        main = rng.randn(b, s, v).astype(np.float32)
        mtp0 = rng.randn(b, s, v).astype(np.float32)
        mtp1 = rng.randn(b, s, v).astype(np.float32)
        # Labels length S + K; all valid and distinguishable so a wrong per-depth
        # shift changes the value.
        labels = np.array([[1, 4, 0, 2, 3], [3, 2, 1, 0, 4]], dtype=np.int64)
        return b, s, v, k, main, mtp0, mtp1, labels

    def _ref_total(self, main, mtp0, mtp1, labels, s, scaling):
        lm = _token_avg_loss(main, labels[:, :s])
        l0 = _token_avg_loss(mtp0, labels[:, 1 : 1 + s])
        l1 = _token_avg_loss(mtp1, labels[:, 2 : 2 + s])
        return lm, l0, l1, lm + scaling * (l0 + l1) / 2.0

    def test_total_loss_matches_reference_with_mtp_added(self):
        b, s, v, k, main, mtp0, mtp1, labels = self._fixture()
        scaling = 0.5
        loss_fn = LanguageLoss(
            config=_LossConfig(
                num_nextn_predict_layers=k,
                add_mtp_loss=True,
                mtp_loss_scaling_factor=scaling,
            )
        )
        logits = [
            paddle.to_tensor(main),
            paddle.to_tensor(mtp0),
            paddle.to_tensor(mtp1),
        ]
        out = loss_fn.forward(logits, paddle.to_tensor(labels))
        _, _, _, ref = self._ref_total(main, mtp0, mtp1, labels, s, scaling)
        self.assertAlmostEqual(float(out.numpy()), ref, places=4)

    def test_add_mtp_false_leaves_value_at_lm_loss(self):
        # add_mtp_loss=False -> main + mtp - mtp.detach(); value equals lm_loss
        # while gradient still flows. Distinct from the added-value branch above.
        b, s, v, k, main, mtp0, mtp1, labels = self._fixture()
        loss_fn = LanguageLoss(
            config=_LossConfig(
                num_nextn_predict_layers=k,
                add_mtp_loss=False,
                mtp_loss_scaling_factor=0.5,
            )
        )
        logits = [
            paddle.to_tensor(main),
            paddle.to_tensor(mtp0),
            paddle.to_tensor(mtp1),
        ]
        out = loss_fn.forward(logits, paddle.to_tensor(labels))
        lm, l0, l1, added = self._ref_total(main, mtp0, mtp1, labels, s, 0.5)
        self.assertAlmostEqual(float(out.numpy()), lm, places=4)
        self.assertNotAlmostEqual(lm, added, places=4)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR or "paddle not installed")
class TestMTPLanguageLossForward(unittest.TestCase):
    """MTPLanguageLoss.forward: builds the per-depth MTP loss list."""

    def test_builds_loss_list_pops_logits_and_shifts_labels(self):
        rng = np.random.RandomState(11)
        b, s, v, k = 2, 3, 5, 2
        mtp0 = rng.randn(b, s, v).astype(np.float32)
        mtp1 = rng.randn(b, s, v).astype(np.float32)
        labels = np.array([[1, 4, 0, 2, 3], [3, 2, 1, 0, 4]], dtype=np.int64)

        loss_fn = MTPLanguageLoss(
            config=_LossConfig(num_nextn_predict_layers=k)
        )
        dict_args = {
            "mtp_logits": [paddle.to_tensor(mtp0), paddle.to_tensor(mtp1)],
            "labels": paddle.to_tensor(labels),
        }
        out = loss_fn.forward(dict_args)

        self.assertIs(out, dict_args)
        self.assertNotIn("mtp_logits", out)
        mtp_loss = out["mtp_loss"]
        self.assertEqual(len(mtp_loss), k)
        # depth d predicts labels[:, d+1 : d+1+S]; a wrong offset moves the value.
        ref0 = _token_avg_loss(mtp0, labels[:, 1 : 1 + s])
        ref1 = _token_avg_loss(mtp1, labels[:, 2 : 2 + s])
        self.assertAlmostEqual(float(mtp_loss[0].numpy()), ref0, places=4)
        self.assertAlmostEqual(float(mtp_loss[1].numpy()), ref1, places=4)


@unittest.skipUnless(IMPORT_OK, IMPORT_ERR or "paddle not installed")
class TestMainLanguageLossForward(unittest.TestCase):
    """MainLanguageLoss.forward: combines lm loss with precomputed MTP losses."""

    def _run(self, add_mtp_loss):
        rng = np.random.RandomState(13)
        b, s, v, k = 2, 3, 5, 2
        main = rng.randn(b, s, v).astype(np.float32)
        labels = np.array([[1, 4, 0, 2, 3], [3, 2, 1, 0, 4]], dtype=np.int64)
        scaling = 1.0
        mtp_vals = [0.3, 0.7]

        loss_fn = MainLanguageLoss(
            config=_LossConfig(
                num_nextn_predict_layers=k,
                add_mtp_loss=add_mtp_loss,
                mtp_loss_scaling_factor=scaling,
            )
        )
        # Snapshot and restore the class-level tracker so the assertion below
        # observes production writes without polluting other tests.
        original = dict(MainLanguageLoss.mtp_loss_tracker)
        self.addCleanup(MainLanguageLoss.mtp_loss_tracker.clear)
        self.addCleanup(
            lambda: MainLanguageLoss.mtp_loss_tracker.update(original)
        )

        dict_args = {
            "mtp_loss": [
                paddle.to_tensor(mtp_vals[0], dtype=paddle.float32),
                paddle.to_tensor(mtp_vals[1], dtype=paddle.float32),
            ],
            "logits": paddle.to_tensor(main),
        }
        out = loss_fn.forward(dict_args, paddle.to_tensor(labels))

        lm = _token_avg_loss(main, labels[:, :s])
        added = lm + scaling * (mtp_vals[0] + mtp_vals[1]) / 2.0
        # Tracker received both detached MTP losses via the real write path.
        self.assertAlmostEqual(
            float(MainLanguageLoss.mtp_loss_tracker["mtp_1_loss"].numpy()),
            mtp_vals[0],
            places=5,
        )
        self.assertAlmostEqual(
            float(MainLanguageLoss.mtp_loss_tracker["mtp_2_loss"].numpy()),
            mtp_vals[1],
            places=5,
        )
        return lm, added, float(out.numpy())

    def test_add_mtp_true_adds_scaled_mean(self):
        lm, added, out = self._run(add_mtp_loss=True)
        self.assertAlmostEqual(out, added, places=4)
        self.assertNotAlmostEqual(lm, added, places=4)

    def test_add_mtp_false_value_equals_lm_loss(self):
        lm, added, out = self._run(add_mtp_loss=False)
        self.assertAlmostEqual(out, lm, places=4)


if __name__ == "__main__":
    unittest.main()
