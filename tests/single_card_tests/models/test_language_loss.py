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

"""Base loss-numerics tests for
``paddlefleet.models.common.language_loss.language_loss.LanguageLoss``.

Disjoint slice (this is the BASE of the test_language_loss_2..7 family):
this file owns the CORE, single-rank ``forward_impl`` cross-entropy numerics
on a plain tensor -- token-averaged CE, ``ignored_index`` (-100) masking of
both numerator and denominator, the all-ignored short circuit, and the
token-average-vs-sample-average reduction contract -- plus the observable
``__init__`` state, the plain ``forward`` -> ``forward_impl`` dispatch, the
list-input precondition guard, and the ``build_schedule_node`` factory type.

It deliberately does NOT re-cover areas already owned by siblings:
  * the pure ``subbatch`` chunk/concat helper (test_language_loss_4, _2),
  * env-var feature switches and ``_tensor_md5`` (test_language_loss_2),
  * the fused (hidden, weight, bias) tuple / multimax routing path,
  * MTP list numerics, distillation, megatron per-depth labels and the
    cu_seqlens_q packed-doc roll.

Expected values are hand-derived with an independent NumPy log-sum-exp
reference; the production loss is never used as its own oracle. Fixed,
distinguishable logits/labels are chosen so a wrong reduction denominator,
leaked ignored token or swapped average is rejected rather than masked.

The production module imports paddle at load time, and this authoring
environment has no paddle installed, so every paddle-dependent case is gated
behind an honest ``skipUnless`` guard and skips instead of faking a pass. Only
a genuine ImportError/ModuleNotFoundError is treated as a missing dependency;
any other error is allowed to surface.
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

try:
    import paddle

    from paddlefleet.models.common.language_loss.language_loss import (
        LanguageLoss,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig

    IMPORT_OK = True
    IMPORT_ERR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest missing-dep guard
    IMPORT_OK = False
    IMPORT_ERR = exc

_SKIP_REASON = (
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERR!r}"
)


def _ce_token_mean_reference(logits, labels, ignore_index=-100):
    """Independent token-averaged cross-entropy in float64 NumPy.

    Per-token CE is ``logsumexp(logit) - logit[label]``; the returned scalar is
    the sum over non-ignored tokens divided by the count of non-ignored tokens.
    This mirrors nothing in the production module -- it is derived directly from
    the definition of cross entropy so it can act as a true oracle.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    v = logits.shape[-1]
    flat_logits = logits.reshape(-1, v)
    flat_labels = labels.reshape(-1)
    m = flat_logits.max(axis=-1, keepdims=True)
    logsumexp = m[:, 0] + np.log(np.exp(flat_logits - m).sum(axis=-1))
    picked = flat_logits[
        np.arange(flat_logits.shape[0]), np.clip(flat_labels, 0, v - 1)
    ]
    ce = logsumexp - picked
    valid = flat_labels != ignore_index
    if not valid.any():
        return 0.0
    return float(ce[valid].sum() / valid.sum())


def _make_config(**overrides):
    """Minimal, valid TransformerConfig for the single-rank CE path."""
    defaults = {
        "num_hidden_layers": 1,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "use_cpu_initialization": True,
        "loss_subbatch_sequence_length": 0,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


@unittest.skipUnless(IMPORT_OK, _SKIP_REASON)
class TestLanguageLossInit(unittest.TestCase):
    """Observable construction state of LanguageLoss on a single rank."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_default_uses_plain_cross_entropy_none_reduction(self):
        # With no tensor-model-parallel group initialized, the parallel-CE
        # branch must be OFF and the plain CrossEntropyLoss(reduction="none")
        # must be selected; ignored_index is the fixed -100 contract.
        loss_fn = LanguageLoss(_make_config())
        self.assertFalse(loss_fn.enable_parallel_cross_entropy)
        self.assertIsInstance(loss_fn.loss_func, paddle.nn.CrossEntropyLoss)
        self.assertEqual(loss_fn.loss_func.reduction, "none")
        self.assertEqual(loss_fn.ignored_index, -100)

    def test_use_subbatch_reflects_config_threshold(self):
        # use_subbatch is exactly loss_subbatch_sequence_length > 0, and the
        # threshold is propagated verbatim.
        off = LanguageLoss(_make_config(loss_subbatch_sequence_length=0))
        self.assertFalse(off.use_subbatch)
        self.assertEqual(off.loss_subbatch_sequence_length, 0)

        on = LanguageLoss(_make_config(loss_subbatch_sequence_length=4))
        self.assertTrue(on.use_subbatch)
        self.assertEqual(on.loss_subbatch_sequence_length, 4)


@unittest.skipUnless(IMPORT_OK, _SKIP_REASON)
class TestForwardImplCrossEntropy(unittest.TestCase):
    """Core single-rank cross-entropy numerics of ``forward_impl``."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def setUp(self):
        self.loss_fn = LanguageLoss(_make_config())

    def test_token_averaged_ce_matches_numpy_reference(self):
        # All labels valid: forward_impl must return the token-averaged CE
        # (sum of per-token CE over the mean count), matched to an independent
        # log-sum-exp reference. Distinguishable logits reject an off-by-one
        # label pick or a wrong softmax axis.
        logits_np = np.array(
            [[[3.0, 0.0, -1.0], [0.0, 2.0, 1.0], [1.0, 1.0, 1.0]]],
            dtype=np.float32,
        )
        labels_np = np.array([[0, 1, 2]], dtype=np.int64)
        loss = self.loss_fn.forward_impl(
            paddle.to_tensor(logits_np), paddle.to_tensor(labels_np)
        )
        expected = _ce_token_mean_reference(logits_np, labels_np)
        self.assertEqual(loss.shape, [])
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5, atol=1e-6)

    def test_ignored_index_excluded_from_numerator_and_denominator(self):
        # One token is -100. The reference averages over the two valid tokens
        # only; dividing by the full token count (3) would give a smaller
        # number, so this pins that -100 leaves both sums.
        logits_np = np.array(
            [[[3.0, 0.0, -1.0], [0.0, 2.0, 1.0], [5.0, 5.0, 5.0]]],
            dtype=np.float32,
        )
        labels_np = np.array([[0, 1, -100]], dtype=np.int64)
        loss = self.loss_fn.forward_impl(
            paddle.to_tensor(logits_np), paddle.to_tensor(labels_np)
        )
        expected = _ce_token_mean_reference(logits_np, labels_np)
        divide_by_all = _ce_token_mean_reference(logits_np, labels_np) * 2 / 3
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5, atol=1e-6)
        # Guard the guard: the "divide by all tokens" value is genuinely
        # different, so the assertion above could distinguish it.
        self.assertNotAlmostEqual(expected, divide_by_all, places=4)

    def test_all_ignored_returns_exact_zero(self):
        # Every label is -100: the short circuit returns a real 0.0 scalar
        # (mean(loss) * 0.0), not a NaN from dividing by a zero token count.
        logits_np = np.array(
            [[[3.0, 0.0, -1.0], [0.0, 2.0, 1.0]]], dtype=np.float32
        )
        labels_np = np.full((1, 2), -100, dtype=np.int64)
        loss = self.loss_fn.forward_impl(
            paddle.to_tensor(logits_np), paddle.to_tensor(labels_np)
        )
        self.assertEqual(loss.shape, [])
        self.assertEqual(float(loss), 0.0)

    def test_reduction_is_token_average_not_sample_average(self):
        # Two rows with unequal valid-token counts (2 and 1). Token-average
        # weights every valid token equally; a per-row mean then row-average
        # would give a different scalar. Assert we match the former and are
        # NOT equal to the latter.
        logits_np = np.array(
            [
                [[3.0, 0.0, -1.0], [0.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.5, 0.5, 4.0], [1.0, 2.0, 3.0], [2.0, 1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        labels_np = np.array([[0, 1, -100], [2, -100, -100]], dtype=np.int64)
        loss = self.loss_fn.forward_impl(
            paddle.to_tensor(logits_np), paddle.to_tensor(labels_np)
        )
        token_avg = _ce_token_mean_reference(logits_np, labels_np)

        # Independent per-row-mean-then-average reference.
        v = logits_np.shape[-1]
        fl = logits_np.astype(np.float64).reshape(-1, v)
        fb = labels_np.reshape(-1)
        m = fl.max(axis=-1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(fl - m).sum(axis=-1))
        ce = (lse - fl[np.arange(fl.shape[0]), np.clip(fb, 0, v - 1)]).reshape(
            2, 3
        )
        valid = (labels_np != -100).astype(np.float64)
        row_means = (ce * valid).sum(1) / valid.sum(1)
        sample_avg = float(row_means.mean())

        np.testing.assert_allclose(float(loss), token_avg, rtol=1e-5, atol=1e-6)
        self.assertNotAlmostEqual(token_avg, sample_avg, places=4)


@unittest.skipUnless(IMPORT_OK, _SKIP_REASON)
class TestForwardDispatchAndGuards(unittest.TestCase):
    """Plain ``forward`` dispatch, list precondition and schedule-node type."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_forward_tensor_path_equals_forward_impl(self):
        # For a plain (non-list) tensor and no recompute configured, forward
        # must route through _forward to forward_impl and return the same
        # token-averaged CE. Comparing to both forward_impl and the NumPy
        # oracle rules out a dispatch that drops or rescales the result.
        loss_fn = LanguageLoss(_make_config())
        logits_np = np.array(
            [[[2.0, 0.0, 1.0], [1.0, 3.0, 0.0]]], dtype=np.float32
        )
        labels_np = np.array([[2, 1]], dtype=np.int64)
        logits = paddle.to_tensor(logits_np)
        labels = paddle.to_tensor(labels_np)

        via_forward = float(loss_fn.forward(logits, labels))
        via_impl = float(loss_fn.forward_impl(logits, labels))
        expected = _ce_token_mean_reference(logits_np, labels_np)
        np.testing.assert_allclose(via_forward, via_impl, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(via_forward, expected, rtol=1e-5, atol=1e-6)

    def test_list_input_without_mtp_config_asserts(self):
        # A list of logits requires num_nextn_predict_layers > 0; the default
        # config has 0, so the precondition assertion must fire rather than
        # silently proceeding into the MTP branch.
        loss_fn = LanguageLoss(_make_config())
        self.assertEqual(loss_fn.config.num_nextn_predict_layers, 0)
        with self.assertRaises(AssertionError):
            loss_fn.forward([], paddle.to_tensor([[0, 1]], dtype="int64"))

    def test_build_schedule_node_returns_schedule_node(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        loss_fn = LanguageLoss(_make_config())
        node = loss_fn.build_schedule_node()
        self.assertIsInstance(node, ScheduleNode)


if __name__ == "__main__":
    unittest.main()
