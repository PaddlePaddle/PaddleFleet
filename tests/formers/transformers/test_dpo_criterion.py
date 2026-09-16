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

"""Behavior tests for the DPO criterion (CPU / 无卡).

These tests drive the real ``DPOCriterion`` / ``AutoDPOCriterion`` entry points
and compare their outputs against independently hand-derived NumPy references.
They verify the DPO training-objective contract rather than tensor shapes:

  * ``dpo_loss`` for every supported ``loss_type`` matches an independent formula.
  * The core DPO property that the loss depends only on
    ``logits = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)``:
    shifting both reference (or both policy) log-probs by a constant leaves the
    loss unchanged, while swapping chosen/rejected flips the logits sign.
  * ``dpo_logps`` per-token log-prob extraction, chosen/rejected range gather,
    label handling (``chosen_labels + rejected_labels``), the ``ignore_eos_token``
    offset, ``average_log_prob`` and ``normalize_logps`` reductions, and sft loss.
  * ``forward`` reference-pass vs policy-pass routing and ``loss = dpo + sft``.

The fused-head, filtered-label and tensor/sequence-parallel logps paths require a
GPU and a real process group; they are declared skipped with reasons below.
"""

import unittest

import numpy as np
import paddle
from paddle import nn

from paddlefleet.transformers.dpo_criterion import (
    AutoDPOCriterion,
    DPOCriterion,
)

paddle.set_device("cpu")


def _log_sigmoid(x):
    # numerically stable log(sigmoid(x)) = -log(1 + exp(-x)); independent of paddle.
    return -np.logaddexp(0.0, -np.asarray(x, dtype=np.float64))


def _neg_log_sigmoid(x):
    return np.logaddexp(0.0, -np.asarray(x, dtype=np.float64))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _relu(x):
    return np.maximum(0.0, np.asarray(x, dtype=np.float64))


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


class _DPOConfig:
    """Lightweight stand-in for the model's dpo_config object."""

    def __init__(
        self,
        loss_type="sigmoid",
        beta=0.1,
        label_smoothing=0.0,
        pref_loss_ratio=1.0,
        sft_loss_ratio=1.0,
        simpo_gamma=0.5,
        dpop_lambda=1.0,
        normalize_logps=False,
    ):
        self.loss_type = loss_type
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.pref_loss_ratio = pref_loss_ratio
        self.sft_loss_ratio = sft_loss_ratio
        self.simpo_gamma = simpo_gamma
        self.dpop_lambda = dpop_lambda
        self.normalize_logps = normalize_logps


class _ModelConfig:
    """Lightweight stand-in for the top-level model config.

    Only the attributes read by DPOCriterion on the CPU / single-rank path are
    set; the fused / filtered / parallel attributes default to the non-parallel
    branch so the standard CrossEntropyLoss logps path is exercised.
    """

    def __init__(
        self,
        dpo_config=None,
        tensor_parallel_output=False,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    ):
        self.dpo_config = dpo_config
        self.tensor_parallel_output = tensor_parallel_output
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.sequence_parallel = sequence_parallel
        self.use_fused_head_and_loss_fn = False
        self.use_filtered_label_loss = False
        self.chunk_size = 1024
        self.vocab_size = 1024
        self.max_sequence_length = 512


def _make_criterion(loss_type="sigmoid", ignore_eos_token=False, **cfg_kwargs):
    dpo_config = _DPOConfig(loss_type=loss_type, **cfg_kwargs)
    model_config = _ModelConfig()
    return DPOCriterion(
        model_config, dpo_config=dpo_config, ignore_eos_token=ignore_eos_token
    )


# Fixed, distinguishable log-prob vectors (per-sample) used across dpo_loss tests.
# chosen != rejected and policy != reference so swaps / shifts are observable.
_PC = np.array([0.5, -0.2, 1.0], dtype=np.float64)  # policy chosen
_PR = np.array([0.3, 0.1, -0.4], dtype=np.float64)  # policy rejected
_RC = np.array([0.4, 0.0, 0.8], dtype=np.float64)  # reference chosen
_RR = np.array([0.2, -0.1, -0.2], dtype=np.float64)  # reference rejected
_BETA = 0.1


def _t(arr):
    return paddle.to_tensor(np.asarray(arr, dtype=np.float32))


class TestDPOLossFormulas(unittest.TestCase):
    """Each supported loss_type is compared to an independent NumPy formula."""

    def _call(self, criterion, pc=_PC, pr=_PR, rc=_RC, rr=_RR):
        loss = criterion.dpo_loss(_t(pc), _t(pr), _t(rc), _t(rr))
        return float(loss.numpy())

    def test_sigmoid_matches_independent_formula(self):
        crit = _make_criterion("sigmoid", beta=_BETA, pref_loss_ratio=1.0)
        logits = (_PC - _PR) - (_RC - _RR)
        expected = np.mean(_neg_log_sigmoid(_BETA * logits))
        self.assertAlmostEqual(self._call(crit), float(expected), places=6)

    def test_label_smoothing_blends_both_terms(self):
        ls = 0.2
        crit = _make_criterion("sigmoid", beta=_BETA, label_smoothing=ls)
        logits = (_PC - _PR) - (_RC - _RR)
        per = (
            -_log_sigmoid(_BETA * logits) * (1 - ls)
            - _log_sigmoid(-_BETA * logits) * ls
        )
        self.assertAlmostEqual(self._call(crit), float(np.mean(per)), places=6)
        # Smoothing must actually change the result vs the ls=0 baseline.
        base = _make_criterion("sigmoid", beta=_BETA, label_smoothing=0.0)
        self.assertNotAlmostEqual(self._call(crit), self._call(base), places=6)

    def test_hinge_matches_relu_margin(self):
        crit = _make_criterion("hinge", beta=_BETA)
        logits = (_PC - _PR) - (_RC - _RR)
        expected = np.mean(_relu(1 - _BETA * logits))
        self.assertAlmostEqual(self._call(crit), float(expected), places=6)

    def test_ipo_matches_squared_formula(self):
        crit = _make_criterion("ipo", beta=_BETA)
        logits = (_PC - _PR) - (_RC - _RR)
        expected = np.mean((logits - 1.0 / (2 * _BETA)) ** 2)
        self.assertAlmostEqual(self._call(crit), float(expected), places=5)

    def test_simpo_subtracts_gamma_over_beta(self):
        crit = _make_criterion("simpo", beta=_BETA, simpo_gamma=0.5)
        logits = (_PC - _PR) - (_RC - _RR) - (0.5 / _BETA)
        expected = np.mean(_neg_log_sigmoid(_BETA * logits))
        self.assertAlmostEqual(self._call(crit), float(expected), places=6)

    def test_dpop_applies_positive_regularizer(self):
        crit = _make_criterion("dpop", beta=_BETA, dpop_lambda=1.0)
        logits = (_PC - _PR) - (_RC - _RR)
        positive_reg = np.clip(_RC - _PC, a_min=0.0, a_max=None)
        inner = logits - 1.0 * positive_reg
        expected = np.mean(_neg_log_sigmoid(_BETA * inner))
        self.assertAlmostEqual(self._call(crit), float(expected), places=6)
        # The penalty is non-trivial here (some samples have policy_chosen<ref).
        self.assertGreater(positive_reg.max(), 0.0)
        plain = _make_criterion("sigmoid", beta=_BETA)
        self.assertNotAlmostEqual(self._call(crit), self._call(plain), places=6)

    def test_nca_pair_matches_three_term_formula(self):
        crit = _make_criterion("nca_pair", beta=_BETA)
        cr = (_PC - _RC) * _BETA
        rr = (_PR - _RR) * _BETA
        per = (
            -_log_sigmoid(cr)
            - 0.5 * _log_sigmoid(-cr)
            - 0.5 * _log_sigmoid(-rr)
        )
        self.assertAlmostEqual(self._call(crit), float(np.mean(per)), places=6)

    def test_sppo_hard_matches_squared_reward_targets(self):
        crit = _make_criterion("sppo_hard", beta=_BETA)
        a = _PC - _RC
        b = _PR - _RR
        per = (a - 0.5 / _BETA) ** 2 + (b + 0.5 / _BETA) ** 2
        self.assertAlmostEqual(self._call(crit), float(np.mean(per)), places=4)

    def test_kto_pair_matches_halos_formula(self):
        crit = _make_criterion("kto_pair", beta=_BETA)
        chosen_logratios = _PC - _RC
        rejected_logratios = _PR - _RR
        chosen_kl = max(0.0, float(np.mean(_PC - _RC)))
        rejected_kl = max(0.0, float(np.mean(_PR - _RR)))
        first = 1 - _sigmoid(_BETA * (chosen_logratios - rejected_kl))
        second = 1 - _sigmoid(_BETA * (chosen_kl - rejected_logratios))
        expected = np.mean(np.concatenate([first, second], axis=0))
        self.assertAlmostEqual(self._call(crit), float(expected), places=6)

    def test_or_matches_log_odds_ratio(self):
        # ORPO log-odds needs strictly negative log-probs (exp < 1).
        crit = _make_criterion("or", beta=_BETA)
        pc = np.array([-0.5, -1.0], dtype=np.float64)
        pr = np.array([-0.8, -1.2], dtype=np.float64)
        rc = np.zeros_like(pc)
        rr = np.zeros_like(pr)
        log_odds = (pc - pr) - (np.log1p(-np.exp(pc)) - np.log1p(-np.exp(pr)))
        expected = np.mean(_neg_log_sigmoid(log_odds))
        self.assertAlmostEqual(
            self._call(crit, pc, pr, rc, rr), float(expected), places=6
        )

    def test_pref_loss_ratio_scales_linearly(self):
        base = _make_criterion("sigmoid", beta=_BETA, pref_loss_ratio=1.0)
        scaled = _make_criterion("sigmoid", beta=_BETA, pref_loss_ratio=2.5)
        self.assertAlmostEqual(
            self._call(scaled), 2.5 * self._call(base), places=6
        )

    def test_unknown_loss_type_raises(self):
        crit = _make_criterion("does_not_exist", beta=_BETA)
        with self.assertRaises(ValueError):
            crit.dpo_loss(_t(_PC), _t(_PR), _t(_RC), _t(_RR))


class TestDPOLossInvariants(unittest.TestCase):
    """The DPO logits depend only on the two log-ratios, giving shift invariance
    and antisymmetry under a chosen/rejected swap for the sigmoid objective."""

    def _loss(self, crit, pc, pr, rc, rr):
        return float(crit.dpo_loss(_t(pc), _t(pr), _t(rc), _t(rr)).numpy())

    def test_reference_constant_shift_leaves_loss_unchanged(self):
        crit = _make_criterion("sigmoid", beta=_BETA)
        base = self._loss(crit, _PC, _PR, _RC, _RR)
        # Adding the same constant to both reference sides cancels in ref_logratios.
        shifted = self._loss(crit, _PC, _PR, _RC + 7.0, _RR + 7.0)
        self.assertAlmostEqual(base, shifted, places=6)
        # A one-sided reference shift must instead change the loss.
        one_sided = self._loss(crit, _PC, _PR, _RC + 7.0, _RR)
        self.assertNotAlmostEqual(base, one_sided, places=6)

    def test_policy_constant_shift_leaves_loss_unchanged(self):
        crit = _make_criterion("sigmoid", beta=_BETA)
        base = self._loss(crit, _PC, _PR, _RC, _RR)
        shifted = self._loss(crit, _PC - 3.0, _PR - 3.0, _RC, _RR)
        self.assertAlmostEqual(base, shifted, places=6)

    def test_chosen_rejected_swap_flips_logits_sign(self):
        crit = _make_criterion("sigmoid", beta=_BETA)
        base = self._loss(crit, _PC, _PR, _RC, _RR)
        # Swapping chosen<->rejected on both policy and reference negates logits.
        swapped = self._loss(crit, _PR, _PC, _RR, _RC)
        logits = (_PC - _PR) - (_RC - _RR)
        expected_swapped = float(np.mean(_neg_log_sigmoid(_BETA * (-logits))))
        self.assertAlmostEqual(swapped, expected_swapped, places=6)
        self.assertNotAlmostEqual(base, swapped, places=6)


# Fixed logps fixture: batch=2, seq=6, vocab=4 with distinguishable logits so
# that a wrong gather range, wrong label combination or wrong reduction shows up.
_LOGITS_NP = (
    (np.arange(2 * 6 * 4, dtype=np.float64).reshape(2, 6, 4) % 7) - 3
) * 0.37
_CHOSEN_LABELS = np.array(
    [[0, 1, 2, 0, 0, 0], [0, 3, 1, 0, 0, 0]], dtype=np.int64
)
_REJECTED_LABELS = np.array(
    [[0, 0, 0, 2, 3, 0], [0, 0, 0, 1, 2, 0]], dtype=np.int64
)
# columns: [batch_idx, chosen_start, chosen_end/rejected_start, rejected_end]
_RESP = np.array([[0, 1, 3, 5], [1, 1, 3, 5]], dtype=np.int64)


def _reference_logps(offset=0):
    """Independently derive per-response chosen/rejected logps and sft loss."""
    combined = _CHOSEN_LABELS + _REJECTED_LABELS
    lsm = _log_softmax(_LOGITS_NP)  # [b, seq, vocab]
    per_token = np.take_along_axis(lsm, combined[:, :, None], axis=-1)[
        :, :, 0
    ]  # [b, seq]
    chosen, rejected = [], []
    for b, cs, ce, re in _RESP:
        chosen.append(per_token[b, cs:ce].sum())
        rejected.append(per_token[b, ce + offset : re].sum())
    chosen = np.array(chosen, dtype=np.float64)
    rejected = np.array(rejected, dtype=np.float64)
    n_chosen_tokens = float((_CHOSEN_LABELS != 0).sum())
    sft = -chosen.sum() / n_chosen_tokens
    return per_token, chosen, rejected, sft


def _logps_tensors():
    return (
        _t(_LOGITS_NP),
        paddle.to_tensor(_CHOSEN_LABELS),
        paddle.to_tensor(_REJECTED_LABELS),
        paddle.to_tensor(_RESP),
    )


class TestDPOLogps(unittest.TestCase):
    """The standard (non-fused, non-filtered) dpo_logps gather path."""

    def test_logps_and_sft_match_independent_reference(self):
        crit = _make_criterion("sigmoid", sft_loss_ratio=1.0)
        logits, cl, rl, resp = _logps_tensors()
        chosen, rejected, sft = crit.dpo_logps(logits, cl, rl, resp)
        _, exp_chosen, exp_rejected, exp_sft = _reference_logps(offset=0)
        np.testing.assert_allclose(
            chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-5
        )
        self.assertAlmostEqual(float(sft.numpy()), float(exp_sft), places=5)
        # chosen and rejected must differ: they gather different token ranges.
        self.assertFalse(np.allclose(chosen.numpy(), rejected.numpy()))

    def test_sft_loss_ratio_scales_sft_term(self):
        crit = _make_criterion("sigmoid", sft_loss_ratio=3.0)
        logits, cl, rl, resp = _logps_tensors()
        _, _, sft = crit.dpo_logps(logits, cl, rl, resp)
        _, _, _, exp_sft = _reference_logps(offset=0)
        self.assertAlmostEqual(
            float(sft.numpy()), 3.0 * float(exp_sft), places=5
        )

    def test_ignore_eos_token_shifts_rejected_start_by_one(self):
        crit = _make_criterion("sigmoid", ignore_eos_token=True)
        logits, cl, rl, resp = _logps_tensors()
        chosen, rejected, _ = crit.dpo_logps(logits, cl, rl, resp)
        _, exp_chosen, exp_rejected, _ = _reference_logps(offset=1)
        # chosen range is unaffected by the offset; rejected drops its first token.
        np.testing.assert_allclose(
            chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-5
        )
        # Confirm the offset actually changed the rejected sum vs offset=0.
        _, _, rejected_no_offset, _ = _reference_logps(offset=0)
        self.assertFalse(np.allclose(exp_rejected, rejected_no_offset))

    def test_average_log_prob_divides_by_response_length(self):
        crit = _make_criterion("ipo")  # ipo/or/simpo use average_log_prob
        logits, cl, rl, resp = _logps_tensors()
        chosen, rejected, _ = crit.dpo_logps(
            logits, cl, rl, resp, average_log_prob=True
        )
        _, exp_chosen, exp_rejected, _ = _reference_logps(offset=0)
        # chosen_len = end-start-offset = 2, rejected_len = end2-end = 2.
        np.testing.assert_allclose(
            chosen.numpy(), exp_chosen / 2.0, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            rejected.numpy(), exp_rejected / 2.0, rtol=1e-5, atol=1e-5
        )

    def test_normalize_logps_scales_by_length_ratio(self):
        # Asymmetric response lengths so the scaling factors are distinguishable.
        logits_np = ((np.arange(1 * 7 * 4).reshape(1, 7, 4) % 5) - 2) * 0.29
        chosen_labels = np.array([[0, 1, 2, 0, 0, 0, 0]], dtype=np.int64)
        rejected_labels = np.array([[0, 0, 0, 3, 1, 2, 0]], dtype=np.int64)
        resp = np.array(
            [[0, 1, 3, 6]], dtype=np.int64
        )  # chosen len 2, rejected len 3
        crit = _make_criterion("sigmoid", normalize_logps=True)
        chosen, rejected, _ = crit.dpo_logps(
            _t(logits_np),
            paddle.to_tensor(chosen_labels),
            paddle.to_tensor(rejected_labels),
            paddle.to_tensor(resp),
        )
        combined = chosen_labels + rejected_labels
        lsm = _log_softmax(logits_np.astype(np.float64))
        per_token = np.take_along_axis(lsm, combined[:, :, None], axis=-1)[
            :, :, 0
        ]
        base_chosen = per_token[0, 1:3].sum()
        base_rejected = per_token[0, 3:6].sum()
        avg_len = (6 - 1) / 2.0  # (end2 - start) / 2
        exp_chosen = base_chosen * (avg_len / 2.0)  # chosen_len = 2
        exp_rejected = base_rejected * (avg_len / 3.0)  # rejected_len = 3
        self.assertAlmostEqual(float(chosen.numpy()[0]), exp_chosen, places=5)
        self.assertAlmostEqual(
            float(rejected.numpy()[0]), exp_rejected, places=5
        )

    def test_logits_label_shape_mismatch_raises(self):
        crit = _make_criterion("sigmoid")
        bad_logits = _t(np.zeros((2, 5, 4)))  # seq=5 != labels seq=6
        _, cl, rl, resp = _logps_tensors()
        with self.assertRaises(ValueError):
            crit.dpo_logps(bad_logits, cl, rl, resp)


class TestAutoDPOCriterion(unittest.TestCase):
    """AutoDPOCriterion computes logps via boolean masks instead of gather."""

    def test_init_uses_plain_cross_entropy(self):
        crit = AutoDPOCriterion(_ModelConfig(), dpo_config=_DPOConfig())
        # Auto path always uses the non-parallel CrossEntropyLoss.
        self.assertIsInstance(crit.logprobs, nn.CrossEntropyLoss)

    def test_mask_logps_match_independent_reference(self):
        crit = AutoDPOCriterion(_ModelConfig(), dpo_config=_DPOConfig())
        logits, cl, rl, resp = _logps_tensors()
        chosen, rejected, sft = crit.dpo_logps(logits, cl, rl, resp)
        _, exp_chosen, exp_rejected, exp_sft = _reference_logps(offset=0)
        np.testing.assert_allclose(
            chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-5
        )
        self.assertAlmostEqual(float(sft.numpy()), float(exp_sft), places=5)

    def test_mask_logps_honor_ignore_eos_offset(self):
        crit = AutoDPOCriterion(
            _ModelConfig(), dpo_config=_DPOConfig(), ignore_eos_token=True
        )
        logits, cl, rl, resp = _logps_tensors()
        _, rejected, _ = crit.dpo_logps(logits, cl, rl, resp)
        _, _, exp_rejected, _ = _reference_logps(offset=1)
        np.testing.assert_allclose(
            rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-5
        )


class TestDPOForward(unittest.TestCase):
    """forward() routing: reference pass returns logps, policy pass returns loss."""

    def test_reference_pass_returns_computed_logps(self):
        crit = _make_criterion("sigmoid")
        logits, cl, rl, resp = _logps_tensors()
        labels = (cl, rl, resp, None, None)
        out = crit(logits, labels)
        self.assertEqual(len(out), 2)  # (reference_chosen, reference_rejected)
        ref_chosen, ref_rejected = out
        _, exp_chosen, exp_rejected, _ = _reference_logps(offset=0)
        np.testing.assert_allclose(
            ref_chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            ref_rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-5
        )

    def test_policy_pass_combines_dpo_and_sft(self):
        crit = _make_criterion("sigmoid", beta=_BETA, sft_loss_ratio=1.0)
        logits, cl, rl, resp = _logps_tensors()
        ref_c = np.array([0.1, -0.2], dtype=np.float64)
        ref_r = np.array([0.3, 0.05], dtype=np.float64)
        labels = (cl, rl, resp, _t(ref_c), _t(ref_r))
        pol_c, pol_r, sft, dpo, loss = crit(logits, labels)
        _, exp_c, exp_r, exp_sft = _reference_logps(offset=0)
        # Independent dpo sigmoid loss from the policy logps and given references.
        logits_dpo = (exp_c - exp_r) - (ref_c - ref_r)
        exp_dpo = np.mean(_neg_log_sigmoid(_BETA * logits_dpo))
        self.assertAlmostEqual(float(dpo.numpy()), float(exp_dpo), places=5)
        self.assertAlmostEqual(float(sft.numpy()), float(exp_sft), places=5)
        # The returned total loss must be the sum of the two reported components.
        self.assertAlmostEqual(
            float(loss.numpy()),
            float(dpo.numpy()) + float(sft.numpy()),
            places=5,
        )
        np.testing.assert_allclose(pol_c.numpy(), exp_c, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(pol_r.numpy(), exp_r, rtol=1e-5, atol=1e-5)


class TestDPOCriterionInit(unittest.TestCase):
    def test_missing_dpo_config_raises(self):
        with self.assertRaises(ValueError):
            DPOCriterion(_ModelConfig(dpo_config=None))

    def test_reads_and_deepcopies_model_dpo_config(self):
        original = _DPOConfig(loss_type="hinge", beta=0.25)
        crit = DPOCriterion(_ModelConfig(dpo_config=original))
        self.assertEqual(crit.dpo_config.loss_type, "hinge")
        self.assertAlmostEqual(crit.dpo_config.beta, 0.25)
        # __init__ deep-copies the config, so later mutation must not leak in.
        original.beta = 999.0
        self.assertAlmostEqual(crit.dpo_config.beta, 0.25)

    def test_explicit_dpo_config_takes_precedence(self):
        model_cfg = _ModelConfig(dpo_config=_DPOConfig(loss_type="hinge"))
        crit = DPOCriterion(model_cfg, dpo_config=_DPOConfig(loss_type="ipo"))
        self.assertEqual(crit.dpo_config.loss_type, "ipo")

    def test_cpu_single_rank_uses_non_parallel_cross_entropy(self):
        crit = DPOCriterion(_ModelConfig(), dpo_config=_DPOConfig())
        self.assertIsInstance(crit.logprobs, nn.CrossEntropyLoss)


class TestDPOUnsupportedEnvironments(unittest.TestCase):
    """Paths that cannot be validated on CPU without a real process group."""

    def test_fused_head_and_loss_fn_path(self):
        raise unittest.SkipTest(
            "use_fused_head_and_loss_fn drives fused_head_and_loss_fn, a "
            "GPU-only fused kernel; not runnable on CPU. Needs single-card env."
        )

    def test_tensor_parallel_parallel_cross_entropy(self):
        raise unittest.SkipTest(
            "tensor_parallel_output + tensor_model_parallel_size>1 selects "
            "ParallelCrossEntropy and cross-rank logits; requires a real TP "
            "process group (multi-card), not CPU single-process."
        )

    def test_filtered_label_loss_sequence_parallel_gather(self):
        raise unittest.SkipTest(
            "use_filtered_label_loss with sequence_parallel uses "
            "sequence_parallel_sparse_mask_labels / AllGatherVarlenOp collective "
            "gather; only meaningful under a real SP+TP process group (multi-card)."
        )


if __name__ == "__main__":
    unittest.main()
