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

"""Behavior tests for the ERNIE-4.5-VL DPO criterion (CPU / 无卡).

These tests drive the real ``ErnieDPOCriterion`` entry points and compare their
outputs against independently hand-derived NumPy references. The focus is the
DPO training-objective contract (chosen/rejected x policy/reference pairing),
and specifically the behaviors where the VL criterion *diverges* from its base
``DPOCriterion``:

  * ``dpo_loss`` takes an extra ``score_deltas`` argument and the ``sigmoid``
    branch subtracts ``offset_alpha/beta * log(score_deltas + 1e-6)`` from the
    logits when ``offset_alpha > 0`` (the base class has no such term).
  * The ``dpop`` objective here adds ``dpop_lambda * relu(ref_chosen -
    policy_chosen)`` as an *external* additive penalty, whereas the base class
    folds that penalty *inside* the log-sigmoid. This test pins the VL form and
    asserts it differs from the base form.
  * ``forward`` unpacks 5 label elements (with ``score_deltas``) when
    ``offset_alpha > 0`` and 4 otherwise, routes reference-pass vs policy-pass,
    selects ``average_log_prob`` for {ipo, or, simpo}, and combines
    ``loss = dpo_loss + sft_loss``.

Every other ``loss_type`` is still anchored to an independent formula because
``ErnieDPOCriterion.dpo_loss`` is a distinct entry point from the base class.

The ``dpo_logps`` numeric path (parallel_matmul -> per-token log-probs ->
response-index range gather -> sft normalization), the fused-head / filtered-
label paths, and the tensor/sequence-parallel paths require a real Paddle
runtime and/or a process group; they are declared skipped with reasons below.
"""

import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.transformers.ernie4_5_moe_vl.model.loss.dpo import (
    ErnieDPOCriterion,
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


class _DPOConfig:
    """Lightweight, fully-concrete stand-in for the model's dpo_config.

    Every attribute read by ``ErnieDPOCriterion.dpo_loss`` / ``forward`` is a
    plain Python value (no MagicMock), so a wrong attribute name surfaces as an
    AttributeError instead of being silently masked.
    """

    def __init__(
        self,
        loss_type="sigmoid",
        beta=0.1,
        offset_alpha=0.0,
        label_smoothing=0.0,
        pref_loss_ratio=1.0,
        sft_loss_ratio=1.0,
        simpo_gamma=0.5,
        dpop_lambda=1.0,
        normalize_logps=False,
    ):
        self.loss_type = loss_type
        self.beta = beta
        self.offset_alpha = offset_alpha
        self.label_smoothing = label_smoothing
        self.pref_loss_ratio = pref_loss_ratio
        self.sft_loss_ratio = sft_loss_ratio
        self.simpo_gamma = simpo_gamma
        self.dpop_lambda = dpop_lambda
        self.normalize_logps = normalize_logps


class _TextConfig:
    def __init__(self, vocab_size=1024, tie_word_embeddings=True):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings


class _ModelConfig:
    """Stand-in for the top-level model config.

    Defaults select the non-parallel, non-fused branch so the real
    ``DPOCriterion.__init__`` builds a plain ``CrossEntropyLoss`` logprobs op.
    """

    def __init__(
        self,
        tensor_parallel_output=False,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    ):
        self.tensor_parallel_output = tensor_parallel_output
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.sequence_parallel = sequence_parallel
        self.use_fused_head_and_loss_fn = False
        self.use_filtered_label_loss = False
        self.max_sequence_length = 512
        self.text_config = _TextConfig()


def _make_criterion(loss_type="sigmoid", **cfg_kwargs):
    dpo_config = _DPOConfig(loss_type=loss_type, **cfg_kwargs)
    # ErnieDPOCriterion inherits DPOCriterion.__init__; construct the real object.
    return ErnieDPOCriterion(_ModelConfig(), dpo_config=dpo_config)


# Fixed, distinguishable per-sample log-prob vectors. chosen != rejected and
# policy != reference, so a chosen/rejected swap or a mis-pairing is observable.
_PC = np.array([0.5, -0.2, 1.0], dtype=np.float64)  # policy chosen
_PR = np.array([0.3, 0.1, -0.4], dtype=np.float64)  # policy rejected
_RC = np.array([0.4, 0.0, 0.8], dtype=np.float64)  # reference chosen
_RR = np.array([0.2, -0.1, -0.2], dtype=np.float64)  # reference rejected
_BETA = 0.1


def _t(arr):
    return paddle.to_tensor(np.asarray(arr, dtype=np.float32))


class TestErnieDPOLossFormulas(unittest.TestCase):
    """Each supported loss_type compared to an independent NumPy formula.

    ``ErnieDPOCriterion.dpo_loss`` takes an extra ``score_deltas`` positional
    arg; it is only consumed by the sigmoid+offset branch, so unrelated losses
    receive a benign ones-vector.
    """

    def _call(
        self, criterion, pc=_PC, pr=_PR, rc=_RC, rr=_RR, score_deltas=None
    ):
        if score_deltas is None:
            score_deltas = np.ones_like(pc)
        loss = criterion.dpo_loss(
            _t(pc), _t(pr), _t(rc), _t(rr), _t(score_deltas)
        )
        return float(loss.numpy())

    def test_sigmoid_matches_independent_formula(self):
        crit = _make_criterion("sigmoid", beta=_BETA)
        logits = (_PC - _PR) - (_RC - _RR)
        expected = np.mean(_neg_log_sigmoid(_BETA * logits))
        self.assertAlmostEqual(self._call(crit), float(expected), places=6)

    def test_offset_alpha_consumes_score_deltas(self):
        # VL-specific: offset_alpha>0 shifts logits by -alpha/beta*log(sd+1e-6).
        alpha = 0.5
        crit = _make_criterion("sigmoid", beta=_BETA, offset_alpha=alpha)
        sd = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        logits = (_PC - _PR) - (_RC - _RR)
        adj = logits - alpha / _BETA * np.log(sd + 1e-6)
        expected = np.mean(_neg_log_sigmoid(_BETA * adj))
        self.assertAlmostEqual(
            self._call(crit, score_deltas=sd), float(expected), places=6
        )
        # Different score_deltas must change the loss (proves consumption).
        other = self._call(crit, score_deltas=sd * 5.0)
        self.assertNotAlmostEqual(
            self._call(crit, score_deltas=sd), other, places=6
        )
        # With offset_alpha=0 the same score_deltas must be ignored.
        plain = _make_criterion("sigmoid", beta=_BETA, offset_alpha=0.0)
        self.assertAlmostEqual(
            self._call(plain, score_deltas=sd),
            self._call(plain, score_deltas=sd * 5.0),
            places=6,
        )

    def test_label_smoothing_blends_both_terms(self):
        ls = 0.2
        crit = _make_criterion("sigmoid", beta=_BETA, label_smoothing=ls)
        # Use an ASYMMETRIC logits fixture: the default _PC/_PR/_RC/_RR give
        # logits {0, -0.4, 0.4}, a set symmetric under negation, so the smoothed
        # and unsmoothed means coincide and the "must differ" check below would
        # be vacuous. Shifting policy-chosen on one coordinate breaks that
        # symmetry (logits -> {0.6, -0.4, 0.4}) so smoothing genuinely moves the
        # mean loss.
        pc = _PC + np.array([0.6, 0.0, 0.0], dtype=np.float64)
        logits = (pc - _PR) - (_RC - _RR)
        per = (
            -_log_sigmoid(_BETA * logits) * (1 - ls)
            - _log_sigmoid(-_BETA * logits) * ls
        )
        self.assertAlmostEqual(
            self._call(crit, pc=pc), float(np.mean(per)), places=6
        )
        base = _make_criterion("sigmoid", beta=_BETA, label_smoothing=0.0)
        self.assertNotAlmostEqual(
            self._call(crit, pc=pc), self._call(base, pc=pc), places=6
        )

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
        # gamma=0 must give a different (unshifted) result.
        crit0 = _make_criterion("simpo", beta=_BETA, simpo_gamma=0.0)
        self.assertNotAlmostEqual(self._call(crit), self._call(crit0), places=6)

    def test_dpop_uses_external_additive_penalty(self):
        # VL-specific: penalty is added OUTSIDE log-sigmoid (base folds it INSIDE).
        crit = _make_criterion("dpop", beta=_BETA, dpop_lambda=1.0)
        logits = (_PC - _PR) - (_RC - _RR)
        positive_reg = np.clip(_RC - _PC, a_min=0.0, a_max=None)
        self.assertGreater(
            positive_reg.max(), 0.0
        )  # penalty is non-trivial here
        vl_expected = np.mean(
            _neg_log_sigmoid(_BETA * logits) + 1.0 * positive_reg
        )
        self.assertAlmostEqual(self._call(crit), float(vl_expected), places=6)
        # The base-class form folds the penalty inside the sigmoid; assert the VL
        # criterion does NOT match it, pinning the divergence.
        base_form = np.mean(
            _neg_log_sigmoid(_BETA * (logits - 1.0 * positive_reg))
        )
        self.assertNotAlmostEqual(self._call(crit), float(base_form), places=6)
        # dpop_lambda must scale the additive penalty.
        crit0 = _make_criterion("dpop", beta=_BETA, dpop_lambda=0.0)
        self.assertNotAlmostEqual(self._call(crit), self._call(crit0), places=6)

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
            self._call(crit, pc, pr, rc, rr, np.ones_like(pc)),
            float(expected),
            places=6,
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
            crit.dpo_loss(_t(_PC), _t(_PR), _t(_RC), _t(_RR), _t(np.ones(3)))


class TestErnieDPOLossInvariants(unittest.TestCase):
    """The DPO logits depend only on the two log-ratios, giving shift
    invariance and a sign flip under a chosen/rejected swap for sigmoid."""

    def _loss(self, crit, pc, pr, rc, rr):
        return float(
            crit.dpo_loss(
                _t(pc), _t(pr), _t(rc), _t(rr), _t(np.ones_like(pc))
            ).numpy()
        )

    def test_reference_constant_shift_leaves_loss_unchanged(self):
        crit = _make_criterion("sigmoid", beta=_BETA)
        base = self._loss(crit, _PC, _PR, _RC, _RR)
        # Same constant on both reference sides cancels in ref_logratios.
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
        # ASYMMETRIC fixture: the default logits {0, -0.4, 0.4} are symmetric
        # under negation, so a chosen/rejected swap (which negates the logits)
        # leaves the mean loss unchanged and the "must differ" check would be
        # vacuous. Shift policy-chosen on one coordinate so the swap actually
        # changes the loss (logits {0.6, -0.4, 0.4} -> {-0.6, 0.4, -0.4}).
        pc = _PC + np.array([0.6, 0.0, 0.0], dtype=np.float64)
        base = self._loss(crit, pc, _PR, _RC, _RR)
        # Swapping chosen<->rejected on both sides negates the logits.
        swapped = self._loss(crit, _PR, pc, _RR, _RC)
        logits = (pc - _PR) - (_RC - _RR)
        expected_swapped = float(np.mean(_neg_log_sigmoid(_BETA * (-logits))))
        self.assertAlmostEqual(swapped, expected_swapped, places=6)
        self.assertNotAlmostEqual(base, swapped, places=6)

    def test_zero_logits_anchor_equals_log_two(self):
        # When policy and reference log-ratios match, logits=0 and the mean
        # sigmoid loss is exactly -log_sigmoid(0) = log(2).
        crit = _make_criterion("sigmoid", beta=_BETA)
        pc = np.array([-1.0, -1.0], dtype=np.float64)
        pr = np.array([-2.0, -2.0], dtype=np.float64)
        rc = np.array([-3.0, -3.0], dtype=np.float64)
        rr = np.array([-4.0, -4.0], dtype=np.float64)  # (pc-pr)=(rc-rr)=1
        self.assertAlmostEqual(
            self._loss(crit, pc, pr, rc, rr), float(np.log(2.0)), places=6
        )


class TestErnieDPOForwardRouting(unittest.TestCase):
    """``forward`` routing, label unpacking and loss composition.

    ``dpo_logps`` is the model-forward collaborator (real parallel_matmul over
    hidden states, needs the modeling/distributed stack); it is replaced with a
    marker that returns distinguishable, input-independent tensors so that the
    *routing* and *composition* logic is what gets verified. ``dpo_loss`` is kept
    REAL, so the chosen/rejected x policy/reference pairing is genuinely checked.
    """

    def setUp(self):
        # Distinguishable policy markers returned by the stubbed dpo_logps.
        self.pcl = np.array([0.5, -0.2, 1.0], dtype=np.float64)
        self.prl = np.array([0.3, 0.1, -0.4], dtype=np.float64)
        self.sft = 0.7
        # Reference log-probs supplied through the labels tuple.
        self.rc = np.array([0.4, 0.0, 0.8], dtype=np.float64)
        self.rr = np.array([0.2, -0.1, -0.2], dtype=np.float64)
        # Dummy logits / label payloads (consumed only by the stubbed dpo_logps).
        self.logits = (_t(np.zeros((3, 4))), _t(np.zeros((4, 5))), None, False)
        self.response_labels = _t(np.zeros((3, 5)))
        self.response_indexs = _t(np.array([[0, 1, 3, 5]]))

    def _install_stub(self, crit, captured):
        def fake_logps(
            logits, response_labels, response_indexs, average_log_prob=False
        ):
            captured["logits"] = logits
            captured["response_labels"] = response_labels
            captured["response_indexs"] = response_indexs
            captured["average_log_prob"] = average_log_prob
            return (_t(self.pcl), _t(self.prl), _t(self.sft))

        return patch.object(crit, "dpo_logps", side_effect=fake_logps)

    def test_reference_pass_returns_reference_logps(self):
        crit = _make_criterion("sigmoid", beta=_BETA)
        captured = {}
        labels = (self.response_labels, self.response_indexs, None, None)
        with self._install_stub(crit, captured):
            result = crit.forward(self.logits, labels)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(
            result[0].numpy(), self.pcl.astype(np.float32), rtol=1e-6
        )
        np.testing.assert_allclose(
            result[1].numpy(), self.prl.astype(np.float32), rtol=1e-6
        )
        # Reference pass must forward the label payload verbatim.
        self.assertIs(captured["response_labels"], self.response_labels)
        self.assertIs(captured["response_indexs"], self.response_indexs)
        self.assertFalse(captured["average_log_prob"])  # sigmoid -> False

    def test_policy_pass_combines_dpo_and_sft(self):
        crit = _make_criterion("sigmoid", beta=_BETA)
        captured = {}
        labels = (
            self.response_labels,
            self.response_indexs,
            _t(self.rc),
            _t(self.rr),
        )
        with self._install_stub(crit, captured):
            result = crit.forward(self.logits, labels)
        self.assertEqual(len(result), 5)
        pcl_o, prl_o, sft_o, dpo_o, loss_o = result
        # Policy log-probs are the stub markers, unswapped.
        np.testing.assert_allclose(
            pcl_o.numpy(), self.pcl.astype(np.float32), rtol=1e-6
        )
        np.testing.assert_allclose(
            prl_o.numpy(), self.prl.astype(np.float32), rtol=1e-6
        )
        # dpo_loss is the REAL sigmoid objective of (policy, reference).
        logits = (self.pcl - self.prl) - (self.rc - self.rr)
        expected_dpo = float(np.mean(_neg_log_sigmoid(_BETA * logits)))
        self.assertAlmostEqual(float(dpo_o.numpy()), expected_dpo, places=6)
        # Total loss is dpo_loss + sft_loss (exact composition).
        self.assertAlmostEqual(
            float(loss_o.numpy()),
            float(dpo_o.numpy()) + float(sft_o.numpy()),
            places=6,
        )
        self.assertAlmostEqual(float(sft_o.numpy()), self.sft, places=6)

    def test_offset_alpha_unpacks_and_threads_score_deltas(self):
        # offset_alpha>0 -> labels carry score_deltas, threaded into dpo_loss.
        alpha = 0.5
        crit = _make_criterion("sigmoid", beta=_BETA, offset_alpha=alpha)
        captured = {}
        sd = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        labels = (
            self.response_labels,
            self.response_indexs,
            _t(sd),
            _t(self.rc),
            _t(self.rr),
        )
        with self._install_stub(crit, captured):
            result = crit.forward(self.logits, labels)
        self.assertEqual(len(result), 5)
        dpo_o = result[3]
        logits = (self.pcl - self.prl) - (self.rc - self.rr)
        adj = logits - alpha / _BETA * np.log(sd + 1e-6)
        expected = float(np.mean(_neg_log_sigmoid(_BETA * adj)))
        self.assertAlmostEqual(float(dpo_o.numpy()), expected, places=6)
        # Must differ from the no-offset value -> score_deltas really flowed in.
        no_offset = float(np.mean(_neg_log_sigmoid(_BETA * logits)))
        self.assertNotAlmostEqual(float(dpo_o.numpy()), no_offset, places=6)

    def test_average_log_prob_selected_by_loss_type(self):
        # {ipo, or, simpo} request length-normalized log-probs; others do not.
        for loss_type, expected in [
            ("sigmoid", False),
            ("hinge", False),
            ("ipo", True),
            ("or", True),
            ("simpo", True),
        ]:
            crit = _make_criterion(loss_type, beta=_BETA)
            captured = {}
            labels = (self.response_labels, self.response_indexs, None, None)
            with self._install_stub(crit, captured):
                crit.forward(self.logits, labels)  # reference pass, no dpo_loss
            self.assertEqual(
                captured["average_log_prob"],
                expected,
                msg=f"average_log_prob for loss_type={loss_type}",
            )


class TestErnieDPOLogpsSkipped(unittest.TestCase):
    """Paths that need a real Paddle runtime and/or a process group.

    These are explicitly skipped (not faked): a mock collective or a CPU hand-
    computation would not prove the real numeric behavior.
    """

    def test_dpo_logps_parallel_matmul_path(self):
        self.skipTest(
            "dpo_logps drives parallel_matmul + CrossEntropyLoss per-token "
            "log-probs and response-index range gather / sft normalization "
            "through the ernie4_5_moe_vl modeling+distributed import chain. It "
            "is CPU-runnable in principle but Paddle is not installed in this "
            "environment, so the numeric contract is left unverified here."
        )

    def test_fused_head_and_filtered_label_paths(self):
        self.skipTest(
            "use_fused_head_and_loss_fn / use_filtered_label_loss branches call "
            "fused GPU kernels; require a single-card (H20) run."
        )

    def test_tensor_and_sequence_parallel_logps(self):
        self.skipTest(
            "TP+SP logps (GatherOp / AllGatherVarlenOp / "
            "sequence_parallel_sparse_mask_labels) require a real multi-rank "
            "process group; not validated on CPU."
        )


if __name__ == "__main__":
    unittest.main()
