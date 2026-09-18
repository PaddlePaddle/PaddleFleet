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

"""Behavior tests for the KTO criterion (CPU / 无卡).

These tests drive the real ``KTOCriterion`` entry points and compare their
outputs against independently hand-derived NumPy references. They verify the
KTO training-objective contract rather than tensor shapes/existence.

KTO is an *unpaired* objective, unlike DPO:

  * ``kto_logps`` gathers per-response chosen / rejected / KL log-probs from
    disjoint token ranges. The number of chosen and rejected responses need
    not match (a batch may contain more of one class), and the KL term is
    computed for *every* response, independent of its chosen/rejected label.
  * ``kto_loss`` uses an asymmetric formula around a single shared, detached KL
    reference: desirable (chosen) examples want ``chosen_logratio - kl`` high,
    undesirable (rejected) examples want ``kl - rejected_logratio`` high. The
    two sides are scaled by ``desirable_weight`` / ``undesirable_weight``.
  * ``forward`` routes the reference pass (returns 3 logps) vs the policy pass
    (returns policy logps + loss + kl).

Environment: written for CPU (无卡) execution; could not be run locally because
no paddle/pytest is installed. Syntax verified with ``python3 -m py_compile``.
The fused-head, filtered-label and tensor-parallel logps paths and the
distributed KL all-gather/clip path require a GPU and a real process group and
are declared skipped with reasons below.
"""

import unittest

import numpy as np
import paddle
from paddle import nn

from paddlefleet.transformers.kto_criterion import KTOCriterion

paddle.set_device("cpu")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _t(arr):
    return paddle.to_tensor(np.asarray(arr, dtype=np.float32))


class _KTOConfig:
    """Lightweight stand-in for the model's kto_config object."""

    def __init__(self, beta=0.1, desirable_weight=1.0, undesirable_weight=1.0):
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight


class _ModelConfig:
    """Lightweight stand-in for the top-level model config.

    Only the attributes read by KTOCriterion on the CPU / single-rank,
    non-fused, non-filtered path are set; the fused / filtered / parallel
    attributes default to the plain CrossEntropyLoss branch.
    """

    def __init__(
        self,
        kto_config=None,
        tensor_parallel_output=False,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        vocab_size=4,
    ):
        self.kto_config = kto_config
        self.tensor_parallel_output = tensor_parallel_output
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.sequence_parallel = sequence_parallel
        self.use_fused_head_and_loss_fn = False
        self.use_filtered_label_loss = False
        self.fused_linear = False
        self.chunk_size = 1024
        self.vocab_size = vocab_size


def _make_criterion(beta=0.1, desirable_weight=1.0, undesirable_weight=1.0):
    kto_config = _KTOConfig(
        beta=beta,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
    )
    return KTOCriterion(_ModelConfig(kto_config=kto_config))


class TestKTOCriterionInit(unittest.TestCase):
    def test_missing_kto_config_raises(self):
        with self.assertRaises(ValueError):
            KTOCriterion(_ModelConfig(kto_config=None))

    def test_reads_and_deepcopies_model_kto_config(self):
        original = _KTOConfig(beta=0.25, desirable_weight=2.0)
        crit = KTOCriterion(_ModelConfig(kto_config=original))
        self.assertAlmostEqual(crit.kto_config.beta, 0.25)
        self.assertAlmostEqual(crit.kto_config.desirable_weight, 2.0)
        # __init__ deep-copies config.kto_config, so later mutation must not leak.
        original.beta = 999.0
        self.assertAlmostEqual(crit.kto_config.beta, 0.25)

    def test_explicit_kto_config_takes_precedence_without_copy(self):
        model_cfg = _ModelConfig(kto_config=_KTOConfig(beta=0.9))
        explicit = _KTOConfig(beta=0.2)
        crit = KTOCriterion(model_cfg, kto_config=explicit)
        self.assertAlmostEqual(crit.kto_config.beta, 0.2)

    def test_cpu_single_rank_uses_non_parallel_cross_entropy(self):
        crit = _make_criterion()
        self.assertIsInstance(crit.logprobs, nn.CrossEntropyLoss)

    def test_defaults_ignore_label_zero_and_no_infohub(self):
        crit = _make_criterion()
        self.assertEqual(crit.ignore_label, 0)
        self.assertFalse(crit.use_infohub)
        # world_size == 1 on CPU: no communication group is built.
        self.assertIsNone(crit.comm_group)


# ---- kto_loss fixtures: unpaired, distinguishable per-response log-probs. ----
# 2 chosen, 1 rejected, 3 KL responses -> exercises the unpaired structure.
_PC = np.array([0.5, -0.2], dtype=np.float64)  # policy chosen  (2 responses)
_RC = np.array([0.4, 0.1], dtype=np.float64)  # reference chosen
_PR = np.array([0.3], dtype=np.float64)  # policy rejected (1 response)
_RR = np.array([-0.1], dtype=np.float64)  # reference rejected
_PKL = np.array([0.2, -0.3, 0.5], dtype=np.float64)  # policy KL (3 responses)
_RKL = np.array([0.1, 0.1, 0.2], dtype=np.float64)  # reference KL
_BETA = 0.1


def _expected_kto_loss(pc, pr, rc, rr, pkl, rkl, beta, dw, uw):
    """Independent NumPy reference for kto_loss on the CPU (world_size==1) path.

    NOTE: on world_size==1 the production code does NOT clip kl to min 0
    (the clip lives only in the dist.get_world_size() > 1 branch), so the
    reference must not clip either to match the path under test.
    """
    pc, pr = np.asarray(pc), np.asarray(pr)
    rc, rr = np.asarray(rc), np.asarray(rr)
    kl = float(np.mean(np.asarray(pkl) - np.asarray(rkl)))
    chosen_losses = (
        np.zeros((0,))
        if pc.size == 0 or rc.size == 0
        else 1.0 - _sigmoid(beta * ((pc - rc) - kl))
    )
    rejected_losses = (
        np.zeros((0,))
        if pr.size == 0 or rr.size == 0
        else 1.0 - _sigmoid(beta * (kl - (pr - rr)))
    )
    losses = np.concatenate([dw * chosen_losses, uw * rejected_losses], axis=0)
    return float(losses.mean()), kl


class TestKTOLoss(unittest.TestCase):
    """kto_loss numerical contract and its chosen/rejected asymmetry."""

    def _call(self, crit, pc=_PC, pr=_PR, rc=_RC, rr=_RR, pkl=_PKL, rkl=_RKL):
        loss, kl = crit.kto_loss(
            _t(pc), _t(pr), _t(pkl), _t(rc), _t(rr), _t(rkl)
        )
        return float(loss.numpy()), float(kl.numpy())

    def test_matches_independent_formula(self):
        crit = _make_criterion(beta=_BETA)
        loss, kl = self._call(crit)
        exp_loss, exp_kl = _expected_kto_loss(
            _PC, _PR, _RC, _RR, _PKL, _RKL, _BETA, 1.0, 1.0
        )
        self.assertAlmostEqual(kl, exp_kl, places=6)
        self.assertAlmostEqual(loss, exp_loss, places=6)

    def test_kl_is_mean_of_kl_logratios(self):
        crit = _make_criterion(beta=_BETA)
        _, kl = self._call(crit)
        self.assertAlmostEqual(kl, float(np.mean(_PKL - _RKL)), places=6)

    def test_weights_scale_only_their_own_side(self):
        # desirable_weight scales chosen losses, undesirable_weight the rejected.
        crit = _make_criterion(
            beta=_BETA, desirable_weight=2.0, undesirable_weight=3.0
        )
        loss, _ = self._call(crit)
        exp_loss, _ = _expected_kto_loss(
            _PC, _PR, _RC, _RR, _PKL, _RKL, _BETA, 2.0, 3.0
        )
        self.assertAlmostEqual(loss, exp_loss, places=6)
        # And the weighted result must differ from the unit-weight baseline.
        base, _ = self._call(_make_criterion(beta=_BETA))
        self.assertNotAlmostEqual(loss, base, places=6)

    def test_chosen_and_rejected_branches_are_asymmetric(self):
        # KTO is NOT symmetric under a chosen<->rejected swap: chosen uses
        # (logratio - kl), rejected uses (kl - logratio). Swapping the roles of
        # the two response sets (and their references) must change the loss.
        crit = _make_criterion(beta=_BETA)
        base, _ = self._call(crit)
        swapped, _ = crit.kto_loss(
            _t(_PR), _t(_PC), _t(_PKL), _t(_RR), _t(_RC), _t(_RKL)
        )
        self.assertNotAlmostEqual(base, float(swapped.numpy()), places=6)

    def test_raising_policy_chosen_lowers_chosen_loss(self):
        # Only chosen present so the mean isolates the chosen branch.
        crit = _make_criterion(beta=1.0)
        low, _ = crit.kto_loss(
            _t([0.0]),
            _t(np.zeros((0,))),
            _t([0.0]),
            _t([0.0]),
            _t(np.zeros((0,))),
            _t([0.0]),
        )
        high, _ = crit.kto_loss(
            _t([5.0]),
            _t(np.zeros((0,))),
            _t([0.0]),
            _t([0.0]),
            _t(np.zeros((0,))),
            _t([0.0]),
        )
        # Higher policy_chosen_logps -> larger (logratio - kl) -> smaller loss.
        self.assertLess(float(high.numpy()), float(low.numpy()))

    def test_raising_policy_rejected_raises_rejected_loss(self):
        # Only rejected present so the mean isolates the rejected branch.
        crit = _make_criterion(beta=1.0)
        low, _ = crit.kto_loss(
            _t(np.zeros((0,))),
            _t([0.0]),
            _t([0.0]),
            _t(np.zeros((0,))),
            _t([0.0]),
            _t([0.0]),
        )
        high, _ = crit.kto_loss(
            _t(np.zeros((0,))),
            _t([5.0]),
            _t([0.0]),
            _t(np.zeros((0,))),
            _t([0.0]),
            _t([0.0]),
        )
        # Higher policy_rejected_logps -> smaller (kl - logratio) -> larger loss.
        self.assertGreater(float(high.numpy()), float(low.numpy()))

    def test_empty_chosen_uses_only_rejected_side(self):
        crit = _make_criterion(beta=_BETA, undesirable_weight=1.0)
        loss, kl = crit.kto_loss(
            _t(np.zeros((0,))),
            _t(_PR),
            _t(_PKL),
            _t(np.zeros((0,))),
            _t(_RR),
            _t(_RKL),
        )
        exp_loss, exp_kl = _expected_kto_loss(
            np.zeros((0,)),
            _PR,
            np.zeros((0,)),
            _RR,
            _PKL,
            _RKL,
            _BETA,
            1.0,
            1.0,
        )
        self.assertAlmostEqual(float(loss.numpy()), exp_loss, places=6)
        self.assertAlmostEqual(float(kl.numpy()), exp_kl, places=6)

    def test_empty_rejected_uses_only_chosen_side(self):
        crit = _make_criterion(beta=_BETA)
        loss, _ = crit.kto_loss(
            _t(_PC),
            _t(np.zeros((0,))),
            _t(_PKL),
            _t(_RC),
            _t(np.zeros((0,))),
            _t(_RKL),
        )
        exp_loss, _ = _expected_kto_loss(
            _PC,
            np.zeros((0,)),
            _RC,
            np.zeros((0,)),
            _PKL,
            _RKL,
            _BETA,
            1.0,
            1.0,
        )
        self.assertAlmostEqual(float(loss.numpy()), exp_loss, places=6)

    def test_negative_kl_is_not_clipped_on_single_rank(self):
        # policy_kl < reference_kl so raw kl = mean(pkl-rkl) < 0. On the CPU /
        # world_size==1 path the code does NOT clip kl to min 0, so the loss
        # must equal the formula with the *negative* kl. (The min=0 clip only
        # exists in the dist.get_world_size() > 1 branch -- a single-card vs
        # multi-card behaviour difference; see report.)
        pkl = np.array([-1.0, -1.0], dtype=np.float64)
        rkl = np.array([0.0, 0.0], dtype=np.float64)
        crit = _make_criterion(beta=_BETA)
        loss, kl = crit.kto_loss(
            _t(_PC), _t(_PR), _t(pkl), _t(_RC), _t(_RR), _t(rkl)
        )
        self.assertAlmostEqual(float(kl.numpy()), -1.0, places=6)
        exp_loss, _ = _expected_kto_loss(
            _PC, _PR, _RC, _RR, pkl, rkl, _BETA, 1.0, 1.0
        )
        self.assertAlmostEqual(float(loss.numpy()), exp_loss, places=6)
        # Guard: had kl been clipped to 0, the loss would be measurably different.
        clipped_loss, _ = _expected_kto_loss(
            _PC,
            _PR,
            _RC,
            _RR,
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            _BETA,
            1.0,
            1.0,
        )
        self.assertNotAlmostEqual(float(loss.numpy()), clipped_loss, places=6)


# ---- kto_logps fixtures: batch=3, seq=6, vocab=4, distinguishable logits. ----
# Response region is tokens [1:3], KL region is tokens [3:5]; token 0 and 5 are
# padding (label 0) and outside every summed range.
_LOGITS_NP = (
    (np.arange(3 * 6 * 4, dtype=np.float64).reshape(3, 6, 4) % 7) - 3
) * 0.37
# Real response labels live in [1:3], zeros elsewhere.
_RESP_LABELS = np.array(
    [[0, 1, 2, 0, 0, 0], [0, 3, 1, 0, 0, 0], [0, 2, 3, 0, 0, 0]], dtype=np.int64
)
# Real KL labels live in [3:5], zeros elsewhere (added element-wise to above).
_KL_LABELS = np.array(
    [[0, 0, 0, 1, 2, 0], [0, 0, 0, 3, 1, 0], [0, 0, 0, 2, 1, 0]], dtype=np.int64
)
# columns: [batch_idx, start, chosen_end/kl_start, kl_end, is_chosen]
# 2 chosen (rows 0, 2) and 1 rejected (row 1) -> deliberately unpaired.
_RESP_IDX = np.array(
    [[0, 1, 3, 5, 1], [1, 1, 3, 5, 0], [2, 1, 3, 5, 1]], dtype=np.int64
)


def _reference_kto_logps():
    """Independently derive per-response chosen / rejected / KL logps."""
    combined = _RESP_LABELS + _KL_LABELS
    lsm = _log_softmax(_LOGITS_NP)  # [b, seq, vocab]
    per_token = np.take_along_axis(lsm, combined[:, :, None], axis=-1)[:, :, 0]
    chosen, rejected, kl = [], [], []
    for b, start, cend, kend, is_chosen in _RESP_IDX:
        rng = per_token[b, start:cend].sum()
        if is_chosen == 1:
            chosen.append(rng)
        else:
            rejected.append(rng)
        kl.append(per_token[b, cend:kend].sum())
    return (
        np.array(chosen, dtype=np.float64),
        np.array(rejected, dtype=np.float64),
        np.array(kl, dtype=np.float64),
    )


def _logps_tensors():
    return (
        _t(_LOGITS_NP),
        paddle.to_tensor(_RESP_LABELS),
        paddle.to_tensor(_KL_LABELS),
        paddle.to_tensor(_RESP_IDX),
    )


class TestKTOLogps(unittest.TestCase):
    """Standard (non-fused, non-filtered) kto_logps gather path."""

    def test_logps_match_independent_reference(self):
        crit = _make_criterion()
        chosen, rejected, kl = crit.kto_logps(*_logps_tensors())
        exp_chosen, exp_rejected, exp_kl = _reference_kto_logps()
        # Unpaired: 2 chosen, 1 rejected, 3 KL entries.
        self.assertEqual(list(chosen.shape), [2])
        self.assertEqual(list(rejected.shape), [1])
        self.assertEqual(list(kl.shape), [3])
        np.testing.assert_allclose(
            chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(kl.numpy(), exp_kl, rtol=1e-5, atol=1e-5)

    def test_kl_range_differs_from_response_range(self):
        # KL logps sum a different token range than chosen/rejected; the two
        # must not coincide (guards against gathering the same span twice).
        crit = _make_criterion()
        chosen, _, kl = crit.kto_logps(*_logps_tensors())
        # Row 0 is chosen with response [1:3]; its KL is [3:5]. Different sums.
        self.assertNotAlmostEqual(
            float(chosen.numpy()[0]), float(kl.numpy()[0]), places=6
        )

    def test_all_rejected_yields_empty_chosen(self):
        # If no response is labelled chosen, chosen_logps is an empty tensor
        # (KTO does not fabricate a pair), while KL still covers every response.
        crit = _make_criterion()
        logits, rl, kll, _ = _logps_tensors()
        idx = paddle.to_tensor(
            np.array(
                [[0, 1, 3, 5, 0], [1, 1, 3, 5, 0], [2, 1, 3, 5, 0]],
                dtype=np.int64,
            )
        )
        chosen, rejected, kl = crit.kto_logps(logits, rl, kll, idx)
        self.assertEqual(list(chosen.shape), [0])
        self.assertEqual(list(rejected.shape), [3])
        self.assertEqual(list(kl.shape), [3])

    def test_logits_label_shape_mismatch_raises(self):
        crit = _make_criterion()
        bad_logits = _t(np.zeros((3, 5, 4)))  # seq=5 != label seq=6
        _, rl, kll, idx = _logps_tensors()
        with self.assertRaises(ValueError):
            crit.kto_logps(bad_logits, rl, kll, idx)

    def test_logits_tuple_is_unwrapped(self):
        # forward may hand logits in as a 1-tuple; kto_logps must take [0].
        crit = _make_criterion()
        logits, rl, kll, idx = _logps_tensors()
        chosen, rejected, kl = crit.kto_logps((logits,), rl, kll, idx)
        exp_chosen, exp_rejected, exp_kl = _reference_kto_logps()
        np.testing.assert_allclose(
            chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(kl.numpy(), exp_kl, rtol=1e-5, atol=1e-5)


class TestKTOForward(unittest.TestCase):
    """forward() routing: reference pass -> logps, policy pass -> loss + kl."""

    def test_reference_pass_returns_three_logps(self):
        crit = _make_criterion()
        logits, rl, kll, idx = _logps_tensors()
        labels = (rl, kll, idx, None, None, None)
        out = crit(logits, labels)
        self.assertEqual(len(out), 3)
        ref_chosen, ref_rejected, ref_kl = out
        exp_chosen, exp_rejected, exp_kl = _reference_kto_logps()
        np.testing.assert_allclose(
            ref_chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            ref_rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(ref_kl.numpy(), exp_kl, rtol=1e-5, atol=1e-5)

    def test_policy_pass_returns_logps_loss_and_kl(self):
        crit = _make_criterion(beta=_BETA)
        logits, rl, kll, idx = _logps_tensors()
        exp_chosen, exp_rejected, exp_kl = _reference_kto_logps()
        # Fixed, independent reference logps of the matching (unpaired) shapes.
        ref_chosen = np.array([0.1, -0.2], dtype=np.float64)
        ref_rejected = np.array([0.3], dtype=np.float64)
        ref_kl = np.array([0.05, 0.0, -0.1], dtype=np.float64)
        labels = (rl, kll, idx, _t(ref_chosen), _t(ref_rejected), _t(ref_kl))
        out = crit(logits, labels)
        self.assertEqual(len(out), 5)
        pol_chosen, pol_rejected, pol_kl, loss, kl = out
        np.testing.assert_allclose(
            pol_chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            pol_rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(pol_kl.numpy(), exp_kl, rtol=1e-5, atol=1e-5)
        # Loss and kl must equal the independent formula fed with the policy
        # logps (from the gather) and the given reference logps.
        exp_loss, exp_kl_val = _expected_kto_loss(
            exp_chosen,
            exp_rejected,
            ref_chosen,
            ref_rejected,
            exp_kl,
            ref_kl,
            _BETA,
            1.0,
            1.0,
        )
        self.assertAlmostEqual(float(kl.numpy()), exp_kl_val, places=5)
        self.assertAlmostEqual(float(loss.numpy()), exp_loss, places=5)


class TestKTOUnsupportedEnvironments(unittest.TestCase):
    """Paths that cannot be validated on CPU without a real process group."""

    def test_fused_head_and_loss_fn_path(self):
        raise unittest.SkipTest(
            "use_fused_head_and_loss_fn drives fused_head_and_loss_fn, a "
            "GPU-only fused kernel; not runnable on CPU. Needs single-card env."
        )

    def test_filtered_label_loss_sequence_parallel_gather(self):
        raise unittest.SkipTest(
            "use_filtered_label_loss with sequence_parallel uses "
            "sequence_parallel_sparse_mask_labels / AllGatherVarlenOp collective "
            "gather; only meaningful under a real SP+TP process group (multi-card)."
        )

    def test_tensor_parallel_parallel_cross_entropy(self):
        raise unittest.SkipTest(
            "tensor_parallel_output + tensor_model_parallel_size>1 selects "
            "ParallelCrossEntropy over cross-rank logits; requires a real TP "
            "process group (multi-card), not CPU single-process."
        )

    def test_distributed_kl_all_gather_and_clip(self):
        raise unittest.SkipTest(
            "kto_loss only all-gathers KL across pipe ranks and clips it to "
            "min 0 when dist.get_world_size() > 1; that branch needs a real "
            "multi-rank process group and cannot be exercised on CPU."
        )


if __name__ == "__main__":
    unittest.main()
