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

"""Behavior tests for paddlefleet.nn.criterion.dpo_loss.

Environment: 无卡 (CPU). Every function exercised here is pure Paddle math on
small hand-built tensors and runs on CPU; there is no GPU-only numeric path in
these targets, so nothing is skipped for hardware reasons.

Scope and what is intentionally NOT covered here (documented, not papered over):
  - The fused-head / tensor-parallel / sequence-parallel / subbatch branches of
    ``dpo_logps`` require a real TP+SP process group and the fused kernels; those
    are distributed/GPU paths and belong to multi-card tests. This file exercises
    the CPU-executable non-fused, non-filtered logps gathering path and the pure
    DPO loss math, with independent hand-derived expectations.

References are hand-derived with numpy (never by calling the function under
test), so a regression in the production formula, in the chosen/rejected vs
policy/reference wiring, in the label range/offset handling, or in the reduction
will be rejected instead of silently passing.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.nn.criterion.dpo_loss import (
    cal_dpo_loss,
    dpo_logps,
    dpo_preprocess_inputs,
    loss_impl,
)


# --- independent numpy references (do NOT call the code under test) ---------
def _log_sigmoid(x):
    # log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x)), numerically stable.
    return -np.logaddexp(0.0, -np.asarray(x, dtype=np.float64))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# --- lightweight real objects (NOT MagicMock, so attribute comparisons are
#     genuine values rather than always-truthy proxies) ------------------------
class _DPOConfig:
    def __init__(
        self,
        loss_type="sigmoid",
        beta=0.1,
        label_smoothing=0.0,
        offset_alpha=0.0,
        simpo_gamma=0.5,
        dpop_lambda=1.0,
        pref_loss_ratio=1.0,
        sft_loss_ratio=0.1,
        normalize_logps=False,
        ignore_eos_token=False,
    ):
        self.loss_type = loss_type
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.offset_alpha = offset_alpha
        self.simpo_gamma = simpo_gamma
        self.dpop_lambda = dpop_lambda
        self.pref_loss_ratio = pref_loss_ratio
        self.sft_loss_ratio = sft_loss_ratio
        self.normalize_logps = normalize_logps
        self.ignore_eos_token = ignore_eos_token


class _ModelConfig:
    def __init__(self, tensor_model_parallel_size=1, sequence_parallel=False):
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.sequence_parallel = sequence_parallel
        self.max_sequence_length = None
        self.vocab_size = None


class _Criterion:
    """Minimal stand-in for the criterion Layer's attribute surface."""

    def __init__(
        self,
        dpo_config,
        loss_func=None,
        config=None,
        use_filtered_label_loss=False,
        use_fused_head_and_loss_fn=False,
        use_subbatch=False,
        loss_subbatch_sequence_length=1024,
        tie_word_embeddings=False,
    ):
        self.dpo_config = dpo_config
        self.loss_func = loss_func
        self.config = config or _ModelConfig()
        self.use_filtered_label_loss = use_filtered_label_loss
        self.use_fused_head_and_loss_fn = use_fused_head_and_loss_fn
        self.use_subbatch = use_subbatch
        self.loss_subbatch_sequence_length = loss_subbatch_sequence_length
        self.tie_word_embeddings = tie_word_embeddings


def _crit(**cfg_kwargs):
    return _Criterion(_DPOConfig(**cfg_kwargs))


class TestDpoPreprocessInputs(unittest.TestCase):
    """dpo_preprocess_inputs decides, by the arity of the logits object, whether
    we are on the plain-logits path or the fused-head path. Assert which slot
    each input lands in, not merely that a call succeeds."""

    def setUp(self):
        # self is unused by the function; any object works as the bound arg.
        self.owner = object()
        self.labels = paddle.arange(8, dtype="int64").reshape([2, 4])

    def test_plain_tensor_passthrough(self):
        logits = paddle.arange(2 * 4 * 8, dtype="float32").reshape([2, 4, 8])
        out_logits, out_labels, hs, w, b, ty = dpo_preprocess_inputs(
            self.owner, logits, self.labels
        )
        # Plain tensor stays as logits; no fused-head tensors produced.
        self.assertIs(out_logits, logits)
        self.assertIs(out_labels, self.labels)
        self.assertIsNone(hs)
        self.assertIsNone(w)
        self.assertIsNone(b)
        self.assertIsNone(ty)

    def test_len4_tuple_is_fused_head_unpack(self):
        hidden = paddle.randn([2, 4, 8])
        weight = paddle.randn([8, 16])
        bias = paddle.randn([16])
        out_logits, out_labels, hs, w, b, ty = dpo_preprocess_inputs(
            self.owner, (hidden, weight, bias, True), self.labels
        )
        # A 4-tuple means "no materialised logits; compute head later": logits
        # must be None and hidden/weight/bias/transpose_y filled from the tuple.
        self.assertIsNone(out_logits)
        self.assertIs(hs, hidden)
        self.assertIs(w, weight)
        self.assertIs(b, bias)
        self.assertEqual(ty, True)
        self.assertIs(out_labels, self.labels)

    def test_len1_tuple_is_recursively_unwrapped(self):
        inner = paddle.randn([2, 4, 8])
        out_logits, _, hs, w, b, ty = dpo_preprocess_inputs(
            self.owner, (inner,), self.labels
        )
        self.assertIs(out_logits, inner)
        for slot in (hs, w, b, ty):
            self.assertIsNone(slot)

    def test_len2_tuple_takes_first_element(self):
        first = paddle.randn([2, 4, 8])
        second = paddle.randn([2, 4, 8])
        out_logits, _, hs, w, b, ty = dpo_preprocess_inputs(
            self.owner, (first, second), self.labels
        )
        # 2-tuple resolves to its first element as logits (second is dropped),
        # and stays on the plain path (no fused-head tensors).
        self.assertIs(out_logits, first)
        for slot in (hs, w, b, ty):
            self.assertIsNone(slot)


class TestLossImpl(unittest.TestCase):
    """loss_impl casts logits to float32 and forwards to self.loss_func without
    negating or shifting labels (negation happens later in dpo_logps)."""

    def test_casts_to_float32_and_forwards_verbatim(self):
        captured = {}

        def loss_func(logits, labels):
            captured["logits_dtype"] = logits.dtype
            captured["labels"] = labels
            return paddle.to_tensor([[1.0], [2.0]], dtype="float32")

        crit = _Criterion(_DPOConfig(), loss_func=loss_func)
        logits = paddle.ones([2, 1], dtype="float16")
        labels = paddle.to_tensor([[3], [4]], dtype="int64")

        out = loss_impl(crit, logits, labels)

        # float16 in -> float32 handed to loss_func.
        self.assertEqual(captured["logits_dtype"], paddle.float32)
        # labels forwarded unchanged (no shift inside loss_impl).
        np.testing.assert_array_equal(
            captured["labels"].numpy(), labels.numpy()
        )
        # return value is loss_func's output verbatim (no negation/reduction).
        np.testing.assert_array_equal(out.numpy(), [[1.0], [2.0]])


# PLACEHOLDER_LOGPS

# Known per-token "loss" the injected loss_func returns. dpo_logps negates it to
# get per-token logprobs, then gathers chosen/rejected ranges. Distinct values
# per (row, col) so a wrong index range / wrong micro-batch row is detectable.
_TOKEN_LOSS = np.array(
    [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    ],
    dtype=np.float32,
)


def _logps_criterion(**dpo_cfg):
    token_loss = paddle.to_tensor(_TOKEN_LOSS).unsqueeze(-1)  # [2, 6, 1]

    def loss_func(logits, labels):
        # loss_func is an injected collaborator, NOT the code under test. It
        # returns a known per-token loss so the gather/offset/reduce logic that
        # dpo_logps actually owns can be checked against an independent numpy
        # derivation below.
        return token_loss

    return _Criterion(_DPOConfig(**dpo_cfg), loss_func=loss_func)


def _ptl():
    # per-token logprobs = -loss_func output, as numpy for independent expected.
    return -_TOKEN_LOSS


class TestDpoLogps(unittest.TestCase):
    """dpo_logps turns per-token logprobs into per-pair chosen/rejected sums.

    Verifies: chosen tokens come from [c_start, c_end), rejected from
    [r_start, r_end); the micro-batch index selects the right row; ignore_eos
    offset trims the last token; sft_loss is the token-averaged chosen NLL; and
    the normalize / average_log_prob length scalings.
    """

    def _inputs(self, response_indexs):
        logits = paddle.randn(
            [2, 6, 5], dtype="float32"
        )  # ignored by loss_func
        labels = paddle.zeros([2, 6], dtype="int64")
        idx = paddle.to_tensor(response_indexs, dtype="int32")
        return logits, labels, idx

    def test_chosen_rejected_ranges_and_sft(self):
        crit = _logps_criterion()
        idx = [[0, 0, 3, 5], [1, 1, 3, 6]]
        logits, labels, idx_t = self._inputs(idx)
        chosen, rejected, sft = dpo_logps(crit, logits, labels, idx_t)

        ptl = _ptl()
        exp_chosen = np.array([ptl[0, 0:3].sum(), ptl[1, 1:3].sum()])
        exp_rejected = np.array([ptl[0, 3:5].sum(), ptl[1, 3:6].sum()])
        np.testing.assert_allclose(
            chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-6
        )
        # sft = -sum(chosen_logps) / sum(chosen token counts) * sft_loss_ratio.
        chosen_lens = np.array([3 - 0, 3 - 1])
        exp_sft = -exp_chosen.sum() / chosen_lens.sum() * 0.1
        np.testing.assert_allclose(sft.numpy(), exp_sft, rtol=1e-5, atol=1e-6)

    def test_range_boundaries_select_correct_tokens(self):
        # Guards the chosen/rejected token windows: moving the boundaries moves
        # exactly which tokens land in each side's sum.
        crit = _logps_criterion()
        c1, r1, _ = dpo_logps(*self._call_args(crit, [[0, 0, 2, 5]]))
        c2, r2, _ = dpo_logps(*self._call_args(crit, [[0, 2, 4, 5]]))
        ptl = _ptl()
        np.testing.assert_allclose(
            c1.numpy(), [ptl[0, 0:2].sum()], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            r1.numpy(), [ptl[0, 2:5].sum()], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            c2.numpy(), [ptl[0, 2:4].sum()], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            r2.numpy(), [ptl[0, 4:5].sum()], rtol=1e-5, atol=1e-6
        )

    def _call_args(self, crit, response_indexs):
        logits, labels, idx = self._inputs(response_indexs)
        return crit, logits, labels, idx

    def test_ignore_eos_offset_trims_last_token(self):
        crit = _logps_criterion(ignore_eos_token=True)
        idx = [[0, 0, 3, 5], [1, 1, 3, 6]]
        logits, labels, idx_t = self._inputs(idx)
        chosen, rejected, sft = dpo_logps(crit, logits, labels, idx_t)

        ptl = _ptl()
        # offset=1: chosen [c_start, c_end-1), rejected [r_start, r_end-1).
        exp_chosen = np.array([ptl[0, 0:2].sum(), ptl[1, 1:2].sum()])
        exp_rejected = np.array([ptl[0, 3:4].sum(), ptl[1, 3:5].sum()])
        np.testing.assert_allclose(
            chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-6
        )
        chosen_lens = np.array([3 - 0 - 1, 3 - 1 - 1])
        exp_sft = -exp_chosen.sum() / chosen_lens.sum() * 0.1
        np.testing.assert_allclose(sft.numpy(), exp_sft, rtol=1e-5, atol=1e-6)

    def test_average_log_prob_divides_by_lengths(self):
        crit = _logps_criterion()
        idx = [[0, 0, 3, 5], [1, 1, 3, 6]]
        logits, labels, idx_t = self._inputs(idx)
        chosen, rejected, sft = dpo_logps(
            crit, logits, labels, idx_t, average_log_prob=True
        )
        ptl = _ptl()
        exp_chosen = np.array(
            [ptl[0, 0:3].sum() / 3.0, ptl[1, 1:3].sum() / 2.0]
        )
        exp_rejected = np.array(
            [ptl[0, 3:5].sum() / 2.0, ptl[1, 3:6].sum() / 3.0]
        )
        np.testing.assert_allclose(
            chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-6
        )
        # sft uses the pre-average chosen sums, divided by token counts.
        raw_chosen = np.array([ptl[0, 0:3].sum(), ptl[1, 1:3].sum()])
        exp_sft = -raw_chosen.sum() / (3 + 2) * 0.1
        np.testing.assert_allclose(sft.numpy(), exp_sft, rtol=1e-5, atol=1e-6)

    def test_normalize_logps_scales_by_avg_over_side_length(self):
        crit = _logps_criterion(normalize_logps=True)
        idx = [[0, 0, 3, 5], [1, 1, 3, 6]]
        logits, labels, idx_t = self._inputs(idx)
        chosen, rejected, _ = dpo_logps(crit, logits, labels, idx_t)
        ptl = _ptl()
        raw_chosen = np.array([ptl[0, 0:3].sum(), ptl[1, 1:3].sum()])
        raw_rejected = np.array([ptl[0, 3:5].sum(), ptl[1, 3:6].sum()])
        avg = np.array([(5 - 0) / 2.0, (6 - 1) / 2.0])
        chosen_len = np.array([3 - 0, 3 - 1])
        rejected_len = np.array([5 - 3, 6 - 3])
        exp_chosen = raw_chosen * avg / chosen_len
        exp_rejected = raw_rejected * avg / rejected_len
        np.testing.assert_allclose(
            chosen.numpy(), exp_chosen, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            rejected.numpy(), exp_rejected, rtol=1e-5, atol=1e-6
        )

    def test_padding_pair_uses_sentinel(self):
        # response_index[3] == 0 marks a padding pair: chosen/rejected become the
        # 100.0 sentinel while the real pair is untouched.
        crit = _logps_criterion()
        idx = [[0, 0, 3, 5], [1, 1, 3, 0]]
        logits, labels, idx_t = self._inputs(idx)
        chosen, rejected, _ = dpo_logps(crit, logits, labels, idx_t)
        ptl = _ptl()
        np.testing.assert_allclose(
            chosen.numpy(), [ptl[0, 0:3].sum(), 100.0], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            rejected.numpy(), [ptl[0, 3:5].sum(), 100.0], rtol=1e-5, atol=1e-6
        )


# Fixed, distinguishable, non-degenerate preference logprobs. Chosen != rejected
# and policy != reference on every element so a swapped side or a dropped
# reference term changes the result.
_PC = np.array([1.0, 0.5, -0.5, 2.0], dtype=np.float32)
_PR = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
_RC = np.array([0.5, 0.0, -1.0, 1.0], dtype=np.float32)
_RR = np.array([0.2, 0.5, -0.5, 0.3], dtype=np.float32)


def _t(arr):
    return paddle.to_tensor(np.asarray(arr, dtype=np.float32))


def _dpo_logits(pc, pr, rc, rr):
    return (pc - pr) - (rc - rr)


class TestCalDpoLoss(unittest.TestCase):
    """cal_dpo_loss maps (policy/reference) x (chosen/rejected) logprobs to a
    scalar per loss_type. Each reference below is a hand-derived numpy formula;
    none call cal_dpo_loss. The DPO log-ratio is (pc-pr)-(rc-rr)."""

    def _call(
        self,
        loss_type,
        pc=_PC,
        pr=_PR,
        rc=_RC,
        rr=_RR,
        score_deltas=None,
        **cfg,
    ):
        crit = _crit(loss_type=loss_type, **cfg)
        sd = _t(score_deltas) if score_deltas is not None else None
        return cal_dpo_loss(crit, _t(pc), _t(pr), _t(rc), _t(rr), sd)

    def test_sigmoid_matches_reference(self):
        beta = 0.1
        logits = _dpo_logits(_PC, _PR, _RC, _RR)
        expected = np.mean(-_log_sigmoid(beta * logits))
        out = self._call("sigmoid")
        self.assertEqual(list(out.shape), [])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_sigmoid_label_smoothing(self):
        beta, ls = 0.1, 0.1
        logits = _dpo_logits(_PC, _PR, _RC, _RR)
        expected = np.mean(
            -_log_sigmoid(beta * logits) * (1 - ls)
            - _log_sigmoid(-beta * logits) * ls
        )
        out = self._call("sigmoid", label_smoothing=0.1)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_sigmoid_offset_alpha_shifts_by_score_deltas(self):
        beta, alpha = 0.1, 0.5
        sd = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
        logits = _dpo_logits(_PC, _PR, _RC, _RR)
        shifted = logits - alpha / beta * np.log(sd + 1e-6)
        expected = np.mean(-_log_sigmoid(beta * shifted))
        out = self._call("sigmoid", score_deltas=sd, offset_alpha=0.5)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_hinge_matches_reference(self):
        beta = 0.1
        logits = _dpo_logits(_PC, _PR, _RC, _RR)
        expected = np.mean(np.maximum(0.0, 1 - beta * logits))
        out = self._call("hinge")
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_ipo_matches_reference(self):
        beta = 0.1
        logits = _dpo_logits(_PC, _PR, _RC, _RR)
        expected = np.mean((logits - 1 / (2 * beta)) ** 2)
        out = self._call("ipo")
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_simpo_subtracts_gamma_over_beta(self):
        beta, gamma = 0.1, 0.5
        logits = _dpo_logits(_PC, _PR, _RC, _RR) - gamma / beta
        expected = np.mean(-_log_sigmoid(beta * logits))
        out = self._call("simpo")
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_dpop_penalizes_chosen_logprob_drop(self):
        # Pick inputs where reference_chosen > policy_chosen so the positive
        # regulariser is active (clip(rc-pc, 0) > 0); a degenerate input would
        # collapse dpop to plain sigmoid and hide the penalty term.
        beta, lam = 0.1, 1.0
        pc = np.array([0.0, 0.0], dtype=np.float32)
        pr = np.array([0.0, 0.0], dtype=np.float32)
        rc = np.array([1.0, 0.0], dtype=np.float32)
        rr = np.array([0.0, 0.0], dtype=np.float32)
        logits = _dpo_logits(pc, pr, rc, rr)
        positive_reg = np.clip(rc - pc, 0.0, None)
        expected = np.mean(-_log_sigmoid(beta * (logits - lam * positive_reg)))
        out = self._call("dpop", pc=pc, pr=pr, rc=rc, rr=rr)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # Sanity: the penalty actually moved the loss vs. the no-penalty case.
        no_pen = np.mean(-_log_sigmoid(beta * logits))
        self.assertGreater(abs(expected - no_pen), 1e-4)

    def test_kto_pair_matches_reference(self):
        beta = 0.1
        chosen_KL = max(np.mean(_PC - _RC), 0.0)
        rejected_KL = max(np.mean(_PR - _RR), 0.0)
        chosen_logratios = _PC - _RC
        rejected_logratios = _PR - _RR
        first = 1 - _sigmoid(beta * (chosen_logratios - rejected_KL))
        second = 1 - _sigmoid(beta * (chosen_KL - rejected_logratios))
        expected = np.mean(np.concatenate([first, second]))
        out = self._call("kto_pair")
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_sppo_hard_matches_reference(self):
        beta = 0.1
        a = _PC - _RC
        b = _PR - _RR
        expected = np.mean((a - 0.5 / beta) ** 2 + (b + 0.5 / beta) ** 2)
        out = self._call("sppo_hard")
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-4)

    def test_nca_pair_matches_reference(self):
        beta = 0.1
        chosen_rewards = (_PC - _RC) * beta
        rejected_rewards = (_PR - _RR) * beta
        expected = np.mean(
            -_log_sigmoid(chosen_rewards)
            - 0.5 * _log_sigmoid(-chosen_rewards)
            - 0.5 * _log_sigmoid(-rejected_rewards)
        )
        out = self._call("nca_pair")
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_or_orpo_is_reference_free(self):
        # ORPO uses only policy logprobs (must be <= 0 for log1p(-exp(.))).
        pc = np.array([-0.5, -1.0, -0.2, -1.5], dtype=np.float32)
        pr = np.array([-1.0, -0.5, -1.2, -0.8], dtype=np.float32)
        rc = np.array([-9.0, -9.0, -9.0, -9.0], dtype=np.float32)  # ignored
        rr = np.array([-3.0, -3.0, -3.0, -3.0], dtype=np.float32)  # ignored
        log_odds = (pc - pr) - (np.log1p(-np.exp(pc)) - np.log1p(-np.exp(pr)))
        expected = np.mean(-_log_sigmoid(log_odds))
        out = self._call("or", pc=pc, pr=pr, rc=rc, rr=rr)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)
        # Reference-free: changing reference logprobs must not change the loss.
        out2 = self._call("or", pc=pc, pr=pr, rc=rc + 5.0, rr=rr - 2.0)
        np.testing.assert_allclose(
            out.numpy(), out2.numpy(), rtol=1e-6, atol=1e-7
        )

    def test_invalid_loss_type_raises(self):
        with self.assertRaises(ValueError):
            self._call("no_such_loss")

    def test_pref_loss_ratio_scales_linearly(self):
        one = self._call("sigmoid", pref_loss_ratio=1.0)
        two = self._call("sigmoid", pref_loss_ratio=2.0)
        np.testing.assert_allclose(
            two.numpy(), 2.0 * one.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_reference_term_is_subtracted(self):
        # Shifting policy_chosen and reference_chosen by the same constant leaves
        # the DPO log-ratio unchanged -> sigmoid loss unchanged. This pins the
        # policy/reference correspondence (reference genuinely subtracted).
        base = self._call("sigmoid")
        shifted_both = self._call("sigmoid", pc=_PC + 0.7, rc=_RC + 0.7)
        np.testing.assert_allclose(
            base.numpy(), shifted_both.numpy(), rtol=1e-5, atol=1e-6
        )
        # Shifting only policy_chosen DOES change the loss.
        shifted_policy = self._call("sigmoid", pc=_PC + 0.7)
        self.assertGreater(abs(base.numpy() - shifted_policy.numpy()), 1e-4)

    def test_swapping_chosen_rejected_changes_preference_loss(self):
        # Strong preference: chosen >> rejected -> low sigmoid loss. Swap the two
        # sides and the loss must rise (guards chosen/rejected wiring; a
        # side-agnostic bug would give equal losses).
        pc = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        pr = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rc = np.zeros(4, dtype=np.float32)
        rr = np.zeros(4, dtype=np.float32)
        good = self._call("sigmoid", pc=pc, pr=pr, rc=rc, rr=rr)
        swapped = self._call("sigmoid", pc=pr, pr=pc, rc=rc, rr=rr)
        self.assertLess(good.numpy(), swapped.numpy())


if __name__ == "__main__":
    unittest.main()
