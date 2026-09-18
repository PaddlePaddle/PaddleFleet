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

"""Behavior tests for paddlefleet.generation decoding utilities.

Scope: the CPU-executable inference/generation logic that governs decode
selection, generation constraints, sequence-end state and running scores.
Every expected value below is derived independently (hand computation or an
independent numpy/formula reference), never by calling the function under
test.  Fixed, distinguishable inputs are used so token swaps, sign branches,
window offsets and batch-end handling are actually observable.

All tests run on CPU (no accelerator required); the exercised code paths are
pure paddle tensor ops with no device-specific numerics.  The real KV-cache
prefill/decode numeric equivalence needs a small model on an accelerator and
is therefore declared out of scope here (see TestKVCacheDecodeNumerics).
"""

import unittest

import numpy as np
import paddle

from paddlefleet.generation.logits_process import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
from paddlefleet.generation.utils import (
    BeamHypotheses,
    BeamSearchScorer,
    GenerationMixin,
    _make_sliding_window_mask,
    get_unfinished_flag,
)

paddle.set_device("cpu")


class TestGetUnfinishedFlag(unittest.TestCase):
    """EOS-masking / batch-end state for get_unfinished_flag.

    Contract: a sequence stays unfinished iff it was already unfinished AND
    its LAST token is not an eos token.  A finished sequence must never be
    reactivated, and only the last column of input_ids is inspected.
    """

    def test_batch_mixed_end_progress(self):
        input_ids = paddle.to_tensor([[1, 2, 0], [1, 2, 3], [4, 5, 0]])
        unfinished = paddle.to_tensor([[True], [True], [False]])
        out = get_unfinished_flag(input_ids, unfinished, eos_token_id=0)
        # row0: last==eos -> stops; row1: last!=eos -> continues;
        # row2: already finished -> stays finished regardless of token.
        self.assertEqual(out.tolist(), [[False], [True], [False]])

    def test_finished_sequence_not_reactivated(self):
        # last token is NOT eos, but the flag is already False; AND-semantics
        # must keep it False (a done sequence cannot be revived).
        input_ids = paddle.to_tensor([[1, 2, 3]])
        unfinished = paddle.to_tensor([[False]])
        out = get_unfinished_flag(input_ids, unfinished, eos_token_id=0)
        self.assertEqual(out.tolist(), [[False]])

    def test_only_last_token_inspected(self):
        # eos appears at position 0 but the last token is not eos: must
        # remain unfinished (function reads input_ids[:, -1:] only).
        input_ids = paddle.to_tensor([[0, 2, 3]])
        unfinished = paddle.to_tensor([[True]])
        out = get_unfinished_flag(input_ids, unfinished, eos_token_id=0)
        self.assertEqual(out.tolist(), [[True]])

    def test_list_of_eos_ids_stops_on_any(self):
        unfinished = paddle.to_tensor([[True]])
        stop = get_unfinished_flag(
            paddle.to_tensor([[1, 2, 5]]), unfinished, eos_token_id=[0, 5]
        )
        cont = get_unfinished_flag(
            paddle.to_tensor([[1, 2, 9]]), unfinished, eos_token_id=[0, 5]
        )
        self.assertEqual(stop.tolist(), [[False]])  # 5 matches
        self.assertEqual(cont.tolist(), [[True]])  # none match

    def test_nested_list_multi_token_only_checks_last(self):
        # NOTE: the docstring advertises multi-token stop sequences such as
        # [[7, 7]], but the implementation recurses over the sublist and ORs
        # per-id, so [[7, 7]] degenerates to "last token == 7" and does NOT
        # require the preceding token to also be 7.  Asserting ACTUAL behavior.
        unfinished = paddle.to_tensor([[True]])
        stops = get_unfinished_flag(
            paddle.to_tensor([[1, 2, 7]]), unfinished, eos_token_id=[[7, 7]]
        )
        keeps = get_unfinished_flag(
            paddle.to_tensor([[1, 7, 2]]), unfinished, eos_token_id=[[7, 7]]
        )
        self.assertEqual(stops.tolist(), [[False]])  # last==7 -> stop
        self.assertEqual(keeps.tolist(), [[True]])  # last==2 -> continue


class TestMinLengthLogitsProcessor(unittest.TestCase):
    """EOS masking with length semantics.

    Below min_length the eos logit is driven to the dtype's most-negative
    sentinel (the implementation's mask representation, not literal -inf);
    all other logits are untouched.  At/above min_length nothing is masked.
    """

    def test_masks_eos_below_min_length(self):
        proc = MinLengthLogitsProcessor(min_length=5, eos_token_id=2)
        logits = paddle.to_tensor([[0.5, 1.0, 2.0, -1.0]], dtype="float32")
        # cur_len == 3 < 5 -> eos column masked.
        out = proc(paddle.to_tensor([[1, 2, 3]]), logits)
        sentinel = float(paddle.finfo(paddle.float32).min)
        expected = np.array([[0.5, 1.0, sentinel, -1.0]], dtype=np.float32)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_no_mask_at_min_length_boundary(self):
        proc = MinLengthLogitsProcessor(min_length=3, eos_token_id=2)
        logits = paddle.to_tensor([[0.5, 1.0, 2.0, -1.0]], dtype="float32")
        # cur_len == 3 is NOT < 3, so no masking (strict boundary).
        out = proc(paddle.to_tensor([[1, 2, 3]]), logits)
        np.testing.assert_array_equal(
            out.numpy(), np.array([[0.5, 1.0, 2.0, -1.0]], dtype=np.float32)
        )

    def test_rejects_negative_eos(self):
        with self.assertRaises(ValueError):
            MinLengthLogitsProcessor(min_length=5, eos_token_id=-1)


class TestRepetitionPenaltyLogitsProcessor(unittest.TestCase):
    """Repeat penalty: seen tokens with positive logit are divided by the
    penalty, seen tokens with negative logit are multiplied by it; unseen
    tokens are untouched, and the penalty is applied once (not compounded)
    when a token repeats."""

    def test_sign_dependent_penalty_and_untouched_positions(self):
        proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)
        logits = paddle.to_tensor(
            [[0.5, 2.0, -1.0, -3.0, 1.0], [-2.0, 0.5, 1.5, 0.3, 4.0]],
            dtype="float32",
        )
        input_ids = paddle.to_tensor([[1, 3], [0, 4]])
        out = proc(input_ids, logits)
        # row0: tok1 2.0>0 ->/2=1.0 ; tok3 -3.0<0 ->*2=-6.0
        # row1: tok0 -2.0<0 ->*2=-4.0 ; tok4 4.0>0 ->/2=2.0
        expected = np.array(
            [[0.5, 1.0, -1.0, -6.0, 1.0], [-4.0, 0.5, 1.5, 0.3, 2.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_repeated_token_penalized_once(self):
        proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)
        logits = paddle.to_tensor([[0.5, 2.0, -1.0]], dtype="float32")
        # token 1 appears twice; penalty must be applied once: 2.0 / 2 = 1.0.
        out = proc(paddle.to_tensor([[1, 1]]), logits)
        np.testing.assert_allclose(
            out.numpy(),
            np.array([[0.5, 1.0, -1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


class TestGetLogitsProcessorOrder(unittest.TestCase):
    """Processor assembly: which processors are included (each gated by its
    own argument) and in what order.  get_logits_processor ignores `self`, so
    a throwaway object is passed as the receiver."""

    def _build(self, **kwargs):
        proc_list = GenerationMixin.get_logits_processor(object(), **kwargs)
        # LogitsProcessorList stores processors in an insertion-ordered dict.
        return list(proc_list._processors.values())

    def test_full_processor_order(self):
        procs = self._build(
            min_length=3,
            max_length=10,
            eos_token_id=2,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            forced_bos_token_id=1,
            forced_eos_token_id=2,
        )
        self.assertEqual(
            [type(p) for p in procs],
            [
                MinLengthLogitsProcessor,
                RepetitionPenaltyLogitsProcessor,
                NoRepeatNGramLogitsProcessor,
                ForcedBOSTokenLogitsProcessor,
                ForcedEOSTokenLogitsProcessor,
            ],
        )

    def test_gating_conditions_consume_arguments(self):
        # repetition_penalty == 1.0 is a no-op -> processor not added;
        # min_length without eos_token_id -> MinLength not added.
        procs = self._build(
            min_length=3,
            eos_token_id=None,
            repetition_penalty=1.0,
            no_repeat_ngram_size=None,
        )
        self.assertEqual(procs, [])

        procs = self._build(min_length=3, eos_token_id=7)
        self.assertEqual([type(p) for p in procs], [MinLengthLogitsProcessor])

    @unittest.expectedFailure
    def test_custom_logits_processors_merge(self):
        # BUG: get_logits_processor's custom-processor branch iterates the
        # LogitsProcessorList (`for processor in processors`) and calls
        # `.extend(...)`, but LogitsProcessorList defines neither __iter__ nor
        # extend.  So passing custom logits_processors raises instead of
        # returning a merged, type-deduplicated list.  This asserts the
        # CORRECT expected behavior and therefore currently fails, exposing
        # the defect (see report). Marked expectedFailure so the real
        # production defect surfaces without editing production code.
        custom = [MinLengthLogitsProcessor(min_length=3, eos_token_id=2)]
        merged = GenerationMixin.get_logits_processor(
            object(),
            min_length=3,
            eos_token_id=2,
            repetition_penalty=1.2,
            logits_processors=custom,
        )
        types = [type(p) for p in list(merged._processors.values())]
        # Built-in MinLength is dropped (same type overridden by custom) and
        # the custom processor plus the non-overridden RepetitionPenalty stay.
        self.assertIn(RepetitionPenaltyLogitsProcessor, types)
        self.assertIs(list(merged._processors.values())[-1], custom[0])


class TestSlidingWindowMask(unittest.TestCase):
    """Causal sliding-window mask: True where attention is allowed.  The full
    boolean matrix is compared against an independently built reference, and
    the past_key_values_length offset shifts the causal window."""

    @staticmethod
    def _reference(batch, seq_len, past, window):
        total = past + seq_len
        ref = np.zeros((seq_len, total), dtype=bool)
        for i in range(seq_len):
            cur = past + i
            start = max(0, cur - window + 1)
            ref[i, start : cur + 1] = True
        return np.tile(ref[None, None], (batch, 1, 1, 1))

    def test_no_past_full_matrix(self):
        mask = _make_sliding_window_mask(
            (1, 4), past_key_values_length=0, window_size=2
        )
        self.assertEqual(mask.shape, [1, 1, 4, 4])
        np.testing.assert_array_equal(mask.numpy(), self._reference(1, 4, 0, 2))

    def test_with_past_offset_full_matrix(self):
        mask = _make_sliding_window_mask(
            (1, 3), past_key_values_length=2, window_size=2
        )
        self.assertEqual(mask.shape, [1, 1, 3, 5])
        np.testing.assert_array_equal(mask.numpy(), self._reference(1, 3, 2, 2))

    def test_batch_rows_identical(self):
        mask = _make_sliding_window_mask(
            (2, 4), past_key_values_length=0, window_size=3
        )
        np.testing.assert_array_equal(mask.numpy(), self._reference(2, 4, 0, 3))
        # tiling must replicate row 0 exactly onto row 1.
        np.testing.assert_array_equal(mask[0].numpy(), mask[1].numpy())


class TestUpdateScoresForGeneration(unittest.TestCase):
    """Running length-normalized score update; finished rows are left as-is."""

    def test_length_weighted_average_skips_finished(self):
        scores = paddle.to_tensor([[-2.0], [-1.0]], dtype="float32")
        next_scores = paddle.to_tensor([[-0.5], [-0.4]], dtype="float32")
        unfinished = paddle.to_tensor([[True], [False]])
        out = GenerationMixin.update_scores_for_generation(
            scores, next_scores, length=3, unfinished_flag=unfinished
        )
        # row0 (unfinished): (-2*3 + -0.5)/4 = -1.625 ; row1 (finished): -1.0
        np.testing.assert_allclose(
            out.numpy(),
            np.array([[-1.625], [-1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


class TestBeamHypotheses(unittest.TestCase):
    """Length-penalty scoring, origin_len consumption, capacity eviction and
    is_done comparison logic."""

    def test_length_penalty_score_and_origin_len(self):
        hyps = BeamHypotheses(
            num_beams=3, length_penalty=1.0, early_stopping=False
        )
        hyp = paddle.to_tensor([1, 2, 3, 4, 5, 6, 7, 8])  # length 8
        hyps.add(hyp, sum_logprobs=-1.0, origin_len=5)
        # score = -1.0 / (((8 - 5 + 5) / 6) ** 1.0) = -1.0 / (8/6) = -0.75
        self.assertAlmostEqual(hyps.beams[0][0], -0.75, places=6)
        self.assertAlmostEqual(hyps.worst_score, -0.75, places=6)
        np.testing.assert_array_equal(hyps.beams[0][1].numpy(), hyp.numpy())

    def test_origin_len_actually_changes_score(self):
        hyp = paddle.to_tensor([1, 2, 3, 4, 5, 6, 7, 8])
        a = BeamHypotheses(
            num_beams=3, length_penalty=1.0, early_stopping=False
        )
        a.add(hyp, sum_logprobs=-1.0, origin_len=5)
        b = BeamHypotheses(
            num_beams=3, length_penalty=1.0, early_stopping=False
        )
        b.add(hyp, sum_logprobs=-1.0, origin_len=2)
        # different origin_len -> different denominator -> different score.
        self.assertNotAlmostEqual(a.beams[0][0], b.beams[0][0], places=6)
        self.assertAlmostEqual(b.beams[0][0], -1.0 / (11.0 / 6.0), places=6)

    def test_capacity_evicts_worst_keeps_best(self):
        # length_penalty=0 makes score == sum_logprobs (denominator ** 0 == 1).
        hyps = BeamHypotheses(
            num_beams=2, length_penalty=0.0, early_stopping=False
        )
        hyps.add(paddle.to_tensor([10, 10]), sum_logprobs=-3.0)  # worst
        hyps.add(paddle.to_tensor([11, 11]), sum_logprobs=-1.0)  # best
        hyps.add(paddle.to_tensor([12, 12]), sum_logprobs=-2.0)  # middle
        self.assertEqual(len(hyps), 2)
        kept_first_tokens = {int(hyp[0]) for _, hyp in hyps.beams}
        self.assertEqual(kept_first_tokens, {11, 12})  # -3.0 beam evicted
        self.assertAlmostEqual(hyps.worst_score, -2.0, places=6)

    def test_is_done_uses_score_comparison(self):
        hyps = BeamHypotheses(
            num_beams=2, length_penalty=1.0, early_stopping=False
        )
        hyps.add(paddle.to_tensor([1, 2]), sum_logprobs=-1.0)
        hyps.add(paddle.to_tensor([1, 3]), sum_logprobs=-2.0)
        # worst_score = -2.0 / (7/6) ; cur_score = best / ((5+5)/6) = best*0.6
        self.assertAlmostEqual(hyps.worst_score, -2.0 / (7.0 / 6.0), places=6)
        # best high -> worst < cur_score -> not done.
        self.assertFalse(hyps.is_done(best_sum_logprobs=0.0, cur_len=5))
        # best very negative -> worst >= cur_score -> done.
        self.assertTrue(hyps.is_done(best_sum_logprobs=-100.0, cur_len=5))

    def test_is_done_false_when_not_full(self):
        hyps = BeamHypotheses(
            num_beams=3, length_penalty=1.0, early_stopping=False
        )
        hyps.add(paddle.to_tensor([1, 2]), sum_logprobs=-1.0)
        self.assertFalse(hyps.is_done(best_sum_logprobs=-100.0, cur_len=5))

    def test_is_done_early_stopping_when_full(self):
        hyps = BeamHypotheses(
            num_beams=2, length_penalty=1.0, early_stopping=True
        )
        hyps.add(paddle.to_tensor([1, 2]), sum_logprobs=-1.0)
        hyps.add(paddle.to_tensor([1, 3]), sum_logprobs=-2.0)
        self.assertTrue(hyps.is_done(best_sum_logprobs=-100.0, cur_len=5))


class TestBeamSearchScorerProcess(unittest.TestCase):
    """Batch-end EOS routing and done-batch padding for BeamSearchScorer."""

    def test_init_rejects_bad_num_beams(self):
        with self.assertRaises(ValueError):
            BeamSearchScorer(batch_size=1, max_length=10, num_beams=1)

    def test_eos_routed_to_hypotheses_others_fill_beams(self):
        scorer = BeamSearchScorer(batch_size=1, max_length=10, num_beams=2)
        # 2 beams, current length 3.
        input_ids = paddle.to_tensor([[1, 2, 3], [1, 2, 4]])
        # candidate list per batch (len 2*num_beams): the eos (0) sits at
        # rank 1 (within top group_size=2) and must go to hypotheses.
        next_tokens = paddle.to_tensor([[5, 0, 6, 7]])
        next_scores = paddle.to_tensor(
            [[-0.1, -0.2, -0.3, -0.4]], dtype="float32"
        )
        next_indices = paddle.to_tensor([[0, 1, 0, 1]])
        out = scorer.process(
            input_ids,
            next_scores,
            next_tokens,
            next_indices,
            pad_token_id=9,
            eos_token_id=0,
        )
        # non-eos tokens 5 then 6 fill the two beam slots, in order.
        self.assertEqual(out["next_beam_tokens"].tolist(), [5, 6])
        np.testing.assert_allclose(
            out["next_beam_scores"].numpy(),
            np.array([-0.1, -0.3], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        # batch_beam_idx = batch_idx*group_size + next_index; both from beam 0.
        self.assertEqual(out["next_beam_indices"].tolist(), [0, 0])
        # the eos candidate (next_index 1 -> input_ids row 1) was stored.
        self.assertEqual(len(scorer._beam_hyps[0]), 1)
        np.testing.assert_array_equal(
            scorer._beam_hyps[0].beams[0][1].numpy(),
            np.array([1, 2, 4]),
        )
        self.assertFalse(scorer.is_done)  # only 1 hyp < num_beams

    def test_done_batch_is_padded_not_reprocessed(self):
        scorer = BeamSearchScorer(batch_size=1, max_length=10, num_beams=2)
        # Arrange a batch that is already done (>= num_beams hypotheses).
        scorer._beam_hyps[0].add(paddle.to_tensor([1, 2, 3]), sum_logprobs=-1.0)
        scorer._beam_hyps[0].add(paddle.to_tensor([1, 2, 4]), sum_logprobs=-2.0)
        scorer._done[0] = 1
        input_ids = paddle.to_tensor([[1, 2, 3], [1, 2, 4]])
        next_tokens = paddle.to_tensor([[5, 6, 7, 8]])
        next_scores = paddle.to_tensor(
            [[-0.1, -0.2, -0.3, -0.4]], dtype="float32"
        )
        next_indices = paddle.to_tensor([[0, 1, 0, 1]])
        out = scorer.process(
            input_ids,
            next_scores,
            next_tokens,
            next_indices,
            pad_token_id=9,
            eos_token_id=0,
        )
        # done batch: tokens padded, scores zeroed, no new tokens promoted.
        self.assertEqual(out["next_beam_tokens"].tolist(), [9, 9])
        np.testing.assert_allclose(
            out["next_beam_scores"].numpy(),
            np.zeros(2, dtype=np.float32),
            rtol=0,
            atol=0,
        )
        self.assertEqual(len(scorer._beam_hyps[0]), 2)  # unchanged


class TestKVCacheDecodeNumerics(unittest.TestCase):
    """KV-cache prefill/decode numeric equivalence is not verified here."""

    @unittest.skip(
        "KV-cache prefill/decode numeric equivalence requires a real small "
        "model on an accelerator (compare incremental decode against a full "
        "forward pass); it cannot be validated with CPU-only deterministic "
        "logits and is out of scope for this no-accelerator suite."
    )
    def test_kv_cache_decode_matches_full_forward(self):
        raise AssertionError("must run on accelerator with a real model")


if __name__ == "__main__":
    unittest.main()
