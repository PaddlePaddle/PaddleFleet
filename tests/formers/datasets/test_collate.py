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

"""Behavior tests for paddlefleet.datasets.collate DPO collation.

These tests use small, content-distinguishable sequences and hand-derived
expected batches. They verify that padding never enters supervision
(response_labels padded with -100 while input/position padded with 0), that
token / position / label fields stay length-synchronized, that packing keeps
per-document attention block-diagonal (no cross-document leakage), and that
response-index accounting differs correctly between the filtered and unfiltered
label-loss modes.
"""

import unittest

import numpy as np

from paddlefleet.datasets.collate import calc_padding_size, dpo_collate_fn
from paddlefleet.datasets.DPODataset import Sequence


class _TrainingArgs:
    """Minimal stub for the training_args consumed by calc_padding_size.

    Only the four attributes that calc_padding_size reads are provided; no
    behavior of the function under test is emulated here.
    """

    def __init__(
        self, cp_size=1, tp_size=1, sequence_parallel=False, fp8=False
    ):
        self.context_parallel_size = cp_size
        self.tensor_model_parallel_size = tp_size
        self.sequence_parallel = sequence_parallel
        self.fp8 = fp8


def _causal_mask(n):
    """Lower-triangular (causal) attention mask as a list-of-lists."""
    return [[1 if j <= i else 0 for j in range(n)] for i in range(n)]


def _seq(
    token_ids,
    position_ids,
    response_labels,
    response_index,
    attention_mask=None,
    attn_mask_startend_row_indices=None,
    score_delta=0.0,
):
    return Sequence(
        token_ids=token_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        response_labels=response_labels,
        response_index=response_index,
        score_delta=score_delta,
        has_mm=[False],
    )


class TestCalcPaddingSize(unittest.TestCase):
    """Alignment math derived independently from the padding formula."""

    def test_no_parallelism_is_identity(self):
        # cp*sp == 1 -> padding_to_size == 1 -> no rounding.
        args = _TrainingArgs(cp_size=1, tp_size=1, sequence_parallel=False)
        self.assertEqual(calc_padding_size(100, args), 100)
        self.assertEqual(calc_padding_size(7, args), 7)

    def test_context_parallel_rounds_up_to_four(self):
        # padding_to_size = 2 (since cp*sp>1) * cp(2) * sp(1) = 4.
        args = _TrainingArgs(cp_size=2, tp_size=1, sequence_parallel=False)
        self.assertEqual(calc_padding_size(5, args), 8)
        self.assertEqual(calc_padding_size(8, args), 8)
        self.assertEqual(calc_padding_size(9, args), 12)

    def test_sequence_parallel_consumes_tp_size(self):
        # sequence_parallel True -> sp_size = tp_size = 2.
        # padding_to_size = 2 * 1 * 2 = 4.
        args = _TrainingArgs(cp_size=1, tp_size=2, sequence_parallel=True)
        self.assertEqual(calc_padding_size(5, args), 8)
        self.assertEqual(calc_padding_size(4, args), 4)

    def test_tp_size_ignored_when_sequence_parallel_off(self):
        # sequence_parallel False -> sp_size forced to 1, tp_size=4 ignored.
        # cp*sp == 1 -> padding_to_size == 1 -> identity.
        args = _TrainingArgs(cp_size=1, tp_size=4, sequence_parallel=False)
        self.assertEqual(calc_padding_size(10, args), 10)

    def test_fp8_rounds_base_block_to_four(self):
        # base padding_to_size 2 -> fp8 rounds to (2+3)//4*4 = 4;
        # then * cp(1) * sp(2) = 8. ceil(5/8)*8 = 8.
        args = _TrainingArgs(
            cp_size=1, tp_size=2, sequence_parallel=True, fp8=True
        )
        self.assertEqual(calc_padding_size(5, args), 8)
        # ceil(9/8)*8 = 16 confirms the block is really 8, not 4.
        self.assertEqual(calc_padding_size(9, args), 16)


class TestDpoCollatePaddingExcludedFromSupervision(unittest.TestCase):
    """Padding must be zero for ids/positions but -100 for labels."""

    def test_single_sequence_padding_values_and_field_sync(self):
        seq = _seq(
            token_ids=[10, 11, 12],
            position_ids=[0, 1, 2],
            response_labels=[-100, -100, 7],
            response_index=[0, 1, 2],
            attention_mask=_causal_mask(3),
        )
        args = _TrainingArgs()  # identity padding -> length stays max_seq_len

        result = dpo_collate_fn(
            [[seq]],
            tokenizer=None,
            training_args=args,
            max_seq_len=6,
            use_filtered_label_loss=False,
        )

        # Content region preserved, tail padded to length 6.
        np.testing.assert_array_equal(
            result["input_ids"][0], [10, 11, 12, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            result["position_ids"][0], [0, 1, 2, 0, 0, 0]
        )
        # Labels pad with -100 so the padded tail is ignored by the loss,
        # while ids/positions pad with 0. This is the padding-not-in-
        # supervision contract.
        np.testing.assert_array_equal(
            result["response_labels"][0], [-100, -100, 7, -100, -100, -100]
        )

        # token / position / label stay length-synchronized.
        self.assertEqual(result["input_ids"].shape, (1, 6))
        self.assertEqual(result["position_ids"].shape, (1, 6))
        self.assertEqual(result["response_labels"].shape, (1, 6))

    def test_no_max_seq_len_uses_batch_max_no_padding(self):
        seq = _seq(
            token_ids=[10, 11, 12],
            position_ids=[0, 1, 2],
            response_labels=[-100, 5, 6],
            response_index=[0, 1, 2],
            attention_mask=_causal_mask(3),
        )
        args = _TrainingArgs()

        result = dpo_collate_fn(
            [[seq]],
            tokenizer=None,
            training_args=args,
            max_seq_len=None,
            use_filtered_label_loss=False,
        )

        # max_seq_len computed from the batch (3) -> no padding appended.
        np.testing.assert_array_equal(result["input_ids"][0], [10, 11, 12])
        np.testing.assert_array_equal(
            result["response_labels"][0], [-100, 5, 6]
        )
        self.assertEqual(result["input_ids"].shape, (1, 3))


class TestDpoCollatePackingNoCrossDocLeakage(unittest.TestCase):
    """Packing two docs into one row keeps attention block-diagonal."""

    def test_block_diagonal_mask_and_field_concatenation(self):
        seq1 = _seq(
            token_ids=[10, 11, 12],
            position_ids=[0, 1, 2],
            response_labels=[-100, -100, 7],
            response_index=[0, 1, 2],
            attention_mask=_causal_mask(3),
        )
        seq2 = _seq(
            token_ids=[20, 21],
            position_ids=[0, 1],
            response_labels=[-100, 9],
            response_index=[0, 1, 2],
            attention_mask=_causal_mask(2),
        )
        args = _TrainingArgs()

        result = dpo_collate_fn(
            [[seq1, seq2]],  # a single packed row containing two documents
            tokenizer=None,
            training_args=args,
            max_seq_len=6,
            use_filtered_label_loss=False,
        )

        # Fields are concatenated per document, positions reset per doc,
        # then padded (ids/pos with 0, labels with -100).
        np.testing.assert_array_equal(
            result["input_ids"][0], [10, 11, 12, 20, 21, 0]
        )
        np.testing.assert_array_equal(
            result["position_ids"][0], [0, 1, 2, 0, 1, 0]
        )
        np.testing.assert_array_equal(
            result["response_labels"][0], [-100, -100, 7, -100, 9, -100]
        )

        # Hand-derived block-diagonal causal mask, padded to 6x6 with 0.
        expected_mask = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(result["attention_mask"].shape, (1, 1, 6, 6))
        np.testing.assert_array_equal(
            result["attention_mask"][0, 0], expected_mask
        )

        # Explicit no-cross-document-leakage assertions: doc-2 rows (3,4) do
        # not attend doc-1 columns (0..2) and vice versa.
        mask = result["attention_mask"][0, 0]
        self.assertTrue(np.all(mask[3:5, 0:3] == 0))
        self.assertTrue(np.all(mask[0:3, 3:5] == 0))


class TestDpoCollateResponseIndexModes(unittest.TestCase):
    """Filtered vs unfiltered label-loss use different offset rules.

    response_index[2] (=2) is deliberately different from the first doc's
    token length (=3) so the two accumulation rules yield distinguishable
    results for the second document.
    """

    def _packed_batch(self):
        seq1 = _seq(
            token_ids=[10, 11, 12],  # length 3
            position_ids=[0, 1, 2],
            response_labels=[-100, -100, 7],
            response_index=[0, 1, 2],  # response_index[2] == 2 != len 3
            attention_mask=_causal_mask(3),
        )
        seq2 = _seq(
            token_ids=[20, 21],  # length 2
            position_ids=[0, 1],
            response_labels=[-100, 9],
            response_index=[0, 1, 2],
            attention_mask=_causal_mask(2),
        )
        return [[seq1, seq2]]

    def test_unfiltered_offsets_by_token_length(self):
        result = dpo_collate_fn(
            self._packed_batch(),
            tokenizer=None,
            training_args=_TrainingArgs(),
            max_seq_len=5,
            use_filtered_label_loss=False,
        )
        # doc-2 shifted by first doc token length (3): [0, 0+3, 1+3, 2+3].
        np.testing.assert_array_equal(
            result["response_indexs"], [[0, 0, 1, 2], [0, 3, 4, 5]]
        )

    def test_filtered_offsets_by_response_end_index(self):
        result = dpo_collate_fn(
            self._packed_batch(),
            tokenizer=None,
            training_args=_TrainingArgs(),
            max_seq_len=5,
            use_filtered_label_loss=True,
        )
        # doc-2 shifted by first doc response_index[2] (2): [0, 0+2, 1+2, 2+2].
        np.testing.assert_array_equal(
            result["response_indexs"], [[0, 0, 1, 2], [0, 2, 3, 4]]
        )


class TestDpoCollateStartendRowIndices(unittest.TestCase):
    """Sparse attn_mask_startend_row_indices path offsets per document."""

    def test_packed_indices_offset_and_tail_fill(self):
        seq1 = _seq(
            token_ids=[10, 11, 12],
            position_ids=[0, 1, 2],
            response_labels=[-100, -100, 7],
            response_index=[0, 1, 2],
            attn_mask_startend_row_indices=[3, 3, 3],
        )
        seq2 = _seq(
            token_ids=[20, 21],
            position_ids=[0, 1],
            response_labels=[-100, 9],
            response_index=[0, 1, 2],
            attn_mask_startend_row_indices=[2, 2],
        )

        result = dpo_collate_fn(
            [[seq1, seq2]],
            tokenizer=None,
            training_args=_TrainingArgs(),
            max_seq_len=6,
            use_filtered_label_loss=False,
        )

        # doc-1 indices [3,3,3] kept; doc-2 indices [2,2] shifted by 3 -> [5,5];
        # tail range(last=5, 6) -> [5] fills the padded row.
        self.assertEqual(
            result["attn_mask_startend_row_indices"].shape, (1, 1, 6, 1)
        )
        np.testing.assert_array_equal(
            result["attn_mask_startend_row_indices"][0, 0, :, 0],
            [3, 3, 3, 5, 5, 5],
        )
        self.assertNotIn("attention_mask", result)
        np.testing.assert_array_equal(
            result["input_ids"][0], [10, 11, 12, 20, 21, 0]
        )


class TestDpoCollateScoreDeltas(unittest.TestCase):
    """score_deltas collected once per sequence, in order."""

    def test_score_deltas_collected_per_sequence(self):
        seq1 = _seq(
            token_ids=[10, 11, 12],
            position_ids=[0, 1, 2],
            response_labels=[-100, -100, 7],
            response_index=[0, 1, 2],
            attention_mask=_causal_mask(3),
            score_delta=0.5,
        )
        seq2 = _seq(
            token_ids=[20, 21],
            position_ids=[0, 1],
            response_labels=[-100, 9],
            response_index=[0, 1, 2],
            attention_mask=_causal_mask(2),
            score_delta=1.5,
        )

        result = dpo_collate_fn(
            [[seq1], [seq2]],  # two rows, one document each
            tokenizer=None,
            training_args=_TrainingArgs(),
            max_seq_len=3,
            use_filtered_label_loss=False,
            use_response_score_delta=True,
        )

        np.testing.assert_allclose(result["score_deltas"], [0.5, 1.5])

    def test_score_deltas_absent_when_flag_off(self):
        seq = _seq(
            token_ids=[10, 11, 12],
            position_ids=[0, 1, 2],
            response_labels=[-100, -100, 7],
            response_index=[0, 1, 2],
            attention_mask=_causal_mask(3),
            score_delta=0.5,
        )
        result = dpo_collate_fn(
            [[seq]],
            tokenizer=None,
            training_args=_TrainingArgs(),
            max_seq_len=3,
            use_filtered_label_loss=False,
            use_response_score_delta=False,
        )
        self.assertNotIn("score_deltas", result)


class TestDpoCollatePaddingFree(unittest.TestCase):
    """padding_free concatenates all docs into one row with no padding."""

    def test_padding_free_concatenates_without_pad_tokens(self):
        seq1 = _seq(
            token_ids=[10, 11, 12],
            position_ids=[0, 1, 2],
            response_labels=[-100, -100, 7],
            response_index=[0, 1, 2],
            attention_mask=_causal_mask(3),
        )
        seq2 = _seq(
            token_ids=[20, 21],
            position_ids=[0, 1],
            response_labels=[-100, 9],
            response_index=[0, 1, 2],
            attention_mask=_causal_mask(2),
        )

        result = dpo_collate_fn(
            [[seq1], [seq2]],  # two rows collapsed into one by padding_free
            tokenizer=None,
            training_args=_TrainingArgs(),
            max_seq_len=None,
            padding_free=True,
            use_filtered_label_loss=False,
        )

        # Single concatenated row, no pad tokens appended (length == 5).
        self.assertEqual(result["input_ids"].shape, (1, 5))
        np.testing.assert_array_equal(
            result["input_ids"][0], [10, 11, 12, 20, 21]
        )
        np.testing.assert_array_equal(
            result["position_ids"][0], [0, 1, 2, 0, 1]
        )
        np.testing.assert_array_equal(
            result["response_labels"][0], [-100, -100, 7, -100, 9]
        )


class TestDpoCollateRequiresMask(unittest.TestCase):
    """Both attention_mask and startend indices None is a hard error."""

    def test_raises_when_no_mask_provided(self):
        seq = _seq(
            token_ids=[10, 11, 12],
            position_ids=[0, 1, 2],
            response_labels=[-100, -100, 7],
            response_index=[0, 1, 2],
            attention_mask=None,
            attn_mask_startend_row_indices=None,
        )
        with self.assertRaises(ValueError):
            dpo_collate_fn(
                [[seq]],
                tokenizer=None,
                training_args=_TrainingArgs(),
                max_seq_len=3,
                use_filtered_label_loss=False,
            )


if __name__ == "__main__":
    unittest.main()
