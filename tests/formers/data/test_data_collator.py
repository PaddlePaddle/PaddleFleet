# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import shutil
import tempfile
import unittest

import numpy as np
import paddle

from paddlefleet.data import default_data_collator
from paddlefleet.data.data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    _numpy_collate_batch,
    _paddle_collate_batch,
    numpy_default_data_collator,
    paddle_default_data_collator,
)


class DataCollatorIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_default_with_dict(self):
        features = [
            {"label": i, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)
        ]
        batch = default_data_collator(features)

        self.assertTrue(
            batch["labels"].equal_all(paddle.to_tensor(list(range(8))))
        )
        self.assertEqual(batch["labels"].dtype, paddle.int64)
        self.assertEqual(batch["inputs"].shape, [8, 6])

        # With label_ids
        features = [
            {"label_ids": [0, 1, 2], "inputs": [0, 1, 2, 3, 4, 5]}
            for i in range(8)
        ]
        batch = default_data_collator(features)
        self.assertTrue(
            batch["labels"].equal_all(paddle.to_tensor([[0, 1, 2]] * 8))
        )
        self.assertEqual(batch["labels"].dtype, paddle.int64)
        self.assertEqual(batch["inputs"].shape, [8, 6])

        # Features can already be tensors
        features = [
            {"label": i, "inputs": np.random.randint(0, 10, [10])}
            for i in range(8)
        ]
        batch = default_data_collator(features)
        self.assertTrue(
            batch["labels"].equal_all(paddle.to_tensor(list(range(8))))
        )
        self.assertEqual(batch["labels"].dtype, paddle.int64)
        self.assertEqual(batch["inputs"].shape, [8, 10])

        # Labels can already be tensors
        features = [
            {
                "label": paddle.to_tensor(i),
                "inputs": np.random.randint(0, 10, [10]),
            }
            for i in range(8)
        ]

        batch = default_data_collator(features)
        self.assertEqual(batch["labels"].dtype, paddle.int64)
        self.assertTrue(
            batch["labels"].equal_all(paddle.to_tensor(list(range(8))))
        )
        self.assertEqual(batch["labels"].dtype, paddle.int64)
        self.assertEqual(batch["inputs"].shape, [8, 10])

    def test_default_classification_and_regression(self):
        data_collator = default_data_collator

        features = [
            {"input_ids": [0, 1, 2, 3, 4], "label": i} for i in range(4)
        ]
        batch = data_collator(features)
        self.assertEqual(batch["labels"].dtype, paddle.int64)

        features = [
            {"input_ids": [0, 1, 2, 3, 4], "label": float(i)} for i in range(4)
        ]
        batch = data_collator(features)
        self.assertEqual(batch["labels"].dtype, paddle.float32)

    def test_default_with_no_labels(self):
        features = [
            {"label": None, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)
        ]
        batch = default_data_collator(features)
        self.assertTrue("labels" not in batch)
        self.assertEqual(batch["inputs"].shape, [8, 6])

        # With label_ids
        features = [
            {"label_ids": None, "inputs": [0, 1, 2, 3, 4, 5]} for i in range(8)
        ]
        batch = default_data_collator(features)
        self.assertTrue("labels" not in batch)
        self.assertEqual(batch["inputs"].shape, [8, 6])


class _DeterministicPadTokenizer:
    """Minimal deterministic tokenizer stub.

    Isolates the (not-under-test) tokenizer while letting the collator's real
    label / padding logic run. It never returns input-independent tensors: it
    preserves the original ``input_ids`` and appends/prepends ``pad_token_id``
    on the configured side, so the resulting batch content stays fully
    hand-checkable and distinguishable per row.
    """

    def __init__(self, pad_token_id=0, padding_side="right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        # ``_paddle_collate_batch`` / ``_numpy_collate_batch`` gate padding on a
        # non-None ``_pad_token``; a plain string is enough for that check.
        self._pad_token = "[PAD]"

    def pad(
        self,
        features,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_attention_mask=None,
    ):
        seqs = [list(f["input_ids"]) for f in features]
        max_len = max(len(s) for s in seqs)
        if pad_to_multiple_of is not None and max_len % pad_to_multiple_of != 0:
            max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of
        padded = []
        for s in seqs:
            gap = [self.pad_token_id] * (max_len - len(s))
            padded.append(s + gap if self.padding_side == "right" else gap + s)
        return {"input_ids": padded}


class DefaultDataCollatorContentTest(unittest.TestCase):
    """Collation must preserve per-sample content and sample order.

    The existing integration tests assert shapes/dtypes and a single repeated
    value; these use content-distinguishable rows so a row reordering or
    cross-sample mixing regression (which keeps the same shape) is caught.
    """

    def test_paddle_preserves_row_content_and_order(self):
        features = [
            {"input_ids": [10, 11, 12], "label": 5},
            {"input_ids": [20, 21, 22], "label": 6},
            {"input_ids": [30, 31, 32], "label": 7},
        ]
        batch = paddle_default_data_collator(features)
        self.assertEqual(
            batch["input_ids"].tolist(),
            [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
        )
        # Label i must stay attached to sample i, not merely be present.
        self.assertEqual(batch["labels"].tolist(), [5, 6, 7])

    def test_paddle_label_ids_row_content(self):
        features = [
            {"input_ids": [1, 2], "label_ids": [9, 8, 7]},
            {"input_ids": [3, 4], "label_ids": [6, 5, 4]},
        ]
        batch = paddle_default_data_collator(features)
        self.assertEqual(batch["labels"].tolist(), [[9, 8, 7], [6, 5, 4]])
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, 4]])

    def test_numpy_preserves_row_content_and_order(self):
        features = [
            {"input_ids": [10, 11, 12], "label": 5},
            {"input_ids": [20, 21, 22], "label": 6},
        ]
        batch = numpy_default_data_collator(features)
        np.testing.assert_array_equal(
            batch["input_ids"], np.array([[10, 11, 12], [20, 21, 22]])
        )
        np.testing.assert_array_equal(batch["labels"], np.array([5, 6]))


class CollateBatchPaddingContentTest(unittest.TestCase):
    """Padding must keep real tokens intact and place pad ids on the right side.

    Content is checked in full (not just shape) for both padding sides, so a
    wrong pad value, wrong padding side, or a corrupted/overwritten real token
    is rejected.
    """

    def test_paddle_right_padding_content(self):
        tok = _DeterministicPadTokenizer(pad_token_id=0, padding_side="right")
        examples = [[10, 11, 12], [20, 21]]
        result = _paddle_collate_batch(examples, tok)
        self.assertEqual(result.tolist(), [[10, 11, 12], [20, 21, 0]])

    def test_paddle_left_padding_content(self):
        tok = _DeterministicPadTokenizer(pad_token_id=0, padding_side="left")
        examples = [[10, 11, 12], [20, 21]]
        result = _paddle_collate_batch(examples, tok)
        self.assertEqual(result.tolist(), [[10, 11, 12], [0, 20, 21]])

    def test_paddle_nonzero_pad_id_only_in_pad_slots(self):
        # A non-zero pad id must appear only where a real token is absent.
        tok = _DeterministicPadTokenizer(pad_token_id=99, padding_side="right")
        examples = [[1, 2, 3, 4], [5, 6]]
        result = _paddle_collate_batch(examples, tok)
        self.assertEqual(result.tolist(), [[1, 2, 3, 4], [5, 6, 99, 99]])

    def test_numpy_right_padding_content(self):
        tok = _DeterministicPadTokenizer(pad_token_id=0, padding_side="right")
        examples = [[10, 11, 12], [20, 21]]
        result = _numpy_collate_batch(examples, tok)
        np.testing.assert_array_equal(
            result, np.array([[10, 11, 12], [20, 21, 0]])
        )

    def test_numpy_left_padding_content(self):
        tok = _DeterministicPadTokenizer(pad_token_id=0, padding_side="left")
        examples = [[10, 11, 12], [20, 21]]
        result = _numpy_collate_batch(examples, tok)
        np.testing.assert_array_equal(
            result, np.array([[10, 11, 12], [0, 20, 21]])
        )


class TokenClassificationSupervisionTest(unittest.TestCase):
    """Label padding must mirror token padding and stay out of supervision.

    ``DataCollatorForTokenClassification.paddle_call`` must pad labels with
    ``label_pad_token_id`` at exactly the positions where ``input_ids`` are
    padded, on both padding sides, so padded slots never carry a supervised
    target and labels stay length-synced with tokens. The tokenizer is a
    deterministic stub (isolated, not under test); the label-masking logic that
    IS under test runs for real.
    """

    def test_right_padding_labels_masked_and_synced(self):
        tok = _DeterministicPadTokenizer(pad_token_id=0, padding_side="right")
        collator = DataCollatorForTokenClassification(
            tokenizer=tok, label_pad_token_id=-100
        )
        features = [
            {"input_ids": [11, 12, 13], "labels": [1, 2, 3]},
            {"input_ids": [21, 22], "labels": [7, 8]},
        ]
        batch = collator.paddle_call(features)
        # Shorter row right-padded with pad id; real tokens preserved.
        self.assertEqual(
            batch["input_ids"].tolist(), [[11, 12, 13], [21, 22, 0]]
        )
        # Real targets preserved; the padded slot carries the ignore index.
        self.assertEqual(batch["labels"].tolist(), [[1, 2, 3], [7, 8, -100]])
        # Padded token slot and masked label slot are the same position.
        self.assertEqual(batch["input_ids"].tolist()[1][2], 0)
        self.assertEqual(batch["labels"].tolist()[1][2], -100)
        # Every label row is length-synced with the padded token sequence.
        self.assertEqual(
            len(batch["labels"].tolist()[1]),
            len(batch["input_ids"].tolist()[1]),
        )

    def test_left_padding_labels_masked_and_synced(self):
        tok = _DeterministicPadTokenizer(pad_token_id=0, padding_side="left")
        collator = DataCollatorForTokenClassification(
            tokenizer=tok, label_pad_token_id=-100
        )
        features = [
            {"input_ids": [11, 12, 13], "labels": [1, 2, 3]},
            {"input_ids": [21, 22], "labels": [7, 8]},
        ]
        batch = collator.paddle_call(features)
        self.assertEqual(
            batch["input_ids"].tolist(), [[11, 12, 13], [0, 21, 22]]
        )
        self.assertEqual(batch["labels"].tolist(), [[1, 2, 3], [-100, 7, 8]])
        # Left side: the leading slot is padded for both tokens and labels.
        self.assertEqual(batch["input_ids"].tolist()[1][0], 0)
        self.assertEqual(batch["labels"].tolist()[1][0], -100)

    def test_custom_label_pad_token_id(self):
        # A non-default ignore index is what must land in pad slots, and it
        # must not collide with any real supervised target value.
        tok = _DeterministicPadTokenizer(pad_token_id=0, padding_side="right")
        collator = DataCollatorForTokenClassification(
            tokenizer=tok, label_pad_token_id=-1
        )
        features = [
            {"input_ids": [5, 6, 7, 8], "labels": [3, 4, 5, 6]},
            {"input_ids": [9], "labels": [2]},
        ]
        batch = collator.paddle_call(features)
        self.assertEqual(
            batch["labels"].tolist(), [[3, 4, 5, 6], [2, -1, -1, -1]]
        )
        self.assertEqual(batch["labels"].tolist()[1][1:], [-1, -1, -1])


class LanguageModelingSupervisionTest(unittest.TestCase):
    """With MLM disabled, labels equal input_ids except padding is ignored.

    ``DataCollatorForLanguageModeling.paddle_call`` (mlm=False) must copy
    ``input_ids`` into ``labels`` and replace padded positions with -100, so
    padding never contributes to the causal-LM supervision signal while every
    real token is supervised as itself.
    """

    def test_mlm_false_masks_padding_in_labels(self):
        tok = _DeterministicPadTokenizer(pad_token_id=0, padding_side="right")
        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
        # Distinct, non-zero tokens so no real token collides with the pad id.
        examples = [[5, 6, 7], [8, 9]]
        batch = collator.paddle_call(examples)
        # Shorter sequence right-padded with pad id 0.
        self.assertEqual(batch["input_ids"].tolist(), [[5, 6, 7], [8, 9, 0]])
        # Labels mirror tokens, but the padded slot is ignored (-100).
        self.assertEqual(batch["labels"].tolist(), [[5, 6, 7], [8, 9, -100]])
        # Unpadded row: every label equals its own input token.
        self.assertEqual(
            batch["labels"].tolist()[0], batch["input_ids"].tolist()[0]
        )
        # The only masked label lines up with the only padded token slot.
        self.assertEqual(batch["input_ids"].tolist()[1][2], 0)
        self.assertEqual(batch["labels"].tolist()[1][2], -100)

    def test_mlm_false_no_padding_keeps_all_supervised(self):
        tok = _DeterministicPadTokenizer(pad_token_id=0, padding_side="right")
        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
        examples = [[3, 4, 5], [6, 7, 8]]  # equal length -> no padding
        batch = collator.paddle_call(examples)
        self.assertEqual(batch["input_ids"].tolist(), [[3, 4, 5], [6, 7, 8]])
        # No pad slots exist, so no label is masked to -100.
        self.assertEqual(batch["labels"].tolist(), [[3, 4, 5], [6, 7, 8]])


if __name__ == "__main__":
    unittest.main()
