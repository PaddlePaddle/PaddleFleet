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

"""CPU-observable behavior tests for LanguageModelEmbedding.

The whole paddlefleet package imports paddle at import time. If paddle (or the
package) is not importable in this environment the tests are honestly skipped;
they are NOT reported as passing.

Design notes:
- ``tensor_parallel.VocabParallelEmbedding`` requires a real tensor-parallel
  process group, so it is the ONE collaborator we replace. We hand it a fixed,
  distinguishable marker tensor and then verify the REAL orchestration the layer
  performs on top of it: the position-embedding lookup (a real ``paddle.nn.Embedding``),
  the element-wise add, the token-type branch, and the fp32 residual cast.
- Expected values are derived independently in plain Python; the layer under
  test is never called to compute its own expected output.
"""

import unittest
from unittest.mock import MagicMock, patch

MODULE = "paddlefleet.models.common.embeddings.language_model_embedding"

try:
    import paddle

    from paddlefleet.models.common.embeddings.language_model_embedding import (
        LanguageModelEmbedding,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig

    IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    paddle = None
    LanguageModelEmbedding = None
    TransformerConfig = None
    IMPORT_ERROR = exc

_SKIP_REASON = (
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERROR!r}"
)

HIDDEN = 4
VOCAB = 10
MAX_SEQ = 6


def _make_config(**overrides):
    """Build a small real TransformerConfig on CPU with no real initialization."""
    defaults = {
        "hidden_size": HIDDEN,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "hidden_dropout_prob": 0.0,
        "perform_initialization": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _mock_embed_returning(marker):
    """A stand-in VocabParallelEmbedding instance that always returns marker."""
    embed = MagicMock()
    embed.return_value = marker
    return embed


def _flatten(x):
    if isinstance(x, list):
        out = []
        for item in x:
            out.extend(_flatten(item))
        return out
    return [x]


@unittest.skipUnless(IMPORT_ERROR is None, _SKIP_REASON)
class TestLanguageModelEmbeddingInit(unittest.TestCase):
    """add_position_embedding / reduce_scatter_embeddings derivation."""

    @patch(
        f"{MODULE}.get_tensor_model_parallel_group_if_none", return_value=None
    )
    @patch(f"{MODULE}.tensor_parallel")
    def _build(self, position_embedding_type, mock_tp, mock_group, **kw):
        mock_tp.VocabParallelEmbedding.return_value = MagicMock()
        return LanguageModelEmbedding(
            config=_make_config(),
            vocab_size=VOCAB,
            max_sequence_length=MAX_SEQ,
            position_embedding_type=position_embedding_type,
            **kw,
        )

    def test_add_position_embedding_only_for_learned_absolute(self):
        # Contract: add_position_embedding is True IFF type == "learned_absolute".
        self.assertTrue(self._build("learned_absolute").add_position_embedding)
        self.assertFalse(self._build("rope").add_position_embedding)
        self.assertFalse(self._build("none").add_position_embedding)

    def test_reduce_scatter_false_without_sequence_parallel(self):
        # tp_size<=1 forces sequence_parallel off, so reduce_scatter must be off
        # regardless of position type / tokentypes.
        self.assertFalse(self._build("rope").reduce_scatter_embeddings)
        self.assertFalse(
            self._build("learned_absolute").reduce_scatter_embeddings
        )

    def test_learned_absolute_creates_position_but_not_tokentype(self):
        emb = self._build("learned_absolute")
        self.assertIsNotNone(emb.position_embeddings)
        self.assertIsNone(emb.tokentype_embeddings)
        self.assertEqual(emb.tokentype_embeddings, None)

    def test_rope_has_no_position_embedding_attribute(self):
        emb = self._build("rope")
        self.assertFalse(hasattr(emb, "position_embeddings"))

    def test_sequence_parallel_requires_scatter(self):
        # Real __init__ assertion, raised before any tensor-parallel work.
        config = _make_config(
            tensor_model_parallel_size=2, sequence_parallel=True
        )
        with (
            patch(f"{MODULE}.tensor_parallel"),
            patch(
                f"{MODULE}.get_tensor_model_parallel_group_if_none",
                return_value=None,
            ),
            self.assertRaises(AssertionError),
        ):
            LanguageModelEmbedding(
                config=config,
                vocab_size=VOCAB,
                max_sequence_length=MAX_SEQ,
                scatter_to_sequence_parallel=False,
            )


@unittest.skipUnless(IMPORT_ERROR is None, _SKIP_REASON)
class TestLanguageModelEmbeddingForward(unittest.TestCase):
    """forward() combines word / position / tokentype embeddings correctly."""

    def assertClose(self, actual, expected, tol=1e-4):
        a = _flatten(actual)
        e = _flatten(expected)
        self.assertEqual(len(a), len(e))
        for x, y in zip(a, e):
            self.assertLessEqual(abs(float(x) - float(y)), tol)

    @patch(
        f"{MODULE}.get_tensor_model_parallel_group_if_none", return_value=None
    )
    @patch(f"{MODULE}.tensor_parallel")
    def test_learned_absolute_adds_position_lookup(self, mock_tp, mock_group):
        # Word embedding is stubbed with a fixed marker; the position embedding
        # is a REAL paddle.nn.Embedding whose weight we pin. Expected = marker +
        # position_weight[position_ids], derived independently in Python below.
        marker_vals = [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        ]
        marker = paddle.to_tensor(marker_vals, dtype="float32")
        mock_tp.VocabParallelEmbedding.return_value = _mock_embed_returning(
            marker
        )

        emb = LanguageModelEmbedding(
            config=_make_config(),
            vocab_size=VOCAB,
            max_sequence_length=MAX_SEQ,
            position_embedding_type="learned_absolute",
        )
        pos_w = [
            [100.0 + 10 * r + c for c in range(HIDDEN)] for r in range(MAX_SEQ)
        ]
        emb.position_embeddings.weight.set_value(
            paddle.to_tensor(pos_w, dtype="float32")
        )

        input_ids = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        pos_ids_list = [[0, 2, 5]]
        position_ids = paddle.to_tensor(pos_ids_list, dtype="int64")

        expected = [
            [
                [
                    marker_vals[0][s][c] + pos_w[pos_ids_list[0][s]][c]
                    for c in range(HIDDEN)
                ]
                for s in range(3)
            ]
        ]
        result = emb(input_ids, position_ids)
        self.assertEqual(result.shape, [1, 3, HIDDEN])
        self.assertClose(result.tolist(), expected)

    @patch(
        f"{MODULE}.get_tensor_model_parallel_group_if_none", return_value=None
    )
    @patch(f"{MODULE}.tensor_parallel")
    def test_rope_does_not_add_position(self, mock_tp, mock_group):
        # With rope, forward must return the word embedding UNCHANGED: no
        # position lookup, no add. A regression that added positions would
        # perturb these exact values.
        marker_vals = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
        marker = paddle.to_tensor(marker_vals, dtype="float32")
        mock_tp.VocabParallelEmbedding.return_value = _mock_embed_returning(
            marker
        )

        emb = LanguageModelEmbedding(
            config=_make_config(),
            vocab_size=VOCAB,
            max_sequence_length=MAX_SEQ,
            position_embedding_type="rope",
        )
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        position_ids = paddle.to_tensor([[3, 4]], dtype="int64")
        result = emb(input_ids, position_ids)
        self.assertClose(result.tolist(), marker_vals)

    @patch(
        f"{MODULE}.get_tensor_model_parallel_group_if_none", return_value=None
    )
    @patch(f"{MODULE}.tensor_parallel")
    def test_tokentype_embedding_is_added(self, mock_tp, mock_group):
        # rope isolates the tokentype branch: expected = marker +
        # tokentype_weight[tokentype_ids].
        marker_vals = [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        ]
        marker = paddle.to_tensor(marker_vals, dtype="float32")
        mock_tp.VocabParallelEmbedding.return_value = _mock_embed_returning(
            marker
        )

        emb = LanguageModelEmbedding(
            config=_make_config(),
            vocab_size=VOCAB,
            max_sequence_length=MAX_SEQ,
            position_embedding_type="rope",
            num_tokentypes=2,
        )
        tt_w = [
            [200.0 + c for c in range(HIDDEN)],
            [210.0 + c for c in range(HIDDEN)],
        ]
        emb.tokentype_embeddings.weight.set_value(
            paddle.to_tensor(tt_w, dtype="float32")
        )
        input_ids = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        position_ids = paddle.to_tensor([[0, 1, 2]], dtype="int64")
        tt_ids_list = [[0, 1, 0]]
        tokentype_ids = paddle.to_tensor(tt_ids_list, dtype="int64")

        expected = [
            [
                [
                    marker_vals[0][s][c] + tt_w[tt_ids_list[0][s]][c]
                    for c in range(HIDDEN)
                ]
                for s in range(3)
            ]
        ]
        result = emb(input_ids, position_ids, tokentype_ids=tokentype_ids)
        self.assertClose(result.tolist(), expected)

    @patch(
        f"{MODULE}.get_tensor_model_parallel_group_if_none", return_value=None
    )
    @patch(f"{MODULE}.tensor_parallel")
    def test_fp32_residual_connection_casts_output(self, mock_tp, mock_group):
        # A float64 word embedding must come out as float32 when
        # fp32_residual_connection is on, with values preserved.
        marker_vals = [[[1.5, 2.5], [3.5, 4.5]]]
        marker = paddle.to_tensor(marker_vals, dtype="float64")
        mock_tp.VocabParallelEmbedding.return_value = _mock_embed_returning(
            marker
        )

        emb = LanguageModelEmbedding(
            config=_make_config(hidden_size=2, fp32_residual_connection=True),
            vocab_size=VOCAB,
            max_sequence_length=MAX_SEQ,
            position_embedding_type="rope",
        )
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        position_ids = paddle.to_tensor([[0, 1]], dtype="int64")
        result = emb(input_ids, position_ids)
        self.assertEqual(result.dtype, paddle.float32)
        self.assertClose(result.tolist(), marker_vals)


@unittest.skipUnless(IMPORT_ERROR is None, _SKIP_REASON)
class TestLanguageModelEmbeddingZeroParameters(unittest.TestCase):
    """zero_parameters() unconditionally touches self.position_embeddings.

    REAL PRODUCTION BUG (not modified here):
    language_model_embedding.py:130 does
        self.position_embeddings.weight.data.fill_(0)
    with no guard on ``add_position_embedding``. For a rope/none embedding the
    ``position_embeddings`` attribute is never created (see __init__ lines
    94-97), so zero_parameters() raises AttributeError. The correct behavior is
    to skip the (absent) position embeddings. This test asserts that correct
    behavior and is marked expectedFailure to record the latent crash; if the
    guard is added upstream it will surface as an unexpected success.
    """

    @unittest.expectedFailure
    @patch(
        f"{MODULE}.get_tensor_model_parallel_group_if_none", return_value=None
    )
    @patch(f"{MODULE}.tensor_parallel")
    def test_zero_parameters_should_not_crash_for_rope(
        self, mock_tp, mock_group
    ):
        mock_tp.VocabParallelEmbedding.return_value = MagicMock()
        emb = LanguageModelEmbedding(
            config=_make_config(),
            vocab_size=VOCAB,
            max_sequence_length=MAX_SEQ,
            position_embedding_type="rope",
        )
        # Expected-correct behavior: no exception when there is no position
        # embedding. Current production code raises AttributeError here.
        emb.zero_parameters()


if __name__ == "__main__":
    unittest.main()
