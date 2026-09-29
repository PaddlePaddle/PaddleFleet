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
"""Behavior tests for ``LanguageModelEmbedding``.

These exercise CPU/GPU-observable behavior of
``paddlefleet.models.common.embeddings.language_model_embedding.LanguageModelEmbedding``
with hand-derived expected values:

  * ``add_position_embedding`` / ``reduce_scatter_embeddings`` are derived from
    the constructor inputs by an independent truth table.
  * ``embedding_weight`` returns the *same* parameter object as ``embed_tokens``.
  * ``forward`` adds the position embedding for ``learned_absolute`` and omits it
    for ``rope`` (verified against the sub-module outputs, which validates the
    composition/branching in ``forward`` -- not the underlying embedding kernel).
  * ``zero_parameters`` on a ``rope`` embedding hits a real production bug
    (see ``test_zero_parameters_rope_should_succeed_but_hits_production_bug``).

The whole ``paddlefleet`` package imports ``paddle`` at import time. When paddle
is not installed the entire module is skipped with an honest reason rather than
faking a pass.
"""

import unittest

try:
    import numpy as np
    import paddle
    from paddle.distributed import fleet

    from paddlefleet.models.common.embeddings.language_model_embedding import (
        LanguageModelEmbedding,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.utils import init_method_normal

    PADDLE_AVAILABLE = True
    IMPORT_ERROR = ""
except (
    ImportError
) as exc:  # only real missing-dependency, not API/compile errors
    PADDLE_AVAILABLE = False
    IMPORT_ERROR = repr(exc)

SKIP_REASON = (
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERROR}"
)

HIDDEN_SIZE = 8
VOCAB_SIZE = 32
MAX_SEQ_LEN = 8


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestLanguageModelEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # VocabParallelEmbedding needs the model-parallel groups initialized.
        if not paddle.distributed.is_initialized():
            fleet.init(is_collective=True)

    def _make(
        self,
        position_embedding_type="learned_absolute",
        num_tokentypes=0,
        sequence_parallel=False,
        scatter_to_sequence_parallel=True,
    ):
        config = TransformerConfig(
            num_hidden_layers=2, hidden_size=HIDDEN_SIZE, num_attention_heads=4
        )
        config.params_dtype = "float32"
        config.perform_initialization = True
        config.embedding_init_method = init_method_normal(0.02)
        config.hidden_dropout_prob = 0.0
        config.fp32_residual_connection = False
        config.sequence_parallel = sequence_parallel
        config.clone_scatter_output_in_embedding = False
        return LanguageModelEmbedding(
            config=config,
            vocab_size=VOCAB_SIZE,
            max_sequence_length=MAX_SEQ_LEN,
            position_embedding_type=position_embedding_type,
            num_tokentypes=num_tokentypes,
            scatter_to_sequence_parallel=scatter_to_sequence_parallel,
        )

    # ------------------------------------------------------------------
    # Constructor-derived flags (independent truth table).
    # ------------------------------------------------------------------
    def test_add_position_embedding_flag(self):
        # add_position_embedding is True iff type == "learned_absolute".
        self.assertTrue(self._make("learned_absolute").add_position_embedding)
        self.assertFalse(self._make("rope").add_position_embedding)
        self.assertFalse(self._make("none").add_position_embedding)
        # Only "learned_absolute" allocates a position_embeddings sub-module.
        self.assertTrue(
            hasattr(self._make("learned_absolute"), "position_embeddings")
        )
        self.assertFalse(hasattr(self._make("rope"), "position_embeddings"))
        self.assertFalse(hasattr(self._make("none"), "position_embeddings"))

    def test_reduce_scatter_flag_truth_table(self):
        # reduce_scatter = (not add_position_embedding) and num_tokentypes <= 0
        #                   and sequence_parallel and scatter_to_sequence_parallel
        # rope, tt=0, sp=False -> False (sequence_parallel is False)
        self.assertFalse(
            self._make(
                "rope", 0, sequence_parallel=False
            ).reduce_scatter_embeddings
        )
        # rope, tt=0, sp=True, scatter=True -> True (all conditions met)
        self.assertTrue(
            self._make(
                "rope", 0, sequence_parallel=True
            ).reduce_scatter_embeddings
        )
        # learned_absolute, sp=True -> False (add_position_embedding is True)
        self.assertFalse(
            self._make(
                "learned_absolute", 0, sequence_parallel=True
            ).reduce_scatter_embeddings
        )
        # rope, tt=2, sp=True -> False (num_tokentypes > 0)
        self.assertFalse(
            self._make(
                "rope", 2, sequence_parallel=True
            ).reduce_scatter_embeddings
        )

    def test_tokentype_embeddings_presence_and_shape(self):
        self.assertIsNone(
            self._make(
                "learned_absolute", num_tokentypes=0
            ).tokentype_embeddings
        )
        emb = self._make("learned_absolute", num_tokentypes=3)
        self.assertIsNotNone(emb.tokentype_embeddings)
        # One row per token type, hidden_size columns.
        self.assertEqual(
            list(emb.tokentype_embeddings.weight.shape), [3, HIDDEN_SIZE]
        )

    def test_sequence_parallel_requires_scatter(self):
        # sequence_parallel with scatter_to_sequence_parallel=False must assert.
        with self.assertRaises(AssertionError):
            self._make(
                "learned_absolute",
                sequence_parallel=True,
                scatter_to_sequence_parallel=False,
            )

    # ------------------------------------------------------------------
    # embedding_weight identity.
    # ------------------------------------------------------------------
    def test_embedding_weight_is_embed_tokens_weight(self):
        emb = self._make("learned_absolute")
        # Same parameter object, not a copy.
        self.assertIs(emb.embedding_weight, emb.embed_tokens.weight)

    # ------------------------------------------------------------------
    # forward composition (branch on add_position_embedding).
    # ------------------------------------------------------------------
    def _fixed_ids(self):
        input_ids = paddle.to_tensor(
            [[1, 5, 3, 2], [9, 0, 4, 7]], dtype="int64"
        )
        position_ids = paddle.to_tensor(
            [[0, 1, 2, 3], [0, 1, 2, 3]], dtype="int64"
        )
        return input_ids, position_ids

    def test_forward_learned_absolute_adds_position(self):
        emb = self._make("learned_absolute")
        emb.eval()  # disable dropout (also configured with prob 0.0)
        input_ids, position_ids = self._fixed_ids()

        out = emb(input_ids, position_ids)
        # Reference: token lookup + position lookup, composed by hand.
        token = emb.embed_tokens(input_ids)
        position = emb.position_embeddings(position_ids)
        ref = token + position

        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-6, atol=1e-6
        )
        # Position must genuinely contribute: output differs from token-only.
        self.assertFalse(np.allclose(out.numpy(), token.numpy()))

    def test_forward_rope_omits_position(self):
        emb = self._make("rope")
        emb.eval()
        input_ids, position_ids = self._fixed_ids()

        out = emb(input_ids, position_ids)
        # rope: forward must return exactly the token embedding (no position add).
        token = emb.embed_tokens(input_ids)
        np.testing.assert_allclose(
            out.numpy(), token.numpy(), rtol=1e-6, atol=1e-6
        )

    # ------------------------------------------------------------------
    # zero_parameters.
    # ------------------------------------------------------------------
    def test_zero_parameters_zeros_token_and_position_weights(self):
        emb = self._make("learned_absolute")
        tok_w = emb.embedding_weight
        pos_w = emb.position_embeddings.weight
        # Initialized (normal) weights start non-zero.
        self.assertNotEqual(float(paddle.abs(tok_w).sum()), 0.0)
        self.assertNotEqual(float(paddle.abs(pos_w).sum()), 0.0)

        emb.zero_parameters()

        np.testing.assert_array_equal(
            tok_w.numpy(), np.zeros_like(tok_w.numpy())
        )
        np.testing.assert_array_equal(
            pos_w.numpy(), np.zeros_like(pos_w.numpy())
        )

    @unittest.expectedFailure
    def test_zero_parameters_rope_should_succeed_but_hits_production_bug(self):
        # CORRECT behavior: with rope there is no learned position embedding, so
        # zero_parameters() should zero embed_tokens and leave without error.
        #
        # REAL PRODUCTION BUG (language_model_embedding.py:130): zero_parameters
        # unconditionally does `self.position_embeddings.weight.data.fill_(0)`,
        # but position_embeddings is only created when add_position_embedding is
        # True. For rope/none this raises AttributeError. We assert the CORRECT
        # behavior and mark the test expectedFailure. Production code is NOT
        # modified.
        emb = self._make("rope")
        emb.zero_parameters()
        w = emb.embedding_weight
        np.testing.assert_array_equal(w.numpy(), np.zeros_like(w.numpy()))


if __name__ == "__main__":
    unittest.main()
