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

"""Behavior tests for paddlefleet.nn.embedding.Embedding.

Scope note: this module is a *factory* that selects and constructs a token
embedding (``nn.Embedding`` for ``default``, ``VocabParallelEmbedding`` for
``vocab_parallel``). It contains NO positional/rotary embedding and NO
weight-tying logic of its own (tying lives in ``nn.lm_head.LMHead`` via
``tie_word_embeddings``); the factory only *forwards* ``weight_attr``, which is
the mechanism that lets a caller inject a shared/tied weight. Tests therefore
verify: real embedding row-lookup (content/identity/position mapping),
config-vs-explicit dimension resolution, kwargs filtering wired into the real
constructor, weight_attr forwarding, and input validation.

Environment: CPU-only (no accelerator). The ``vocab_parallel`` branch requires
a tensor-parallel process group / Fleet init, so its numerics are skipped here
with a reason; only its CPU-testable control logic (type selection, kwarg
filtering) is asserted.
"""

import unittest

import numpy as np
import paddle
import paddle.nn as nn

from paddlefleet.nn.embedding import Embedding
from paddlefleet.transformers.configuration_utils import PretrainedConfig


def _small_config(**overrides):
    """Build a real PretrainedConfig with small, CPU-friendly dims."""
    params = {"vocab_size": 16, "hidden_size": 8}
    params.update(overrides)
    return PretrainedConfig(**params)


class TestEmbeddingTypeSelection(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def test_default_type_when_no_tensor_parallel(self):
        # tp_size == 1 (and the <=1 boundary) must map to the plain embedding.
        self.assertEqual(
            Embedding.get_embedding_type(
                _small_config(tensor_model_parallel_size=1)
            ),
            "default",
        )

    def test_vocab_parallel_type_when_tensor_parallel(self):
        for tp in (2, 4, 8):
            cfg = _small_config(tensor_model_parallel_size=tp)
            self.assertEqual(
                Embedding.get_embedding_type(cfg),
                "vocab_parallel",
                msg=f"tp={tp} should select vocab_parallel",
            )


class TestEmbeddingProcessKwargs(unittest.TestCase):
    """process_kwargs must filter exactly the keys each backend rejects.

    Independent expected dicts are hand-written; the asymmetry between the two
    branches (default drops only mp_group; vocab_parallel drops padding_idx and
    sparse but *keeps* mp_group) is the real contract under test.
    """

    def test_default_strips_only_mp_group(self):
        kwargs = {
            "mp_group": "g",
            "padding_idx": 0,
            "sparse": True,
            "other": 42,
        }
        result = Embedding.process_kwargs("default", **kwargs)
        self.assertEqual(
            result, {"padding_idx": 0, "sparse": True, "other": 42}
        )

    def test_vocab_parallel_strips_padding_and_sparse_keeps_mp_group(self):
        kwargs = {
            "mp_group": "g",
            "padding_idx": 0,
            "sparse": True,
            "other": 42,
        }
        result = Embedding.process_kwargs("vocab_parallel", **kwargs)
        self.assertEqual(result, {"mp_group": "g", "other": 42})

    def test_missing_keys_are_noop(self):
        # pop(..., None) must not fail and must leave unrelated keys intact.
        self.assertEqual(
            Embedding.process_kwargs("vocab_parallel", other=42), {"other": 42}
        )
        self.assertEqual(
            Embedding.process_kwargs("default", other=42), {"other": 42}
        )


class TestEmbeddingCreate(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def test_default_resolves_dims_from_config(self):
        cfg = _small_config(vocab_size=16, hidden_size=8)
        emb = Embedding.create(cfg)
        self.assertIsInstance(emb, nn.Embedding)
        self.assertEqual(emb.weight.shape, [16, 8])

    def test_explicit_dims_override_config(self):
        cfg = _small_config(vocab_size=16, hidden_size=8)
        emb = Embedding.create(cfg, num_embeddings=5, embedding_dim=3)
        self.assertIsInstance(emb, nn.Embedding)
        self.assertEqual(emb.weight.shape, [5, 3])

    def test_lookup_is_exact_row_gather(self):
        # Real forward: each (batch, position) id must gather its exact weight
        # row, preserving order and mapping repeated ids to identical rows.
        cfg = _small_config(vocab_size=6, hidden_size=4)
        emb = Embedding.create(cfg)
        known = np.arange(6 * 4, dtype="float32").reshape([6, 4])
        emb.weight.set_value(paddle.to_tensor(known))

        ids = np.array([[2, 0, 5], [3, 3, 1]], dtype="int64")
        out = emb(paddle.to_tensor(ids)).numpy()

        expected = known[ids]  # independent gather via numpy indexing
        np.testing.assert_array_equal(out, expected)
        # repeated id -> identical row (identity/position mapping intact)
        np.testing.assert_array_equal(out[1, 0], out[1, 1])
        np.testing.assert_array_equal(out[1, 0], known[3])

    def test_weight_attr_is_forwarded_for_shared_weights(self):
        # The factory forwards weight_attr; this is what lets a caller inject a
        # shared/tied weight. Assign a known matrix and confirm the built
        # embedding actually uses it (both stored weight and lookup output).
        cfg = _small_config(vocab_size=5, hidden_size=3)
        shared = np.arange(5 * 3, dtype="float32").reshape([5, 3]) + 100.0
        attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Assign(shared)
        )
        emb = Embedding.create(cfg, weight_attr=attr)
        np.testing.assert_array_equal(emb.weight.numpy(), shared)

        ids = np.array([[4, 1, 0]], dtype="int64")
        out = emb(paddle.to_tensor(ids)).numpy()
        np.testing.assert_array_equal(out, shared[ids])

    def test_mp_group_stripped_so_default_constructor_accepts_it(self):
        # Behavioral proof that process_kwargs is wired into create: nn.Embedding
        # has no mp_group parameter, so if create failed to strip it the call
        # would raise TypeError. Success here means the kwarg was removed.
        cfg = _small_config(vocab_size=6, hidden_size=4)
        emb = Embedding.create(cfg, mp_group="unused_for_default")
        self.assertIsInstance(emb, nn.Embedding)
        self.assertEqual(emb.weight.shape, [6, 4])

    def test_padding_idx_forwarded_to_default_embedding(self):
        # padding_idx is retained for the default backend and consumed by
        # nn.Embedding, which zero-initializes that row.
        cfg = _small_config(vocab_size=6, hidden_size=4)
        emb = Embedding.create(cfg, padding_idx=0)
        np.testing.assert_array_equal(
            emb.weight.numpy()[0], np.zeros(4, dtype="float32")
        )

    def test_missing_vocab_size_and_num_embeddings_raises(self):
        cfg = _small_config(vocab_size=None)
        with self.assertRaisesRegex(ValueError, "num_embeddings"):
            Embedding.create(cfg, num_embeddings=None)

    def test_missing_hidden_size_and_embedding_dim_raises(self):
        # vocab present so validation reaches the embedding_dim check.
        cfg = _small_config(vocab_size=16, hidden_size=None)
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            Embedding.create(cfg, embedding_dim=None)


class TestEmbeddingVocabParallel(unittest.TestCase):
    def test_vocab_parallel_numerics_require_multicard(self):
        # get_embedding_type/process_kwargs for vocab_parallel are covered on CPU
        # above. Constructing and running VocabParallelEmbedding needs a real
        # tensor-parallel process group (Fleet init) and multiple ranks to
        # verify vocab sharding + all-reduce; not available in a CPU no-card run.
        self.skipTest(
            "VocabParallelEmbedding forward/sharding numerics require a "
            "multi-card tensor-parallel process group (Fleet init)."
        )


if __name__ == "__main__":
    unittest.main()
