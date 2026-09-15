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

"""Behavior tests for paddlefleet.nn.lm_head.LMHead.

Environment: 无卡 (CPU, single process). With tensor_model_parallel_size == 1
and sequence_parallel == False, LMHead.forward runs the real production path
LMHead.forward -> calc_lm_head_logits -> parallel_matmul, which on CPU reduces
to `paddle.matmul(hidden, weight, transpose_y=True)` (+ bias). This lets us
verify the projection direction, weight tying and bias consumption numerically
against an independent numpy reference. Vocab tensor-parallel and
sequence-parallel gather numerics require a real Fleet process group and are
explicitly skipped (see test_vocab_parallel_logits_numerics).
"""

import unittest

import numpy as np
import paddle

from paddlefleet.nn.lm_head import LMHead
from paddlefleet.transformers import LlamaConfig


def _small_config(**overrides):
    """Build a small *real* LlamaConfig so the production init/forward run."""
    kwargs = dict(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
    )
    kwargs.update(overrides)
    return LlamaConfig(**kwargs)


class TestLMHeadInit(unittest.TestCase):
    def test_init_default_no_bias(self):
        config = _small_config()
        head = LMHead(config)

        # weight is [vocab_size, hidden_size]; single card => not distributed.
        self.assertEqual(
            list(head.weight.shape), [config.vocab_size, config.hidden_size]
        )
        self.assertIsNone(head.bias)
        self.assertFalse(head.vocab_parallel)
        self.assertFalse(head.weight.is_distributed)

    def test_init_with_bias_is_zero_initialized(self):
        config = _small_config()
        config.lm_head_bias = True
        head = LMHead(config)

        self.assertIsNotNone(head.bias)
        self.assertEqual(list(head.bias.shape), [config.vocab_size])
        # Bias uses a Constant(0.0) initializer: verify the actual content,
        # not merely that a parameter exists.
        np.testing.assert_array_equal(
            head.bias.numpy(), np.zeros([config.vocab_size], dtype="float32")
        )

    def test_init_rejects_indivisible_vocab_for_tp(self):
        # vocab_size (7) not divisible by tp_size (2) must raise, otherwise
        # vocab sharding would silently drop/duplicate rows.
        config = _small_config(vocab_size=7)
        config.tensor_model_parallel_size = 2
        with self.assertRaises(ValueError):
            LMHead(config)


class TestLMHeadForward(unittest.TestCase):
    def test_forward_logits_match_manual_projection(self):
        config = _small_config()
        head = LMHead(config)
        vocab, hidden = config.vocab_size, config.hidden_size

        # Distinguishable (non-degenerate) weight and hidden states.
        weight = (
            np.arange(vocab * hidden, dtype="float32").reshape(vocab, hidden)
            * 0.01
            - 0.3
        )
        head.weight.set_value(paddle.to_tensor(weight))
        hs = (
            np.arange(2 * 3 * hidden, dtype="float32").reshape(2, 3, hidden)
            * 0.05
            - 0.5
        )

        logits = head(paddle.to_tensor(hs))

        # Independent reference: logits[b,s,v] = sum_h hidden[b,s,h]*weight[v,h].
        # This pins the transpose_y=True direction; a wrong axis would fail
        # even though the shape would still be [2, 3, vocab].
        expected = np.einsum("bsh,vh->bsv", hs, weight)
        self.assertEqual(list(logits.shape), [2, 3, vocab])
        np.testing.assert_allclose(
            logits.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_bias_is_added_and_broadcast(self):
        config = _small_config()
        config.lm_head_bias = True
        head = LMHead(config)
        vocab, hidden = config.vocab_size, config.hidden_size

        weight = (
            np.arange(vocab * hidden, dtype="float32").reshape(vocab, hidden)
            * 0.02
            - 0.1
        )
        bias = np.linspace(-1.0, 1.0, vocab).astype("float32")
        head.weight.set_value(paddle.to_tensor(weight))
        head.bias.set_value(paddle.to_tensor(bias))
        hs = (
            np.arange(2 * 3 * hidden, dtype="float32").reshape(2, 3, hidden)
            * 0.03
            - 0.4
        )

        logits = head(paddle.to_tensor(hs))

        # bias must be consumed and broadcast over batch/seq.
        expected = np.einsum("bsh,vh->bsv", hs, weight) + bias
        np.testing.assert_allclose(
            logits.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_weight_tying_produces_gram_logits(self):
        # Tie lm_head to a token embedding, then feed the embedding rows back
        # as hidden states. With shared weights logits[i, v] is the dot product
        # embedding[id_i] . embedding[v], and the diagonal (v == id_i) is the
        # squared L2 norm of that embedding row -- a property that only holds
        # when the classifier weight truly equals the embedding table.
        config = _small_config()
        head = LMHead(config)
        vocab, hidden = config.vocab_size, config.hidden_size

        emb_w = (
            np.arange(vocab * hidden, dtype="float32").reshape(vocab, hidden)
            * 0.1
            - 0.2
        )
        embedding = paddle.nn.Embedding(vocab, hidden)
        embedding.weight.set_value(paddle.to_tensor(emb_w))

        # Apply the tie and confirm it took effect.
        head.weight.set_value(embedding.weight)
        np.testing.assert_array_equal(head.weight.numpy(), emb_w)

        ids = [0, 3, 5]
        hidden_states = embedding(paddle.to_tensor([ids]))  # [1, 3, hidden]
        logits = head(hidden_states)  # [1, 3, vocab]

        expected = np.einsum("bsh,vh->bsv", hidden_states.numpy(), emb_w)
        np.testing.assert_allclose(
            logits.numpy(), expected, rtol=1e-5, atol=1e-5
        )

        # Diagonal identity from tying: own-token logit == ||embedding row||^2.
        for i, token in enumerate(ids):
            self.assertAlmostEqual(
                float(logits[0, i, token]),
                float((emb_w[token] ** 2).sum()),
                places=4,
            )

    def test_fused_head_returns_raw_weight_and_hidden(self):
        # With use_fused_head_and_loss_fn the head must NOT project; it forwards
        # (hidden_states, weight, bias, True) so a downstream fused kernel does
        # the matmul + loss. Verify identity of the passed-through objects.
        config = _small_config()
        config.lm_head_bias = True
        config.use_fused_head_and_loss_fn = True
        head = LMHead(config)

        hs = paddle.to_tensor(
            np.arange(1 * 2 * config.hidden_size, dtype="float32").reshape(
                1, 2, config.hidden_size
            )
        )
        result = head(hs)

        self.assertEqual(len(result), 4)
        hidden_out, weight_out, bias_out, flag = result
        self.assertIs(hidden_out, hs)  # returned unchanged, no projection
        self.assertIs(weight_out, head.weight)
        self.assertIs(bias_out, head.bias)
        self.assertIs(flag, True)


class TestLMHeadRepr(unittest.TestCase):
    def test_extra_repr_reports_shape_and_parallel(self):
        config = _small_config()
        head = LMHead(config)
        text = head.extra_repr()
        self.assertIn(f"hidden_size={config.hidden_size}", text)
        self.assertIn(f"vocab_size={config.vocab_size}", text)
        self.assertIn("vocab_parallel=False", text)


class TestLMHeadDistributed(unittest.TestCase):
    @unittest.skip(
        "Vocab tensor-parallel logits (tensor_model_parallel_size > 1) and "
        "sequence-parallel gather run the fleet _c_identity/_c_concat path in "
        "parallel_matmul. Their per-rank sharding and cross-rank concat "
        "numerics require a real Fleet process group / multi-card run and "
        "cannot be validated on CPU in a single process."
    )
    def test_vocab_parallel_logits_numerics(self):
        pass


if __name__ == "__main__":
    unittest.main()
