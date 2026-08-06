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

"""Parity between generate()'s full-prefill and prefill+KV-cache decode paths.

``GreedyGenerator.generate`` has two ways to produce the same autoregressive
sequence:

* ``no_cache=True`` -> :meth:`_generate_no_cache`, which re-runs a *full
  prefill* over ``prompt + all-tokens-so-far`` at every step; and
* the default KV-cache path, which prefills once and then feeds a single new
  token per step through the incremental-decode branch (the code the HySparse
  ``_is_incremental_decode`` / ``_hy_sparse_decode_block_indices`` /
  ``DynamicKVCache`` machinery serves).

Both feed the last-token logits into the identical ``_sample_next_token``, so a
greedy (``temperature=0``) run of one must reproduce the other token for token,
and the per-step ``return_log_probs`` must match up to bf16 recompute noise. A
divergence means the incremental-decode path (KV read-back, position ids, cache
bookkeeping) disagrees with the ground-truth full recompute.

This exercises a real ``GPTModel`` end to end, so unlike
``test_hysparse_decode_unit.py`` it needs a CUDA device (standard
``DotProductAttention``; no SM10.x/FlashMask/DSA required) and is *skipped on
CPU-only CI runners* -- which the reviewer explicitly accepted for this case.
"""

import functools
import unittest

import paddle
from paddle.distributed import fleet

from paddlefleet.generation.greedy_generator import GreedyGenerator
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig


def _fleet_init():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    fleet.init(is_collective=True, strategy=strategy)


if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
    _fleet_init()


def _make_model(
    vocab_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 2,
    max_seq_len: int = 32,
):
    """Build a tiny GPTModel suitable for inference tests."""
    paddle.manual_seed(0)
    config = GPTConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        rotary_base=10000,
        vocab_size=vocab_size,
        rotary_percent=1.0,
        rope_scaling=1.0,
        position_embedding_type="rope",
        num_attention_heads=4,
        intermediate_size=hidden_size * 2,
        max_sequence_length=max_seq_len,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        tie_word_embeddings=True,
    )
    return gpt_builder(config, num_stages=1), config


@unittest.skipUnless(
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
    "generate() runs the real GPTModel forward, which needs a CUDA device",
)
class TestGenerateCacheParity(unittest.TestCase):
    """Full-prefill (no_cache) vs prefill+incremental-decode must agree."""

    @classmethod
    def setUpClass(cls):
        cls.vocab_size = 64
        cls.model, cls.config = _make_model(vocab_size=cls.vocab_size)
        cls.gen = GreedyGenerator(cls.model)
        cls.input_ids = paddle.to_tensor([[1, 5, 10, 3]], dtype="int64")
        cls.max_new_tokens = 8

    def _greedy(self, no_cache: bool, return_log_probs: bool = False):
        return self.gen.generate(
            self.input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=None,
            temperature=0.0,
            top_k=0,
            top_p=0.0,
            return_log_probs=return_log_probs,
            no_cache=no_cache,
        )

    def test_greedy_tokens_match(self):
        """Greedy tokens from both paths must be identical."""
        full = self._greedy(no_cache=True)
        cached = self._greedy(no_cache=False)
        self.assertEqual(full.tolist(), cached.tolist())

    def test_prompt_preserved_in_both_paths(self):
        prompt_len = self.input_ids.shape[1]
        full = self._greedy(no_cache=True)
        cached = self._greedy(no_cache=False)
        self.assertEqual(full[:, :prompt_len].tolist(), self.input_ids.tolist())
        self.assertEqual(
            cached[:, :prompt_len].tolist(), self.input_ids.tolist()
        )

    def test_log_probs_match(self):
        """Per-step log-probs agree up to bf16 recompute noise.

        The full-prefill path recomputes attention over the whole sequence
        every step; the cache path reads stored KV and attends incrementally.
        Same math, so the chosen-token log-probs should track closely -- the
        only gap is bf16 reduction-order noise, measured at 0.0 here with the
        CI determinism flags. ``1e-2`` keeps a safety margin for other GPUs
        while still being tight enough to flag a real decode-path regression.
        """
        full_tokens, full_lp = self._greedy(
            no_cache=True, return_log_probs=True
        )
        cached_tokens, cached_lp = self._greedy(
            no_cache=False, return_log_probs=True
        )
        # Tokens must match exactly before comparing their log-probs.
        self.assertEqual(full_tokens.tolist(), cached_tokens.tolist())
        self.assertEqual(len(full_lp), len(cached_lp))
        for b in range(len(full_lp)):
            self.assertEqual(len(full_lp[b]), len(cached_lp[b]))
            for step, (a, c) in enumerate(zip(full_lp[b], cached_lp[b])):
                self.assertAlmostEqual(
                    a,
                    c,
                    delta=1e-2,
                    msg=f"log-prob mismatch batch={b} step={step}",
                )


if __name__ == "__main__":
    unittest.main()
