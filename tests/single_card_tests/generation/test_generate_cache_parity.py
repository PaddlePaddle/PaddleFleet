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

The parity assertions are shared by :class:`_CacheParityTests` and applied to
one model per attention flavour, because the two ``generate`` paths diverge for
different reasons in each:

* :class:`TestGenerateCacheParity` -- a tiny dense ``GPTModel`` with standard
  ``DotProductAttention``, exercising the plain full-attention KV cache.
* :class:`TestGenerateCacheParityHySparse` -- the HySparse MLA-absorbed MQA
  stack (sliding-window + block-sparse branches, two attention sinks), which is
  what the ``_is_incremental_decode`` / ``_hy_sparse_decode_block_indices``
  machinery actually exists for. Skipped unless the TileLang kernels can run.

This exercises a real ``GPTModel`` end to end, so unlike
``test_hysparse_decode_unit.py`` it needs a CUDA device and is *skipped on
CPU-only CI runners* -- which the reviewer explicitly accepted for this case.
"""

import functools
import unittest

import paddle
import paddle.nn.functional as F
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


# ---- HySparse MQA model ---------------------------------------------------

HYSPARSE_HIDDEN = 2048
HYSPARSE_HEADS = 64
HYSPARSE_WINDOW = 128
HYSPARSE_BLOCK = 64
HYSPARSE_TOPK = 16
# 24 key blocks > TOPK, so the block-sparse branch really selects a subset (a
# prompt short enough to fit in TOPK blocks would select everything and hide
# selection bugs in the decode path).
HYSPARSE_PROMPT_LEN = 1536


def _hysparse_available() -> bool:
    """Whether the TileLang HySparse kernels can run (needs SM 10.x)."""
    if (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.device_count() == 0
        or paddle.device.cuda.get_device_capability()[0] != 10
    ):
        return False
    try:
        from paddlefleet.tilelang_ops.hysparse import (  # noqa: F401
            sliding_window_mqa_attention,
        )
        from paddlefleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (  # noqa: F401
            block_sparse_mqa_attention_tl,
        )
    except (ImportError, RuntimeError):
        return False
    return True


def _make_hysparse_model(vocab_size: int = 256, num_layers: int = 2):
    """Build the HySparse MLA-absorbed MQA model as a full ``GPTModel``.

    The attention dims and HySparse flags mirror
    ``test_hysparse_decode_sink_parity.TestHySparseDecodeSinkParity._build_config``
    so this covers the same architecture end to end, with one deviation:
    ``num_key_value_heads`` must equal ``num_attention_heads`` because
    :class:`MultiLatentAttention` rejects GQA. That costs no coverage -- MLA
    absorbs KV into ``kv_lora_rank`` and hardcodes ``num_key_value_heads=1`` for
    ``core_attention``, so the config field is unused on this path.

    ``window_attn_skip_freq=2`` makes layer 0 full attention and layer 1 SWA,
    which is the pairing ``HySparseTransformerLayer`` requires: the full layer
    publishes ``shared_key`` / ``block_indices`` that the SWA layer consumes.
    ``gpt_builder`` picks ``HySparseTransformerLayer`` + ``MQASelfAttention``
    from ``multi_latent_attention`` + ``enable_hy_sparse_attention`` alone.
    """
    paddle.manual_seed(0)
    config = GPTConfig(
        num_hidden_layers=num_layers,
        vocab_size=vocab_size,
        max_sequence_length=HYSPARSE_PROMPT_LEN + 64,
        hidden_size=HYSPARSE_HIDDEN,
        intermediate_size=HYSPARSE_HIDDEN * 2,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        head_dim=192,
        num_attention_heads=HYSPARSE_HEADS,
        num_key_value_heads=HYSPARSE_HEADS,
        gated_attention=True,
        gated_attn_use_q_lora=True,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        qk_nope_head_dim=192,
        v_head_dim=256,
        kv_lora_rank=448,
        rope_theta=640000,
        use_qk_norm=True,
        multi_latent_attention=True,
        rope_type="rope",
        add_swa_attention_sink_bias=True,
        sliding_window=[HYSPARSE_WINDOW, HYSPARSE_WINDOW],
        swa_head_dim=192,
        swa_v_head_dim=256,
        swa_num_attention_heads=HYSPARSE_HEADS,
        swa_num_key_value_heads=HYSPARSE_HEADS,
        window_attn_skip_freq=2,
        enable_hy_sparse_attention=True,
        hy_sparse_full_attn_use_tilelang=True,
        hy_sparse_block_sparse_use_tilelang=True,
        hy_sparse_block_size=HYSPARSE_BLOCK,
        hy_sparse_topk=HYSPARSE_TOPK,
    )
    model = gpt_builder(config, num_stages=1)
    # The HySparse TileLang kernels take bf16 K/V directly and reject fp32
    # master weights, so the O2 cast has to be baked into the parameters
    # instead of relying on ``generate``'s ``auto_cast`` alone.
    model = paddle.amp.decorate(models=model, level="O2", dtype="bfloat16")
    return model, config


# ---- KDA (linear attention) model -----------------------------------------

KDA_HIDDEN = 64
KDA_HEAD_DIM = 16
KDA_HEADS = 4


def _make_kda_model(vocab_size: int = 64, num_layers: int = 2):
    """Build a tiny all-KDA ``GPTModel``.

    ``layer_types`` is the only selector for the attention flavour
    (``gpt_layer_specs.get_gpt_decoder_layers_spec``), so every layer is asked
    for ``kimi_delta_attention`` here -- a mixed stack would let a standard
    attention layer's KV cache carry the parity and hide a KDA state bug.

    The default configuration is intentional: when FLA is available,
    ``use_fused_kernels`` is true for the no-cache path, while cache calls use
    the implemented paddle-native recurrence.

    Unlike the HySparse model this needs no ``paddle.amp.decorate``: KDA already
    pins its state, conv weight, ``A_log``, ``dt_bias`` and out-norm to fp32 and
    upcasts the recurrence itself.
    """
    paddle.manual_seed(0)
    config = GPTConfig(
        num_hidden_layers=num_layers,
        hidden_size=KDA_HIDDEN,
        vocab_size=vocab_size,
        max_sequence_length=64,
        intermediate_size=KDA_HIDDEN * 2,
        num_attention_heads=KDA_HEADS,
        hidden_act=F.silu,
        normalization="RMSNorm",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        position_embedding_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        rope_scaling=1.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        layer_types=["kimi_delta_attention"] * num_layers,
        linear_conv_kernel_dim=4,
        # KDA requires key_head_dim == value_head_dim.
        linear_key_head_dim=KDA_HEAD_DIM,
        linear_value_head_dim=KDA_HEAD_DIM,
        linear_num_key_heads=KDA_HEADS,
        linear_num_value_heads=KDA_HEADS,
        linear_gate_lora_rank=KDA_HEAD_DIM,
        linear_use_full_rank_gate=True,
        linear_gate_lower_bound=-5.0,
    )
    return gpt_builder(config, num_stages=1), config


class _CacheParityTests:
    """Full-prefill (no_cache) vs prefill+incremental-decode must agree.

    Subclasses set up ``model`` / ``gen`` / ``input_ids`` for one attention
    flavour; ``log_prob_delta`` is the per-step log-prob tolerance.
    """

    max_new_tokens = 8
    log_prob_delta = 1e-2

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
        only gap is bf16 reduction-order noise. ``log_prob_delta`` keeps a
        safety margin for other GPUs while still being tight enough to flag a
        real decode-path regression.
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
                    delta=self.log_prob_delta,
                    msg=f"log-prob mismatch batch={b} step={step}",
                )


@unittest.skipUnless(
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
    "generate() runs the real GPTModel forward, which needs a CUDA device",
)
class TestGenerateCacheParity(_CacheParityTests, unittest.TestCase):
    """Tiny dense model with standard ``DotProductAttention``."""

    @classmethod
    def setUpClass(cls):
        cls.vocab_size = 64
        cls.model, cls.config = _make_model(vocab_size=cls.vocab_size)
        cls.gen = GreedyGenerator(cls.model)
        cls.input_ids = paddle.to_tensor([[1, 5, 10, 3]], dtype="int64")


@unittest.skipUnless(
    _hysparse_available(),
    "HySparse TileLang kernels require SM 10.x and an importable backend",
)
class TestGenerateCacheParityHySparse(_CacheParityTests, unittest.TestCase):
    """HySparse MLA-absorbed MQA stack (sliding-window + block-sparse + sinks).

    ``max_new_tokens`` is smaller than the dense case because every no-cache
    step re-prefills all ``HYSPARSE_PROMPT_LEN`` tokens through the TileLang
    kernels. ``log_prob_delta`` is looser because the two sink softmaxes and the
    score-dependent top-k block selection amplify bf16 noise: the max per-step
    deviation measured here is 1.4e-2, against 0.0 for the dense stack. 5e-2
    leaves margin for other GPUs while staying far below the gap a mis-read KV
    cache or a dropped sink would open.
    """

    max_new_tokens = 6
    log_prob_delta = 5e-2

    @classmethod
    def setUpClass(cls):
        cls.vocab_size = 256
        cls.model, cls.config = _make_hysparse_model(vocab_size=cls.vocab_size)
        cls.gen = GreedyGenerator(cls.model)
        paddle.seed(7)
        cls.input_ids = paddle.randint(
            0, cls.vocab_size, [1, HYSPARSE_PROMPT_LEN], dtype="int64"
        )


@unittest.skipUnless(
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
    "generate() runs the real GPTModel forward, which needs a CUDA device",
)
class TestGenerateCacheParityKDA(_CacheParityTests, unittest.TestCase):
    """Tiny all-KDA (linear attention) model.

    The KDA decode path summarises the prefix into a fixed-size recurrent state
    plus a short conv sliding window (no growing K/V cache). This validates that
    the cached single-step decode reproduces the full-sequence recompute end to
    end through ``GreedyGenerator.generate()``.

    The recurrence itself is fp32, but ``generate`` wraps both paths in
    ``paddle.amp.auto_cast(level="O2", dtype="bfloat16")``, so the surrounding
    MLP / RMSNorm / logit projection run in bf16 in both paths. The full-prefill
    path recomputes them from scratch every step while the cache path
    accumulates incrementally, and the two orderings round differently: measured
    max per-step deviation here is 1.4e-2, against 0.0 for the dense stack.
    ``log_prob_delta = 3e-2`` leaves margin for other GPUs while staying far
    below the gap a mis-read recurrent state would open. Tokens must still match
    exactly, which is the real cache-consistency signal.
    """

    log_prob_delta = 3e-2

    @classmethod
    def setUpClass(cls):
        cls.vocab_size = 64
        cls.model, cls.config = _make_kda_model(vocab_size=cls.vocab_size)
        cls.gen = GreedyGenerator(cls.model)
        cls.input_ids = paddle.to_tensor([[4, 14, 25, 24]], dtype="int64")


if __name__ == "__main__":
    unittest.main()
