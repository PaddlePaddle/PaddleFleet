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

"""Behavior tests for DeepSeekProjection MFU / FLOPs / param projection.

Module under test: paddlefleet.transformers.deepseek_v3.mfu_utils

The projection class is pure integer/float arithmetic (no Paddle, no device),
so these are CPU-only ("无卡") tests. Following the configuration & run
infrastructure module rules, parameter-count and FLOPs projections must be
pinned to exact independently hand-derived values rather than asserted `> 0`
(asserting positivity is an explicit antipattern: it passes even when whole
components -- embedding, experts, gate, attention recompute -- are dropped).

Every expected value below is computed by hand from a tiny config and written
as a literal, so production is never used to generate its own reference.
"""

import unittest
from types import SimpleNamespace

from paddlefleet.transformers.deepseek_v3.mfu_utils import DeepSeekProjection


# ---------------------------------------------------------------------------
# Tiny, hand-computable config. Every dimension is small and distinct so an
# error in any single term changes the exact totals asserted below.
#
#   vocab_size (V)              = 100
#   seq_length (S)              = 8
#   hidden_size (D)             = 16
#   intermediate_size (I)       = 32   dense-layer FFN hidden
#   moe_intermediate_size (M)   = 12   expert FFN hidden
#   num_hidden_layers (L)       = 3
#   first_k_dense_replace (Ld)  = 1    dense layers before MoE layers
#   num_attention_heads (H)     = 2
#   qk_nope_head_dim (qn)       = 4    (== v_head_dim in this projection)
#   q_lora_rank (ql)            = 6
#   kv_lora_rank (kvl)          = 5
#   qk_rope_head_dim (qr)       = 3
#   n_shared_experts (Es)       = 1
#   n_routed_experts (Er)       = 4
#   num_experts_per_tok (topk)  = 2
# ---------------------------------------------------------------------------
def _cfg(**overrides):
    base = {
        "vocab_size": 100,
        "seq_length": 8,
        "hidden_size": 16,
        "intermediate_size": 32,
        "moe_intermediate_size": 12,
        "num_hidden_layers": 3,
        "first_k_dense_replace": 1,
        "num_attention_heads": 2,
        "qk_nope_head_dim": 4,
        "q_lora_rank": 6,
        "kv_lora_rank": 5,
        "qk_rope_head_dim": 3,
        "n_shared_experts": 1,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _train_options(causal_mask=True, fused_atten=True):
    return SimpleNamespace(causal_mask=causal_mask, fused_atten=fused_atten)


class TestDeepSeekProjectionInit(unittest.TestCase):
    """The constructor must unpack the right config fields into internals."""

    def test_config_fields_mapped_to_internals(self):
        proj = DeepSeekProjection(_cfg())
        # Field-by-field: catches any swapped / mis-assigned unpacking.
        self.assertEqual(proj._vocab_size, 100)
        self.assertEqual(proj._max_seq_len, 8)  # sourced from seq_length
        self.assertEqual(proj._dim, 16)
        self.assertEqual(proj._intermediate_size, 32)
        self.assertEqual(proj._moe_intermediate_size, 12)
        self.assertEqual(proj._n_layers, 3)
        self.assertEqual(proj._n_dense_layers, 1)  # from first_k_dense_replace
        self.assertEqual(proj._n_heads, 2)
        self.assertEqual(proj._qk_nope_head_dim, 4)
        self.assertEqual(proj._q_lora_rank, 6)
        self.assertEqual(proj._kv_lora_rank, 5)
        self.assertEqual(proj._qk_rope_head_dim, 3)
        self.assertEqual(proj._n_experts_shared, 1)
        self.assertEqual(proj._n_experts_routed, 4)
        self.assertEqual(proj._router_top_k, 2)  # from num_experts_per_tok

    def test_defaults_when_no_train_options(self):
        proj = DeepSeekProjection(_cfg(), train_options=None)
        self.assertTrue(proj._causal_mask)
        self.assertTrue(proj._fused_atten)

    def test_train_options_propagate(self):
        proj = DeepSeekProjection(
            _cfg(), _train_options(causal_mask=False, fused_atten=False)
        )
        self.assertFalse(proj._causal_mask)
        self.assertFalse(proj._fused_atten)

    def test_train_options_mixed(self):
        proj = DeepSeekProjection(
            _cfg(), _train_options(causal_mask=True, fused_atten=False)
        )
        self.assertTrue(proj._causal_mask)
        self.assertFalse(proj._fused_atten)


class TestDeepSeekProjectionParams(unittest.TestCase):
    """Exact parameter counts, hand-derived from the tiny config.

    Per-layer components (base config, q_lora_rank=6):
      Attention params = 516
        proj_q  = D*ql + ql*H*qn + ql*H*qr = 96 + 48 + 36 = 180
        down_kv = D*kvl                     = 80
        up_k    = kvl*H*qn                  = 40
        rope_k  = D*qr                      = 48
        up_v    = kvl*H*qn                  = 40
        o_proj  = H*qn*D                    = 128
      norm       = 2*D + q_lora_rank        = 32 + 6 = 38
      ffn_dense  = D*I*3                     = 1536
      ffn (MoE)  = D*M*3 * (Er+Es)           = 576 * 5 = 2880
      ffn_active = D*M*3 * (Es+topk)         = 576 * 3 = 1728
      gate       = D*Er                      = 64
      embedding  = V*D                       = 1600
      final_norm = D                         = 16

    total     = 1600 + 1*(516+38+1536) + 2*(516+38+2880+64) + 16 = 10702
    activated = 1600 + 1*(516+38+1536) + 2*(516+38+1728+64) + 16 =  8398
    """

    def test_params_with_embedding_exact(self):
        proj = DeepSeekProjection(_cfg())
        num_params, num_activated = proj.get_num_params(include_embedding=True)
        self.assertEqual(num_params, 10702)
        self.assertEqual(num_activated, 8398)
        # Activated (top-k + shared experts) must be strictly below the full
        # sparse count; equality would mean the routing sparsity was ignored.
        self.assertLess(num_activated, num_params)

    def test_params_without_embedding_exact(self):
        proj = DeepSeekProjection(_cfg())
        num_params, num_activated = proj.get_num_params(include_embedding=False)
        self.assertEqual(num_params, 9102)
        self.assertEqual(num_activated, 6798)

    def test_embedding_delta_is_vocab_times_hidden(self):
        # Independent invariant: the only difference between the two calls is
        # the word-token-embedding term V*D. Guards against embedding being
        # dropped or double counted in either total.
        proj = DeepSeekProjection(_cfg())
        with_emb, act_with = proj.get_num_params(include_embedding=True)
        no_emb, act_no = proj.get_num_params(include_embedding=False)
        self.assertEqual(with_emb - no_emb, 100 * 16)
        self.assertEqual(act_with - act_no, 100 * 16)

    def test_params_q_lora_rank_none_exact(self):
        # No Q low-rank: proj_q = D*H*(qn+qr) = 16*2*7 = 224 -> atten = 560,
        # and the extra latent RMSNorm uses kv_lora_rank -> norm = 32+5 = 37.
        #   total     = 1600 + (560+37+1536) + 2*(560+37+2880+64) + 16 = 10831
        #   activated = 1600 + (560+37+1536) + 2*(560+37+1728+64) + 16 =  8527
        proj = DeepSeekProjection(_cfg(q_lora_rank=None))
        num_params, num_activated = proj.get_num_params(include_embedding=True)
        self.assertEqual(num_params, 10831)
        self.assertEqual(num_activated, 8527)

    def test_params_single_expert_disables_moe_branch(self):
        # n_routed=1, n_shared=0 -> n_experts == 1, so the MoE branch is not
        # taken: no gate, FFN not replicated, activated == full count.
        #   total = 1600 + (516+38+1536) + 2*(516+38+576) + 16 = 5966
        proj = DeepSeekProjection(_cfg(n_routed_experts=1, n_shared_experts=0))
        num_params, num_activated = proj.get_num_params(include_embedding=True)
        self.assertEqual(num_params, 5966)
        self.assertEqual(num_activated, 5966)
        self.assertEqual(num_params, num_activated)


class TestDeepSeekProjectionFwdFlops(unittest.TestCase):
    """Exact forward FLOPs, hand-derived from the tiny config (batch_size=1).

    Per attention block (factor 2 = MAC -> FLOPs):
      proj_q  = 2*S*D*ql + 2*S*ql*qn*H + 2*S*ql*qr*H = 1536+768+576 = 2880
      proj_k  = 2*S*D*kvl + 2*S*kvl*qn*H + 2*S*D*qr  = 1280+640+768 = 2688
      proj_v  = 2*S*qn*H*D                            = 2048
      sdpa    = 4*S^2*D // 2 (causal)                 = 4096 // 2 = 2048
      o_proj  = 2*S*D^2                               = 4096
      atten_fwd = 2880+2688+2048 + 2048 + 4096        = 13760

    FFN per layer:
      moe_ffn   = (2*S*D*M)*3 * (Es+topk) + gate(2*S*D*Er) = 27648 + 1024 = 28672
      dense_ffn = (2*S*D*I)*3                               = 24576
    logits      = 2*S*D*V                                   = 25600

    fwd = 1*(13760+24576) + 2*(13760+28672) + 25600 = 148800
    """

    def test_fwd_flops_exact(self):
        proj = DeepSeekProjection(_cfg())
        self.assertEqual(proj.get_num_flop_fwd(batch_size=1), 148800)

    def test_fwd_flops_scale_linearly_with_batch(self):
        proj = DeepSeekProjection(_cfg())
        f1 = proj.get_num_flop_fwd(batch_size=1)
        self.assertEqual(proj.get_num_flop_fwd(batch_size=2), 2 * f1)
        self.assertEqual(proj.get_num_flop_fwd(batch_size=4), 4 * f1)
        self.assertEqual(proj.get_num_flop_fwd(batch_size=2), 297600)

    def test_fwd_flops_q_lora_rank_none_exact(self):
        # proj_q collapses to 2*S*D*H*(qn+qr) = 3584 -> atten_fwd = 14464.
        #   fwd = 1*(14464+24576) + 2*(14464+28672) + 25600 = 150912
        proj = DeepSeekProjection(_cfg(q_lora_rank=None))
        self.assertEqual(proj.get_num_flop_fwd(batch_size=1), 150912)

    def test_causal_mask_halves_sdpa(self):
        # With causal_mask=False the SDPA term is NOT floor-divided by 2, so
        # each of the 3 attention blocks gains 2048 FLOPs -> +6144 total.
        proj = DeepSeekProjection(_cfg(), _train_options(causal_mask=False))
        self.assertEqual(proj.get_num_flop_fwd(batch_size=1), 148800 + 6144)

    def test_single_expert_disables_moe_flop_branch(self):
        # n_experts == 1 -> no expert replication, no gate FLOPs.
        #   moe_ffn collapses to plain (2*S*D*M)*3 = 9216 (no *3 experts, no gate)
        #   fwd = 1*(13760+24576) + 2*(13760+9216) + 25600 = 109888
        proj = DeepSeekProjection(_cfg(n_routed_experts=1, n_shared_experts=0))
        self.assertEqual(proj.get_num_flop_fwd(batch_size=1), 109888)


class TestDeepSeekProjectionQKAndBwdFlops(unittest.TestCase):
    """QK-recompute and backward FLOPs."""

    def test_qk_fwd_flops_exact(self):
        # num_flop_qk = L * (2*S^2*D) // 2 (causal) = 3*2048 // 2 = 3072
        proj = DeepSeekProjection(_cfg())
        self.assertEqual(proj._get_num_flop_QK_fwd(batch_size=1), 3072)

    def test_qk_fwd_flops_causal_off_not_halved(self):
        proj = DeepSeekProjection(_cfg(), _train_options(causal_mask=False))
        self.assertEqual(proj._get_num_flop_QK_fwd(batch_size=1), 6144)

    def test_bwd_flops_fused_adds_qk_recompute(self):
        # Flash-attention path: bwd = 2*fwd + QK-recompute.
        #   2*148800 + 3072 = 300672
        proj = DeepSeekProjection(_cfg(), _train_options(fused_atten=True))
        self.assertEqual(proj.get_num_flop_bwd(batch_size=1), 300672)

    def test_bwd_flops_no_fused_is_twice_fwd(self):
        proj = DeepSeekProjection(_cfg(), _train_options(fused_atten=False))
        self.assertEqual(proj.get_num_flop_bwd(batch_size=1), 297600)

    def test_fused_minus_nofused_equals_qk_recompute(self):
        # Independent delta: the ONLY difference between the two backward
        # variants is the recomputed QK^T forward FLOPs (belongs to backward,
        # never to forward). Guards against the recompute term leaking or
        # being dropped.
        fused = DeepSeekProjection(_cfg(), _train_options(fused_atten=True))
        plain = DeepSeekProjection(_cfg(), _train_options(fused_atten=False))
        delta = fused.get_num_flop_bwd(batch_size=1) - plain.get_num_flop_bwd(
            batch_size=1
        )
        self.assertEqual(delta, fused._get_num_flop_QK_fwd(batch_size=1))
        self.assertEqual(delta, 3072)


class TestDeepSeekProjectionFlopPerToken(unittest.TestCase):
    """Per-token FLOPs uses the fwd + bwd(=2*fwd) = 3*fwd simplification."""

    def test_flop_per_token_exact(self):
        # fwd(1) / seq_len * 3 = 148800 / 8 * 3 = 55800.0
        proj = DeepSeekProjection(_cfg())
        self.assertEqual(proj.get_num_flop_per_token(), 55800.0)


if __name__ == "__main__":
    unittest.main()
