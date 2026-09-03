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
"""Coverage for the ``use_accuracy_compatible`` router branches added in
commit 80a72f9: MG-aligned seq aux-loss and gather_nd top-k weight lookup."""

import unittest
from unittest.mock import patch

import numpy as np
import paddle


def _mg_seq_aux_loss_ref(probs, routing_map, top_k, num_experts, batch_size):
    """Independent reference mirroring Megatron-LM.

    ``switch_load_balancing_loss_func`` (moe_utils.py) computes
    ``sum(agg_probs_per_expert * tokens_per_expert) * E / (topk * T^2)`` and
    ``_apply_seq_aux_loss`` (router.py) divides it by ``bsz``, where
    ``get_tokens_per_expert_and_token_count`` sets
    ``T = tokens_per_expert.sum() / (topk * bsz)`` (the valid routing token
    count per line) once a padding mask is in play.
    """
    p = probs.numpy().reshape([batch_size, -1, num_experts]).astype("float64")
    rm = (
        routing_map.numpy()
        .reshape([batch_size, -1, num_experts])
        .astype("float64")
    )
    tokens_per_expert = rm.sum(axis=1)  # [B, E]
    aggregated = p.sum(axis=1)  # [B, E]
    total_num_tokens = tokens_per_expert.sum() / (top_k * batch_size)
    loss = (aggregated * tokens_per_expert).sum() * (
        num_experts / (top_k * total_num_tokens * total_num_tokens)
    )
    return loss / batch_size


def _make_router_config(**overrides):
    from paddlefleet.transformer.transformer_config import TransformerConfig

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "topk_method": "greedy",
        "norm_topk_prob": True,
        "scoring_func": "softmax",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "routed_scaling_factor_learnable": False,
        "moe_router_force_load_balancing": False,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_deep_gemm": False,
        "router_aux_loss_coef": 0.01,
        "router_z_loss_coef": None,
        "moe_n_hash_layers": 0,
        "use_accuracy_compatible": True,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


_CP_PATCH = (
    "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size"
)


class TestRouterAccuracyCompatible(unittest.TestCase):
    @patch(_CP_PATCH, return_value=1)
    def test_use_accuracy_compatible_flag_set(self, _cp):
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        router = StandardMoERouter(_make_router_config())
        self.assertTrue(router.use_accuracy_compatible)

    @patch(_CP_PATCH, return_value=1)
    def test_seq_aux_loss_scalar_and_matches_default(self, _cp):
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        B, S, E, K = 2, 3, 4, 2
        paddle.seed(0)
        probs = paddle.nn.functional.softmax(paddle.randn([B * S, E]), axis=-1)
        idx = paddle.topk(probs, k=K, axis=-1).indices
        routing_map = paddle.zeros([B * S, E]).put_along_axis_(
            idx, paddle.to_tensor(1.0), axis=-1
        )

        aligned = StandardMoERouter(_make_router_config())
        default = StandardMoERouter(
            _make_router_config(use_accuracy_compatible=False)
        )
        loss_a = aligned._cal_seq_aux_loss(probs, K, routing_map, S, B)
        loss_d = default._cal_seq_aux_loss(probs, K, routing_map, S, B)
        self.assertEqual(loss_a.shape, [])
        # The aligned branch only reorders float ops; the value must match the
        # reference (batch-mean) formula of the default branch.
        np.testing.assert_allclose(
            loss_a.numpy(), loss_d.numpy(), rtol=1e-5, atol=1e-6
        )

    @patch(_CP_PATCH, return_value=1)
    def test_seq_aux_loss_is_batch_mean(self, _cp):
        """Duplicating a sequence across the batch must not change the loss."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        S, E, K = 3, 4, 2
        paddle.seed(3)
        probs1 = paddle.nn.functional.softmax(paddle.randn([S, E]), axis=-1)
        idx1 = paddle.topk(probs1, k=K, axis=-1).indices
        rm1 = paddle.zeros([S, E]).put_along_axis_(
            idx1, paddle.to_tensor(1.0), axis=-1
        )

        router = StandardMoERouter(_make_router_config())
        loss_b1 = router._cal_seq_aux_loss(probs1, K, rm1, S, 1)
        loss_b2 = router._cal_seq_aux_loss(
            paddle.concat([probs1, probs1], axis=0),
            K,
            paddle.concat([rm1, rm1], axis=0),
            S,
            2,
        )
        np.testing.assert_allclose(
            loss_b2.numpy(), loss_b1.numpy(), rtol=1e-6, atol=1e-6
        )

    @patch(_CP_PATCH, return_value=1)
    def test_seq_aux_loss_padding_uses_valid_token_count(self, _cp):
        """With padding rows the denominator must be the valid routing token
        count per line (MG semantics), not the fixed sequence length."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        B, S, E, K = 3, 5, 4, 2
        valid_lens = [5, 3, 1]
        paddle.seed(5)
        probs = paddle.nn.functional.softmax(paddle.randn([B, S, E]), axis=-1)
        idx = paddle.topk(probs, k=K, axis=-1).indices
        rm = paddle.zeros([B, S, E]).put_along_axis_(
            idx, paddle.to_tensor(1.0), axis=-1
        )
        ids = paddle.zeros([B, S], dtype="int64")
        for b, n in enumerate(valid_lens):
            ids[b, :n] = 1
            if n < S:
                probs[b, n:] = 0.0
                rm[b, n:] = 0.0

        router = StandardMoERouter(_make_router_config())
        loss = router._cal_seq_aux_loss(
            probs.reshape([B * S, E]),
            K,
            rm.reshape([B * S, E]),
            S,
            B,
            input_ids=ids,
        )

        expected = _mg_seq_aux_loss_ref(
            probs.reshape([B * S, E]), rm.reshape([B * S, E]), K, E, B
        )
        np.testing.assert_allclose(
            loss.numpy(), np.float32(expected), rtol=1e-5, atol=1e-7
        )
        # Guard against normalizing by the fixed sequence length: the mean
        # valid length is 3, so an S=5 normalization is off by (3/5)^2.
        mean_valid = sum(valid_lens) / B
        seq_len_normalized = expected * (mean_valid / S) ** 2
        self.assertFalse(
            np.allclose(
                loss.numpy(),
                np.float32(seq_len_normalized),
                rtol=1e-3,
                atol=1e-8,
            )
        )

    @patch(_CP_PATCH, return_value=1)
    def test_seq_aux_loss_matches_mg_reference_without_padding(self, _cp):
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        B, S, E, K = 2, 4, 4, 2
        paddle.seed(6)
        probs = paddle.nn.functional.softmax(paddle.randn([B * S, E]), axis=-1)
        idx = paddle.topk(probs, k=K, axis=-1).indices
        rm = paddle.zeros([B * S, E]).put_along_axis_(
            idx, paddle.to_tensor(1.0), axis=-1
        )

        router = StandardMoERouter(_make_router_config())
        loss = router._cal_seq_aux_loss(probs, K, rm, S, B)
        expected = _mg_seq_aux_loss_ref(probs, rm, K, E, B)
        np.testing.assert_allclose(
            loss.numpy(), np.float32(expected), rtol=1e-5, atol=1e-7
        )

    @patch(_CP_PATCH, return_value=1)
    def test_seq_aux_loss_matches_default_router(self, _cp):
        """The recomputed routing_map means the aligned loss is a well-defined
        scalar; the default router path must also produce a finite scalar."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        B, S, E, K = 2, 4, 4, 2
        paddle.seed(1)
        probs = paddle.nn.functional.softmax(paddle.randn([B * S, E]), axis=-1)
        idx = paddle.topk(probs, k=K, axis=-1).indices
        routing_map = paddle.zeros([B * S, E]).put_along_axis_(
            idx, paddle.to_tensor(1.0), axis=-1
        )

        aligned = StandardMoERouter(_make_router_config())
        default = StandardMoERouter(
            _make_router_config(use_accuracy_compatible=False)
        )
        loss_a = aligned._cal_seq_aux_loss(probs, K, routing_map, S, B)
        loss_d = default._cal_seq_aux_loss(probs, K, routing_map, S, B)
        self.assertEqual(loss_a.shape, [])
        self.assertEqual(loss_d.shape, [])
        self.assertTrue(bool(paddle.isfinite(loss_a).all().numpy()))
        self.assertTrue(bool(paddle.isfinite(loss_d).all().numpy()))

    @patch(_CP_PATCH, return_value=1)
    def test_topk_noaux_tc_gather_nd_matches_default(self, _cp):
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        E, K = 4, 2
        paddle.seed(2)
        scores = paddle.nn.functional.softmax(paddle.randn([5, E]), axis=-1)

        aligned = StandardMoERouter(_make_router_config(topk_method="noaux_tc"))
        default = StandardMoERouter(
            _make_router_config(
                topk_method="noaux_tc", use_accuracy_compatible=False
            )
        )
        # Force identical correction bias so routing decisions match.
        default.e_score_correction_bias.set_value(
            aligned.e_score_correction_bias
        )

        tw_a, ti_a = aligned._topk_noaux_tc(
            scores, k=K, n_group=1, topk_group=1
        )
        tw_d, ti_d = default._topk_noaux_tc(
            scores, k=K, n_group=1, topk_group=1
        )
        self.assertEqual(tw_a.shape, [5, K])
        np.testing.assert_array_equal(ti_a.numpy(), ti_d.numpy())
        np.testing.assert_allclose(
            tw_a.numpy(), tw_d.numpy(), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
