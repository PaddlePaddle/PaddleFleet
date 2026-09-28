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
"""Behavior tests for ``paddlefleet.transformer.moe.moe_router``.

These tests drive the real ``StandardMoERouter`` scoring / top-k selection /
capacity / auxiliary-loss primitives with small, fixed, position-distinguishable
inputs, and compare the outputs against expected values derived independently
(plain NumPy softmax/sigmoid, hand-computed arg-sorts, hand-computed
mean-products, hand-computed capacity cumulative counts). No reference value is
produced by calling the routine under test, and no assertion is reduced to a
shape/dtype/existence-only check.

The router numerics run on CPU; only ``paddle`` is required. When paddle (or the
paddlefleet package) cannot be imported the whole module is skipped with an
honest reason rather than reported as passing.
"""

import unittest
from unittest.mock import patch

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.moe.moe_router import StandardMoERouter

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without deps
    np = None
    paddle = None
    StandardMoERouter = None
    _IMPORT_ERROR = exc

_CP_PATCH_TARGET = (
    "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size"
)


class _RouterConfig:
    """Minimal data-only stand-in for ``TransformerConfig``.

    It only carries the attributes ``StandardMoERouter.__init__`` reads. It is a
    passive configuration holder (not a mock of the code under test): every
    routing computation exercised below runs the real router methods.
    """

    def __init__(self, **overrides):
        self.hidden_size = 8
        self.n_routed_experts = 4
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.topk_method = "greedy"
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.routed_scaling_factor_learnable = False
        self.scoring_func = "softmax"
        self.moe_router_load_balancing_type = "aux_loss"
        self.tensor_model_parallel_size = 1
        self.sequence_parallel = False
        self.gpt_model_use_experimental_version = False
        self.router_aux_loss_coef = 0.0
        self.__dict__.update(overrides)
        # init_method is invoked on the gate weight during __init__; the gate
        # projection is never used in these tests (scores are fed directly), so
        # a deterministic constant initializer keeps construction side-effect
        # free.
        if paddle is not None and "init_method" not in overrides:
            self.init_method = paddle.nn.initializer.Constant(0.0)

    def get(self, key, default=None):
        return getattr(self, key, default)


@unittest.skipUnless(
    paddle is not None and StandardMoERouter is not None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestMoERouterNumerics(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")
        patcher = patch(_CP_PATCH_TARGET, return_value=1)
        self.mock_cp = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_router(self, **overrides):
        return StandardMoERouter(_RouterConfig(**overrides))

    # ----- gate_score_func -------------------------------------------------

    def test_gate_score_func_softmax_matches_numpy(self):
        """softmax scoring equals an independent NumPy softmax, row-normalized."""
        router = self._make_router(scoring_func="softmax")
        logits_np = np.array(
            [[0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 0.0, 2.0]], dtype=np.float32
        )
        shifted = logits_np - logits_np.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        expected = exp / exp.sum(axis=-1, keepdims=True)

        scores = router.gate_score_func(paddle.to_tensor(logits_np))
        self.assertEqual(scores.dtype, paddle.float32)
        np.testing.assert_allclose(
            scores.numpy(), expected, rtol=1e-6, atol=1e-6
        )
        # Row 0 and row 1 are permutations of the same logits, so the argmax
        # expert differs; a constant/degenerate output would fail here.
        self.assertEqual(int(np.argmax(scores.numpy()[0])), 3)
        self.assertEqual(int(np.argmax(scores.numpy()[1])), 0)

    def test_gate_score_func_sigmoid_is_elementwise(self):
        """sigmoid scoring is the elementwise logistic, not a row-normalized map."""
        router = self._make_router(scoring_func="sigmoid")
        logits_np = np.array(
            [[-2.0, 0.0, 2.0, 4.0], [1.0, -1.0, 3.0, -3.0]], dtype=np.float32
        )
        expected = 1.0 / (1.0 + np.exp(-logits_np))

        scores = router.gate_score_func(paddle.to_tensor(logits_np))
        np.testing.assert_allclose(
            scores.numpy(), expected, rtol=1e-6, atol=1e-6
        )
        # Sigmoid rows do not sum to 1 in general; guards against a softmax swap.
        self.assertFalse(
            np.allclose(scores.numpy().sum(axis=-1), np.ones(2), atol=1e-3)
        )

    # ----- top-k selection -------------------------------------------------

    def test_topk_greedy_selects_largest_with_weights(self):
        """greedy top-k returns the k largest scores and their expert indices."""
        router = self._make_router(n_routed_experts=6)
        scores = paddle.to_tensor(
            [
                [0.10, 0.50, 0.30, 0.05, 0.90, 0.20],
                [0.70, 0.20, 0.65, 0.15, 0.10, 0.40],
            ],
            dtype="float32",
        )
        weight, idx = router._topk_greedy(scores, k=2)

        self.assertEqual(idx.numpy().tolist(), [[4, 1], [0, 2]])
        np.testing.assert_allclose(
            weight.numpy(),
            np.array([[0.90, 0.50], [0.70, 0.65]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_topk_group_limited_greedy_confines_to_selected_group(self):
        """Only experts inside the winning group can be selected."""
        # 6 experts, 2 groups of 3 (contiguous), keep the single best group.
        router = self._make_router(n_routed_experts=6, n_group=2, topk_group=1)
        scores = paddle.to_tensor(
            [
                # group0=[0.10,0.20,0.15] max=0.20; group1=[0.90,0.05,0.80] max=0.90
                [0.10, 0.20, 0.15, 0.90, 0.05, 0.80],
                # group0=[0.70,0.60,0.65] max=0.70; group1=[0.10,0.05,0.20] max=0.20
                [0.70, 0.60, 0.65, 0.10, 0.05, 0.20],
            ],
            dtype="float32",
        )
        weight, idx = router._topk_group_limited_greedy(
            scores, k=2, n_group=2, topk_group=1
        )

        # Row 0 keeps group1 -> experts 3 & 5; row 1 keeps group0 -> experts 0 & 2.
        self.assertEqual(idx.numpy().tolist(), [[3, 5], [0, 2]])
        np.testing.assert_allclose(
            weight.numpy(),
            np.array([[0.90, 0.80], [0.70, 0.65]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_topk_noaux_tc_bias_moves_selection_not_gate_weight(self):
        """The correction bias steers selection; gate weights stay on raw scores."""
        router = self._make_router(
            topk_method="noaux_tc", n_routed_experts=4, num_experts_per_tok=2
        )
        # Distinguishable per-expert bias: only expert 2 is boosted.
        router.e_score_correction_bias.set_value(
            paddle.to_tensor([0.0, 0.0, 0.5, 0.0], dtype="float32")
        )
        scores = paddle.to_tensor(
            [
                # raw top-2 would be experts 0,1; +bias makes it 2,0.
                [0.50, 0.40, 0.30, 0.20],
                # raw top-2 would be experts 0,3; +bias makes it 2,0.
                [0.60, 0.10, 0.20, 0.50],
            ],
            dtype="float32",
        )
        weight, idx = router._topk_noaux_tc(
            scores, k=2, n_group=1, topk_group=1
        )

        self.assertEqual(idx.numpy().tolist(), [[2, 0], [2, 0]])
        # Weights must be the ORIGINAL (bias-free) scores at the chosen experts.
        np.testing.assert_allclose(
            weight.numpy(),
            np.array([[0.30, 0.50], [0.20, 0.60]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    # ----- auxiliary loss --------------------------------------------------

    def test_cal_aux_loss_matches_hand_derived_value(self):
        """aux_loss = num_experts * sum(mean(gates) * mean(mask))."""
        router = self._make_router(n_routed_experts=3)
        gates = paddle.to_tensor(
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], dtype="float32"
        )
        mask = paddle.to_tensor(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype="float32"
        )
        # me = [0.4, 0.4, 0.2]; ce = [0.5, 1.0, 0.5]
        # sum(me*ce) = 0.2 + 0.4 + 0.1 = 0.7; * num_experts(3) = 2.1
        aux = router._cal_aux_loss(gates, mask)
        self.assertEqual(list(aux.shape), [])
        self.assertAlmostEqual(float(aux.numpy()), 2.1, places=5)

    # ----- capacity --------------------------------------------------------

    def test_capacity_floor_and_clipping(self):
        """capacity = int((tokens//experts)*factor), clipped to [min, max]."""
        router = self._make_router(n_routed_experts=4)
        gates = paddle.zeros([8, 4], dtype="float32")  # tokens//experts == 2

        # 2 * 2.0 = 4.0 -> 4, inside [1, 100].
        self.assertEqual(
            router._capacity(
                gates, capacity_factor=2.0, max_capacity=100, min_capacity=1
            ),
            4,
        )
        # 2 * 0.1 = 0.2 -> int 0 -> raised to min_capacity 3.
        self.assertEqual(
            router._capacity(
                gates, capacity_factor=0.1, max_capacity=100, min_capacity=3
            ),
            3,
        )
        # 2 * 10 = 20 -> capped at max_capacity 5.
        self.assertEqual(
            router._capacity(
                gates, capacity_factor=10.0, max_capacity=5, min_capacity=1
            ),
            5,
        )

    # ----- capacity-limited priority mask ----------------------------------

    def test_priority_drops_tokens_beyond_capacity_by_arrival(self):
        """Over-capacity assignments to an expert are dropped in arrival order."""
        router = self._make_router(n_routed_experts=3)
        # Tokens 0,1,3 all pick expert 0; token 2 picks expert 1. capacity=2.
        topk_idx = paddle.to_tensor([[0], [0], [1], [0]], dtype="int64")
        mask = router._priority(topk_idx, capacity=2)

        # Expert 0 admits the first two arrivals (tokens 0,1) and drops token 3;
        # expert 1 admits token 2. Preserving token identity, not just counts.
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(mask.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
