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

"""CPU-observable behavior tests for the ERNIE MoE layer building blocks.

Target production code:
``src/paddlefleet/cli/train/ernie_pretrain/models/moe/moe_layer.py``

Scope (unit-test-rules.md "模型层 / MoE" and "配置与运行基础设施"):
  * ``dispatching`` -- masked scatter-add that places each token's (mask-scaled)
    features into its routed expert-capacity slot, summed over the ``k`` routing
    choices. The oracle pins the exact content of every output slot, including
    the ``overwrite=False`` ACCUMULATE behavior when two of a token's k choices
    (or two tokens) land on the same slot, and the per-choice mask scaling. A
    wrong slot, a dropped choice, or an ignored mask changes the numbers.
  * ``combining_fused`` (hard-gate path) -- ``F.embedding`` gather that selects,
    for each token, the expert-output row named by ``scatter_index``. Verified
    with a non-identity index map so a wrong gather is caught, plus the contract
    that in the hard-gate path ``combine_weights`` is intentionally ignored.
  * ``MoEStatics`` -- per-layer routing statistics module. Verified that the
    ``e_score_correction_bias`` starts at all-zeros (a neutral routing bias, NOT
    a final combine weight) with a live gradient, and that ``expert_usage`` is a
    zero-initialized int64 counter with gradients stopped. Shapes are ``[1, N]``
    for ``N = moe_num_experts``.
  * ``GateOutput`` -- namedtuple field-order contract ``(aux, z, logits)`` that
    downstream positional unpacking relies on.

The all-to-all PyLayers, the fused-FP8 experts and the full ``MOELayer.forward``
call GPU custom ops (``moe_gate_dispatch``, ``moe_combine``, ``fleet`` collectives)
and are NOT exercised here -- the local environment has no GPU and no paddle. The
whole module imports ``paddle`` at load time, so when paddle is absent every test
skips with an honest reason instead of reporting a fake pass. All expected values
below are hand-derived and independent of the production implementation.
"""

import types
import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.cli.train.ernie_pretrain.models.moe.moe_layer import (
        GateOutput,
        MoEStatics,
        combining_fused,
        dispatching,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or a paddle-dependent import) is missing
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "moe_layer imports paddle at module load; paddle is not importable in this "
    f"environment ({_IMPORT_ERROR})"
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDispatching(unittest.TestCase):
    """``dispatching`` scatters mask-scaled tokens into expert-capacity slots."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_scatter_add_places_and_scales_and_accumulates(self):
        # 2 tokens, d_model=2, k=2 routing choices, 2 experts x capacity 2 = 4
        # slots. Routing (hand-chosen, slot indices are absolute):
        #   token0 -> slot 0 (mask 1.0),  slot 3 (mask 0.5)
        #   token1 -> slot 0 (mask 2.0),  slot 2 (mask 1.0)
        # Both tokens' first choice land on slot 0, so slot 0 must ACCUMULATE.
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        scatter_index = paddle.to_tensor([[0, 3], [0, 2]], dtype="int64")
        dispatch_mask = paddle.to_tensor(
            [[1.0, 0.5], [2.0, 1.0]], dtype="float32"
        )

        out = dispatching(
            x, dispatch_mask, scatter_index, num_experts=2, capacity=2
        )

        # Hand-derived per slot:
        #   slot0 = x0*1.0 + x1*2.0 = [1,2] + [6,8]   = [7, 10]
        #   slot1 = (unused)                          = [0, 0]
        #   slot2 = x1*1.0                            = [3, 4]
        #   slot3 = x0*0.5                            = [0.5, 1.0]
        expected = np.array(
            [[7.0, 10.0], [0.0, 0.0], [3.0, 4.0], [0.5, 1.0]], dtype="float32"
        )
        self.assertEqual(list(out.shape), [4, 2])
        np.testing.assert_allclose(out.numpy(), expected, rtol=0, atol=1e-6)

    def test_zero_mask_choice_contributes_nothing(self):
        # A single token routed to two slots; the second choice has mask 0.0,
        # so only the first slot receives content. This pins that the mask
        # actually scales the contribution (mask 0 -> slot stays zero), not
        # merely a presence indicator.
        x = paddle.to_tensor([[5.0, -3.0]], dtype="float32")
        scatter_index = paddle.to_tensor([[1, 2]], dtype="int64")
        dispatch_mask = paddle.to_tensor([[1.0, 0.0]], dtype="float32")

        out = dispatching(
            x, dispatch_mask, scatter_index, num_experts=1, capacity=3
        )

        expected = np.array(
            [[0.0, 0.0], [5.0, -3.0], [0.0, 0.0]], dtype="float32"
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=0, atol=1e-6)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestCombiningFusedHardGate(unittest.TestCase):
    """``combining_fused`` hard-gate path gathers rows named by scatter_index."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_hard_gate_gathers_named_rows(self):
        # expert_output rows carry distinct content. scatter_index[token] = the
        # single row that token should pick. Non-identity map [2, 0, 1] so a
        # wrong gather axis or an identity implementation is visible.
        expert_output = paddle.to_tensor(
            [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype="float32"
        )
        scatter_index = paddle.to_tensor([[2], [0], [1]], dtype="int64")

        out = combining_fused(
            expert_output,
            combine_weights=None,
            scatter_index=scatter_index,
            hard_gate=True,
        )

        expected = np.array(
            [[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]], dtype="float32"
        )
        self.assertEqual(list(out.shape), [3, 2])
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_hard_gate_ignores_combine_weights(self):
        # Contract: on the hard-gate path the routed value is a pure gather;
        # combine_weights is not consumed. Two different weight tensors must
        # yield byte-identical output for the same index map.
        expert_output = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype="float32"
        )
        scatter_index = paddle.to_tensor([[1], [0]], dtype="int64")

        out_a = combining_fused(
            expert_output,
            combine_weights=paddle.to_tensor([[0.25]]),
            scatter_index=scatter_index,
            hard_gate=True,
        )
        out_b = combining_fused(
            expert_output,
            combine_weights=paddle.to_tensor([[9.0]]),
            scatter_index=scatter_index,
            hard_gate=True,
        )

        expected = np.array([[3.0, 4.0], [1.0, 2.0]], dtype="float32")
        np.testing.assert_array_equal(out_a.numpy(), expected)
        np.testing.assert_array_equal(out_b.numpy(), out_a.numpy())


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestMoEStatics(unittest.TestCase):
    """``MoEStatics`` initial routing-statistics state."""

    def setUp(self):
        paddle.set_device("cpu")

    def _build(self, num_experts):
        config = types.SimpleNamespace(moe_num_experts=num_experts)
        return MoEStatics(config, layer_idx=0)

    def test_correction_bias_starts_neutral_with_gradient(self):
        stats = self._build(num_experts=4)

        bias = stats.e_score_correction_bias
        self.assertEqual(list(bias.shape), [1, 4])
        # The correction bias is a routing-selection bias, added to gate logits;
        # it must start neutral (all zeros), NOT as a combine weight. A non-zero
        # init would skew routing from step 0.
        np.testing.assert_array_equal(
            bias.numpy(), np.zeros((1, 4), dtype=bias.numpy().dtype)
        )
        # It is a trainable parameter (routing bias is updated), so gradients
        # must be enabled.
        self.assertFalse(bias.stop_gradient)

    def test_expert_usage_is_zero_int64_counter_without_gradient(self):
        stats = self._build(num_experts=3)

        usage = stats.expert_usage
        self.assertEqual(list(usage.shape), [1, 3])
        self.assertEqual(usage.dtype, paddle.int64)
        np.testing.assert_array_equal(
            usage.numpy(), np.zeros((1, 3), dtype="int64")
        )
        # Usage is an accumulated statistic, not a learnable parameter.
        self.assertTrue(usage.stop_gradient)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestGateOutput(unittest.TestCase):
    """``GateOutput`` namedtuple field-order contract."""

    def test_field_order_and_positional_mapping(self):
        # Downstream code unpacks GateOutput positionally, so both the field
        # names and their order are load-bearing.
        self.assertEqual(GateOutput._fields, ("aux", "z", "logits"))
        out = GateOutput("AUX", "Z", "LOGITS")
        self.assertEqual(out.aux, "AUX")
        self.assertEqual(out.z, "Z")
        self.assertEqual(out.logits, "LOGITS")


if __name__ == "__main__":
    unittest.main()
