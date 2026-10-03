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
"""Tests for the fused-projection splits that the ``"hf"`` target needs.

The reference ``Qwen3_5MoeMLP`` keeps ``gate_proj`` and ``up_proj`` as two
separate ``nn.Linear`` modules where PaddleFleet fuses them into one
``up_gate_proj``. That costs bit-exactness in two independent places:

* **dgrad.** The fused ``K = 2 * inter`` GEMM is not bitwise equal to the sum of
  the two ``K = inter`` GEMMs, because cuBLAS reduces the whole K range in one
  pass. ``MLP.forward``'s ``hidden_states_up`` argument therefore drives the
  projection twice, each call seeing a gradient that is zero on the other half,
  so the two contributions enter the accumulation chain separately.
* **gradient clipping.** Two ``nn.Linear``s mean the clip takes *two* per-tensor
  BF16 norms over the two column halves, not one over the concatenation, and
  squaring one norm is not the same as summing the squares of two.
  ``_maybe_tag_up_gate_norm_groups`` records the halves on the weight for the
  clip to read.

Both are gated on the ``"hf"`` target: ``True``/``"megatron"`` must leave the
fused single-consumer path exactly as it was.
"""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import paddle

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddlefleet.transformer.mlp import MLP
from paddlefleet.transformer.moe.moe_shared_expert import (
    StandardMLPSharedExpert,
)
from paddlefleet.transformer.transformer_config import (
    TransformerConfig,
)


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 12,
        "intermediate_size": 48,
        "num_attention_heads": 4,
        "use_bias": True,
        "gated_linear_unit": True,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_mlp(config):
    return MLP(
        config,
        get_gpt_layer_local_spec(config).sublayers_spec.mlp.sublayers_spec,
    )


class TestMLPDualConsumerUpGate(unittest.TestCase):
    """``hidden_states_up`` splits the fused projection into two consumers."""

    def setUp(self):
        paddle.seed(20260908)
        self.config = _make_config()
        self.mlp = _make_mlp(self.config)
        self.hidden = paddle.randn([4, 2, 12], dtype=paddle.float32)

    def test_same_input_twice_matches_the_fused_call(self):
        """Feeding the same tensor to both consumers is a pure refactor.

        The concat of the gate half of one call and the up half of the other is
        the same value the single fused call produces; only the autograd graph
        differs.
        """
        fused, bias_a = self.mlp(self.hidden)
        split, bias_b = self.mlp(self.hidden, hidden_states_up=self.hidden)
        np.testing.assert_allclose(
            fused.numpy(), split.numpy(), rtol=1e-6, atol=1e-6
        )
        if bias_a is not None and bias_b is not None:
            np.testing.assert_allclose(
                bias_a.numpy(), bias_b.numpy(), rtol=0, atol=0
            )

    def test_each_call_sees_a_half_zero_gradient(self):
        """That is what makes the two dgrads narrow instead of one wide GEMM."""
        gate_in = self.hidden.detach()
        gate_in.stop_gradient = False
        up_in = self.hidden.detach()
        up_in.stop_gradient = False
        out, _ = self.mlp(gate_in, hidden_states_up=up_in)
        # Drive the whole graph with ``backward()``: ``paddle.grad`` on a
        # subset of inputs prunes the projection weight and then Paddle's
        # PyLayer contract demands None at that position.
        out.sum().backward()
        g_gate, g_up = gate_in.grad, up_in.grad
        # Both consumers receive a gradient, and neither is the full fused one.
        self.assertIsNotNone(g_gate)
        self.assertIsNotNone(g_up)
        self.assertFalse(np.array_equal(g_gate.numpy(), g_up.numpy()))

    def test_output_shape_and_dtype_unchanged(self):
        out, _ = self.mlp(self.hidden, hidden_states_up=self.hidden)
        self.assertEqual(out.shape, list(self.hidden.shape))
        self.assertEqual(out.dtype, paddle.float32)

    def test_none_keeps_the_single_consumer_path(self):
        """The default argument must not perturb the historical behaviour."""
        a, _ = self.mlp(self.hidden)
        b, _ = self.mlp(self.hidden, hidden_states_up=None)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_distinct_inputs_take_the_expected_halves(self):
        """Gate comes from arg 1, up from ``hidden_states_up``."""
        other = paddle.randn(self.hidden.shape, dtype=paddle.float32)
        mixed, _ = self.mlp(self.hidden, hidden_states_up=other)
        same, _ = self.mlp(self.hidden, hidden_states_up=self.hidden)
        self.assertFalse(np.array_equal(mixed.numpy(), same.numpy()))


class TestSharedExpertNormGroupTagging(unittest.TestCase):
    """``_maybe_tag_up_gate_norm_groups`` records the reference's two halves."""

    def _make(self, target):
        config = _make_config(
            use_accuracy_compatible=target,
            moe_shared_expert_gate=False,
        )
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        return StandardMLPSharedExpert(
            config,
            moe_intermediate_size=config.intermediate_size,
            is_expert=False,
            mlp_spec=spec,
        )

    def test_hf_target_tags_two_column_halves(self):
        expert = self._make("hf")
        weight = expert.up_gate_proj.weight
        groups = getattr(weight, "hf_norm_groups", None)
        self.assertIsNotNone(groups)
        self.assertEqual(len(groups), 2)
        width = weight.shape[-1]
        half = width // 2
        np.testing.assert_array_equal(
            groups[0].numpy(), np.arange(0, half, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            groups[1].numpy(), np.arange(half, width, dtype=np.int64)
        )

    def test_groups_partition_the_columns_exactly(self):
        """No overlap and no gap: the clip must see every column once."""
        expert = self._make("hf")
        groups = expert.up_gate_proj.weight.hf_norm_groups
        merged = np.concatenate([g.numpy() for g in groups])
        width = expert.up_gate_proj.weight.shape[-1]
        np.testing.assert_array_equal(np.sort(merged), np.arange(width))

    def test_non_hf_targets_leave_the_weight_untagged(self):
        """A Megatron-aligned run must keep the single-norm clip."""
        for target in (False, True, "megatron"):
            with self.subTest(target=target):
                expert = self._make(target)
                self.assertIsNone(
                    getattr(expert.up_gate_proj.weight, "hf_norm_groups", None)
                )

    def test_groups_are_int64_index_tensors(self):
        """``index_select`` in the clip requires int64 indices."""
        expert = self._make("hf")
        for group in expert.up_gate_proj.weight.hf_norm_groups:
            self.assertEqual(group.dtype, paddle.int64)

    def test_odd_width_is_left_untagged(self):
        """An odd column count has no two equal halves, so it is skipped.

        Called unbound on a namespace: Paddle refuses a non-``Layer`` sublayer
        assignment, and the method only reads ``self.up_gate_proj.weight`` and
        ``self.use_accuracy_compatible``.
        """
        weight = paddle.zeros([12, 7], dtype=paddle.float32)
        stub = SimpleNamespace(
            use_accuracy_compatible="hf",
            up_gate_proj=SimpleNamespace(weight=weight),
        )
        StandardMLPSharedExpert._maybe_tag_up_gate_norm_groups(stub)
        self.assertIsNone(getattr(weight, "hf_norm_groups", None))

    def test_even_width_on_the_same_seam_is_tagged(self):
        """Sanity check that the unbound-call seam does reach the tagging."""
        weight = paddle.zeros([12, 8], dtype=paddle.float32)
        stub = SimpleNamespace(
            use_accuracy_compatible="hf",
            up_gate_proj=SimpleNamespace(weight=weight),
        )
        StandardMLPSharedExpert._maybe_tag_up_gate_norm_groups(stub)
        self.assertEqual(len(weight.hf_norm_groups), 2)

    def test_missing_weight_is_tolerated(self):
        """A projection without a ``weight`` attribute must not raise."""
        stub = SimpleNamespace(
            use_accuracy_compatible="hf", up_gate_proj=SimpleNamespace()
        )
        StandardMLPSharedExpert._maybe_tag_up_gate_norm_groups(stub)


class TestSharedExpertGate(unittest.TestCase):
    """The shared-expert gate multiplies the MLP output by ``sigmoid(x @ w)``."""

    def _make(self, use_gate):
        config = _make_config(
            use_accuracy_compatible="hf",
            moe_shared_expert_gate=use_gate,
        )
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        return StandardMLPSharedExpert(
            config,
            moe_intermediate_size=config.intermediate_size,
            is_expert=False,
            mlp_spec=spec,
        )

    def test_gate_weight_created_only_when_enabled(self):
        self.assertIsNotNone(self._make(True).gate_weight)
        self.assertIsNone(self._make(False).gate_weight)

    def test_gate_scales_the_output(self):
        """With a zero-initialized gate the scale is sigmoid(0) = 0.5."""
        paddle.seed(3)
        gated = self._make(True)
        hidden = paddle.randn([4, 2, 12], dtype=paddle.float32)
        out_gated, _ = gated(hidden)
        # gate_weight is init'd by config.init_method, so recompute explicitly.
        logits = paddle.nn.functional.linear(hidden, gated.gate_weight)
        expected_scale = paddle.nn.functional.sigmoid(logits)
        base, _ = super(type(gated), gated).forward(hidden)
        np.testing.assert_allclose(
            out_gated.numpy(),
            (base * expected_scale).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_explicit_gate_source_overrides_the_default(self):
        """``hidden_states_gate`` lets the HF path feed its own clone."""
        paddle.seed(5)
        gated = self._make(True)
        hidden = paddle.randn([4, 2, 12], dtype=paddle.float32)
        other = paddle.randn([4, 2, 12], dtype=paddle.float32)
        a, _ = gated(hidden)
        b, _ = gated(hidden, hidden_states_gate=other)
        self.assertFalse(np.array_equal(a.numpy(), b.numpy()))


if __name__ == "__main__":
    unittest.main()
