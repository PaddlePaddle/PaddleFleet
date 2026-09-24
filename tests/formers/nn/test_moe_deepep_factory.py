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

"""Behavior tests for QuickAccessMoEFactory.create_from_model_name.

Environment: 无卡 (CPU). The factory belongs to the "模型与训练目标 / MoE" module
(config -> real ModularMoELayer assembly). These tests build a REAL ModularMoELayer
through the real factory and inspect the object it returns: resolved config values,
the chosen gate / communication / expert implementations, and their wiring.

The factory takes ``expert_class`` as an explicit dependency-injection parameter, so
tests inject a minimal *real* nn.Layer expert (``_RecordingExpert``) with the exact
signature ModularMoELayer uses to instantiate experts. This is a legitimate
collaborator, not a mock of the unit under test -- ModularMoELayer, its gate and its
communication layer are all constructed for real and asserted on directly.

Not covered here (documented, not silently skipped):
  * EP forward / token dispatch / expert compute numerics -- require a real expert
    parallel process group and GPUs (多卡). Only construction/selection is the
    factory's responsibility.
  * The inference_topk_method branch of ModularMoELayer -- topk_method is frozen at
    __init__ time while nn.Layer.training defaults to True, so the eval branch is
    unreachable through this constructor path (see the train-topk test).
"""

import unittest

from paddle import nn

from paddlefleet.nn.moe_deepep.modular_moe_layer import ModularMoELayer
from paddlefleet.nn.moe_deepep.moe_communication import (
    AllToAllMoECommunication,
    DeepEPMoECommunication,
)
from paddlefleet.nn.moe_deepep.moe_factory import QuickAccessMoEFactory
from paddlefleet.nn.moe_deepep.moe_gate import StandardMoEGate
from paddlefleet.transformers.configuration_utils import PretrainedConfig


class _RecordingExpert(nn.Layer):
    """Minimal real expert with the signature ModularMoELayer instantiates experts
    with: ``(config, intermediate_size, fuse_up_gate)``. Records what it received so
    tests can verify factory -> layer -> expert wiring without pulling in the heavy
    real Linear stack."""

    def __init__(self, config, intermediate_size, fuse_up_gate):
        super().__init__()
        self.captured_config = config
        self.captured_intermediate_size = intermediate_size
        self.captured_fuse_up_gate = fuse_up_gate

    def forward(self, x):  # pragma: no cover - not exercised in these tests
        return x


def _make_config(**overrides):
    """Build a real PretrainedConfig with the minimum fields the factory reads as
    plain attributes (hidden_size, moe_intermediate_size, model_type). Extra MoE keys
    are passed through and become attributes."""
    base = {
        "hidden_size": 64,
        "moe_intermediate_size": 32,
        "model_type": "qwen2_moe",
        "num_experts": 4,
        "num_experts_per_tok": 2,
    }
    base.update(overrides)
    return PretrainedConfig(**base)


def _create(
    config,
    expert_class=_RecordingExpert,
    gate_activation="softmax",
    expert_activation="silu",
    train_topk_method="greedy",
    inference_topk_method="greedy",
    transpose_gate_weight=False,
):
    return QuickAccessMoEFactory.create_from_model_name(
        pretrained_config=config,
        expert_class=expert_class,
        gate_activation=gate_activation,
        expert_activation=expert_activation,
        train_topk_method=train_topk_method,
        inference_topk_method=inference_topk_method,
        transpose_gate_weight=transpose_gate_weight,
    )


class TestQuickAccessMoEFactory(unittest.TestCase):
    def test_returns_real_modular_moe_layer(self):
        layer = _create(_make_config())
        self.assertIsInstance(layer, ModularMoELayer)
        # Wiring: experts are the injected class, one entry per expert on this rank.
        self.assertEqual(len(layer.experts), 4)
        self.assertTrue(
            all(isinstance(e, _RecordingExpert) for e in layer.experts)
        )
        self.assertIs(layer.expert_class, _RecordingExpert)

    def test_missing_model_type_raises_value_error(self):
        config = _make_config(model_type=None)
        with self.assertRaises(ValueError) as ctx:
            _create(config)
        self.assertIn("Cannot determine model type", str(ctx.exception))

    def test_num_experts_uses_direct_value(self):
        layer = _create(_make_config(num_experts=6))
        # Independent expectation: num_experts is present -> used verbatim.
        self.assertEqual(layer.num_experts, 6)
        self.assertEqual(len(layer.experts), 6)

    def test_num_experts_falls_back_to_n_routed_experts(self):
        # num_experts absent; n_routed_experts is the next key in the chain.
        config = PretrainedConfig(
            hidden_size=64,
            moe_intermediate_size=32,
            model_type="mixtral",
            num_experts_per_tok=2,
            n_routed_experts=6,
        )
        layer = _create(config)
        self.assertEqual(layer.num_experts, 6)
        self.assertEqual(len(layer.experts), 6)

    def test_num_experts_per_tok_moe_k_fallback_is_used(self):
        # REAL-BEHAVIOR: the factory resolves
        #   config.get("num_experts_per_tok", config.get("moe_k", -1))
        # Here ``num_experts_per_tok`` is absent from the config and is not a
        # populated attribute, so .get() falls through to the moe_k fallback,
        # yielding moe_k (3). Asserting the actual resolved value.
        config = PretrainedConfig(
            hidden_size=64,
            moe_intermediate_size=32,
            model_type="test_moe",
            num_experts=4,
            moe_k=3,
        )
        layer = _create(config)
        self.assertEqual(layer.num_experts_per_tok, 3)

    def test_num_shared_experts_value_and_shared_expert_sizing(self):
        layer = _create(_make_config(n_shared_experts=2))
        self.assertEqual(layer.num_shared_experts, 2)
        self.assertIsInstance(layer.shared_experts, _RecordingExpert)
        # Shared expert intermediate size = moe_intermediate_size * num_shared_experts.
        self.assertEqual(
            layer.shared_experts.captured_intermediate_size, 32 * 2
        )

    def test_no_shared_experts_when_zero(self):
        layer = _create(
            _make_config()
        )  # n_shared_experts absent -> resolves to 0
        self.assertEqual(layer.num_shared_experts, 0)
        self.assertIsNone(layer.shared_experts)

    def test_expert_activation_comes_from_config_hidden_act(self):
        # QUIRK: the `expert_activation` argument is placed only into moe_config
        # (which the layer/gate never read). The layer's real expert_activation is
        # resolved from config: hidden_act -> expert_activation -> "silu". Here the
        # passed arg "relu" is ignored in favor of config.hidden_act="gelu".
        layer = _create(
            _make_config(hidden_act="gelu"), expert_activation="relu"
        )
        self.assertEqual(layer.expert_activation, "gelu")

    def test_expert_activation_defaults_to_silu(self):
        # No hidden_act and no expert_activation on config -> "silu".
        layer = _create(_make_config(), expert_activation="relu")
        self.assertEqual(layer.expert_activation, "silu")

    def test_gate_activation_flows_into_real_gate(self):
        layer = _create(_make_config(), gate_activation="sigmoid")
        self.assertEqual(layer.gate_activation, "sigmoid")
        self.assertIsInstance(layer.gate, StandardMoEGate)
        # gate_activation is consumed as the gate's scoring function.
        self.assertEqual(layer.gate.scoring_func, "sigmoid")

    def test_transpose_gate_weight_controls_gate_weight_shape(self):
        straight = _create(_make_config(), transpose_gate_weight=False)
        self.assertFalse(straight.transpose_gate_weight)
        self.assertFalse(straight.gate.transpose_gate_weight)
        # not transposed: [expert_hidden_size, num_experts]
        self.assertEqual(list(straight.gate.weight.shape), [64, 4])

        transposed = _create(_make_config(), transpose_gate_weight=True)
        self.assertTrue(transposed.gate.transpose_gate_weight)
        # transposed: [num_experts, expert_hidden_size]
        self.assertEqual(list(transposed.gate.weight.shape), [4, 64])

    def test_train_topk_method_selected_at_construction(self):
        # nn.Layer.training defaults True at __init__, so the train method is chosen
        # and frozen. inference_topk_method is unreachable through this path.
        layer = _create(
            _make_config(),
            train_topk_method="group_limited_greedy",
            inference_topk_method="greedy",
        )
        self.assertEqual(layer.topk_method, "group_limited_greedy")
        self.assertEqual(layer.gate.topk_method, "group_limited_greedy")

    def test_norm_topk_prob_passthrough_and_default(self):
        explicit = _create(_make_config(norm_topk_prob=False))
        self.assertFalse(explicit.norm_topk_prob)
        self.assertFalse(explicit.gate.norm_topk_prob)

        default = _create(
            _make_config()
        )  # norm_topk_prob absent -> default True
        self.assertTrue(default.norm_topk_prob)

    def test_default_communication_is_alltoall_from_config(self):
        # ModularMoELayer's own literal default is "deepep", but it is shadowed:
        # moe_token_dispatcher_type is a PretrainedConfig field defaulting to
        # "alltoall" (LlmMetaConfig), so the resolved dispatcher is alltoall.
        layer = _create(_make_config())
        self.assertEqual(layer.moe_token_dispatcher_type, "alltoall")
        self.assertIsInstance(layer.communication, AllToAllMoECommunication)

    def test_deepep_communication_selected(self):
        layer = _create(_make_config(moe_token_dispatcher_type="deepep"))
        self.assertEqual(layer.moe_token_dispatcher_type, "deepep")
        self.assertIsInstance(layer.communication, DeepEPMoECommunication)

    def test_unsupported_dispatcher_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _create(_make_config(moe_token_dispatcher_type="bogus"))
        self.assertIn("Unsupported communication type", str(ctx.exception))

    def test_token_dispatcher_none_without_expert_parallel(self):
        # No fleet init -> expert_model_parallel_size == 1 -> no flex token dispatcher.
        layer = _create(_make_config())
        self.assertEqual(layer.expert_model_parallel_size, 1)
        self.assertIsNone(layer.token_dispatcher)

    def test_routed_expert_receives_moe_intermediate_size_and_fuse_up_gate(
        self,
    ):
        layer = _create(_make_config(moe_intermediate_size=48))
        expert = layer.experts[0]
        self.assertEqual(expert.captured_intermediate_size, 48)
        self.assertTrue(expert.captured_fuse_up_gate)

    def test_model_type_propagates_to_layer(self):
        layer = _create(_make_config(model_type="qwen3_moe"))
        self.assertEqual(layer.model_type, "qwen3_moe")

    def test_default_standard_expert_construction_is_broken(self):
        # REAL BUG: with expert_class=None the layer falls back to StandardMLPExpert,
        # then instantiates every expert as
        #   StandardMLPExpert(config=..., intermediate_size=..., fuse_up_gate=...).
        # But StandardMLPExpert.__init__(self, config, moe_intermediate_size) accepts
        # neither `intermediate_size` nor `fuse_up_gate` and requires
        # `moe_intermediate_size`, so construction raises TypeError. The default expert
        # path is therefore unusable; only an injected compatible expert_class works.
        # This characterization test pins the current broken contract and must be
        # updated once StandardMLPExpert's signature is fixed.
        with self.assertRaises(TypeError):
            _create(_make_config(), expert_class=None)


if __name__ == "__main__":
    unittest.main()
