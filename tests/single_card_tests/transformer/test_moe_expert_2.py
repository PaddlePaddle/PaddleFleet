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

"""Behaviour tests for paddlefleet.transformer.moe.moe_expert.

These exercise the config-driven construction logic of ``GroupedMLPExpert``
(fused gate/up weight layout, gated-vs-plain activation selection, and the
guards that reject unsupported configs) plus the shared-config-safety contract
of ``StandardMLPExpert``. The gated activation closure is compared against an
independent NumPy SwiGLU reference rather than against paddle's own
``F.silu`` output, so a wrong branch or a swapped gate/value half is caught.

Heavy imports are guarded: on a host without paddle/paddlefleet the whole
module skips with an honest reason instead of erroring at collection time.
"""

import unittest

import numpy as np

try:
    import paddle
    import paddle.nn.functional as F

    from paddlefleet.transformer.moe.moe_expert import (
        GroupedMLPExpert,
        StandardMLPExpert,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.utils import (
        init_method_normal,
        scaled_init_method_normal,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    F = None
    _IMPORT_ERROR = exc


requires_paddle = unittest.skipUnless(
    paddle is not None,
    f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR}",
)


def _make_config(**overrides):
    """Build a small TransformerConfig for a single-process MoE expert.

    ``perform_initialization=False`` keeps the fused weights at their
    ``Constant(0.0)`` seed so construction is deterministic; the numeric tests
    below drive the activation closure with their own inputs and never read the
    (zeroed) weights, so this does not weaken them.
    """
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "num_attention_heads": 4,
        "intermediate_size": 16,
        "moe_intermediate_size": 16,
        "use_bias": False,
        "gated_linear_unit": True,
        "hidden_act": F.silu,
        "fp8": False,
        "recompute_granularity": None,
        "perform_initialization": False,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _numpy_silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


@requires_paddle
class TestGroupedMLPExpertWeightLayout(unittest.TestCase):
    """The fused grouped-gemm weight shapes are derived from the config."""

    def test_gated_weight_shapes(self):
        # SwiGLU fuses gate+up into weight1, so its last dim is doubled while
        # weight2 (down-proj) keeps a single intermediate width.
        # moe_expert.py:266-285 build these from hidden_size / moe_intermediate.
        config = _make_config(
            hidden_size=8, moe_intermediate_size=16, gated_linear_unit=True
        )
        expert = GroupedMLPExpert(
            num_local_experts=3, config=config, moe_deep_gemm=False
        )
        self.assertEqual(list(expert.weight1.shape), [3, 8, 32])
        self.assertEqual(list(expert.weight2.shape), [3, 16, 8])
        self.assertEqual(expert.weight1.dtype, paddle.bfloat16)
        self.assertEqual(expert.weight2.dtype, paddle.bfloat16)

    def test_plain_weight_shapes_not_doubled(self):
        # Without a gated unit weight1's last dim equals the intermediate size
        # (no doubling); weight2 is unchanged. Distinguishes the GLU branch.
        config = _make_config(
            hidden_size=8, moe_intermediate_size=16, gated_linear_unit=False
        )
        expert = GroupedMLPExpert(
            num_local_experts=3, config=config, moe_deep_gemm=False
        )
        self.assertEqual(list(expert.weight1.shape), [3, 8, 16])
        self.assertEqual(list(expert.weight2.shape), [3, 16, 8])

    def test_intermediate_size_per_partition_override(self):
        # An explicit per-partition intermediate size (EP intermediate shard)
        # overrides config.moe_intermediate_size for both fused weights.
        # moe_expert.py:217-221 select the override.
        config = _make_config(hidden_size=8, moe_intermediate_size=16)
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            intermediate_size_per_partition=4,
        )
        self.assertEqual(expert.intermediate_size_per_partition, 4)
        self.assertEqual(list(expert.weight1.shape), [2, 8, 8])
        self.assertEqual(list(expert.weight2.shape), [2, 4, 8])


@requires_paddle
class TestGroupedMLPExpertActivation(unittest.TestCase):
    """The activation closure selected in __init__ matches its config."""

    def test_gated_activation_is_swiglu(self):
        # GLU branch: chunk the fused projection into (gate, up) on the last
        # dim, apply hidden_act to gate, multiply by up. Compared against an
        # independent NumPy SwiGLU so a gate/up swap or wrong act is rejected.
        config = _make_config(gated_linear_unit=True, hidden_act=F.silu)
        expert = GroupedMLPExpert(
            num_local_experts=1, config=config, moe_deep_gemm=False
        )
        fused = np.array(
            [[-1.0, 2.0, 0.5, -3.0], [0.25, -0.5, 4.0, 1.0]],
            dtype=np.float32,
        )
        gate, up = fused[:, :2], fused[:, 2:]
        expected = _numpy_silu(gate) * up

        out = expert.activation_func(paddle.to_tensor(fused, dtype="float32"))
        self.assertEqual(list(out.shape), [2, 2])
        np.testing.assert_allclose(
            out.astype("float32").numpy(), expected, rtol=1e-5, atol=1e-6
        )

    def test_plain_activation_applies_elementwise(self):
        # Non-GLU branch: activation_func is exactly config.hidden_act, applied
        # to the full tensor with no chunking (output width is preserved).
        config = _make_config(gated_linear_unit=False, hidden_act=F.silu)
        expert = GroupedMLPExpert(
            num_local_experts=1, config=config, moe_deep_gemm=False
        )
        data = np.array([[-1.0, 2.0, 0.5, -3.0]], dtype=np.float32)
        expected = _numpy_silu(data)

        out = expert.activation_func(paddle.to_tensor(data, dtype="float32"))
        self.assertEqual(list(out.shape), [1, 4])
        np.testing.assert_allclose(
            out.astype("float32").numpy(), expected, rtol=1e-5, atol=1e-6
        )


@requires_paddle
class TestGroupedMLPExpertConfigGuards(unittest.TestCase):
    """Unsupported configs are rejected at construction time."""

    def test_bias_is_rejected(self):
        # Grouped GEMM has no bias path. moe_expert.py:222-224 asserts.
        config = _make_config(use_bias=True)
        with self.assertRaises(AssertionError):
            GroupedMLPExpert(
                num_local_experts=2, config=config, moe_deep_gemm=False
            )

    def test_unsupported_gated_activation_raises(self):
        # GLU only supports silu / gelu / situ; relu must be rejected.
        # moe_expert.py:255-259 raises ValueError. The original coverage test
        # only read config attributes and never reached this guard.
        config = _make_config(gated_linear_unit=True, hidden_act=F.relu)
        with self.assertRaises(ValueError):
            GroupedMLPExpert(
                num_local_experts=2, config=config, moe_deep_gemm=False
            )

    def test_backward_dw_is_noop(self):
        # backward_dw exists for API parity with (TE)GroupedMLP and must be a
        # side-effect-free no-op returning None. moe_expert.py:472-476.
        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=2, config=config, moe_deep_gemm=False
        )
        w1_before = expert.weight1.astype("float32").numpy().copy()
        w2_before = expert.weight2.astype("float32").numpy().copy()
        self.assertIsNone(expert.backward_dw())
        np.testing.assert_array_equal(
            expert.weight1.astype("float32").numpy(), w1_before
        )
        np.testing.assert_array_equal(
            expert.weight2.astype("float32").numpy(), w2_before
        )


@requires_paddle
class TestStandardMLPExpert(unittest.TestCase):
    """StandardMLPExpert wraps MLP and must not mutate the shared config."""

    def _mlp_spec(self, config):
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )

        return get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec

    def test_differing_intermediate_size_leaves_config_untouched(self):
        # When moe_intermediate_size != config.intermediate_size the expert
        # deep-copies the config before overriding it (moe_expert.py:1018-1029);
        # the caller's config.intermediate_size must be preserved.
        config = _make_config(intermediate_size=16, moe_intermediate_size=16)
        expert = StandardMLPExpert(
            config=config,
            moe_intermediate_size=32,
            is_expert=True,
            mlp_spec=self._mlp_spec(config),
        )
        self.assertIsInstance(expert, StandardMLPExpert)
        self.assertEqual(config.intermediate_size, 16)

    def test_matching_intermediate_size_uses_config(self):
        # Equal sizes take the direct branch (no deepcopy); config is unchanged.
        config = _make_config(intermediate_size=16, moe_intermediate_size=16)
        expert = StandardMLPExpert(
            config=config,
            moe_intermediate_size=16,
            is_expert=True,
            mlp_spec=self._mlp_spec(config),
        )
        self.assertIsInstance(expert, StandardMLPExpert)
        self.assertEqual(config.intermediate_size, 16)


if __name__ == "__main__":
    unittest.main()
