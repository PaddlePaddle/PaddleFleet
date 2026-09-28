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
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.muon_utils import ortho_gate_up
from paddlefleet.transformer.transformer_config import TransformerConfig


class TestParallelMLP(unittest.TestCase):
    transformer_config = TransformerConfig(
        num_hidden_layers=2,
        hidden_size=12,
        intermediate_size=48,
        num_attention_heads=4,
        use_bias=True,
    )
    expected_num_weights = 1212

    def setUp(self):
        self.mlp = MLP(
            self.transformer_config,
            get_gpt_layer_local_spec(
                self.transformer_config
            ).sublayers_spec.mlp.sublayers_spec,
        )

    def test_constructor(self):
        assert isinstance(self.mlp, MLP)

        num_weights = sum([p.numel() for p in self.mlp.parameters()])
        assert num_weights == self.expected_num_weights

    def test_forward_backward(self):
        mlp = self.mlp
        # [sequence length, batch size, hidden size]
        hidden_states = paddle.ones((32, 12, mlp.config.hidden_size))
        hidden_states.stop_gradient = False

        # add 0.0 to make hidden_states non-leaf
        output, output_bias = mlp(hidden_states + 0.0)
        assert output.shape[0] == 32
        assert output.shape[1] == 12
        assert output.shape[2] == mlp.config.hidden_size
        assert output.dtype == paddle.float32
        assert output_bias.shape[0] == mlp.config.hidden_size

        paddle.autograd.backward((output, output_bias))
        assert hidden_states.grad is not None


class TestBiasFusedGatedMLP(TestParallelMLP):
    transformer_config = TransformerConfig(
        num_hidden_layers=2,
        hidden_size=12,
        intermediate_size=48,
        num_attention_heads=4,
        bias_activation_fusion=True,
        gated_linear_unit=True,
        use_bias=True,
    )
    expected_num_weights = 1836


# ---------------------------------------------------------------------------
# Additional compliant behavior tests.
#
# The classes above only assert output/bias shapes and "grad is not None".
# The classes below drive the real ``MLP.forward`` / ``MLP.__init__`` /
# ``MLP.muon_slice_specs`` entry points and compare against independent numpy
# references built by hand from the module's own (real, initialized) weights.
# ``F.relu`` is used as the activation so the reference is a plain, transcendental
# -free formula (``max(x, 0)``) that does not depend on any Paddle activation
# implementation; a "forgot the activation" regression is still caught because
# the fixtures feed values with both signs. Runs on CPU; requires Paddle exactly
# like the existing classes in this file (no separate skip gating exists here).
# ---------------------------------------------------------------------------


def _mlp_from_config(config):
    """Build a dense ``MLP`` from a config using the local GPT layer spec."""
    spec = get_gpt_layer_local_spec(config).sublayers_spec.mlp.sublayers_spec
    mlp = MLP(config, spec)
    mlp.eval()
    return mlp


def _real_weights(mlp):
    """Extract the module's real parameters as float64 numpy for references."""
    return (
        mlp.up_gate_proj.weight.numpy().astype("float64"),
        mlp.up_gate_proj.bias.numpy().astype("float64"),
        mlp.down_proj.weight.numpy().astype("float64"),
        mlp.down_proj.bias.numpy().astype("float64"),
    )


class TestDenseMLPForwardReference(unittest.TestCase):
    """Numeric check of the dense (non-gated, non-fused) forward path.

    Contract: ``out = relu(x @ W_up + b_up) @ W_down`` with ``skip_bias_add``,
    so the down-projection bias is returned *separately* and is NOT folded into
    ``out``. Weight/bias wiring, matmul orientation and the split of the bias
    into the second return value are all observed.
    """

    def _config(self):
        return TransformerConfig(
            num_hidden_layers=2,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            use_bias=True,
            gated_linear_unit=False,
            bias_activation_fusion=False,
            hidden_act=F.relu,
        )

    def test_dense_relu_forward_matches_independent_reference(self):
        paddle.seed(2024)
        mlp = _mlp_from_config(self._config())
        x = paddle.randn([3, 2, 8], dtype="float32")

        out, out_bias = mlp(x)

        w_up, b_up, w_down, b_down = _real_weights(mlp)
        xn = x.numpy().astype("float64")
        intermediate = np.maximum(xn @ w_up + b_up, 0.0)
        ref = intermediate @ w_down

        self.assertEqual(out.shape, [3, 2, 8])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)
        # skip_bias_add: down_proj bias comes back on the side, un-added.
        self.assertEqual(out_bias.shape, [8])
        np.testing.assert_allclose(
            out_bias.numpy(), b_down, rtol=1e-6, atol=1e-7
        )


class TestDenseMLPPerTokenScale(unittest.TestCase):
    """Per-token scaling folds the down-proj bias into the output.

    With ``per_token_scale`` the dense path multiplies the activation by the
    per-token factor, and because ``output_bias is not None`` the bias is added
    in as ``bias * scale`` and the returned bias becomes ``None``. A non-uniform
    scale is used so a "scale ignored" or "wrong broadcast axis" regression is
    visible.
    """

    def test_per_token_scale_scales_and_folds_bias(self):
        paddle.seed(5)
        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            use_bias=True,
            gated_linear_unit=False,
            bias_activation_fusion=False,
            hidden_act=F.relu,
        )
        mlp = _mlp_from_config(config)
        x = paddle.randn([3, 2, 8], dtype="float32")
        scale = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 0.5], [2.0, 4.0]], dtype="float32"
        )

        out, out_bias = mlp(x, per_token_scale=scale)

        w_up, b_up, w_down, b_down = _real_weights(mlp)
        xn = x.numpy().astype("float64")
        scale_n = scale.numpy().astype("float64")
        intermediate = np.maximum(xn @ w_up + b_up, 0.0) * scale_n[..., None]
        ref = intermediate @ w_down + b_down[None, None, :] * scale_n[..., None]

        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)
        # bias was folded into the output, so nothing is returned on the side.
        self.assertIsNone(out_bias)


class TestGatedMLPForwardReference(unittest.TestCase):
    """Numeric check of the non-fused gated (GLU) path.

    Contract: with the fused gate/up projection split in half along the last
    axis, ``out = (relu(gate) * (linear + glu_linear_offset)) @ W_down`` where
    ``gate`` is the first half and ``linear`` the second half of
    ``x @ W_up + b_up``. ``activation_func_clamp_value`` clamps ``gate`` from
    above and ``linear`` to ``[-val, val]``. Both knobs are verified to actually
    change the output for the chosen fixture, not merely to be accepted.
    """

    def _config(self, offset=0.0, clamp=None):
        return TransformerConfig(
            num_hidden_layers=2,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            use_bias=True,
            gated_linear_unit=True,
            bias_activation_fusion=False,
            hidden_act=F.relu,
            glu_linear_offset=offset,
            activation_func_clamp_value=clamp,
        )

    def _reference(self, mlp, xn, offset, clamp):
        w_up, b_up, w_down, b_down = _real_weights(mlp)
        z = xn @ w_up + b_up
        half = z.shape[-1] // 2
        gate = z[..., :half]
        linear = z[..., half:]
        if clamp is not None:
            gate = np.minimum(gate, clamp)
            linear = np.clip(linear, -clamp, clamp)
        hidden = np.maximum(gate, 0.0) * (linear + offset)
        return hidden @ w_down, b_down

    def test_gated_relu_forward_matches_reference(self):
        paddle.seed(7)
        mlp = _mlp_from_config(self._config(offset=0.0))
        x = paddle.randn([3, 2, 8], dtype="float32")

        out, out_bias = mlp(x)

        xn = x.numpy().astype("float64")
        ref, b_down = self._reference(mlp, xn, offset=0.0, clamp=None)
        self.assertEqual(out.shape, [3, 2, 8])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            out_bias.numpy(), b_down, rtol=1e-6, atol=1e-7
        )

    def test_glu_linear_offset_is_consumed(self):
        paddle.seed(11)
        offset = 0.7
        mlp = _mlp_from_config(self._config(offset=offset))
        x = paddle.randn([3, 2, 8], dtype="float32")

        out, _ = mlp(x)

        xn = x.numpy().astype("float64")
        ref_with, _ = self._reference(mlp, xn, offset=offset, clamp=None)
        ref_without, _ = self._reference(mlp, xn, offset=0.0, clamp=None)
        np.testing.assert_allclose(out.numpy(), ref_with, rtol=1e-4, atol=1e-5)
        # The offset genuinely moves the output for this fixture.
        self.assertFalse(
            np.allclose(ref_with, ref_without, rtol=1e-4, atol=1e-5)
        )

    def test_activation_clamp_value_is_consumed(self):
        paddle.seed(13)
        # The gate/linear pre-activations here are small (weights use
        # init std 0.02 over hidden_size=8, so |z| ~ 0.05), so the clamp
        # threshold must sit inside that range to actually bite. 0.01 is well
        # below the activation spread, guaranteeing both the min-clamp on the
        # gate and the [-v, v] clip on the linear term change the output.
        clamp = 0.01
        mlp = _mlp_from_config(self._config(offset=0.0, clamp=clamp))
        x = paddle.randn([3, 2, 8], dtype="float32")

        out, _ = mlp(x)

        xn = x.numpy().astype("float64")
        ref_clamped, _ = self._reference(mlp, xn, offset=0.0, clamp=clamp)
        ref_unclamped, _ = self._reference(mlp, xn, offset=0.0, clamp=None)
        np.testing.assert_allclose(
            out.numpy(), ref_clamped, rtol=1e-4, atol=1e-5
        )
        # The clamp actually bites on this fixture (values exceed 0.01).
        self.assertFalse(
            np.allclose(ref_clamped, ref_unclamped, rtol=1e-4, atol=1e-5)
        )


class TestMLPExpertConstruction(unittest.TestCase):
    """MoE-expert construction requires an explicit ``intermediate_size``."""

    def _spec(self, config):
        return get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec

    def test_expert_without_intermediate_size_raises(self):
        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            use_bias=True,
        )
        spec = self._spec(config)
        with self.assertRaises(ValueError):
            MLP(config, spec, is_expert=True, intermediate_size=None)

    def test_expert_with_intermediate_size_builds(self):
        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            use_bias=True,
        )
        spec = self._spec(config)
        mlp = MLP(config, spec, is_expert=True, intermediate_size=16)
        # up_gate_proj input == hidden_size, down_proj output == hidden_size.
        self.assertEqual(mlp.up_gate_proj.weight.shape, [8, 16])
        self.assertEqual(mlp.down_proj.weight.shape, [16, 8])


class TestMLPGatedWeightShapes(unittest.TestCase):
    """Gated linear unit doubles the fused gate/up projection width."""

    def test_gated_unit_doubles_up_gate_width(self):
        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            use_bias=True,
            gated_linear_unit=True,
        )
        mlp = _mlp_from_config(config)
        # gate/up fused -> 2 * intermediate columns; down proj keeps inter rows.
        self.assertEqual(mlp.up_gate_proj.weight.shape, [8, 32])
        self.assertEqual(mlp.down_proj.weight.shape, [16, 8])


class TestMLPMuonSliceSpecs(unittest.TestCase):
    """``muon_slice_specs`` only emits a spec for gated units when asked."""

    def _mlp(self, gated):
        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            use_bias=True,
            gated_linear_unit=gated,
        )
        return _mlp_from_config(config)

    def test_no_spec_when_not_gated(self):
        mlp = self._mlp(gated=False)
        self.assertEqual(mlp.muon_slice_specs({"muon_ffn_split": True}), {})

    def test_no_spec_when_flag_absent_or_false(self):
        mlp = self._mlp(gated=True)
        self.assertEqual(mlp.muon_slice_specs({}), {})
        self.assertEqual(mlp.muon_slice_specs({"muon_ffn_split": False}), {})

    def test_spec_when_gated_and_flag_set(self):
        mlp = self._mlp(gated=True)
        specs = mlp.muon_slice_specs({"muon_ffn_split": True})
        self.assertEqual(set(specs), {"up_gate_proj.weight"})
        ortho_fn, extra_kwargs = specs["up_gate_proj.weight"]
        self.assertIs(ortho_fn, ortho_gate_up)
        self.assertEqual(extra_kwargs, {})


class TestMLPBiasActivationFusionErrors(unittest.TestCase):
    """Unsupported fused activations raise instead of silently mis-computing."""

    def test_non_gated_relu_fusion_raises(self):
        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            use_bias=True,
            gated_linear_unit=False,
            bias_activation_fusion=True,
            hidden_act=F.relu,
        )
        mlp = _mlp_from_config(config)
        x = paddle.randn([3, 2, 8], dtype="float32")
        with self.assertRaises(ValueError):
            mlp(x)


class TestMLPSublayersSpecDataclass(unittest.TestCase):
    """``MLPSublayersSpec`` field defaults and assignment."""

    def test_defaults_are_none(self):
        spec = MLPSublayersSpec()
        self.assertIsNone(spec.up_gate_proj)
        self.assertIsNone(spec.hidden_act)
        self.assertIsNone(spec.down_proj)

    def test_fields_store_supplied_objects(self):
        spec = MLPSublayersSpec(
            up_gate_proj=paddle.nn.Linear,
            hidden_act=F.gelu,
            down_proj=paddle.nn.Linear,
        )
        self.assertIs(spec.up_gate_proj, paddle.nn.Linear)
        self.assertIs(spec.hidden_act, F.gelu)
        self.assertIs(spec.down_proj, paddle.nn.Linear)


if __name__ == "__main__":
    unittest.main()
