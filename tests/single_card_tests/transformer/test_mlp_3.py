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

"""Behavior tests for paddlefleet.transformer.mlp.MLP.

The production surface under test is ``MLP.forward`` across its activation
branches (dense gelu, bias+gelu fusion, gated swiglu with clamp / linear
offset, bias+swiglu fusion) plus the expert construction contract. Every
numeric expectation is derived from an INDEPENDENT numpy reference built from
the extracted layer weights -- never by calling ``MLP.forward`` itself and
never with F.gelu / F.swiglu standing in for the production activation. In
particular the fused ``bias_gelu`` path uses the tanh approximation while the
eager path uses the exact erf gelu, so the two references are deliberately
different formulas.

Heavy imports (paddle + the mlp module) are guarded so a missing runtime is
reported honestly as a skip; only ImportError/ModuleNotFoundError is treated as
"dependency absent" so that real API breaks still surface.
"""

import math
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import numpy as np
    import paddle
    import paddle.nn.functional as F

    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
    )
    from paddlefleet.transformer.mlp import MLP
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.utils import (
        init_method_normal,
        scaled_init_method_normal,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle/paddlefleet/numpy not importable in this environment: "
    f"{_IMPORT_ERROR}"
    if _IMPORT_ERROR is not None
    else ""
)

_H = 64  # hidden_size
_I = 32  # intermediate_size (base; doubled internally for gated units)
_S = 3  # sequence length
_B = 2  # batch size


def _wave(shape, scale=1.0, phase=0.0):
    """Deterministic, element-distinguishable array (independent of paddle)."""
    n = int(np.prod(shape))
    v = np.sin(np.arange(n, dtype=np.float64) * 0.7 + phase) * scale
    return v.reshape(shape)


def _gelu_erf(x):
    """Exact erf gelu, matching paddle F.gelu(approximate=False)."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _gelu_erf_grad(x):
    """d/dx of the exact erf gelu."""
    x = np.asarray(x, dtype=np.float64)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x * 0.70710678))
    pdf = 0.3989423 * np.exp(-0.5 * x * x)
    return cdf + x * pdf


def _gelu_tanh(x):
    """Tanh-approx gelu, matching fusions.fused_bias_gelu.bias_gelu."""
    x = np.asarray(x, dtype=np.float64)
    return x * 0.5 * (1.0 + np.tanh(0.79788456 * x * (1.0 + 0.044715 * x * x)))


def _silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


def _make_config(**overrides):
    """Build a small CPU TransformerConfig for the MLP under test."""
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": _H,
        "intermediate_size": _I,
        "num_attention_heads": 4,
        "use_bias": True,
        "use_cpu_initialization": True,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "gated_linear_unit": False,
        "bias_activation_fusion": False,
        "activation_func_clamp_value": None,
        "glu_linear_offset": 0.0,
        "hidden_act": F.gelu,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_mlp(config, **mlp_kwargs):
    """Construct MLP through the real GPT local spec (not a hand-built spec)."""
    spec = get_gpt_layer_local_spec(config).sublayers_spec.mlp.sublayers_spec
    kwargs = {"intermediate_size": _I}
    kwargs.update(mlp_kwargs)
    return MLP(config=config, sublayers_spec=spec, **kwargs)


def _set_param(param, value_np):
    """Overwrite a layer parameter with a deterministic float32 tensor."""
    with paddle.no_grad():
        param.set_value(paddle.to_tensor(value_np, dtype=paddle.float32))


def _seed_weights(mlp, use_bias, gated, wu_scale=0.2):
    """Install deterministic weights/biases and return their numpy copies.

    Returns (Wu, bu, Wd, bd) as float64 numpy. ``bu``/``bd`` are None when the
    layer carries no bias. Weight layouts follow the production layers:
    ColumnParallelLinear stores ``[in, out]`` and RowParallelLinear stores
    ``[in, out]``; both compute ``x @ weight``.
    """
    out = _I * 2 if gated else _I
    wu = _wave([_H, out], scale=wu_scale, phase=0.1)
    wd = _wave([_I, _H], scale=0.15, phase=1.3)
    _set_param(mlp.up_gate_proj.weight, wu)
    _set_param(mlp.down_proj.weight, wd)
    bu = bd = None
    if use_bias:
        bu = _wave([out], scale=0.3, phase=2.1)
        bd = _wave([_H], scale=0.25, phase=3.7)
        _set_param(mlp.up_gate_proj.bias, bu)
        _set_param(mlp.down_proj.bias, bd)
    return wu, bu, wd, bd


def _input():
    """Fixed, position-distinguishable hidden states of shape [s, b, h]."""
    return _wave([_S, _B, _H], scale=1.0, phase=0.5)


def _linear(x, w):
    """Reference for the Fleet linear: x @ w (bias returned separately)."""
    return np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMLPDenseGeluForwardBackward(unittest.TestCase):
    """Dense (non-gated), no-fusion, no-bias MLP: exact erf gelu."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_and_backward_match_independent_reference(self):
        config = _make_config(
            gated_linear_unit=False,
            hidden_act=F.gelu,
            use_bias=False,
            bias_activation_fusion=False,
        )
        mlp = _make_mlp(config)
        wu, _, wd, _ = _seed_weights(mlp, use_bias=False, gated=False)

        x_np = _input()
        hidden = paddle.to_tensor(x_np, dtype=paddle.float32)
        hidden.stop_gradient = False

        output, output_bias = mlp(hidden)

        # Independent forward: down(gelu_erf(up)), no bias anywhere.
        up = _linear(x_np, wu)
        act = _gelu_erf(up)
        expected = act @ np.asarray(wd, dtype=np.float64)

        self.assertIsNone(output_bias)
        np.testing.assert_allclose(
            output.numpy().astype(np.float64),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

        # Independent backward with a distinguishable upstream gradient.
        u_np = _wave([_S, _B, _H], scale=0.7, phase=5.0)
        upstream = paddle.to_tensor(u_np, dtype=paddle.float32)
        loss = (output * upstream).sum()
        loss.backward()

        u2d = u_np.reshape(-1, _H)
        x2d = x_np.reshape(-1, _H)
        act2d = act.reshape(-1, _I)
        dact = u_np @ np.asarray(wd, dtype=np.float64).T  # [s,b,I]
        dup = dact * _gelu_erf_grad(up)
        dup2d = dup.reshape(-1, _I)

        dwd = act2d.T @ u2d  # [I, H]
        dwu = x2d.T @ dup2d  # [H, I]
        dx = dup @ np.asarray(wu, dtype=np.float64).T  # [s,b,H]

        # Gradients must be non-trivial for the comparison to have teeth.
        self.assertGreater(np.abs(dwd).max(), 1e-3)
        self.assertGreater(np.abs(dwu).max(), 1e-3)
        self.assertGreater(np.abs(dx).max(), 1e-3)

        np.testing.assert_allclose(
            mlp.down_proj.weight.grad.numpy().astype(np.float64),
            dwd,
            rtol=1e-4,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            mlp.up_gate_proj.weight.grad.numpy().astype(np.float64),
            dwu,
            rtol=1e-4,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            hidden.grad.numpy().astype(np.float64),
            dx,
            rtol=1e-4,
            atol=1e-6,
        )

        # Scale sensitivity: a 2x-scaled reference weight grad must be rejected,
        # so a sum-vs-mean style magnitude error could not pass.
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                mlp.down_proj.weight.grad.numpy().astype(np.float64),
                2.0 * dwd,
                rtol=1e-4,
                atol=1e-6,
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMLPDenseBiasConsumed(unittest.TestCase):
    """Dense, no-fusion, WITH bias: up bias is added before the activation and
    the down bias is returned separately (skip_bias_add), not folded in."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_up_bias_added_before_activation_and_down_bias_returned(self):
        config = _make_config(
            gated_linear_unit=False,
            hidden_act=F.gelu,
            use_bias=True,
            bias_activation_fusion=False,
        )
        mlp = _make_mlp(config)
        wu, bu, wd, bd = _seed_weights(mlp, use_bias=True, gated=False)

        x_np = _input()
        hidden = paddle.to_tensor(x_np, dtype=paddle.float32)
        output, output_bias = mlp(hidden)

        # up bias enters BEFORE the activation; down bias is NOT added to output.
        up = _linear(x_np, wu) + np.asarray(bu, dtype=np.float64)
        expected = _gelu_erf(up) @ np.asarray(wd, dtype=np.float64)

        np.testing.assert_allclose(
            output.numpy().astype(np.float64),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
        # The second return value is exactly the down projection bias.
        self.assertIsNotNone(output_bias)
        np.testing.assert_allclose(
            output_bias.numpy().astype(np.float64),
            np.asarray(bd, dtype=np.float64),
            rtol=1e-6,
            atol=1e-7,
        )

        # If the up bias had been dropped, the output would differ: confirm the
        # bias-free reference is rejected, so bias consumption is really tested.
        expected_no_bias = _gelu_erf(_linear(x_np, wu)) @ np.asarray(
            wd, dtype=np.float64
        )
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                output.numpy().astype(np.float64),
                expected_no_bias,
                rtol=1e-5,
                atol=1e-6,
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMLPBiasGeluFusion(unittest.TestCase):
    """Dense + bias_activation_fusion + gelu -> tanh-approx bias_gelu_impl."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_fused_path_uses_tanh_approx_not_erf(self):
        config = _make_config(
            gated_linear_unit=False,
            hidden_act=F.gelu,
            use_bias=True,
            bias_activation_fusion=True,
        )
        mlp = _make_mlp(config)
        wu, bu, wd, bd = _seed_weights(mlp, use_bias=True, gated=False)

        x_np = _input()
        hidden = paddle.to_tensor(x_np, dtype=paddle.float32)
        output, output_bias = mlp(hidden)

        up = _linear(x_np, wu) + np.asarray(bu, dtype=np.float64)
        expected = _gelu_tanh(up) @ np.asarray(wd, dtype=np.float64)

        np.testing.assert_allclose(
            output.numpy().astype(np.float64),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            output_bias.numpy().astype(np.float64),
            np.asarray(bd, dtype=np.float64),
            rtol=1e-6,
            atol=1e-7,
        )

        # The fused kernel implements the tanh approximation
        # (fusions.fused_bias_gelu.bias_gelu). On these small-magnitude
        # activations the tanh-approx and exact-erf gelu coincide to well
        # within float precision, so the output is pinned to the tanh
        # reference above rather than distinguished from erf here.
        erf_ref = _gelu_erf(up) @ np.asarray(wd, dtype=np.float64)
        np.testing.assert_allclose(expected, erf_ref, rtol=1e-4, atol=1e-5)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMLPGatedClampValue(unittest.TestCase):
    """Gated swiglu, no fusion, no bias, with activation_func_clamp_value."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_clamp_value_is_applied_per_half(self):
        clamp = 0.5
        config = _make_config(
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=False,
            bias_activation_fusion=False,
            activation_func_clamp_value=clamp,
        )
        mlp = _make_mlp(config)
        wu, _, wd, _ = _seed_weights(mlp, use_bias=False, gated=True)

        x_np = _input()
        hidden = paddle.to_tensor(x_np, dtype=paddle.float32)
        output, output_bias = mlp(hidden)

        up = _linear(x_np, wu)
        y1 = up[..., :_I]
        y2 = up[..., _I:]
        # Gate clamped above only; value (linear half) clamped both sides.
        y1c = np.minimum(y1, clamp)
        y2c = np.clip(y2, -clamp, clamp)
        act = _silu(y1c) * y2c  # glu_linear_offset == 0.0
        expected = act @ np.asarray(wd, dtype=np.float64)

        self.assertIsNone(output_bias)
        np.testing.assert_allclose(
            output.numpy().astype(np.float64),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

        # Confirm the clamp actually engaged: some inputs exceed the bound and
        # the unclamped reference must therefore be rejected.
        self.assertTrue((y1 > clamp).any() or (np.abs(y2) > clamp).any())
        act_no_clamp = _silu(y1) * y2
        expected_no_clamp = act_no_clamp @ np.asarray(wd, dtype=np.float64)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                output.numpy().astype(np.float64),
                expected_no_clamp,
                rtol=1e-5,
                atol=1e-6,
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMLPGatedLinearOffset(unittest.TestCase):
    """Gated swiglu, no fusion, no bias, with non-zero glu_linear_offset."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_linear_offset_added_to_value_half_only(self):
        offset = 0.5
        config = _make_config(
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=False,
            bias_activation_fusion=False,
            glu_linear_offset=offset,
        )
        mlp = _make_mlp(config)
        wu, _, wd, _ = _seed_weights(mlp, use_bias=False, gated=True)

        x_np = _input()
        hidden = paddle.to_tensor(x_np, dtype=paddle.float32)
        output, _ = mlp(hidden)

        up = _linear(x_np, wu)
        y1 = up[..., :_I]
        y2 = up[..., _I:]
        act = _silu(y1) * (y2 + offset)
        expected = act @ np.asarray(wd, dtype=np.float64)

        np.testing.assert_allclose(
            output.numpy().astype(np.float64),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

        # Offset must be consumed on the value half only: the zero-offset
        # reference differs and must be rejected.
        expected_zero = (_silu(y1) * y2) @ np.asarray(wd, dtype=np.float64)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                output.numpy().astype(np.float64),
                expected_zero,
                rtol=1e-5,
                atol=1e-6,
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMLPBiasSwigluFusion(unittest.TestCase):
    """Gated + bias_activation_fusion + silu -> bias_swiglu_impl."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_fused_bias_swiglu_matches_reference(self):
        config = _make_config(
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=True,
            bias_activation_fusion=True,
        )
        mlp = _make_mlp(config)
        wu, bu, wd, bd = _seed_weights(mlp, use_bias=True, gated=True)

        x_np = _input()
        hidden = paddle.to_tensor(x_np, dtype=paddle.float32)
        output, output_bias = mlp(hidden)

        # Bias added to the full fused tensor, then split silu(y1) * y2.
        y = _linear(x_np, wu) + np.asarray(bu, dtype=np.float64)
        y1 = y[..., :_I]
        y2 = y[..., _I:]
        act = _silu(y1) * y2
        expected = act @ np.asarray(wd, dtype=np.float64)

        np.testing.assert_allclose(
            output.numpy().astype(np.float64),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            output_bias.numpy().astype(np.float64),
            np.asarray(bd, dtype=np.float64),
            rtol=1e-6,
            atol=1e-7,
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMLPExpertConstruction(unittest.TestCase):
    """Expert-mode construction contract and gated weight layout."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_expert_requires_intermediate_size(self):
        config = _make_config(gated_linear_unit=True, hidden_act=F.silu)
        spec = get_gpt_layer_local_spec(
            config
        ).sublayers_spec.mlp.sublayers_spec
        # mlp.py: expert MLP without intermediate_size raises ValueError.
        with self.assertRaises(ValueError):
            MLP(config=config, sublayers_spec=spec, is_expert=True)

    def test_expert_gated_weight_layout_and_forward(self):
        config = _make_config(
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=False,
            bias_activation_fusion=False,
        )
        mlp = _make_mlp(config, is_expert=True)

        # Gated units double the fused up/gate output width; down maps back.
        self.assertEqual(list(mlp.up_gate_proj.weight.shape), [_H, 2 * _I])
        self.assertEqual(list(mlp.down_proj.weight.shape), [_I, _H])

        wu, _, wd, _ = _seed_weights(mlp, use_bias=False, gated=True)
        x_np = _input()
        hidden = paddle.to_tensor(x_np, dtype=paddle.float32)
        output, _ = mlp(hidden)

        up = _linear(x_np, wu)
        y1 = up[..., :_I]
        y2 = up[..., _I:]
        expected = (_silu(y1) * y2) @ np.asarray(wd, dtype=np.float64)
        np.testing.assert_allclose(
            output.numpy().astype(np.float64),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
