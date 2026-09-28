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

"""Behavior tests for the SwiGLU activation-fusion surface of the MLP block.

Scope is the CPU-executable numeric surface referenced by the coverage source:

  * ``paddlefleet.fusions.fused_bias_swiglu.weighted_bias_swiglu_impl`` -- the
    per-token-weighted swiglu fusion (clamped and un-clamped), forward and
    backward, plus its bias-not-supported contract.
  * ``paddlefleet.fusions.fused_bias_swiglu.bias_swiglu_impl`` -- the biased
    swiglu fusion (clamped and un-clamped), forward and backward.
  * ``paddlefleet.transformer.mlp.MLP.forward`` dispatch: that a silu + gated +
    ``bias_activation_fusion`` config with a ``per_token_scale`` routes through
    the weighted fusion, applies the per-token scale, and that
    ``activation_func_clamp_value`` actually changes the produced activation.

Every numeric expectation is derived from an INDEPENDENT reference: forward
from an explicit numpy silu/clamp formulation, backward from paddle autograd
through the naive ``silu(gate) * value`` graph -- never by calling the
production fused kernel (whose backward is a hand-written PyLayer) to build its
own expected value. Inputs are fixed (seeded) and chosen to straddle the clamp
bound so a dropped-clamp or wrong-scale bug cannot survive on degenerate input.

Heavy imports (paddle + the fused module + the MLP) are guarded so a missing
runtime is reported honestly as a skip; only ImportError/ModuleNotFoundError
counts as "dependency absent" so that real API breaks still surface.
"""

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

    from paddlefleet.fusions.fused_bias_swiglu import (
        bias_swiglu_impl,
        weighted_bias_swiglu_impl,
    )
    from paddlefleet.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
    )
    from paddlefleet.transformer.mlp import MLP
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.utils import (
        init_method_normal,
        scaled_init_method_normal,
    )

    paddle.set_device("cpu")
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


def _np_silu(x):
    """SiLU(x) = x * sigmoid(x), computed in float64 for the reference."""
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


def _np_swiglu_activation(y_np, weights_np=None, bias_np=None, clamp=None):
    """Independent numpy reference for the (clamped) (bias/weighted) swiglu.

    Splits the last dim into gate/value halves, optionally clamps gate to
    (-inf, clamp] and value to [-clamp, clamp], computes ``silu(gate) * value``,
    and optionally scales by per-token weights. Written from the GPT-OSS clamp
    spec directly; it never calls the production fused function.
    """
    z = np.asarray(y_np, dtype=np.float64)
    if bias_np is not None:
        z = z + np.asarray(bias_np, dtype=np.float64)
    half = z.shape[-1] // 2
    gate = z[..., :half]
    value = z[..., half:]
    if clamp is not None:
        gate = np.minimum(gate, clamp)
        value = np.clip(value, -clamp, clamp)
    act = _np_silu(gate) * value
    if weights_np is not None:
        act = act * np.asarray(weights_np, dtype=np.float64)
    return act


def _paddle_naive_swiglu(y, weights=None, bias=None, clamp=None):
    """Autograd reference graph: naive ``silu(gate) * value`` in paddle.

    Used only for the *backward* comparison. It differentiates the plain
    elementwise expression, so the production PyLayer's hand-written backward
    (clamp masking, weight-grad reduction, ``swiglu_grad`` C++ op) is checked
    against an independent gradient rather than against itself.
    """
    z = y if bias is None else y + bias
    g, v = paddle.chunk(z, 2, axis=-1)
    if clamp is not None:
        g = paddle.clip(g, max=float(clamp))
        v = paddle.clip(v, min=-float(clamp), max=float(clamp))
    act = F.silu(g) * v
    if weights is not None:
        act = act * weights
    return act


def _make_config(clamp_value, **overrides):
    """A minimal silu + gated + bias_activation_fusion TransformerConfig.

    ``use_bias=False`` keeps the projection bias out of the activation input so
    the per-token-weighted fusion path (which rejects a non-None bias) is the
    one exercised, matching a MoE-style weighted expert MLP.
    """
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "use_bias": False,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "gated_linear_unit": True,
        "bias_activation_fusion": True,
        "activation_func_clamp_value": clamp_value,
        "glu_linear_offset": 0.0,
        "hidden_act": F.silu,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _build_mlp(clamp_value):
    config = _make_config(clamp_value)
    spec = get_gpt_layer_local_spec(config)
    sublayers = spec.sublayers_spec.mlp.sublayers_spec
    return MLP(config=config, sublayers_spec=sublayers)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestWeightedBiasSwigluImpl(unittest.TestCase):
    """weighted_bias_swiglu_impl: per-token-weighted (clamped) swiglu fusion."""

    def _fixed_inputs(self):
        # b=2, s=3, 2*half=8 (half=4). Values straddle a clamp of 1.0 so the
        # clamp branch is genuinely exercised (not a degenerate no-op).
        rng = np.random.RandomState(20240917)
        y = (rng.rand(2, 3, 8).astype(np.float32) * 6.0) - 3.0
        w = (rng.rand(2, 3, 1).astype(np.float32) * 1.5) + 0.25
        return y, w

    def test_forward_unclamped_matches_numpy(self):
        y_np, w_np = self._fixed_inputs()
        y = paddle.to_tensor(y_np)
        w = paddle.to_tensor(w_np)
        out = weighted_bias_swiglu_impl(y, None, w, clamp_value=None)
        ref = _np_swiglu_activation(y_np, weights_np=w_np, clamp=None)
        self.assertEqual(list(out.shape), [2, 3, 4])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-6)

    def test_forward_clamped_matches_numpy_and_differs_from_unclamped(self):
        y_np, w_np = self._fixed_inputs()
        clamp = 1.0
        # Guard: the fixture must actually exceed the clamp on both halves,
        # otherwise the clamp branch would be indistinguishable.
        self.assertTrue((y_np[..., :4] > clamp).any())
        self.assertTrue((np.abs(y_np[..., 4:]) > clamp).any())

        y = paddle.to_tensor(y_np)
        w = paddle.to_tensor(w_np)
        out = weighted_bias_swiglu_impl(y, None, w, clamp_value=clamp)
        ref_clamped = _np_swiglu_activation(y_np, weights_np=w_np, clamp=clamp)
        ref_unclamped = _np_swiglu_activation(y_np, weights_np=w_np, clamp=None)
        np.testing.assert_allclose(
            out.numpy(), ref_clamped, rtol=1e-5, atol=1e-6
        )
        # clamp_value is consumed: the clamped result is not the un-clamped one.
        self.assertFalse(
            np.allclose(ref_clamped, ref_unclamped, rtol=1e-5, atol=1e-6)
        )

    def test_backward_unclamped_matches_autograd_reference(self):
        y_np, w_np = self._fixed_inputs()
        upstream = np.random.RandomState(7).rand(2, 3, 4).astype(np.float32)
        up = paddle.to_tensor(upstream)

        y = paddle.to_tensor(y_np)
        y.stop_gradient = False
        w = paddle.to_tensor(w_np)
        w.stop_gradient = False
        out = weighted_bias_swiglu_impl(y, None, w, clamp_value=None)
        out.backward(up)

        y_ref = paddle.to_tensor(y_np)
        y_ref.stop_gradient = False
        w_ref = paddle.to_tensor(w_np)
        w_ref.stop_gradient = False
        ref = _paddle_naive_swiglu(y_ref, weights=w_ref, clamp=None)
        ref.backward(up)

        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(w.grad)
        self.assertGreater(np.abs(y_ref.grad.numpy()).max(), 1e-3)
        np.testing.assert_allclose(
            y.grad.numpy(), y_ref.grad.numpy(), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            w.grad.numpy(), w_ref.grad.numpy(), rtol=1e-4, atol=1e-5
        )

    def test_backward_clamped_zeros_grad_where_clamped(self):
        y_np, w_np = self._fixed_inputs()
        clamp = 1.0
        upstream = np.random.RandomState(11).rand(2, 3, 4).astype(np.float32)
        up = paddle.to_tensor(upstream)

        y = paddle.to_tensor(y_np)
        y.stop_gradient = False
        w = paddle.to_tensor(w_np)
        w.stop_gradient = False
        out = weighted_bias_swiglu_impl(y, None, w, clamp_value=clamp)
        out.backward(up)

        y_ref = paddle.to_tensor(y_np)
        y_ref.stop_gradient = False
        w_ref = paddle.to_tensor(w_np)
        w_ref.stop_gradient = False
        ref = _paddle_naive_swiglu(y_ref, weights=w_ref, clamp=clamp)
        ref.backward(up)

        np.testing.assert_allclose(
            y.grad.numpy(), y_ref.grad.numpy(), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            w.grad.numpy(), w_ref.grad.numpy(), rtol=1e-4, atol=1e-5
        )
        # Gate gradient must be exactly zero where the gate was clamped.
        gate_clamped = y_np[..., :4] > clamp
        gate_grad = y.grad.numpy()[..., :4]
        np.testing.assert_array_equal(
            gate_grad[gate_clamped], np.zeros(int(gate_clamped.sum()))
        )

    def test_bias_not_supported_raises(self):
        y_np, w_np = self._fixed_inputs()
        y = paddle.to_tensor(y_np)
        w = paddle.to_tensor(w_np)
        bias = paddle.zeros([8], dtype="float32")
        with self.assertRaises(NotImplementedError):
            weighted_bias_swiglu_impl(y, bias, w, clamp_value=None)

    def test_rank4_input_rejected(self):
        y = paddle.zeros([2, 2, 2, 8], dtype="float32")
        w = paddle.ones([2, 2, 2, 1], dtype="float32")
        with self.assertRaises(AssertionError):
            weighted_bias_swiglu_impl(y, None, w, clamp_value=None)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestBiasSwigluImpl(unittest.TestCase):
    """bias_swiglu_impl: biased (clamped) swiglu fusion, forward and backward.

    A full-shape bias is used (rather than a broadcast per-feature bias) so the
    independent reference and the returned bias gradient share an unambiguous
    shape; the swiglu math and the clamp branch are what is under test here.
    """

    def _fixed_inputs(self):
        rng = np.random.RandomState(1234)
        y = (rng.rand(6, 8).astype(np.float32) * 6.0) - 3.0
        bias = (rng.rand(6, 8).astype(np.float32) * 2.0) - 1.0
        return y, bias

    def test_forward_unclamped_matches_numpy(self):
        y_np, b_np = self._fixed_inputs()
        out = bias_swiglu_impl(
            paddle.to_tensor(y_np), paddle.to_tensor(b_np), clamp_value=None
        )
        ref = _np_swiglu_activation(y_np, bias_np=b_np, clamp=None)
        self.assertEqual(list(out.shape), [6, 4])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-6)

    def test_forward_clamped_matches_numpy_and_differs(self):
        y_np, b_np = self._fixed_inputs()
        clamp = 1.0
        z = y_np + b_np
        self.assertTrue((z[..., :4] > clamp).any())
        self.assertTrue((np.abs(z[..., 4:]) > clamp).any())
        out = bias_swiglu_impl(
            paddle.to_tensor(y_np), paddle.to_tensor(b_np), clamp_value=clamp
        )
        ref_clamped = _np_swiglu_activation(y_np, bias_np=b_np, clamp=clamp)
        ref_unclamped = _np_swiglu_activation(y_np, bias_np=b_np, clamp=None)
        np.testing.assert_allclose(
            out.numpy(), ref_clamped, rtol=1e-5, atol=1e-6
        )
        self.assertFalse(
            np.allclose(ref_clamped, ref_unclamped, rtol=1e-5, atol=1e-6)
        )

    def _run_backward(self, clamp):
        y_np, b_np = self._fixed_inputs()
        upstream = np.random.RandomState(99).rand(6, 4).astype(np.float32)
        up = paddle.to_tensor(upstream)

        y = paddle.to_tensor(y_np)
        y.stop_gradient = False
        b = paddle.to_tensor(b_np)
        b.stop_gradient = False
        out = bias_swiglu_impl(y, b, clamp_value=clamp)
        out.backward(up)

        y_ref = paddle.to_tensor(y_np)
        y_ref.stop_gradient = False
        b_ref = paddle.to_tensor(b_np)
        b_ref.stop_gradient = False
        ref = _paddle_naive_swiglu(y_ref, bias=b_ref, clamp=clamp)
        ref.backward(up)

        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(b.grad)
        self.assertGreater(np.abs(y_ref.grad.numpy()).max(), 1e-3)
        np.testing.assert_allclose(
            y.grad.numpy(), y_ref.grad.numpy(), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            b.grad.numpy(), b_ref.grad.numpy(), rtol=1e-4, atol=1e-5
        )

    def test_backward_unclamped_matches_autograd_reference(self):
        self._run_backward(clamp=None)

    def test_backward_clamped_matches_autograd_reference(self):
        self._run_backward(clamp=1.0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMLPActivationDispatch(unittest.TestCase):
    """MLP.forward routes silu + gated + per_token_scale through the weighted
    swiglu fusion, applies the scale, and honours activation_func_clamp_value.

    The projection layers are genuine collaborators (not under test): the
    reference reuses the real up_gate/down projections to obtain the pre- and
    post-activation anchors, but computes the activation itself from the
    independent numpy formulation, so a wrong branch / dropped clamp / dropped
    scale in the dispatch is caught while a projection bug is out of scope.
    """

    def _hidden_and_scale(self):
        paddle.seed(2024)
        # Large hidden magnitude so the fused projection output straddles the
        # clamp bound, keeping the clamp branch non-degenerate.
        hidden = paddle.randn([2, 3, 64], dtype="float32") * 4.0
        scale_np = (
            np.random.RandomState(5).rand(2, 3).astype(np.float32) * 1.5 + 0.5
        )
        return hidden, paddle.to_tensor(scale_np), scale_np

    def _reference(self, mlp, hidden, scale_np, clamp):
        inter, up_bias = mlp.up_gate_proj(hidden)
        self.assertIsNone(up_bias)  # use_bias=False -> no bias into fusion
        act_np = _np_swiglu_activation(
            inter.numpy(), weights_np=scale_np[..., None], clamp=clamp
        )
        act_t = paddle.to_tensor(act_np.astype("float32"))
        ref_out, down_bias = mlp.down_proj(act_t)
        self.assertIsNone(down_bias)
        return inter.numpy(), ref_out.numpy()

    def test_weighted_dispatch_unclamped(self):
        mlp = _build_mlp(clamp_value=None)
        mlp.eval()
        hidden, scale, scale_np = self._hidden_and_scale()
        _, ref_out = self._reference(mlp, hidden, scale_np, clamp=None)

        out, out_bias = mlp(hidden, per_token_scale=scale)
        self.assertIsNone(out_bias)
        self.assertEqual(list(out.shape), [2, 3, 64])
        np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-4, atol=1e-5)

    def test_weighted_dispatch_consumes_scale(self):
        mlp = _build_mlp(clamp_value=None)
        mlp.eval()
        hidden, scale, scale_np = self._hidden_and_scale()
        _, ref_out = self._reference(mlp, hidden, scale_np, clamp=None)
        # A reference that ignores the scale (uses 1.0) must NOT match, proving
        # per_token_scale actually flows into the activation.
        ones = np.ones_like(scale_np)
        _, ref_ignoring_scale = self._reference(mlp, hidden, ones, clamp=None)

        out, _ = mlp(hidden, per_token_scale=scale)
        np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-4, atol=1e-5)
        self.assertFalse(
            np.allclose(ref_out, ref_ignoring_scale, rtol=1e-4, atol=1e-5)
        )

    def test_weighted_dispatch_clamped_changes_output(self):
        clamp = 0.05
        mlp = _build_mlp(clamp_value=clamp)
        mlp.eval()
        hidden, scale, scale_np = self._hidden_and_scale()
        inter_np, ref_clamped = self._reference(mlp, hidden, scale_np, clamp)
        # Guard: the projection output must exceed the clamp so the branch is
        # genuinely exercised.
        self.assertTrue((inter_np[..., :128] > clamp).any())
        self.assertTrue((np.abs(inter_np[..., 128:]) > clamp).any())
        _, ref_unclamped = self._reference(mlp, hidden, scale_np, clamp=None)

        out, _ = mlp(hidden, per_token_scale=scale)
        np.testing.assert_allclose(
            out.numpy(), ref_clamped, rtol=1e-4, atol=1e-5
        )
        # activation_func_clamp_value reaches the fusion: clamped != unclamped.
        self.assertFalse(
            np.allclose(ref_clamped, ref_unclamped, rtol=1e-4, atol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
