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

import os
import sys
import unittest

# Bootstrap: make the repo `src/` importable when the test is run standalone
# (CI normally puts it on PYTHONPATH; this keeps the file runnable directly).
_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

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

    PADDLE_AVAILABLE = True
    _IMPORT_ERROR = ""
except (
    ImportError,
    ModuleNotFoundError,
) as exc:  # CPU-only env may lack paddle
    PADDLE_AVAILABLE = False
    _IMPORT_ERROR = repr(exc)


_SKIP_REASON = (
    "paddle / paddlefleet not importable in this CPU-only environment: "
    + _IMPORT_ERROR
)


def _np_silu(x):
    # SiLU / swish: x * sigmoid(x). Implemented independently in float64 so it
    # cannot share a bug with the production activation path.
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class _MLPTestBase(unittest.TestCase):
    """Shared fixtures for the swiglu MLP behavior tests.

    All references are derived by hand from the raw projection weights
    (``up_gate_proj.weight`` / ``down_proj.weight``) using numpy matmuls and an
    independent SiLU, never by calling ``MLP.forward`` or its sub-linears. The
    weights are overwritten with known, non-degenerate random content so a
    zero-initialized (``perform_initialization=False``) build cannot make the
    comparison pass trivially.
    """

    def setUp(self):
        # Force CPU and restore whatever device was selected before.
        self._orig_device = paddle.device.get_device()
        self.addCleanup(paddle.device.set_device, self._orig_device)
        paddle.device.set_device("cpu")
        paddle.seed(20240517)

    def _make_config(self, **overrides):
        defaults = {
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "use_bias": False,
            "init_method": init_method_normal(0.02),
            "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
            "gated_linear_unit": True,
            "bias_activation_fusion": False,
            "activation_func_clamp_value": None,
            "glu_linear_offset": 0.0,
            "hidden_act": F.silu,
        }
        defaults.update(overrides)
        return TransformerConfig(**defaults)

    def _make_spec(self, config):
        spec = get_gpt_layer_local_spec(config)
        return spec.sublayers_spec.mlp.sublayers_spec

    def _build_mlp(self, config, **mlp_kwargs):
        mlp = MLP(
            config=config,
            sublayers_spec=self._make_spec(config),
            **mlp_kwargs,
        )
        # Assign known, distinguishable weights so the reference is non-trivial.
        with paddle.no_grad():
            wu = paddle.randn(mlp.up_gate_proj.weight.shape) * 0.3
            wd = paddle.randn(mlp.down_proj.weight.shape) * 0.3
            mlp.up_gate_proj.weight.set_value(wu)
            mlp.down_proj.weight.set_value(wd)
        # Sanity: no tensor-parallel bias sneaking in for use_bias=False configs.
        self.assertGreater(
            float(paddle.abs(mlp.up_gate_proj.weight).max()), 0.0
        )
        self.assertGreater(float(paddle.abs(mlp.down_proj.weight).max()), 0.0)
        return mlp

    def _ref_swiglu(self, x_np, wu_np, wd_np, offset=0.0, per_token_scale=None):
        """Independent numpy reference for the gated (swiglu) MLP forward."""
        inter = np.asarray(x_np, dtype=np.float64) @ np.asarray(
            wu_np, dtype=np.float64
        )
        half = inter.shape[-1] // 2
        gate = inter[..., :half]
        linear = inter[..., half:]
        act = _np_silu(gate) * (linear + offset)
        if per_token_scale is not None:
            act = act * np.asarray(per_token_scale, dtype=np.float64)[..., None]
        return act @ np.asarray(wd_np, dtype=np.float64)


class TestSwigluForward(_MLPTestBase):
    def test_swiglu_forward_matches_numpy_reference(self):
        config = self._make_config()
        mlp = self._build_mlp(config)
        x = paddle.randn([2, 4, 16])

        out, out_bias = mlp(x)

        ref = self._ref_swiglu(
            x.numpy(),
            mlp.up_gate_proj.weight.numpy(),
            mlp.down_proj.weight.numpy(),
        )
        self.assertEqual(list(out.shape), [2, 4, 16])
        self.assertIsNone(out_bias)  # use_bias=False => no returned bias
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), ref, rtol=1e-4, atol=1e-5
        )

    def test_gate_half_drives_silu_not_linear_half(self):
        # Guards the chunk order: SiLU is applied to the FIRST half (gate) and
        # the SECOND half is the linear term. If the halves were swapped the
        # output would change, so we assert the correct-order reference matches
        # and the swapped-order reference does NOT.
        config = self._make_config()
        mlp = self._build_mlp(config)
        x = paddle.randn([2, 4, 16])
        out = mlp(x)[0].numpy().astype(np.float64)

        wu = mlp.up_gate_proj.weight.numpy()
        wd = mlp.down_proj.weight.numpy()
        correct = self._ref_swiglu(x.numpy(), wu, wd)

        inter = x.numpy().astype(np.float64) @ wu.astype(np.float64)
        half = inter.shape[-1] // 2
        swapped = (_np_silu(inter[..., half:]) * inter[..., :half]) @ wd.astype(
            np.float64
        )

        np.testing.assert_allclose(out, correct, rtol=1e-4, atol=1e-5)
        self.assertGreater(
            np.abs(correct - swapped).max(),
            1e-3,
            "gate/linear halves are distinguishable for this fixture",
        )

    def test_glu_linear_offset_added_to_linear_half_only(self):
        offset = 0.75
        config = self._make_config(glu_linear_offset=offset)
        mlp = self._build_mlp(config)
        x = paddle.randn([2, 4, 16])

        out = mlp(x)[0].numpy().astype(np.float64)
        wu = mlp.up_gate_proj.weight.numpy()
        wd = mlp.down_proj.weight.numpy()

        ref_offset = self._ref_swiglu(x.numpy(), wu, wd, offset=offset)
        ref_no_offset = self._ref_swiglu(x.numpy(), wu, wd, offset=0.0)

        np.testing.assert_allclose(out, ref_offset, rtol=1e-4, atol=1e-5)
        # The offset must actually change the result (added to linear half).
        self.assertGreater(np.abs(ref_offset - ref_no_offset).max(), 1e-3)

    def test_per_token_scale_scales_activation(self):
        config = self._make_config()
        mlp = self._build_mlp(config)
        x = paddle.randn([2, 4, 16])
        per_token_scale = paddle.randn([2, 4])

        out = (
            mlp(x, per_token_scale=per_token_scale)[0]
            .numpy()
            .astype(np.float64)
        )
        wu = mlp.up_gate_proj.weight.numpy()
        wd = mlp.down_proj.weight.numpy()

        ref = self._ref_swiglu(
            x.numpy(), wu, wd, per_token_scale=per_token_scale.numpy()
        )
        ref_unscaled = self._ref_swiglu(x.numpy(), wu, wd)

        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
        # Per-token scale must actually be consumed (broadcast over hidden dim).
        self.assertGreater(np.abs(ref - ref_unscaled).max(), 1e-3)


class TestNonGatedForward(_MLPTestBase):
    def test_non_gated_applies_activation_directly(self):
        # gated_linear_unit=False: up_gate_proj is NOT doubled and the
        # activation is applied to the whole projection (no chunk / no linear
        # half).
        config = self._make_config(gated_linear_unit=False, hidden_act=F.silu)
        mlp = self._build_mlp(config)
        x = paddle.randn([2, 4, 16])

        out = mlp(x)[0].numpy().astype(np.float64)
        wu = mlp.up_gate_proj.weight.numpy()
        wd = mlp.down_proj.weight.numpy()
        # up_gate_proj must have width == intermediate_size (32), not doubled.
        self.assertEqual(list(mlp.up_gate_proj.weight.shape), [16, 32])

        inter = x.numpy().astype(np.float64) @ wu.astype(np.float64)
        ref = _np_silu(inter) @ wd.astype(np.float64)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


class TestCustomSizes(_MLPTestBase):
    def test_custom_input_size_forward(self):
        config = self._make_config()
        mlp = self._build_mlp(config, input_size=8)
        self.assertEqual(mlp.up_gate_proj.weight.shape[0], 8)
        x = paddle.randn([2, 4, 8])

        out = mlp(x)[0].numpy().astype(np.float64)
        ref = self._ref_swiglu(
            x.numpy(),
            mlp.up_gate_proj.weight.numpy(),
            mlp.down_proj.weight.numpy(),
        )
        self.assertEqual(list(out.shape), [2, 4, 16])
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)

    def test_custom_hidden_size_output(self):
        config = self._make_config()
        mlp = self._build_mlp(config, hidden_size=8)
        self.assertEqual(mlp.down_proj.weight.shape[1], 8)
        x = paddle.randn([2, 4, 16])

        out = mlp(x)[0].numpy().astype(np.float64)
        ref = self._ref_swiglu(
            x.numpy(),
            mlp.up_gate_proj.weight.numpy(),
            mlp.down_proj.weight.numpy(),
        )
        self.assertEqual(list(out.shape), [2, 4, 8])
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


class TestSplitGateUpConsumers(_MLPTestBase):
    def test_hidden_states_up_splits_gate_and_linear_sources(self):
        # forward(hidden_states, hidden_states_up=...) must take the SiLU'd
        # gate half from ``hidden_states`` and the linear half from
        # ``hidden_states_up``.
        config = self._make_config()
        mlp = self._build_mlp(config)
        hs = paddle.randn([2, 4, 16])
        hs_up = paddle.randn([2, 4, 16])

        out = mlp(hs, hidden_states_up=hs_up)[0].numpy().astype(np.float64)

        wu = mlp.up_gate_proj.weight.numpy().astype(np.float64)
        wd = mlp.down_proj.weight.numpy().astype(np.float64)
        inter_gate = hs.numpy().astype(np.float64) @ wu
        inter_up = hs_up.numpy().astype(np.float64) @ wu
        half = inter_gate.shape[-1] // 2
        act = _np_silu(inter_gate[..., :half]) * inter_up[..., half:]
        ref = act @ wd
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)

        # Distinguishing: swapping the two inputs must change the output,
        # otherwise the wiring of gate/linear sources would be unverified.
        swapped = mlp(hs_up, hidden_states_up=hs)[0].numpy().astype(np.float64)
        self.assertGreater(np.abs(out - swapped).max(), 1e-3)


class TestForwardBackward(_MLPTestBase):
    def test_gradients_match_independent_reference(self):
        config = self._make_config()
        mlp = self._build_mlp(config)

        x = paddle.randn([2, 4, 16])
        x.stop_gradient = False
        upstream = paddle.randn([2, 4, 16])

        out, _ = mlp(x)
        paddle.autograd.backward([out], [upstream])

        # Independent paddle reference built from primitive ops on cloned,
        # differentiable copies of the same weights and input.
        xr = x.detach().clone()
        xr.stop_gradient = False
        wu = mlp.up_gate_proj.weight.detach().clone()
        wu.stop_gradient = False
        wd = mlp.down_proj.weight.detach().clone()
        wd.stop_gradient = False

        inter = paddle.matmul(xr, wu)
        gate, linear = paddle.chunk(inter, 2, axis=-1)
        act = F.silu(gate) * linear
        ref_out = paddle.matmul(act, wd)
        paddle.autograd.backward([ref_out], [upstream])

        # Forward parity first.
        np.testing.assert_allclose(
            out.numpy().astype(np.float64),
            ref_out.numpy().astype(np.float64),
            rtol=1e-4,
            atol=1e-5,
        )

        # Every gradient in the contract must exist, be non-trivial, and match
        # scale-sensitively (allclose, not cosine/norm).
        pairs = [
            (x.grad, xr.grad),
            (mlp.up_gate_proj.weight.grad, wu.grad),
            (mlp.down_proj.weight.grad, wd.grad),
        ]
        for actual, expected in pairs:
            self.assertIsNotNone(actual)
            self.assertIsNotNone(expected)
            expected_np = expected.numpy().astype(np.float64)
            self.assertGreater(np.abs(expected_np).max(), 1e-4)
            np.testing.assert_allclose(
                actual.numpy().astype(np.float64),
                expected_np,
                rtol=1e-3,
                atol=1e-5,
            )


class TestConstructionValidation(_MLPTestBase):
    def test_expert_without_intermediate_size_raises(self):
        config = self._make_config()
        with self.assertRaises(ValueError):
            MLP(
                config=config,
                sublayers_spec=self._make_spec(config),
                is_expert=True,
            )

    def test_non_expert_without_config_intermediate_size_raises(self):
        config = self._make_config()
        spec = self._make_spec(config)
        # Non-expert path falls back to config.intermediate_size; when that is
        # also None the constructor must raise (mlp.py: intermediate_size guard).
        config.intermediate_size = None
        with self.assertRaises(ValueError):
            MLP(config=config, sublayers_spec=spec, is_expert=False)


class TestBackwardDwOrchestration(_MLPTestBase):
    def test_backward_dw_flushes_both_projections_once(self):
        # MLP.backward_dw must flush the deferred weight-grad of BOTH
        # projections exactly once, down_proj before up_gate_proj
        # (mlp.py MLP.backward_dw). The default local-spec projections are
        # plain Column/RowParallelLinear that do not implement backward_dw
        # (that hook only exists on the dw-overlap linear variants), so we
        # install real recorder methods on the two collaborators and drive the
        # genuine MLP.backward_dw body. Recording call order + count is the
        # full observable contract of this argument-less orchestration method.
        config = self._make_config()
        mlp = self._build_mlp(config)

        calls = []
        mlp.down_proj.backward_dw = lambda: calls.append("down")
        mlp.up_gate_proj.backward_dw = lambda: calls.append("up")

        mlp.backward_dw()

        # Each collaborator flushed exactly once, in the documented order.
        self.assertEqual(calls, ["down", "up"])


if __name__ == "__main__":
    unittest.main()
