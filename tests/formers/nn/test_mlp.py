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

"""Behaviour tests for paddlefleet.nn.mlp.MLP (SwiGLU-style gated FFN).

Scope: model layer, no-card (CPU). We keep the real MLP forward/gating math
in the verification chain and compare against independently hand-derived NumPy
references computed from tiny, sign-varied weights. tensor_model_parallel_size
is fixed to 1 so Linear.create resolves to the CPU-capable ``paddle.nn.Linear``;
the fused ``paddle.nn.functional.swiglu`` kernel numeric path is skipped with a
recorded reason because it may require a GPU backend.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.nn.mlp import MLP
from paddlefleet.transformers import LlamaConfig


def _silu(z):
    # SiLU / swish, computed independently of paddle.
    return z / (1.0 + np.exp(-z))


def _relu(z):
    return np.maximum(z, 0.0)


class TestMLPForward(unittest.TestCase):
    """Numeric behaviour of the real MLP forward on CPU."""

    def setUp(self):
        paddle.set_device("cpu")
        paddle.seed(0)

        # Tiny dimensions so the whole forward is hand-derivable.
        self.hidden = 3
        self.inter = 4

        # Fixed, sign-varied input; last-dim gating is what matters.
        self.x = np.array(
            [[0.5, -1.0, 2.0], [-0.3, 1.5, 0.2]], dtype=np.float32
        )

        # Distinguishable projection weights (paddle Linear weight is
        # [in_features, out_features]; y = x @ W + b).
        self.Wg = np.array(
            [
                [0.10, -0.20, 0.30, 0.05],
                [0.20, 0.10, -0.15, 0.25],
                [-0.05, 0.30, 0.10, -0.20],
            ],
            dtype=np.float32,
        )
        self.Wu = np.array(
            [
                [0.20, 0.10, -0.10, 0.15],
                [-0.25, 0.20, 0.05, 0.10],
                [0.30, -0.10, 0.20, 0.05],
            ],
            dtype=np.float32,
        )
        self.Wd = np.array(
            [
                [0.10, 0.20, -0.10],
                [0.15, -0.20, 0.10],
                [-0.05, 0.25, 0.20],
                [0.20, 0.10, -0.15],
            ],
            dtype=np.float32,
        )

    def _make_config(
        self, hidden_act="silu", mlp_bias=False, fuse_swiglu=False
    ):
        # Build from the real production config entry, single-rank (CPU nn.Linear).
        config = LlamaConfig()
        config.hidden_size = self.hidden
        config.intermediate_size = self.inter
        config.tensor_model_parallel_size = 1
        config.sequence_parallel = False
        config.mlp_bias = mlp_bias
        config.fuse_swiglu = fuse_swiglu
        config.hidden_act = hidden_act
        return config

    @staticmethod
    def _set_linear(layer, weight_np, bias_np=None):
        layer.weight.set_value(
            paddle.to_tensor(weight_np, dtype=layer.weight.dtype)
        )
        if bias_np is not None:
            layer.bias.set_value(
                paddle.to_tensor(bias_np, dtype=layer.bias.dtype)
            )

    def test_forward_silu_gating_matches_hand_derived(self):
        """gate/up SwiGLU: out = down((silu(x@Wg)) * (x@Wu))."""
        mlp = MLP(self._make_config(hidden_act="silu"))
        self._set_linear(mlp.gate_proj, self.Wg)
        self._set_linear(mlp.up_proj, self.Wu)
        self._set_linear(mlp.down_proj, self.Wd)

        gate = self.x @ self.Wg
        up = self.x @ self.Wu
        expected = (_silu(gate) * up) @ self.Wd

        out = mlp(paddle.to_tensor(self.x)).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_activation_is_actually_consumed(self):
        """hidden_act must drive gating: relu zeros negative gate entries.

        The gate pre-activation contains negatives, so relu and silu produce
        materially different outputs. We assert the relu-derived reference and
        also that it differs from the silu reference, proving act_fn is used
        rather than hard-coded.
        """
        mlp = MLP(self._make_config(hidden_act="relu"))
        self._set_linear(mlp.gate_proj, self.Wg)
        self._set_linear(mlp.up_proj, self.Wu)
        self._set_linear(mlp.down_proj, self.Wd)

        gate = self.x @ self.Wg
        up = self.x @ self.Wu
        self.assertTrue((gate < 0).any(), "fixture must exercise relu clamp")

        expected_relu = (_relu(gate) * up) @ self.Wd
        expected_silu = (_silu(gate) * up) @ self.Wd
        # The two activations must genuinely diverge on this fixture.
        self.assertGreater(np.abs(expected_relu - expected_silu).max(), 1e-3)

        out = mlp(paddle.to_tensor(self.x)).numpy()
        np.testing.assert_allclose(out, expected_relu, rtol=1e-5, atol=1e-6)

    def test_forward_with_bias_applied_at_every_projection(self):
        """mlp_bias=True: each Linear adds its bias in the real math path."""
        mlp = MLP(self._make_config(mlp_bias=True))
        self.assertTrue(mlp.has_bias)

        bg = np.array([0.01, -0.02, 0.03, -0.04], dtype=np.float32)
        bu = np.array([-0.05, 0.06, -0.07, 0.08], dtype=np.float32)
        bd = np.array([0.09, -0.10, 0.11], dtype=np.float32)
        self._set_linear(mlp.gate_proj, self.Wg, bg)
        self._set_linear(mlp.up_proj, self.Wu, bu)
        self._set_linear(mlp.down_proj, self.Wd, bd)

        gate = self.x @ self.Wg + bg
        up = self.x @ self.Wu + bu
        expected = (_silu(gate) * up) @ self.Wd + bd

        out = mlp(paddle.to_tensor(self.x)).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

        # Bias genuinely changes the result vs. the no-bias reference.
        no_bias = (_silu(self.x @ self.Wg) * (self.x @ self.Wu)) @ self.Wd
        self.assertGreater(np.abs(out - no_bias).max(), 1e-3)

    def test_fuse_up_gate_chunk_order(self):
        """fuse_up_gate: single proj chunked into (gate, up) on the last axis.

        The two halves use distinguishable weights, so a swapped gate/up split
        would fail. Verifies gate = first half, up = second half, then
        out = down(silu(gate) * up).
        """
        mlp = MLP(self._make_config(hidden_act="silu"), fuse_up_gate=True)
        self.assertTrue(mlp.fuse_up_gate)

        # up_gate_proj: [hidden, inter*2]; first inter cols -> gate, rest -> up.
        Wgu = np.concatenate([self.Wg, self.Wu], axis=-1).astype(np.float32)
        self.assertEqual(
            list(mlp.up_gate_proj.weight.shape), [self.hidden, self.inter * 2]
        )
        self._set_linear(mlp.up_gate_proj, Wgu)
        self._set_linear(mlp.down_proj, self.Wd)

        fused = self.x @ Wgu
        gate = fused[:, : self.inter]
        up = fused[:, self.inter :]
        expected = (_silu(gate) * up) @ self.Wd

        out = mlp(paddle.to_tensor(self.x)).numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

        # A swapped-half interpretation would give a different result: guard it.
        swapped = (_silu(up) * gate) @ self.Wd
        self.assertGreater(np.abs(expected - swapped).max(), 1e-3)

    @unittest.skip(
        "fuse_swiglu routes through paddle.nn.functional.swiglu, a fused "
        "kernel whose numerics we do not verify on CPU here; requires a "
        "single-card (GPU) run to confirm the fused path. The unfused gating "
        "math is covered by test_forward_silu_gating_matches_hand_derived."
    )
    def test_forward_fuse_swiglu_numeric(self):
        pass


if __name__ == "__main__":
    unittest.main()
