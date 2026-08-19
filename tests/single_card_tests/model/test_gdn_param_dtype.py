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

"""``GatedDeltaNet`` state-parameter dtype (``A_log`` / ``dt_bias``).

The reference implementation declares both as plain parameters, so the official
checkpoint stores them in the model dtype and they are promoted to FP32 only at
the ``softplus`` / ``exp`` boundary. Creating FP32 leaves instead changes the
parameter, gradient and optimizer-state dtype relative to that checkpoint, so
``use_accuracy_compatible`` honors ``params_dtype`` while the default path keeps
the historical FP32 leaves.
"""

from __future__ import annotations

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlefleet.transformer.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

H, B, S = 64, 2, 16
A_INIT_RANGE = (1, 16)


class NoBiasLinear(nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias_attr=False)

    def forward(self, x):
        return self.linear(x), None

    def backward_dw(self):
        pass


class SimpleRMSNorm(nn.Layer):
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[normalized_shape],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = eps

    def forward(self, x):
        x_float = x.astype(paddle.float32)
        rms = paddle.rsqrt(
            x_float.pow(2).mean(axis=-1, keepdim=True) + self.eps
        )
        return (x_float * rms * self.weight.astype(paddle.float32)).astype(
            x.dtype
        )


class _FakeGroup:
    ranks = [0]
    nranks = 1
    rank = 0


class _FakePGCollection:
    def __init__(self):
        self.tp = _FakeGroup()


def _make_gdn(*, use_accuracy_compatible=False, params_dtype=None):
    kwargs = {}
    if params_dtype is not None:
        kwargs["params_dtype"] = params_dtype
    config = TransformerConfig(
        hidden_size=H,
        num_attention_heads=4,
        num_hidden_layers=2,
        hidden_act=F.silu,
        rms_norm_eps=1e-5,
        normalization="RMSNorm",
        deterministic_mode=True,
        use_accuracy_compatible=use_accuracy_compatible,
        **kwargs,
    )
    spec = GatedDeltaNetSublayersSpec(
        in_proj=NoBiasLinear,
        out_norm=SimpleRMSNorm,
        out_proj=NoBiasLinear,
    )
    return GatedDeltaNet(
        config=config,
        sublayers_spec=spec,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=A_INIT_RANGE,
        pg_collection=_FakePGCollection(),
        conv_kernel_dim=4,
        key_head_dim=16,
        value_head_dim=16,
        num_key_heads=4,
        num_value_heads=4,
    )


class TestStateParamDtypeDefault(unittest.TestCase):
    """Default path: FP32 leaves regardless of ``params_dtype``."""

    def test_fp32_config_gives_fp32_leaves(self):
        gdn = _make_gdn()
        self.assertEqual(gdn.A_log.dtype, paddle.float32)
        self.assertEqual(gdn.dt_bias.dtype, paddle.float32)

    def test_bf16_config_still_gives_fp32_leaves(self):
        gdn = _make_gdn(params_dtype="bfloat16")
        self.assertEqual(gdn.A_log.dtype, paddle.float32)
        self.assertEqual(gdn.dt_bias.dtype, paddle.float32)

    def test_accuracy_flag_alone_follows_fp32_default_params_dtype(self):
        """``use_accuracy_compatible`` selects ``params_dtype``, which is FP32
        by default — so the flag on its own must not change anything."""
        gdn = _make_gdn(use_accuracy_compatible=True)
        self.assertEqual(gdn.A_log.dtype, paddle.float32)
        self.assertEqual(gdn.dt_bias.dtype, paddle.float32)


class TestStateParamDtypeAccuracyCompatible(unittest.TestCase):
    """``use_accuracy_compatible`` + BF16 ``params_dtype`` gives BF16 leaves."""

    def setUp(self):
        self.gdn = _make_gdn(
            use_accuracy_compatible=True, params_dtype="bfloat16"
        )

    def test_leaves_are_bf16(self):
        self.assertEqual(self.gdn.A_log.dtype, paddle.bfloat16)
        self.assertEqual(self.gdn.dt_bias.dtype, paddle.bfloat16)

    def test_dt_bias_initialized_to_ones(self):
        np.testing.assert_allclose(
            self.gdn.dt_bias.astype("float32").numpy(),
            np.ones(self.gdn.num_v_heads_local_tp),
        )

    def test_a_log_init_assigned_in_leaf_dtype(self):
        """``paddle.log(A)`` is FP32; without the cast the assign would either
        fail or silently widen the leaf back to FP32."""
        a_log = self.gdn.A_log.astype("float32").numpy()
        self.assertTrue(np.isfinite(a_log).all())
        lo, hi = np.log(A_INIT_RANGE[0]), np.log(A_INIT_RANGE[1])
        # BF16 has ~3 decimal digits, so allow a rounding margin on the bounds.
        self.assertTrue((a_log >= lo - 1e-2).all())
        self.assertTrue((a_log <= hi + 1e-2).all())
        self.assertEqual(self.gdn.A_log.dtype, paddle.bfloat16)

    def test_a_log_values_are_bf16_rounded(self):
        """The leaf really is stored at BF16 precision, not FP32 in disguise."""
        a_log = self.gdn.A_log.astype("float32")
        round_tripped = a_log.astype("bfloat16").astype("float32")
        np.testing.assert_array_equal(a_log.numpy(), round_tripped.numpy())

    def test_gradients_are_bf16(self):
        x = paddle.randn([B, S, H])
        x.stop_gradient = False
        out, _ = self.gdn(x, attention_mask=None)
        out.sum().backward()
        for name, param in (
            ("A_log", self.gdn.A_log),
            ("dt_bias", self.gdn.dt_bias),
        ):
            self.assertIsNotNone(param.grad, f"{name} received no gradient")
            self.assertEqual(param.grad.dtype, paddle.bfloat16, name)

    def test_forward_promotes_at_the_computation_boundary(self):
        """BF16 leaves must not produce inf/nan: ``A_log`` is exponentiated and
        ``dt_bias`` goes through ``softplus``, both in FP32."""
        x = paddle.randn([B, S, H])
        out, _ = self.gdn(x, attention_mask=None)
        self.assertTrue(paddle.isfinite(out.astype("float32")).all().item())

    def test_reset_parameters_is_idempotent_in_bf16(self):
        before = self.gdn.A_log.astype("float32").numpy().copy()
        self.gdn.reset_parameters()
        after = self.gdn.A_log.astype("float32").numpy()
        self.assertEqual(self.gdn.A_log.dtype, paddle.bfloat16)
        # Re-initialization redraws from the same uniform range.
        self.assertTrue(np.isfinite(after).all())
        self.assertEqual(before.shape, after.shape)


class TestStateParamDtypeParity(unittest.TestCase):
    def test_both_modes_build_and_run(self):
        """Whichever leaf dtype is selected, the layer stays trainable."""
        for kwargs in (
            {},
            {"use_accuracy_compatible": True, "params_dtype": "bfloat16"},
        ):
            with self.subTest(**kwargs):
                gdn = _make_gdn(**kwargs)
                x = paddle.randn([B, S, H])
                x.stop_gradient = False
                out, _ = gdn(x, attention_mask=None)
                out.sum().backward()
                self.assertEqual(list(out.shape), [B, S, H])
                self.assertTrue(
                    paddle.isfinite(x.grad.astype("float32")).all().item()
                )


if __name__ == "__main__":
    unittest.main()
