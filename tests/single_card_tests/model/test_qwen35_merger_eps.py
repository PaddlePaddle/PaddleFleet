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

"""The Qwen3.5 patch merger's ``LayerNorm`` epsilon is pinned to 1e-6.

The reference merger hardcodes ``nn.LayerNorm(..., eps=1e-6)``. The merger spec
used to pass no ``eps`` at all, so it silently inherited
``WrappedPaddleNorm``'s ``eps=1e-5`` default — the model-wide ``rms_norm_eps``
was not consulted either, so declaring it on the config did not help.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import paddle
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.models.qwen3_5.layer_specs import get_qwen3_5_vision_spec
from paddlefleet.models.qwen3_vl.patch_merger import Qwen3VLVisionPathMerger
from paddlefleet.transformer.transformer_config import TransformerConfig

MERGER_EPS = 1e-6
WRAPPED_NORM_DEFAULT_EPS = 1e-5


@dataclass
class _VisionConfig(TransformerConfig):
    """Trimmed-down stand-in for ``Qwen3_5VisionProvider``."""

    patch_size: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    num_position_embeddings: int = 64
    hidden_size: int = 32
    out_hidden_size: int = 64
    in_channels: int = 3
    spatial_merge_size: int = 2
    spatial_patch_size: int = 16
    temporal_patch_size: int = 2
    intermediate_size: int = 64
    num_attention_heads: int = 4
    num_hidden_layers: int = 2
    gated_linear_unit: bool = False
    bias_activation_fusion: bool = False
    normalization: str = "LayerNorm"
    rms_norm_eps: float = 1e-6
    model_version: str = "qwen3_5"


def _make_config(**overrides):
    config = _VisionConfig(**overrides)
    config.hidden_act = F.gelu
    return config


def _merger_spec(config) -> LayerSpec:
    return get_qwen3_5_vision_spec(config).sublayers_spec.merger


def _build_merger(config):
    return build_spec_layer(_merger_spec(config))


class TestMergerNormSpec(unittest.TestCase):
    """The epsilon is carried explicitly on the spec, not left to a default."""

    def test_norm_is_a_layerspec_with_explicit_eps(self):
        norm_spec = _merger_spec(_make_config()).sublayers_spec.norm
        self.assertIsInstance(
            norm_spec,
            LayerSpec,
            "the merger norm must be a LayerSpec so eps can be attached",
        )
        self.assertEqual(norm_spec.extra_kwargs.get("eps"), MERGER_EPS)

    def test_eps_is_not_the_wrapped_norm_default(self):
        norm_spec = _merger_spec(_make_config()).sublayers_spec.norm
        self.assertNotEqual(
            norm_spec.extra_kwargs["eps"], WRAPPED_NORM_DEFAULT_EPS
        )


class TestMergerNormBuilt(unittest.TestCase):
    """The built layer really uses 1e-6."""

    def test_built_norm_eps(self):
        merger = _build_merger(_make_config())
        self.assertIsInstance(merger, Qwen3VLVisionPathMerger)
        self.assertAlmostEqual(
            float(merger.norm.variance_epsilon), MERGER_EPS, places=12
        )

    def test_eps_does_not_track_rms_norm_eps(self):
        """A different model-wide epsilon must not move the merger's."""
        for rms_norm_eps in (1e-5, 1e-4, 1e-8):
            with self.subTest(rms_norm_eps=rms_norm_eps):
                merger = _build_merger(_make_config(rms_norm_eps=rms_norm_eps))
                self.assertAlmostEqual(
                    float(merger.norm.variance_epsilon), MERGER_EPS, places=12
                )

    def test_norm_is_layernorm_not_rmsnorm(self):
        """Qwen3.5's vision tower normalizes with LayerNorm."""
        merger = _build_merger(_make_config())
        self.assertTrue(hasattr(merger.norm, "bias"))

    def test_rmsnorm_config_still_gets_the_same_eps(self):
        """The ``rms_norm=`` switch selects the class; eps stays pinned."""
        merger = _build_merger(_make_config(normalization="RMSNorm"))
        self.assertAlmostEqual(
            float(merger.norm.variance_epsilon), MERGER_EPS, places=12
        )


class TestMergerNormNumerics(unittest.TestCase):
    def test_forward_matches_reference_layernorm_at_1e_6(self):
        config = _make_config()
        merger = _build_merger(config)
        x = paddle.randn([16, config.hidden_size])

        normed = merger.norm(x)
        expected = F.layer_norm(
            x,
            normalized_shape=[config.hidden_size],
            weight=merger.norm.weight,
            bias=merger.norm.bias,
            epsilon=MERGER_EPS,
        )
        self.assertTrue(paddle.allclose(normed, expected, atol=1e-6).item())

    def test_forward_differs_from_1e_5_on_low_variance_input(self):
        """1e-6 vs 1e-5 is observable when the variance is small."""
        config = _make_config()
        merger = _build_merger(config)
        x = paddle.randn([16, config.hidden_size]) * 1e-3

        normed = merger.norm(x)
        at_1e_5 = F.layer_norm(
            x,
            normalized_shape=[config.hidden_size],
            weight=merger.norm.weight,
            bias=merger.norm.bias,
            epsilon=WRAPPED_NORM_DEFAULT_EPS,
        )
        self.assertFalse(
            paddle.allclose(normed, at_1e_5, atol=1e-4).item(),
            "the merger epsilon is indistinguishable from the old default",
        )


class TestMergerActivation(unittest.TestCase):
    """Cross-check: the merger MLP runs exact gelu, not ``config.hidden_act``.

    ``Qwen3VLVisionPathMerger`` declares ``hidden_act=F.gelu`` in its
    ``MLPSublayersSpec``; the reference merger uses exact GELU while the vision
    config carries ``gelu_pytorch_tanh``.
    """

    def test_merger_mlp_uses_spec_gelu(self):
        config = _make_config()
        config.hidden_act = F.silu  # stand-in for a differing config activation
        merger = _build_merger(config)
        self.assertIs(merger.mlp.hidden_act, F.gelu)
        self.assertIs(config.hidden_act, F.silu)


if __name__ == "__main__":
    unittest.main()
