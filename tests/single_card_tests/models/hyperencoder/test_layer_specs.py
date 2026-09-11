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

"""Unit tests for the HyperEncoder layer-spec builders."""

import os
import types
import unittest
from contextlib import contextmanager
from unittest import mock

import paddle  # noqa: F401

from paddlefleet.models.hyperencoder import layer_specs
from paddlefleet.models.hyperencoder.layer_specs import (
    _is_moe_layer,
    get_hyperencoder_block_spec,
    get_hyperencoder_layer_specs,
)
from paddlefleet.models.hyperencoder.norm import (
    HyperEncoderRMSNorm,
)


@contextmanager
def _env(**kv):
    saved = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cfg(**over):
    base = {
        "num_hidden_layers": 2,
        "moe_layer_freq": [0, 1],
        "n_routed_experts": 4,
        "moe_expert_fusion": True,
        "_attn_implementation": "eager",
    }
    base.update(over)
    return types.SimpleNamespace(**base)


def _make_spec(with_core_attention=True):
    """Fabricate a spec object shaped like get_gpt_layer_local_spec's return."""
    sub = types.SimpleNamespace(
        input_layernorm=None, post_attention_layernorm=None
    )
    if with_core_attention:
        attn_sub = types.SimpleNamespace(core_attention=object())
        sub.self_attn = types.SimpleNamespace(sublayers_spec=attn_sub)
    else:
        sub.self_attn = types.SimpleNamespace(
            sublayers_spec=types.SimpleNamespace()
        )
    return types.SimpleNamespace(sublayers_spec=sub)


class TestIsMoeLayer(unittest.TestCase):
    def test_list(self):
        cfg = types.SimpleNamespace(moe_layer_freq=[0, 1, 0])
        self.assertFalse(_is_moe_layer(cfg, 0))
        self.assertTrue(_is_moe_layer(cfg, 1))

    def test_tuple(self):
        cfg = types.SimpleNamespace(moe_layer_freq=(1, 0))
        self.assertTrue(_is_moe_layer(cfg, 0))

    def test_int_raises(self):
        cfg = types.SimpleNamespace(moe_layer_freq=2)
        with self.assertRaises(TypeError):
            _is_moe_layer(cfg, 0)


class TestLayerSpecsEager(unittest.TestCase):
    def test_missing_eager_raises(self):
        with _env(HYPERBODY_ENCODER_ATTN_BACKEND=None):
            cfg = _cfg(_attn_implementation="sdpa")
            with self.assertRaises(ValueError):
                get_hyperencoder_layer_specs(cfg)

    def test_happy_path_replaces_norms(self):
        with _env(HYPERBODY_ENCODER_ATTN_BACKEND=None):
            cfg = _cfg()
            with mock.patch.object(
                layer_specs,
                "get_gpt_layer_local_spec",
                side_effect=lambda **kw: _make_spec(),
            ) as build:
                specs = get_hyperencoder_layer_specs(cfg)
            self.assertEqual(len(specs), 2)
            for s in specs:
                self.assertIs(
                    s.sublayers_spec.input_layernorm, HyperEncoderRMSNorm
                )
                self.assertIs(
                    s.sublayers_spec.post_attention_layernorm,
                    HyperEncoderRMSNorm,
                )
            # MoE layer index 1 should receive num_experts; index 0 should not.
            num_experts = [
                c.kwargs["num_experts"] for c in build.call_args_list
            ]
            self.assertEqual(num_experts, [None, 4])


class TestLayerSpecsTriton(unittest.TestCase):
    def test_triton_installs_core_attention(self):
        from paddlefleet.transformer.prefix_lm_triton_core import (
            PrefixLMTritonCore,
        )

        with _env(HYPERBODY_ENCODER_ATTN_BACKEND="triton"):
            cfg = _cfg()
            with mock.patch.object(
                layer_specs,
                "get_gpt_layer_local_spec",
                side_effect=lambda **kw: _make_spec(),
            ):
                specs = get_hyperencoder_layer_specs(cfg)
            for s in specs:
                self.assertIs(
                    s.sublayers_spec.self_attn.sublayers_spec.core_attention,
                    PrefixLMTritonCore,
                )

    def test_triton_missing_core_attention_raises(self):
        with _env(HYPERBODY_ENCODER_ATTN_BACKEND="triton"):
            cfg = _cfg()
            with (
                mock.patch.object(
                    layer_specs,
                    "get_gpt_layer_local_spec",
                    side_effect=lambda **kw: _make_spec(
                        with_core_attention=False
                    ),
                ),
                self.assertRaises(RuntimeError),
            ):
                get_hyperencoder_layer_specs(cfg)


class TestBlockSpec(unittest.TestCase):
    def test_block_spec_wraps_layers_and_final_norm(self):
        with _env(HYPERBODY_ENCODER_ATTN_BACKEND=None):
            cfg = _cfg()
            with mock.patch.object(
                layer_specs,
                "get_gpt_layer_local_spec",
                side_effect=lambda **kw: _make_spec(),
            ):
                block = get_hyperencoder_block_spec(cfg)
            self.assertEqual(len(block.layer_specs), 2)
            self.assertIs(block.layer_norm, HyperEncoderRMSNorm)


if __name__ == "__main__":
    unittest.main()
