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

"""``MLPSublayersSpec.hidden_act`` must win over ``config.hidden_act``.

Modules such as the Qwen3-VL / Qwen3.5 patch merger and the Kimi-K2.5 tpool
merge declare their own activation in the spec precisely because it differs from
the model-wide ``config.hidden_act``. That declaration used to be dropped on the
floor, so the merger silently ran the text model's activation.
"""

from __future__ import annotations

import dataclasses
import unittest

import paddle
import paddle.nn.functional as F

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.transformer_config import TransformerConfig

S, BATCH, HIDDEN = 8, 4, 12


def _make_config(**overrides):
    kwargs = {
        "num_hidden_layers": 2,
        "hidden_size": HIDDEN,
        "intermediate_size": 48,
        "num_attention_heads": 4,
        # Keep the plain activation path: the fused branches only accept
        # gelu/swiglu and would mask which activation object is in play.
        "gated_linear_unit": False,
        "bias_activation_fusion": False,
        "use_bias": False,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _base_spec(config) -> MLPSublayersSpec:
    return get_gpt_layer_local_spec(config).sublayers_spec.mlp.sublayers_spec


def _make_mlp(config, hidden_act="unset"):
    spec = _base_spec(config)
    if hidden_act != "unset":
        spec = dataclasses.replace(spec, hidden_act=hidden_act)
    return MLP(config, spec)


class _ActivationOwner:
    """Provides a *bound method* activation, which must be unwrapped."""

    def act(self, x):
        return F.gelu(x)


class TestSpecHiddenActPrecedence(unittest.TestCase):
    def test_spec_none_falls_back_to_config(self):
        config = _make_config(hidden_act=F.silu)
        mlp = _make_mlp(config, hidden_act=None)
        self.assertIs(mlp.hidden_act, F.silu)

    def test_spec_absent_falls_back_to_config(self):
        """``MLPSublayersSpec.hidden_act`` defaults to ``None``."""
        config = _make_config(hidden_act=F.silu)
        self.assertIsNone(_base_spec(config).hidden_act)
        self.assertIs(_make_mlp(config).hidden_act, F.silu)

    def test_spec_overrides_config(self):
        config = _make_config(hidden_act=F.silu)
        mlp = _make_mlp(config, hidden_act=F.gelu)
        self.assertIs(mlp.hidden_act, F.gelu)
        self.assertIs(config.hidden_act, F.silu)

    def test_spec_bound_method_is_unwrapped(self):
        config = _make_config(hidden_act=F.silu)
        owner = _ActivationOwner()
        mlp = _make_mlp(config, hidden_act=owner.act)
        self.assertIs(mlp.hidden_act, _ActivationOwner.act)
        self.assertFalse(hasattr(mlp.hidden_act, "__self__"))

    def test_config_bound_method_still_unwrapped(self):
        """The pre-existing unwrap must keep applying to the fallback value."""
        config = _make_config()
        config.hidden_act = _ActivationOwner().act
        mlp = _make_mlp(config, hidden_act=None)
        self.assertIs(mlp.hidden_act, _ActivationOwner.act)


class TestSpecHiddenActForward(unittest.TestCase):
    """The spec activation must be the one actually applied in ``forward``."""

    def _run(self, mlp):
        x = paddle.ones((S, BATCH, HIDDEN))
        out, _ = mlp(x)
        return out

    def test_forward_uses_spec_activation(self):
        calls = []

        def tracking_act(x):
            calls.append(x.shape)
            return F.gelu(x)

        config = _make_config(hidden_act=F.silu)
        self._run(_make_mlp(config, hidden_act=tracking_act))
        self.assertEqual(len(calls), 1, "spec activation was not invoked")

    def test_forward_does_not_use_config_activation_when_spec_set(self):
        calls = []

        def tracking_config_act(x):
            calls.append(x.shape)
            return F.silu(x)

        config = _make_config()
        config.hidden_act = tracking_config_act
        self._run(_make_mlp(config, hidden_act=F.gelu))
        self.assertEqual(calls, [], "config activation leaked into forward")

    def test_spec_activation_changes_the_numbers(self):
        """Same weights, different declared activation → different output."""
        config = _make_config(hidden_act=F.silu)
        silu_mlp = _make_mlp(config, hidden_act=None)
        gelu_mlp = _make_mlp(config, hidden_act=F.gelu)
        gelu_mlp.set_state_dict(silu_mlp.state_dict())

        silu_out = self._run(silu_mlp)
        gelu_out = self._run(gelu_mlp)
        self.assertFalse(
            paddle.allclose(silu_out, gelu_out, atol=1e-6).item(),
            "spec activation had no effect on the output",
        )

    def test_spec_gelu_matches_config_gelu(self):
        """Declaring gelu in the spec == declaring it on the config."""
        via_config = _make_mlp(_make_config(hidden_act=F.gelu), hidden_act=None)
        via_spec = _make_mlp(_make_config(hidden_act=F.silu), hidden_act=F.gelu)
        via_spec.set_state_dict(via_config.state_dict())
        self.assertTrue(
            paddle.allclose(
                self._run(via_config), self._run(via_spec), atol=1e-6
            ).item()
        )


class TestGatedPathKnownGap(unittest.TestCase):
    """Documents that the GLU branch still reads ``config.hidden_act``.

    ``MLP.forward``'s inner ``glu()`` closure calls
    ``self.config.hidden_act(x_glu)`` rather than ``self.hidden_act``, so a
    spec-level activation is ignored when ``gated_linear_unit=True``. No current
    caller hits that combination — the patch merger and the tpool merge are both
    non-GLU — so this is left as-is rather than silently changing GLU models. If
    that call is ever switched to ``self.hidden_act``, this test should fail and
    be deleted.
    """

    def test_glu_branch_ignores_spec_activation(self):
        calls = []

        def tracking_act(x):
            calls.append(x.shape)
            return F.gelu(x)

        config = _make_config(gated_linear_unit=True, hidden_act=F.silu)
        mlp = _make_mlp(config, hidden_act=tracking_act)
        # The constructor honors the spec ...
        self.assertIs(mlp.hidden_act, tracking_act)
        mlp(paddle.ones((S, BATCH, HIDDEN)))
        # ... but the GLU closure does not consult it.
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
