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

"""Behavior tests for the SFT workflow entry module.

Module under test: ``paddlefleet.cli.train.sft.workflow``. In the repository
module map this belongs to the "Trainer 训练引擎" layer -- a workflow assembles
the model/config/data and drives the Trainer. Two pure, CPU-runnable decision
functions are exercised here with hand-derived expectations:

* ``freeze_param_except_mtp(model, config)`` -- iterates ``model.state_dict()``,
  parses the layer index out of each parameter name, and sets
  ``param.stop_gradient``. The Multi-Token-Prediction (MTP) layers are the
  contiguous index range ``[num_hidden_layers, num_hidden_layers +
  mtp_num_layers)``; parameters inside that range are UN-frozen
  (``stop_gradient=False``) and every other parameter -- including ones whose
  name carries no ``model.layers.<idx>`` component at all -- is frozen
  (``stop_gradient=True``). The expected freeze/unfreeze map below is derived by
  hand from that intent, never by re-running the production classifier.

* ``create_peft_model(model_args, ...)`` -- when ``model_args.lora`` is false
  the model must be returned untouched (same object identity, no LoRA wrapper).
  The LoRA-enabled branch builds a real ``LoRAModel`` around a live Paddle
  model and belongs to a single-card test; it is intentionally not covered here.

Real, minimal stand-in objects (not MagicMock) back the model/config/args so the
genuine attribute reads and writes run: a MagicMock would answer ``hasattr``
truthy for any name and its ``.items()`` / ``.stop_gradient`` would not reflect
real assignment, masking regressions.

These tests run on CPU. Importing the production module pulls the ``paddlefleet``
package, which imports Paddle at import time; the local environment has no
Paddle, so the import raises ``ImportError`` and every test skips (recorded, not
silently passed). No production code is modified by this file.
"""

import unittest

try:
    import paddle  # noqa: F401

    from paddlefleet.cli.train.sft.workflow import (
        create_peft_model,
        freeze_param_except_mtp,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # Paddle backend / paddlefleet not installed.
    paddle = None
    create_peft_model = None
    freeze_param_except_mtp = None
    _IMPORT_ERROR = exc


class _Param:
    """A real parameter stand-in exposing only ``stop_gradient``.

    ``freeze_param_except_mtp`` writes this attribute directly, so a plain
    object with a mutable ``stop_gradient`` records the production write
    faithfully.
    """

    def __init__(self, stop_gradient):
        self.stop_gradient = stop_gradient


class _StateModel:
    """Minimal model whose ``state_dict()`` returns a real name->param dict."""

    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class _Config:
    def __init__(self, num_hidden_layers, mtp_num_layers):
        self.num_hidden_layers = num_hidden_layers
        self.mtp_num_layers = mtp_num_layers


class _SkipIfNoPaddle(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                "paddlefleet.cli.train.sft.workflow import failed "
                "(Paddle unavailable in this environment): "
                "{!r}".format(_IMPORT_ERROR)
            )


class TestFreezeParamExceptMtp(_SkipIfNoPaddle):
    def test_freeze_classifies_by_mtp_layer_range(self):
        # num_hidden_layers=4, mtp_num_layers=2  =>  MTP layer indices {4, 5}.
        # Expected freeze state hand-derived from that range:
        #   * indices 4,5           -> MTP     -> stop_gradient False (unfrozen)
        #   * indices 0,3,6         -> non-MTP -> stop_gradient True  (frozen)
        #   * names with no layer idx (embed / lm_head) -> None match
        #                                     -> non-MTP -> stop_gradient True
        # Initial values are chosen so that EVERY parameter must change state;
        # a no-op implementation, an "always freeze", or an "always unfreeze"
        # implementation each fails on at least one entry.
        config = _Config(num_hidden_layers=4, mtp_num_layers=2)
        params = {
            "model.embed_tokens.weight": _Param(stop_gradient=False),
            "model.layers.0.self_attn.q_proj.weight": _Param(
                stop_gradient=False
            ),
            "model.layers.3.mlp.down_proj.weight": _Param(stop_gradient=False),
            "model.layers.4.self_attn.k_proj.weight": _Param(
                stop_gradient=True
            ),
            "model.layers.5.mlp.gate.weight": _Param(stop_gradient=True),
            "model.layers.6.self_attn.v_proj.weight": _Param(
                stop_gradient=False
            ),
            "lm_head.weight": _Param(stop_gradient=False),
        }
        expected_stop_gradient = {
            "model.embed_tokens.weight": True,
            "model.layers.0.self_attn.q_proj.weight": True,
            "model.layers.3.mlp.down_proj.weight": True,
            "model.layers.4.self_attn.k_proj.weight": False,
            "model.layers.5.mlp.gate.weight": False,
            "model.layers.6.self_attn.v_proj.weight": True,
            "lm_head.weight": True,
        }

        freeze_param_except_mtp(_StateModel(params), config)

        actual = {name: p.stop_gradient for name, p in params.items()}
        self.assertEqual(actual, expected_stop_gradient)

    def test_mtp_range_boundary_shifts_with_config(self):
        # num_hidden_layers=6, mtp_num_layers=1  =>  MTP layer index {6} only.
        # Layer 5 (last base layer) frozen, layer 6 (sole MTP) unfrozen,
        # layer 7 (past the MTP window) frozen. All start frozen=False so the
        # only observable change is the single unfreeze at index 6.
        config = _Config(num_hidden_layers=6, mtp_num_layers=1)
        params = {
            "model.layers.5.mlp.up_proj.weight": _Param(stop_gradient=False),
            "model.layers.6.mtp.embed.weight": _Param(stop_gradient=False),
            "model.layers.7.mtp.extra.weight": _Param(stop_gradient=False),
        }

        freeze_param_except_mtp(_StateModel(params), config)

        self.assertTrue(
            params["model.layers.5.mlp.up_proj.weight"].stop_gradient
        )
        self.assertFalse(
            params["model.layers.6.mtp.embed.weight"].stop_gradient
        )
        self.assertTrue(params["model.layers.7.mtp.extra.weight"].stop_gradient)

    def test_zero_mtp_layers_freezes_everything(self):
        # mtp_num_layers=0  =>  empty MTP range  =>  every parameter frozen,
        # including a name whose index equals num_hidden_layers (would be the
        # first MTP layer only if the window were non-empty).
        config = _Config(num_hidden_layers=2, mtp_num_layers=0)
        params = {
            "model.layers.0.self_attn.q_proj.weight": _Param(
                stop_gradient=False
            ),
            "model.layers.1.mlp.gate.weight": _Param(stop_gradient=False),
            "model.layers.2.mtp.embed.weight": _Param(stop_gradient=False),
        }

        freeze_param_except_mtp(_StateModel(params), config)

        self.assertTrue(all(p.stop_gradient for p in params.values()))


class _ModelArgs:
    def __init__(self, lora):
        self.lora = lora


class _MarkerModel:
    """A distinguishable model object used to prove pass-through identity."""

    def __init__(self):
        self.marker = "original-unwrapped-model"


class TestCreatePeftModel(_SkipIfNoPaddle):
    def test_returns_input_model_untouched_when_lora_disabled(self):
        # With LoRA disabled the production function must short-circuit and hand
        # back the very same model object -- no LoRAModel wrapper. Identity
        # (assertIs) plus an intact marker attribute proves it was neither
        # replaced nor wrapped. The LoRA-enabled branch needs a live Paddle
        # model with real parameters and is left to a single-card test.
        model = _MarkerModel()
        result = create_peft_model(
            _ModelArgs(lora=False),
            training_args=object(),
            dtype="bfloat16",
            model=model,
        )
        self.assertIs(result, model)
        self.assertEqual(result.marker, "original-unwrapped-model")


if __name__ == "__main__":
    unittest.main()
