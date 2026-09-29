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
"""Behavior tests for ``paddlefleet.gpt_builders``.

Focus is the spec-function selection control flow:

* ``_get_transformer_layer_spec_func`` binds specific config fields to the
  ``get_gpt_layer_local_spec`` keyword arguments; in particular
  ``num_experts`` is fed from ``config.n_routed_experts`` (not a same-named
  field), so a field swap must be observable.
* ``gpt_builder`` picks ``get_gpt_decoder_layers_spec`` when
  ``config.n_routed_experts`` is truthy OR ``config.layer_types is not None``,
  and otherwise builds one local spec per hidden layer with a head offset.

Only genuine, non-tested leaf collaborators (``get_gpt_spec``,
``build_spec_layer``, ``get_gpt_layer_local_spec``,
``get_gpt_decoder_layers_spec``, ``_get_effective_mtp_layers``) are replaced,
each with a distinguishable response, so the branch logic, the per-layer loop
and ``_get_transformer_layer_spec_func`` all execute for real.
"""

import os
import sys
import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

# Mirror the repo layout so the package is importable when it is not pip
# installed (``src`` layout). Import errors are surfaced as an honest skip.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from paddlefleet import gpt_builders
    from paddlefleet.gpt_builders import (
        _get_transformer_layer_spec_func,
        gpt_builder,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not available in this env
    gpt_builders = None
    _get_transformer_layer_spec_func = None
    gpt_builder = None
    _IMPORT_ERROR = exc


def _make_config(**overrides):
    """A lightweight config carrying every attribute gpt_builder reads.

    Values are placeholders for the fields that only flow through the
    (replaced) ``get_gpt_spec``; the fields that drive control flow are set
    explicitly by each test via ``overrides``.
    """
    base = {
        "moe_token_dispatcher_type": None,
        "n_routed_experts": 0,
        "layer_types": None,
        "num_hidden_layers": 3,
        "num_empty_layers_add_in_head": 2,
        "num_empty_layers_add_in_tail": 0,
        "separate_mtp_headloss": False,
        "use_qk_norm": False,
        "multi_latent_attention": False,
        "normalization": "RMSNorm",
        "vocab_size": 1000,
        "tie_word_embeddings": False,
        "max_sequence_length": 128,
        "position_embedding_type": "rope",
        "rotary_percent": 1.0,
        "rope_theta": 10000.0,
        "swa_rope_theta": 10000.0,
        "rope_scaling": None,
        "parallel_output": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@unittest.skipUnless(
    gpt_builders is not None,
    f"requires paddle/paddlefleet: {_IMPORT_ERROR}",
)
class TestGetTransformerLayerSpecFunc(unittest.TestCase):
    """The partial must bind each config field to the correct keyword."""

    def test_maps_config_fields_to_partial_keywords(self):
        # Distinct sentinels: any field swap changes an identity check below.
        qk = object()
        experts = object()
        mla = object()
        norm = object()
        config = SimpleNamespace(
            use_qk_norm=qk,
            n_routed_experts=experts,
            multi_latent_attention=mla,
            normalization=norm,
        )

        func = _get_transformer_layer_spec_func(config)

        self.assertIsInstance(func, partial)
        self.assertIs(func.func, gpt_builders.get_gpt_layer_local_spec)
        self.assertEqual(func.args, ())
        # Hand-derived mapping; note num_experts <- config.n_routed_experts.
        self.assertEqual(
            set(func.keywords),
            {
                "config",
                "use_qk_norm",
                "num_experts",
                "multi_latent_attention",
                "normalization",
            },
        )
        self.assertIs(func.keywords["config"], config)
        self.assertIs(func.keywords["use_qk_norm"], qk)
        self.assertIs(func.keywords["num_experts"], experts)
        self.assertIs(func.keywords["multi_latent_attention"], mla)
        self.assertIs(func.keywords["normalization"], norm)


@unittest.skipUnless(
    gpt_builders is not None,
    f"requires paddle/paddlefleet: {_IMPORT_ERROR}",
)
class TestGptBuilderSpecSelection(unittest.TestCase):
    """gpt_builder must pick the right spec source for each config flag set."""

    def _run(self, config):
        built = object()

        def fake_local(**kwargs):
            # Distinguishable per-layer marker carrying the layer number.
            return ("local-spec", kwargs["layer_number"])

        def fake_decoder(cfg, **kwargs):
            return ("decoder-spec", cfg)

        with (
            mock.patch.object(
                gpt_builders, "_get_effective_mtp_layers", return_value=0
            ),
            mock.patch.object(
                gpt_builders, "get_gpt_layer_local_spec", side_effect=fake_local
            ) as m_local,
            mock.patch.object(
                gpt_builders,
                "get_gpt_decoder_layers_spec",
                side_effect=fake_decoder,
            ) as m_decoder,
            mock.patch.object(
                gpt_builders, "get_gpt_spec", return_value="gpt-spec"
            ) as m_spec,
            mock.patch.object(
                gpt_builders, "build_spec_layer", return_value=built
            ) as m_build,
        ):
            result = gpt_builder(config, loss_fn=object())

        return SimpleNamespace(
            result=result,
            built=built,
            m_local=m_local,
            m_decoder=m_decoder,
            m_spec=m_spec,
            m_build=m_build,
        )

    def test_dense_branch_builds_local_spec_per_layer_with_head_offset(self):
        config = _make_config(
            n_routed_experts=0,
            layer_types=None,
            num_hidden_layers=3,
            num_empty_layers_add_in_head=2,
        )

        run = self._run(config)

        # Dense path must not consult the decoder-block builder at all.
        run.m_decoder.assert_not_called()
        # One local spec per hidden layer, offset by the head empty layers:
        # real_layer_number = layer_number + num_empty_layers_add_in_head.
        self.assertEqual(run.m_local.call_count, 3)
        for call, expected_layer in zip(run.m_local.call_args_list, [2, 3, 4]):
            kwargs = call.kwargs
            self.assertEqual(kwargs["layer_number"], expected_layer)
            self.assertIs(kwargs["config"], config)
            self.assertIs(kwargs["num_experts"], config.n_routed_experts)
            self.assertIs(kwargs["use_qk_norm"], config.use_qk_norm)
            self.assertIs(
                kwargs["multi_latent_attention"],
                config.multi_latent_attention,
            )
            self.assertEqual(kwargs["normalization"], config.normalization)
        # The assembled per-layer specs (in order) reach get_gpt_spec.
        forwarded = run.m_spec.call_args.kwargs["transformer_layers_spec"]
        self.assertEqual(
            forwarded,
            [("local-spec", 2), ("local-spec", 3), ("local-spec", 4)],
        )
        self.assertIs(run.result, run.built)

    def test_routed_experts_selects_decoder_block_spec(self):
        config = _make_config(n_routed_experts=8, layer_types=None)

        run = self._run(config)

        # Truthy n_routed_experts takes the decoder-block branch; the dense
        # per-layer local-spec path must be skipped entirely.
        run.m_local.assert_not_called()
        run.m_decoder.assert_called_once()
        self.assertIs(run.m_decoder.call_args.args[0], config)
        self.assertEqual(
            run.m_decoder.call_args.kwargs["normalization"],
            config.normalization,
        )
        forwarded = run.m_spec.call_args.kwargs["transformer_layers_spec"]
        self.assertEqual(forwarded, ("decoder-spec", config))
        self.assertIs(run.result, run.built)

    def test_layer_types_forces_decoder_branch_without_experts(self):
        # n_routed_experts is falsy, so only the ``layer_types is not None``
        # operand of the OR can select the decoder branch here.
        config = _make_config(
            n_routed_experts=0,
            layer_types=["full_attention", "linear_attention"],
        )

        run = self._run(config)

        run.m_local.assert_not_called()
        run.m_decoder.assert_called_once()
        self.assertIs(run.m_decoder.call_args.args[0], config)
        forwarded = run.m_spec.call_args.kwargs["transformer_layers_spec"]
        self.assertEqual(forwarded, ("decoder-spec", config))
        self.assertIs(run.result, run.built)


if __name__ == "__main__":
    unittest.main()
