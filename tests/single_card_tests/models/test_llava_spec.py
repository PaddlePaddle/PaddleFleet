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

"""CPU-only unit tests for ``paddlefleet.models.multimodal.llava_spec``.

Scope / disjointness. This file owns the assembly contract of
``decoder_model_with_local_default_spec`` and the exact field/keyword contract
of the two dataclasses it feeds (``TransformerLayerSublayersSpec`` and
``SelfAttentionSublayersSpec``), keyed to the symbols the coverage source
imports (the spec factory, ``get_mlp_layer_spec``, ``TransformerLayer`` /
``TransformerLayerSublayersSpec``, ``LNImpl`` -> ``FusedLayerNorm`` and
``get_bias_dropout_add``). Sibling files in a ``test_llava_spec_*`` family
would own heavier paths; this file stays disjoint.

Honesty / environment. The module under test imports Paddle at load time, and
Paddle is not installed here, so every test skips with an explicit reason
rather than reporting a pass. The dataclass-contract tests import only the
stable dataclasses (independent of the broken factory import) and would run on
a single-card box that has Paddle.

Real production bugs (documented, never patched here):

* ``llava_spec.py:21`` imports ``get_mlp_layer_spec`` from
  ``paddlefleet.models.gpt.gpt_layer_specs``; that module only defines
  ``get_mlp_layer_spec_for_backend``. The import fails at module load, so the
  factory is unreachable. Tests that need correct end-to-end behaviour assert
  the correct outcome and are marked ``expectedFailure``.
* ``llava_spec.py:55`` passes ``self_attention=`` to
  ``TransformerLayerSublayersSpec`` whose real field is ``self_attn``.
* ``llava_spec.py:59,61`` pass ``linear_qkv=`` / ``linear_proj=`` to
  ``SelfAttentionSublayersSpec`` whose real fields are ``qkv_proj`` / ``o_proj``.
* ``llava_spec.py:43`` declares ``qk_layernorm`` but never consumes it, so the
  flag is silently ignored (contrast ``get_gpt_layer_local_spec`` which maps a
  qk-norm flag onto ``q_norm`` / ``k_norm``).
"""

import dataclasses
import importlib
import os
import sys
import unittest
from unittest import mock

_REPO_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

_LLAVA_SPEC_MODULE = "paddlefleet.models.multimodal.llava_spec"

# Stable dependencies that llava_spec itself relies on, minus the broken import.
# Only genuine missing-dependency failures (ImportError/ModuleNotFoundError,
# e.g. Paddle absent) are recorded as a skip reason; other exception types are
# left to propagate as real failures instead of being mislabelled "no dep".
_DEPS_ERR = None
try:
    import paddle  # noqa: F401
    from paddle.distributed.fleet.meta_parallel import LayerSpec  # noqa: F401

    from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
    from paddlefleet.fusions.fused_layer_norm import FusedLayerNorm
    from paddlefleet.models.gpt import gpt_layer_specs
    from paddlefleet.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from paddlefleet.transformer.attention import (
        SelfAttention,
        SelfAttentionSublayersSpec,
    )
    from paddlefleet.transformer.dot_product_attention import (
        DotProductAttention,
    )
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerSublayersSpec,
    )
except (ImportError, ModuleNotFoundError) as exc:
    _DEPS_ERR = exc

_DEPS_OK = _DEPS_ERR is None
_DEPS_SKIP = (
    "" if _DEPS_OK else f"paddle / paddlefleet not importable: {_DEPS_ERR!r}"
)


class TestGptLayerSpecsExportsMlpHelper(unittest.TestCase):
    """The factory depends on ``get_mlp_layer_spec`` existing in gpt specs."""

    @unittest.expectedFailure
    def test_get_mlp_layer_spec_symbol_available(self):
        # Correct expectation: llava_spec.py imports
        # ``get_mlp_layer_spec`` from this module, so the symbol must exist.
        # It does not (only ``get_mlp_layer_spec_for_backend`` is defined),
        # which is the real import-time bug -> expectedFailure.
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        self.assertTrue(
            hasattr(gpt_layer_specs, "get_mlp_layer_spec"),
            "gpt_layer_specs is missing get_mlp_layer_spec "
            "(only get_mlp_layer_spec_for_backend exists); "
            "llava_spec.py:21 import is broken.",
        )

    def test_backend_helper_is_the_only_variant_present(self):
        # Independent, passing companion: pin the symbol that *does* exist so a
        # future rename is caught and the skip reason above stays accurate.
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        self.assertTrue(
            hasattr(gpt_layer_specs, "get_mlp_layer_spec_for_backend")
        )
        self.assertFalse(hasattr(gpt_layer_specs, "get_mlp_layer_spec"))


class TestLlavaSpecModuleImportable(unittest.TestCase):
    """The module must import cleanly for the factory to be usable at all."""

    @unittest.expectedFailure
    def test_module_imports_without_error(self):
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        # Correct expectation: importing the module succeeds. Currently raises
        # ImportError on the missing get_mlp_layer_spec symbol.
        importlib.import_module(_LLAVA_SPEC_MODULE)


class TestTransformerLayerSublayersSpecContract(unittest.TestCase):
    """Field contract the factory feeds into TransformerLayerSublayersSpec."""

    def test_field_names_match_expected_contract(self):
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        names = {
            f.name for f in dataclasses.fields(TransformerLayerSublayersSpec)
        }
        # Hand-derived from the dataclass definition: attention slot is
        # ``self_attn`` (NOT ``self_attention``, which llava_spec.py:55 uses).
        expected_subset = {
            "input_layernorm",
            "self_attn",
            "self_attn_bda",
            "post_attention_layernorm",
            "mlp",
            "mlp_bda",
        }
        self.assertTrue(expected_subset.issubset(names), names)
        self.assertNotIn("self_attention", names)

    def test_rejects_self_attention_keyword(self):
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        # The exact keyword llava_spec.py:55 passes is not a valid field, so a
        # dataclass construction with it raises TypeError. This proves the
        # factory, if its import were fixed, would still fail here.
        with self.assertRaises(TypeError):
            TransformerLayerSublayersSpec(self_attention=object())

    def test_accepts_documented_self_attn_keyword(self):
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        sentinel = object()
        spec = TransformerLayerSublayersSpec(self_attn=sentinel)
        self.assertIs(spec.self_attn, sentinel)


class TestSelfAttentionSublayersSpecContract(unittest.TestCase):
    """Field contract the factory feeds into SelfAttentionSublayersSpec."""

    def test_field_names_match_expected_contract(self):
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        names = {f.name for f in dataclasses.fields(SelfAttentionSublayersSpec)}
        self.assertEqual(
            names,
            {
                "qkv_proj",
                "core_attention",
                "o_proj",
                "q_norm",
                "k_norm",
                "gate_proj",
            },
        )
        # llava_spec.py:59,61 use these non-existent field names.
        self.assertNotIn("linear_qkv", names)
        self.assertNotIn("linear_proj", names)

    def test_rejects_linear_qkv_and_linear_proj_keywords(self):
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        with self.assertRaises(TypeError):
            SelfAttentionSublayersSpec(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                linear_proj=RowParallelLinear,
            )

    def test_accepts_documented_projection_keywords(self):
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        spec = SelfAttentionSublayersSpec(
            qkv_proj=ColumnParallelLinear,
            core_attention=DotProductAttention,
            o_proj=RowParallelLinear,
        )
        self.assertIs(spec.qkv_proj, ColumnParallelLinear)
        self.assertIs(spec.core_attention, DotProductAttention)
        self.assertIs(spec.o_proj, RowParallelLinear)


class TestDecoderSpecEndToEnd(unittest.TestCase):
    """End-to-end assembly of decoder_model_with_local_default_spec.

    All methods import the (currently broken) module and are marked
    expectedFailure; each encodes the fully-correct structure so that fixing
    the three assembly bugs turns them green (an unexpected success is then
    flagged, prompting a test update).
    """

    def _load_factory(self):
        module = importlib.import_module(_LLAVA_SPEC_MODULE)
        return module

    @unittest.expectedFailure
    def test_default_spec_full_structure(self):
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        module = self._load_factory()
        marker = object()
        with mock.patch.object(
            module, "get_mlp_layer_spec", return_value=marker
        ) as mlp_helper:
            spec = module.decoder_model_with_local_default_spec()

        # get_mlp_layer_spec is the only collaborator isolated here; its
        # return value must be consumed as the mlp slot, and it must be called
        # with the hand-derived default arguments.
        mlp_helper.assert_called_once_with(
            use_te=False, num_experts=None, moe_expert_fusion=False
        )

        self.assertIs(spec.module, TransformerLayer)
        sub = spec.submodules
        self.assertIsInstance(sub, TransformerLayerSublayersSpec)
        self.assertIs(sub.input_layernorm, FusedLayerNorm)
        self.assertIs(sub.post_attention_layernorm, FusedLayerNorm)
        self.assertIs(sub.self_attn_bda, get_bias_dropout_add)
        self.assertIs(sub.mlp_bda, get_bias_dropout_add)
        self.assertIs(sub.mlp, marker)

        attn = sub.self_attn
        self.assertIs(attn.module, SelfAttention)
        self.assertEqual(attn.params, {"attn_mask_type": AttnMaskType.causal})
        self.assertIs(attn.submodules.qkv_proj, ColumnParallelLinear)
        self.assertIs(attn.submodules.core_attention, DotProductAttention)
        self.assertIs(attn.submodules.o_proj, RowParallelLinear)

    @unittest.expectedFailure
    def test_moe_expert_arguments_forwarded_and_consumed(self):
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        module = self._load_factory()
        marker = object()
        with mock.patch.object(
            module, "get_mlp_layer_spec", return_value=marker
        ) as mlp_helper:
            spec = module.decoder_model_with_local_default_spec(
                num_experts=8, moe_expert_fusion=True
            )
        mlp_helper.assert_called_once_with(
            use_te=False, num_experts=8, moe_expert_fusion=True
        )
        self.assertIs(spec.submodules.mlp, marker)

    @unittest.expectedFailure
    def test_qk_layernorm_flag_is_actually_applied(self):
        if not _DEPS_OK:
            self.skipTest(_DEPS_SKIP)
        module = self._load_factory()
        with mock.patch.object(
            module, "get_mlp_layer_spec", return_value=object()
        ):
            spec = module.decoder_model_with_local_default_spec(
                qk_layernorm=True
            )
        # Correct expectation: qk_layernorm=True installs a q_norm on the
        # attention sublayers (as get_gpt_layer_local_spec does). The factory
        # never consumes the flag, so q_norm stays None -> expectedFailure.
        self.assertIsNotNone(spec.submodules.self_attn.submodules.q_norm)


if __name__ == "__main__":
    unittest.main()
