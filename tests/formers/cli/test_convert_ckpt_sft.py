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

"""Behavior tests for the DeepSeek checkpoint -> HF SFT weight converter.

Module under test:
``paddlefleet.cli.train.deepseek_v3_pretrain.utils.convert_ckpt_to_sft``.

This belongs to the "Checkpoint and weight management" layer: the converter
rewrites Paddle parameter *names* into Hugging Face names, decides which
tensors must be transposed, and splits fused gate/up projections into two
owned halves. The load-bearing contract is *key mapping, ownership and tensor
content*, not shape alone -- a mapping that swapped gate/up, dropped the layer
or expert index, or transposed the wrong tensors would still preserve shapes.

What these tests verify with independent, hand-derived oracles:

* ``paddle_name_to_hf_names`` produces the exact HF name list for every branch
  (embedding / norm / lm_head, the ``.norm_weight`` / ``router`` /
  ``input_layernorm`` drop filters, the ``custom_name_map`` attention entries,
  MoE experts, shared experts, dense MLP, and the fused ``gate_up_fused_proj``
  v2 forms), plus the ``deepseek_v2`` -> ``model`` fallback. Layer and expert
  indices are varied so a mapping that ignored them would fail.
* The ``_handle_*`` helpers return the owned key list or ``None`` for
  non-matching inputs.
* ``_is_need_transpose`` selects exactly the documented suffixes, including the
  subtle split between bare ``mlp.w1`` (transposed) and an expert
  ``...w1.weight`` (not transposed).
* The compiled regexes match/reject the intended strings and capture indices.
* ``prepare_tensor`` (real ``torch`` tensors, CPU) maps the key, applies the
  transpose decision, splits a fused weight into gate=first-half /
  up=second-half with distinguishable content, and accounts byte size. Inputs
  share shape but differ in content so an ownership/half swap is rejected.

A real production defect is flagged via ``expectedFailure`` (no code edited):
the ``custom_name_map`` entry ``self_attn.input_layernorm.weight`` is
unreachable because an earlier ``if "input_layernorm" in paddle_name`` guard
returns ``[]`` first, so that weight is silently dropped instead of renamed.

These tests run on CPU. The real package import is attempted first; because the
``paddlefleet`` package ``__init__`` imports Paddle, on a Paddle-less no-card
host that raises ImportError and we load the *same* production source file
directly from disk (full package import therefore unverified here). If the
module's own ``safetensors.torch`` dependency is missing, tests skip. No
production code is modified.
"""

import importlib
import importlib.util
import os
import unittest

import numpy as np

_PROD_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "src",
        "paddlefleet",
        "cli",
        "train",
        "deepseek_v3_pretrain",
        "utils",
        "convert_ckpt_to_sft.py",
    )
)


def _load_production_module():
    """Load the real converter module; prefer package import, fall back to file.

    Returns ``(module, error)``. The package import exercises the exact entry
    point callers use; when Paddle (pulled by ``paddlefleet/__init__``) is
    absent it raises ImportError and we exec the identical production source
    file directly. Only a genuine dependency ImportError is surfaced for skip.
    """
    try:
        from paddlefleet.cli.train.deepseek_v3_pretrain.utils import (
            convert_ckpt_to_sft as mod,
        )

        return mod, None
    except ImportError:
        pass
    try:
        spec = importlib.util.spec_from_file_location(
            "paddlefleet_convert_ckpt_to_sft_under_test", _PROD_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, None
    except ImportError as exc:
        return None, exc


_CONV, _LOAD_ERROR = _load_production_module()

try:
    import torch

    _TORCH_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on host
    torch = None
    _TORCH_ERROR = exc


class _Base(unittest.TestCase):
    def setUp(self):
        if _CONV is None:
            self.skipTest(
                f"converter import failed (dependency unavailable): {_LOAD_ERROR!r}"
            )


class TestPaddleNameToHfNames(_Base):
    """Exact HF name lists for every branch of ``paddle_name_to_hf_names``."""

    def test_embed_tokens_exact(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names("deepseek_v2.embed_tokens.weight"),
            ["model.embed_tokens.weight"],
        )

    def test_norm_exact(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names("deepseek_v2.norm.weight"),
            ["model.norm.weight"],
        )

    def test_lm_head_exact(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names("lm_head.weight"), ["lm_head.weight"]
        )

    def test_norm_weight_filter_drops(self):
        # The drop filter requires a literal dot: ".norm_weight". A dotted
        # segment is dropped ...
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.4.self_attn.norm_weight"
            ),
            [],
        )

    def test_norm_weight_filter_is_dot_sensitive(self):
        # ... but "q_norm_weight" (underscore, no dot) is NOT dropped; it
        # falls through to the deepseek_v2 -> model fallback rename.
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.4.self_attn.q_norm_weight"
            ),
            ["model.layers.4.self_attn.q_norm_weight"],
        )

    def test_router_filter_drops(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.3.mlp.router.weight"
            ),
            [],
        )

    def test_no_layer_match_drops(self):
        self.assertEqual(_CONV.paddle_name_to_hf_names("some.random.name"), [])

    def test_custom_map_fused_rms_norm_to_input_layernorm(self):
        # fused rms_norm_weight -> input_layernorm.weight; the name itself does
        # NOT contain "input_layernorm" so it survives the drop filter.
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.3.self_attn."
                "fused_rms_norm_linear.rms_norm_weight"
            ),
            ["model.layers.3.input_layernorm.weight"],
        )

    def test_custom_map_attention_entries(self):
        # Hand-derived from custom_name_map; layer index 6 must propagate.
        cases = {
            "self_attn.memory_recompute_att.kv_ln_weight": "self_attn.kv_a_layernorm.weight",
            "self_attn.fused_rms_norm_linear.kv_down_weight": "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.memory_recompute_att.kv_up_weight": "self_attn.kv_b_proj.weight",
            "self_attn.memory_recompute_att.q_ln_weight": "self_attn.q_a_layernorm.weight",
            "self_attn.fused_rms_norm_linear.q_down_weight": "self_attn.q_a_proj.weight",
            "self_attn.memory_recompute_att.q_up_weight": "self_attn.q_b_proj.weight",
        }
        for rest, mapped in cases.items():
            with self.subTest(rest=rest):
                self.assertEqual(
                    _CONV.paddle_name_to_hf_names(
                        "deepseek_v2.layers.6." + rest
                    ),
                    ["model.layers.6." + mapped],
                )

    def test_expert_w1_exact_and_order(self):
        # gate_proj first, up_proj second; layer 5 / expert 2 both embedded.
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.5.mlp.experts.2.w1.weight"
            ),
            [
                "model.layers.5.mlp.experts.2.gate_proj.weight",
                "model.layers.5.mlp.experts.2.up_proj.weight",
            ],
        )

    def test_expert_w2_exact(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.5.mlp.experts.2.w2.weight"
            ),
            ["model.layers.5.mlp.experts.2.down_proj.weight"],
        )

    def test_expert_index_is_not_ignored(self):
        # Same shape/structure, different expert id -> different key.
        a = _CONV.paddle_name_to_hf_names(
            "deepseek_v2.layers.5.mlp.experts.2.w2.weight"
        )
        b = _CONV.paddle_name_to_hf_names(
            "deepseek_v2.layers.5.mlp.experts.7.w2.weight"
        )
        self.assertEqual(a, ["model.layers.5.mlp.experts.2.down_proj.weight"])
        self.assertEqual(b, ["model.layers.5.mlp.experts.7.down_proj.weight"])
        self.assertNotEqual(a, b)

    def test_shared_expert_w1_exact(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.5.mlp.shared_experts.w1.weight"
            ),
            [
                "model.layers.5.mlp.shared_experts.gate_proj.weight",
                "model.layers.5.mlp.shared_experts.up_proj.weight",
            ],
        )

    def test_shared_expert_w2_exact(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.5.mlp.shared_experts.w2.weight"
            ),
            ["model.layers.5.mlp.shared_experts.down_proj.weight"],
        )

    def test_dense_mlp_w1_exact(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names("deepseek_v2.layers.3.mlp.w1"),
            [
                "model.layers.3.mlp.gate_proj.weight",
                "model.layers.3.mlp.up_proj.weight",
            ],
        )

    def test_dense_mlp_w2_exact(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names("deepseek_v2.layers.3.mlp.w2"),
            ["model.layers.3.mlp.down_proj.weight"],
        )

    def test_dense_mlp_gate_up_fused_exact(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.3.mlp.gate_up_fused_proj.weight"
            ),
            [
                "model.layers.3.mlp.gate_proj.weight",
                "model.layers.3.mlp.up_proj.weight",
            ],
        )

    def test_shared_expert_gate_up_fused_exact(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.5.mlp.shared_experts."
                "gate_up_fused_proj.weight"
            ),
            [
                "model.layers.5.mlp.shared_experts.gate_proj.weight",
                "model.layers.5.mlp.shared_experts.up_proj.weight",
            ],
        )

    def test_expert_gate_up_fused_v2_exact(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.5.mlp.experts.3.gate_up_fused_proj.weight"
            ),
            [
                "model.layers.5.mlp.experts.3.gate_proj.weight",
                "model.layers.5.mlp.experts.3.up_proj.weight",
            ],
        )

    def test_fallback_replaces_prefix(self):
        # Unmatched attention projections fall through to deepseek_v2 -> model.
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.3.self_attn.o_proj.weight"
            ),
            ["model.layers.3.self_attn.o_proj.weight"],
        )
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.3.self_attn.q_proj.weight"
            ),
            ["model.layers.3.self_attn.q_proj.weight"],
        )

    def test_input_layernorm_current_behavior_is_dropped(self):
        # Documents the *current* observable behavior: the substring guard
        # drops any name containing "input_layernorm" before custom_name_map.
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.3.self_attn.input_layernorm.weight"
            ),
            [],
        )


class TestHandleExpertWeights(_Base):
    def test_w1_returns_owned_pair(self):
        self.assertEqual(
            _CONV._handle_expert_weights(
                "model.layers.3", "mlp.experts.5.w1.weight"
            ),
            [
                "model.layers.3.mlp.experts.5.gate_proj.weight",
                "model.layers.3.mlp.experts.5.up_proj.weight",
            ],
        )

    def test_w2_returns_down_proj(self):
        self.assertEqual(
            _CONV._handle_expert_weights(
                "model.layers.3", "mlp.experts.5.w2.weight"
            ),
            ["model.layers.3.mlp.experts.5.down_proj.weight"],
        )

    def test_non_expert_returns_none(self):
        self.assertIsNone(
            _CONV._handle_expert_weights(
                "model.layers.3", "self_attn.q_proj.weight"
            )
        )


class TestHandleSharedExpertWeights(_Base):
    def test_w1_pair(self):
        self.assertEqual(
            _CONV._handle_shared_expert_weights(
                "model.layers.3", "mlp.shared_experts.w1.weight"
            ),
            [
                "model.layers.3.mlp.shared_experts.gate_proj.weight",
                "model.layers.3.mlp.shared_experts.up_proj.weight",
            ],
        )

    def test_w2_down(self):
        self.assertEqual(
            _CONV._handle_shared_expert_weights(
                "model.layers.3", "mlp.shared_experts.w2.weight"
            ),
            ["model.layers.3.mlp.shared_experts.down_proj.weight"],
        )

    def test_non_match_none(self):
        self.assertIsNone(
            _CONV._handle_shared_expert_weights(
                "model.layers.3", "self_attn.q_proj.weight"
            )
        )


class TestHandleMlpWeights(_Base):
    def test_w1_pair(self):
        self.assertEqual(
            _CONV._handle_mlp_weights("model.layers.3", "mlp.w1"),
            [
                "model.layers.3.mlp.gate_proj.weight",
                "model.layers.3.mlp.up_proj.weight",
            ],
        )

    def test_w2_down(self):
        self.assertEqual(
            _CONV._handle_mlp_weights("model.layers.3", "mlp.w2"),
            ["model.layers.3.mlp.down_proj.weight"],
        )

    def test_non_match_none(self):
        self.assertIsNone(
            _CONV._handle_mlp_weights(
                "model.layers.3", "self_attn.q_proj.weight"
            )
        )


class TestIsNeedTranspose(_Base):
    def test_documented_transpose_suffixes(self):
        for key in (
            "deepseek_v2.layers.3.self_attn."
            "fused_rms_norm_linear.kv_down_weight",
            "deepseek_v2.layers.3.self_attn.memory_recompute_att.kv_up_weight",
            "deepseek_v2.layers.3.self_attn.o_proj.weight",
            "deepseek_v2.layers.3.self_attn."
            "fused_rms_norm_linear.q_down_weight",
            "deepseek_v2.layers.3.self_attn.memory_recompute_att.q_up_weight",
            "deepseek_v2.layers.3.mlp.w1",
            "deepseek_v2.layers.3.mlp.w2",
            "deepseek_v2.layers.3.mlp.gate.weight",
            "deepseek_v2.layers.3.eh_proj.weight",
            "lm_head.weight",
        ):
            with self.subTest(key=key):
                self.assertTrue(_CONV._is_need_transpose(key))

    def test_non_transpose_keys(self):
        for key in (
            "deepseek_v2.layers.3.self_attn.q_proj.weight",
            "deepseek_v2.layers.3.self_attn.k_proj.weight",
            "deepseek_v2.norm.weight",
            "deepseek_v2.embed_tokens.weight",
        ):
            with self.subTest(key=key):
                self.assertFalse(_CONV._is_need_transpose(key))

    def test_bare_w1_transposes_but_expert_weight_does_not(self):
        # Suffix rule: bare "...mlp.w1" ends with "w1" (transpose), while an
        # expert "...w1.weight" ends with ".weight" (no transpose).
        self.assertTrue(_CONV._is_need_transpose("deepseek_v2.layers.3.mlp.w1"))
        self.assertFalse(
            _CONV._is_need_transpose(
                "deepseek_v2.layers.3.mlp.experts.2.w1.weight"
            )
        )

    def test_gate_up_fused_proj_not_transposed(self):
        self.assertFalse(
            _CONV._is_need_transpose(
                "deepseek_v2.layers.3.mlp.gate_up_fused_proj.weight"
            )
        )


class TestRegexPatterns(_Base):
    def test_layer_re_captures_index_and_rest(self):
        m = _CONV._LAYER_RE.match(
            "deepseek_v2.layers.5.self_attn.q_proj.weight"
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "5")
        self.assertEqual(m.group(2), "self_attn.q_proj.weight")

    def test_layer_re_no_match(self):
        self.assertIsNone(_CONV._LAYER_RE.match("other.pattern"))

    def test_expert_w1_w2_capture(self):
        self.assertEqual(
            _CONV._EXPERT_W1_RE.match("mlp.experts.3.w1.weight").group(1), "3"
        )
        self.assertEqual(
            _CONV._EXPERT_W2_RE.match("mlp.experts.7.w2.weight").group(1), "7"
        )
        # Bare form without ".weight" also matches.
        self.assertIsNotNone(_CONV._EXPERT_W1_RE.match("mlp.experts.3.w1"))

    def test_expert_re_rejects_non_digit_and_trailing(self):
        self.assertIsNone(_CONV._EXPERT_W1_RE.match("mlp.experts.x.w1.weight"))
        self.assertIsNone(
            _CONV._EXPERT_W1_RE.match("mlp.experts.3.w1.weight.extra")
        )

    def test_shared_expert_patterns(self):
        self.assertIsNotNone(
            _CONV._SHARE_EXPERT_W1_RE.match("mlp.shared_experts.w1.weight")
        )
        self.assertIsNotNone(
            _CONV._SHARE_EXPERT_W2_RE.match("mlp.shared_experts.w2.weight")
        )

    def test_v2_fused_patterns(self):
        self.assertEqual(
            _CONV._EXPERT_W1_RE_v2.match(
                "mlp.experts.5.gate_up_fused_proj.weight"
            ).group(1),
            "5",
        )
        self.assertIsNotNone(
            _CONV._SHARE_EXPERT_W1_RE_v2.match(
                "mlp.shared_experts.gate_up_fused_proj.weight"
            )
        )


@unittest.skipUnless(_CONV is not None, "converter import unavailable")
@unittest.skipUnless(torch is not None, "torch unavailable")
class TestPrepareTensor(unittest.TestCase):
    """``prepare_tensor`` with real CPU torch tensors: mapping + transpose +
    fused split ownership + byte accounting. Content is a distinguishable
    ``arange`` so a gate/up swap or missing transpose is rejected."""

    @staticmethod
    def _mat(rows, cols):
        return torch.arange(rows * cols, dtype=torch.float32).reshape(
            rows, cols
        )

    def test_single_key_no_transpose_preserves_content(self):
        value = self._mat(4, 8)  # norm weight is not transposed
        result, size = _CONV.prepare_tensor("deepseek_v2.norm.weight", value)
        self.assertEqual(set(result), {"model.norm.weight"})
        np.testing.assert_array_equal(
            result["model.norm.weight"].numpy(), value.numpy()
        )
        self.assertEqual(size, 32 * value.element_size())

    def test_single_key_transpose_applies(self):
        value = self._mat(4, 8)  # lm_head.weight IS transposed
        result, size = _CONV.prepare_tensor("lm_head.weight", value)
        self.assertEqual(set(result), {"lm_head.weight"})
        np.testing.assert_array_equal(
            result["lm_head.weight"].numpy(), value.numpy().T
        )
        self.assertEqual(size, 32 * value.element_size())

    def test_fused_w1_transpose_then_split_ownership(self):
        # mlp.w1 -> [gate, up]; transposed first, then split along dim 0.
        value = self._mat(4, 8)  # -> transpose (8, 4) -> halves of 4 rows
        result, size = _CONV.prepare_tensor(
            "deepseek_v2.layers.3.mlp.w1", value
        )
        gate = "model.layers.3.mlp.gate_proj.weight"
        up = "model.layers.3.mlp.up_proj.weight"
        self.assertEqual(set(result), {gate, up})
        vt = value.numpy().T
        np.testing.assert_array_equal(result[gate].numpy(), vt[:4])
        np.testing.assert_array_equal(result[up].numpy(), vt[4:])
        # Halves must differ: an ownership swap would be detectable.
        self.assertFalse(
            np.array_equal(result[gate].numpy(), result[up].numpy())
        )
        self.assertEqual(size, (16 + 16) * value.element_size())

    def test_fused_gate_up_no_transpose_split_ownership(self):
        # gate_up_fused_proj is NOT transposed: split the original dim 0.
        value = self._mat(8, 4)
        result, size = _CONV.prepare_tensor(
            "deepseek_v2.layers.3.mlp.gate_up_fused_proj.weight", value
        )
        gate = "model.layers.3.mlp.gate_proj.weight"
        up = "model.layers.3.mlp.up_proj.weight"
        self.assertEqual(set(result), {gate, up})
        arr = value.numpy()
        np.testing.assert_array_equal(result[gate].numpy(), arr[:4])
        np.testing.assert_array_equal(result[up].numpy(), arr[4:])
        self.assertEqual(size, (16 + 16) * value.element_size())

    def test_expert_w1_split_keeps_index_no_transpose(self):
        value = self._mat(8, 4)  # expert w1.weight is not transposed
        result, size = _CONV.prepare_tensor(
            "deepseek_v2.layers.5.mlp.experts.2.w1.weight", value
        )
        gate = "model.layers.5.mlp.experts.2.gate_proj.weight"
        up = "model.layers.5.mlp.experts.2.up_proj.weight"
        self.assertEqual(set(result), {gate, up})
        arr = value.numpy()
        np.testing.assert_array_equal(result[gate].numpy(), arr[:4])
        np.testing.assert_array_equal(result[up].numpy(), arr[4:])
        self.assertEqual(size, (16 + 16) * value.element_size())

    def test_zero_key_returns_empty(self):
        value = self._mat(4, 8)
        result, size = _CONV.prepare_tensor("some.norm_weight.param", value)
        self.assertEqual(result, {})
        self.assertEqual(size, 0)


class TestInputLayernormMappingDefect(_Base):
    """Flags a real production defect without editing the module.

    ``custom_name_map`` contains an entry
    ``"self_attn.input_layernorm.weight": "input_layernorm.weight"`` expressing
    the intent to rename that weight. But ``paddle_name_to_hf_names`` runs
    ``if "input_layernorm" in paddle_name: return []`` *before* consulting the
    map, so the entry is unreachable and the weight is silently dropped rather
    than renamed. This test encodes the intended mapping and is expected to
    fail against current code; it documents the dead map entry.
    """

    @unittest.expectedFailure
    def test_input_layernorm_should_map_not_drop(self):
        self.assertEqual(
            _CONV.paddle_name_to_hf_names(
                "deepseek_v2.layers.3.self_attn.input_layernorm.weight"
            ),
            ["model.layers.3.input_layernorm.weight"],
        )


if __name__ == "__main__":
    unittest.main()
