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
"""Behavior tests for paddlefleet.transformer.moe.fusion_layer_utils.

These exercise the pure-Python orchestration of the fused-MoE helper nodes:
ordered cache slot mapping (UnZipNode / ZipNode), and MlpNode's construction
guards plus the per-expert padding / token-offset math. Expected values are
hand-derived from the alignment rule (ceil(n/align)*align, cumulative offsets)
and never produced by calling the code under test.

The kernel-backed forward/backward of these nodes (moe_permute /
moe_unpermute / grouped-gemm) requires a GPU and is out of scope here; that
numeric contract must be covered on real hardware.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import paddle  # noqa: F401  # module under test imports paddle at import time.

    from paddlefleet.transformer.moe.fusion_layer_utils import (
        FP8_ALIGN,
        MlpNode,
        UnZipNode,
        ZipNode,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest, narrow guard
    FP8_ALIGN = MlpNode = UnZipNode = ZipNode = None
    _IMPORT_ERROR = exc

_GEMM_NODE = (
    "paddlefleet.transformer.moe.fusion_layer_utils."
    "ExpertsGroupGemmContiguousNode"
)
_SKIP_REASON = (
    "fusion_layer_utils requires paddle/paddlefleet_ops, unavailable here: "
    f"{_IMPORT_ERROR!r}"
)


def _make_custom_map(tokens_per_expert):
    """Minimal stand-in for the token-dispatcher owner.

    Only the attributes MlpNode.__init__ actually reads are provided, so the
    real getattr defaults (moe_rank=0, experts=None, num_experts_per_device=
    len(tokens_per_expert), _activation_type='swiglu') take effect instead of
    being masked by an auto-attribute mock.
    """
    comm = SimpleNamespace(tokens_per_expert=list(tokens_per_expert))
    dispatcher = SimpleNamespace(_comm_manager=comm)
    return SimpleNamespace(token_dispatcher=dispatcher)


@unittest.skipIf(_IMPORT_ERROR is not None, _SKIP_REASON)
class TestUnZipNodeCacheSlots(unittest.TestCase):
    """UnZipNode stores exactly two cache tensors in a fixed order."""

    def test_init_defaults_and_name_consumed(self):
        dispatcher = object()
        node = UnZipNode(dispatcher)
        self.assertIs(node.token_dispatcher, dispatcher)
        self.assertEqual(node.name, "unzip")
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)
        # name is a real ctor argument, not hard-coded.
        self.assertEqual(UnZipNode(dispatcher, name="alt").name, "alt")

    def test_set_and_cached_tensors_preserve_slot_order(self):
        node = UnZipNode(object())
        probs, rowmap = object(), object()
        node.set_cached_tensors([probs, rowmap])
        # First slot -> unzipped_probs, second -> rowmap. Distinct sentinels
        # would expose a swapped assignment.
        self.assertIs(node.unzipped_probs, probs)
        self.assertIs(node.zipped_expertwise_rowmap, rowmap)
        cached = node.cached_tensors()
        self.assertEqual(len(cached), 2)
        self.assertIs(cached[0], probs)
        self.assertIs(cached[1], rowmap)

    def test_clear_and_reset_null_both_slots(self):
        for method in ("clear_cached_tensors", "reset_state"):
            node = UnZipNode(object())
            node.unzipped_probs = object()
            node.zipped_expertwise_rowmap = object()
            getattr(node, method)()
            self.assertIsNone(node.unzipped_probs, method)
            self.assertIsNone(node.zipped_expertwise_rowmap, method)
            self.assertEqual(node.cached_tensors(), [None, None], method)


@unittest.skipIf(_IMPORT_ERROR is not None, _SKIP_REASON)
class TestZipNodeCacheSlots(unittest.TestCase):
    """ZipNode holds no cache tensors and rejects any non-empty payload."""

    def test_init_defaults_and_empty_cache(self):
        dispatcher = object()
        node = ZipNode(dispatcher)
        self.assertIs(node.token_dispatcher, dispatcher)
        self.assertEqual(node.name, "zip")
        self.assertEqual(node.cached_tensors(), [])

    def test_set_cached_tensors_accepts_empty_only(self):
        node = ZipNode(object())
        node.set_cached_tensors([])  # contract: empty is fine
        self.assertEqual(node.cached_tensors(), [])
        with self.assertRaises(AssertionError):
            node.set_cached_tensors([object()])

    def test_clear_is_noop_and_keeps_cache_empty(self):
        node = ZipNode(object())
        self.assertIsNone(node.clear_cached_tensors())
        self.assertEqual(node.cached_tensors(), [])


@unittest.skipIf(_IMPORT_ERROR is not None, _SKIP_REASON)
class TestMlpNodePaddingAndOffsets(unittest.TestCase):
    """MlpNode pads each expert's token count up to the alignment and builds
    cumulative token offsets from those padded counts."""

    @mock.patch(_GEMM_NODE)
    def test_fp8_default_alignment_padding_and_offsets(self, gemm_cls):
        # Default path: use_fp8_mlp=True -> alignment == FP8_ALIGN (128).
        self.assertEqual(FP8_ALIGN, 128)
        tokens_per_expert = [1, 128, 129, 256, 300]
        node = MlpNode(
            _make_custom_map(tokens_per_expert), num_experts_per_tok=2
        )

        # ceil(n/128)*128 per expert, chosen so values straddle the boundary
        # (1->128, 128->128, 129->256, 256->256, 300->384): a per-value bug or
        # a wrong alignment cannot collapse to the same list.
        self.assertEqual(node.moe_permute_padding_alignment, 128)
        self.assertEqual(
            node.padding_token_per_experts, [128, 128, 256, 256, 384]
        )
        self.assertEqual(node.token_offsets, [0, 128, 256, 512, 768, 1152])
        # num_experts_per_tok is stored as router_topk (argument consumed).
        self.assertEqual(node.router_topk, 2)
        # Default (no static/auto subbatch) selects the single grouped-gemm node.
        self.assertIs(node.experts_group_gemm_node, gemm_cls.return_value)
        self.assertIsInstance(node.unzip_node, UnZipNode)
        self.assertIsInstance(node.zip_node, ZipNode)

    @mock.patch(_GEMM_NODE)
    def test_accuracy_compatible_bf16_uses_alignment_one(self, gemm_cls):
        # use_accuracy_compatible=True AND not fp8 AND not grouped_gemm -> align 1,
        # so padded counts equal the raw per-expert token counts unchanged.
        tokens_per_expert = [2, 4, 1, 3]
        node = MlpNode(
            _make_custom_map(tokens_per_expert),
            num_experts_per_tok=3,
            use_fp8_mlp=False,
            moe_expert_fusion=False,
            use_accuracy_compatible=True,
        )
        self.assertEqual(node.moe_permute_padding_alignment, 1)
        self.assertEqual(node.padding_token_per_experts, [2, 4, 1, 3])
        self.assertEqual(node.token_offsets, [0, 2, 6, 7, 10])
        self.assertEqual(node.router_topk, 3)


@unittest.skipIf(_IMPORT_ERROR is not None, _SKIP_REASON)
class TestMlpNodeConstructionGuards(unittest.TestCase):
    """MlpNode enforces prerequisite flags before building anything. Each test
    patches the grouped-gemm collaborator so that, if a guard were removed,
    construction would succeed and the missing raise would be detected."""

    @mock.patch(_GEMM_NODE)
    def test_recompute_moe_premute_requires_gate_up(self, _gemm_cls):
        with self.assertRaisesRegex(
            AssertionError, "recompute_moe_gate_up must be enabled"
        ):
            MlpNode(
                _make_custom_map([2, 2]),
                num_experts_per_tok=2,
                recompute_moe_premute=True,
            )

    @mock.patch(_GEMM_NODE)
    def test_static_subbatch_must_be_multiple_of_align(self, _gemm_cls):
        # 100 % 128 != 0 -> the alignment assertion fires first.
        with self.assertRaises(AssertionError):
            MlpNode(
                _make_custom_map([2, 2]),
                num_experts_per_tok=2,
                moe_subbatch_token_num_after_dispatch=100,
            )

    @mock.patch(_GEMM_NODE)
    def test_static_subbatch_aligned_still_requires_dequant(self, _gemm_cls):
        # 128 clears the %128 and fusion==deep_gemm checks, then the distinct
        # dequant_input guard must fire (not the alignment one).
        with self.assertRaisesRegex(
            AssertionError, "dequant_input must be enabled"
        ):
            MlpNode(
                _make_custom_map([2, 2]),
                num_experts_per_tok=2,
                moe_subbatch_token_num_after_dispatch=128,
            )

    @mock.patch(_GEMM_NODE)
    def test_invalid_auto_subbatch_mode_rejected(self, _gemm_cls):
        with self.assertRaisesRegex(
            ValueError, "auto_subbatch_mode must be one of"
        ):
            MlpNode(
                _make_custom_map([2, 2]),
                num_experts_per_tok=2,
                auto_subbatch_mode="not_a_mode",
            )

    @mock.patch(_GEMM_NODE)
    def test_pre_permute_requires_expert_fusion(self, _gemm_cls):
        with self.assertRaisesRegex(
            AssertionError, "pre_permute.*requires.*moe_expert_fusion"
        ):
            MlpNode(
                _make_custom_map([2, 2]),
                num_experts_per_tok=2,
                use_auto_subbatch=True,
                auto_subbatch_mode="pre_permute",
                moe_expert_fusion=False,
            )


if __name__ == "__main__":
    unittest.main()
