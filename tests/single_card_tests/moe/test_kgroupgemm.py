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

"""Behavior tests for ExpertsGroupGemmContiguousNode (k-grouped GEMM path).

Target: paddlefleet.transformer.moe.fp8_utils.ExpertsGroupGemmContiguousNode.

Two contracts are pinned here, both CPU-only and independent of the fp8 /
deep_gemm GPU kernels:

  * gen_m_indices: the contiguous grouped-GEMM layout depends on mapping each
    dispatched token to the id of the expert that owns it. For per-expert
    counts [c0, c1, ...] the row layout must be c0 copies of 0, then c1 copies
    of 1, and so on, as int32. A wrong repeat, a dropped/extra count, or a
    non-int32 dtype silently feeds the grouped GEMM the wrong expert routing,
    so we compare against a hand-built expectation with distinguishable,
    unequal counts (including a zero-token expert in the middle).

  * __init__ weight-source selection: the node must bind the *fused*
    grouped_gemm_experts exactly on the (fusion and (bf16 or deep_gemm)) path
    and the per-expert experts list otherwise. We bind two distinct sentinel
    objects and assert by identity which one the node kept, plus the derived
    is_split_group_gemm flag, so binding the wrong collection is rejected
    (hasattr alone would not catch a swap).

Paddle is imported under a guard so environments without paddle/paddlefleet
skip honestly instead of reporting a false pass.
"""

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.fp8_utils import (
        ExpertsGroupGemmContiguousNode,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet_ops not installed
    paddle = None
    ExpertsGroupGemmContiguousNode = None
    _IMPORT_ERROR = exc


def _make_custom_map(experts, grouped_gemm_experts):
    """Minimal stand-in for the MoE layer the node reads in __init__.

    __init__ only reads .experts / .grouped_gemm_experts (and optional
    layer_number/config via getattr) on the non-per-expert paths, so plain
    sentinels are enough to observe which collection the node binds.
    """
    return SimpleNamespace(
        experts=experts,
        grouped_gemm_experts=grouped_gemm_experts,
    )


@unittest.skipUnless(
    ExpertsGroupGemmContiguousNode is not None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestGenMIndices(unittest.TestCase):
    """gen_m_indices maps contiguous tokens back to their owning expert."""

    def _node(self):
        # fusion + deep_gemm binds grouped_gemm_experts and needs no GPU here.
        return ExpertsGroupGemmContiguousNode(
            _make_custom_map(object(), object()),
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )

    def test_repeats_expert_ids_by_token_count(self):
        """counts [2,0,3,1] -> [0,0,2,2,2,3] as int32 (expert 1 gets none)."""
        out = self._node().gen_m_indices([2, 0, 3, 1])
        self.assertEqual(out.dtype, paddle.int32)
        np.testing.assert_array_equal(
            out.numpy(), np.array([0, 0, 2, 2, 2, 3], dtype=np.int32)
        )

    def test_tensor_input_matches_list_input(self):
        """A paddle int tensor of counts yields the same mapping as a list."""
        counts = [1, 4]
        expected = np.array([0, 1, 1, 1, 1], dtype=np.int32)
        from_list = self._node().gen_m_indices(counts)
        from_tensor = self._node().gen_m_indices(
            paddle.to_tensor(counts, dtype="int64")
        )
        np.testing.assert_array_equal(from_list.numpy(), expected)
        np.testing.assert_array_equal(from_tensor.numpy(), expected)

    def test_empty_counts_returns_empty_int32(self):
        """No experts -> an empty int32 tensor, not a crash."""
        out = self._node().gen_m_indices([])
        self.assertEqual(list(out.shape), [0])
        self.assertEqual(out.dtype, paddle.int32)


@unittest.skipUnless(
    ExpertsGroupGemmContiguousNode is not None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestWeightSourceSelection(unittest.TestCase):
    """__init__ binds the fused vs per-expert weights per the flag matrix."""

    def _build(self, *, fusion, fp8, deep_gemm):
        experts = SimpleNamespace(tag="per_expert")
        grouped = SimpleNamespace(tag="grouped")
        node = ExpertsGroupGemmContiguousNode(
            _make_custom_map(experts, grouped),
            use_fp8_mlp=fp8,
            moe_deep_gemm=deep_gemm,
            moe_expert_fusion=fusion,
        )
        return node, experts, grouped

    def test_selection_matrix(self):
        """Expected owner per (fusion, fp8, deep_gemm), expert_id=None.

        Condition for the per-expert path is
        ``not fusion or (fp8 and not deep_gemm)``; otherwise the fused
        grouped_gemm_experts is used.
        """
        cases = [
            # (fusion, fp8, deep_gemm) -> "grouped" or "experts"
            ((False, True, True), "experts"),
            ((True, True, False), "experts"),
            ((True, True, True), "grouped"),
            ((True, False, True), "grouped"),
            ((True, False, False), "grouped"),
        ]
        for (fusion, fp8, deep_gemm), owner in cases:
            with self.subTest(fusion=fusion, fp8=fp8, deep_gemm=deep_gemm):
                node, experts, grouped = self._build(
                    fusion=fusion, fp8=fp8, deep_gemm=deep_gemm
                )
                if owner == "grouped":
                    self.assertIs(node.grouped_gemm_experts, grouped)
                    self.assertFalse(hasattr(node, "experts"))
                else:
                    self.assertIs(node.experts, experts)
                    self.assertFalse(hasattr(node, "grouped_gemm_experts"))
                # is_split_group_gemm is derived purely from fusion.
                self.assertEqual(node.is_split_group_gemm, not fusion)
                self.assertEqual(node.moe_expert_fusion, fusion)

    def test_non_fused_wraps_single_expert_when_expert_id_given(self):
        """With expert_id set on the per-expert path, only that expert is kept."""
        experts = ["e0", "e1", "e2"]
        node = ExpertsGroupGemmContiguousNode(
            _make_custom_map(experts, SimpleNamespace(tag="grouped")),
            use_fp8_mlp=True,
            moe_deep_gemm=False,
            moe_expert_fusion=True,
            expert_id=1,
        )
        self.assertEqual(node.experts, ["e1"])
        self.assertEqual(node.expert_id, 1)


@unittest.skipUnless(
    ExpertsGroupGemmContiguousNode is not None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestActivationTypeValidation(unittest.TestCase):
    """__init__ rejects unknown activations up front instead of silently
    falling through to SwiGLU in every downstream branch."""

    def test_unknown_activation_type_raises(self):
        with self.assertRaises(ValueError):
            ExpertsGroupGemmContiguousNode(
                _make_custom_map(object(), object()),
                use_fp8_mlp=True,
                moe_deep_gemm=True,
                moe_expert_fusion=True,
                activation_type="not_an_activation",
            )

    def test_supported_activation_type_is_stored(self):
        node = ExpertsGroupGemmContiguousNode(
            _make_custom_map(object(), object()),
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
            activation_type="geglu",
        )
        self.assertEqual(node.activation_type, "geglu")


if __name__ == "__main__":
    unittest.main()
