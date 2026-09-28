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
"""Behavior tests for paddlefleet.transformer.moe.fp8_utils.

Scope is the CPU-executable, non-collective surface of the FP8 expert node:

  * ``_get_fp8_weight_and_scale`` on-the-fly stacked reshape+transpose, its
    cached-transpose short circuit, the UE8M0 scale selection, and the
    divisibility guard.
  * ``ExpertsGroupGemmContiguousNode`` constructor branch selection
    (per-expert slice vs grouped-expert view, split vs fused flag) and its
    two eager-fail contracts (sub-batch alignment, activation name).
  * ``gen_m_indices`` expert-id expansion for list / tensor / empty counts.

Every numeric expectation is derived from an INDEPENDENT numpy reference
(per-expert block transpose written from the spec, explicit repeat loop),
never by calling the production helper under test. Distinguishable,
non-square blocks are used so a transpose-direction or block-ordering bug
cannot survive.

Heavy imports (paddle + the fp8 module) are guarded so a missing runtime is
reported honestly as a skip; only ImportError/ModuleNotFoundError counts as
"dependency absent" so that real API breaks still surface.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.moe.fp8_utils import (
        FP8_ALIGN,
        ExpertsGroupGemmContiguousNode,
        _get_fp8_weight_and_scale,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle/paddlefleet/numpy not importable in this environment: "
    f"{_IMPORT_ERROR}"
    if _IMPORT_ERROR is not None
    else ""
)


def _reference_stacked_transpose(mat, expert_num):
    """Independent reference for the stacked per-expert transpose.

    The production path does ``reshape([E, h0, h1]).transpose([0,2,1])
    .reshape([-1, h0])``. Here we instead slice each expert's ``[h0, h1]``
    block and transpose it with ``numpy.T``, then stack the resulting
    ``[h1, h0]`` blocks. Same math, independent implementation, so a shared
    reshape/transpose bug would not hide on both sides.
    """
    mat = np.asarray(mat, dtype=np.float64)
    total_rows, h1 = mat.shape
    h0 = total_rows // expert_num
    blocks = []
    for e in range(expert_num):
        block = mat[e * h0 : (e + 1) * h0, :]  # [h0, h1]
        blocks.append(block.T)  # [h1, h0]
    return np.concatenate(blocks, axis=0)  # [expert_num * h1, h0]


def _reference_m_indices(counts):
    """Independent reference for gen_m_indices: expert e appears counts[e] times."""
    out = []
    for expert_id, n in enumerate(counts):
        out.extend([expert_id] * int(n))
    return out


class _CustomMap:
    """Minimal custom_map with distinguishable expert identities."""

    def __init__(self):
        # Distinguishable, order-sensitive expert identities.
        self.experts = ["expert-A", "expert-B", "expert-C"]
        self.grouped_gemm_experts = "grouped-view"


class _CachedWeight:
    """Offline-quant weight carrier: only the non-transposed stack is cached."""

    def __init__(self, weight_stacked, scale_stacked, shape, scale_transpose):
        self.shape = shape
        self.fp8_weight_stacked = weight_stacked
        self.fp8_scale_stacked = scale_stacked
        self.fp8_weight_stacked_transpose = None
        self.fp8_scale_stacked_transpose = scale_transpose


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetFp8WeightAndScale(unittest.TestCase):
    def test_onthefly_transpose_transposes_weight_and_scale_per_expert(self):
        # E=2 non-square 2x3 blocks: transpose direction + block order matter.
        weight_stacked = paddle.arange(12, dtype="float32").reshape([4, 3])
        scale_stacked = paddle.arange(
            start=12, end=24, dtype="float32"
        ).reshape([4, 3])
        # A distinct precomputed-transpose scale that MUST be ignored when
        # use_ue8m0 is falsy (proves the on-the-fly path was taken).
        decoy_scale_t = paddle.full([99, 99], -7.0, dtype="float32")
        weight = _CachedWeight(
            weight_stacked,
            scale_stacked,
            shape=[2, 3],  # expert_num inferred = 4 // 2 = 2
            scale_transpose=decoy_scale_t,
        )

        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(
            weight, transpose=True, num_expert=None, use_ue8m0=None
        )

        expected_weight = _reference_stacked_transpose(
            weight_stacked.numpy(), expert_num=2
        )
        expected_scale = _reference_stacked_transpose(
            scale_stacked.numpy(), expert_num=2
        )
        # Hand-derived literal anchor for the weight, independent of numpy path.
        np.testing.assert_array_equal(
            fp8_weight.numpy(),
            np.array(
                [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(fp8_weight.numpy(), expected_weight)
        # Scale is transposed on the fly (NOT the decoy precomputed one).
        np.testing.assert_array_equal(fp8_scale.numpy(), expected_scale)
        self.assertFalse(
            np.array_equal(
                fp8_scale.numpy(), np.full([99, 99], -7.0, dtype=np.float32)
            )
        )

    def test_ue8m0_uses_precomputed_scale_but_still_transposes_weight(self):
        # explicit num_expert=3, 2x2 blocks.
        weight_stacked = paddle.arange(12, dtype="float32").reshape([6, 2])
        scale_stacked = paddle.arange(18, dtype="float32").reshape([6, 3])
        precomputed_scale_t = paddle.arange(
            start=100, end=118, dtype="float32"
        ).reshape([9, 2])
        weight = _CachedWeight(
            weight_stacked,
            scale_stacked,
            shape=[2, 2],
            scale_transpose=precomputed_scale_t,
        )

        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(
            weight, transpose=True, num_expert=3, use_ue8m0=True
        )

        expected_weight = _reference_stacked_transpose(
            weight_stacked.numpy(), expert_num=3
        )
        np.testing.assert_array_equal(fp8_weight.numpy(), expected_weight)
        np.testing.assert_array_equal(
            fp8_weight.numpy(),
            np.array(
                [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11]],
                dtype=np.float32,
            ),
        )
        # UE8M0: scale is the precomputed transpose object, verbatim.
        self.assertIs(fp8_scale, weight.fp8_scale_stacked_transpose)
        np.testing.assert_array_equal(
            fp8_scale.numpy(), precomputed_scale_t.numpy()
        )

    def test_cached_transpose_short_circuits_both_weight_and_scale(self):
        # When fp8_weight_stacked_transpose is present, the on-the-fly
        # reshape/transpose is skipped and the cached pair is returned as-is.
        weight_stacked = paddle.arange(12, dtype="float32").reshape([6, 2])
        scale_stacked = paddle.arange(18, dtype="float32").reshape([6, 3])
        cached_w_t = paddle.arange(start=200, end=212, dtype="float32").reshape(
            [6, 2]
        )
        cached_s_t = paddle.arange(start=300, end=318, dtype="float32").reshape(
            [9, 2]
        )
        weight = _CachedWeight(
            weight_stacked,
            scale_stacked,
            shape=[2, 2],
            scale_transpose=cached_s_t,
        )
        weight.fp8_weight_stacked_transpose = cached_w_t

        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(
            weight, transpose=True, num_expert=3, use_ue8m0=True
        )

        self.assertIs(fp8_weight, cached_w_t)
        self.assertIs(fp8_scale, cached_s_t)
        # And the raw non-transposed stack was NOT returned.
        self.assertIsNot(fp8_weight, weight.fp8_weight_stacked)

    def test_non_transpose_returns_raw_stacked_pair(self):
        weight_stacked = paddle.arange(12, dtype="float32").reshape([6, 2])
        scale_stacked = paddle.arange(18, dtype="float32").reshape([6, 3])
        weight = _CachedWeight(
            weight_stacked,
            scale_stacked,
            shape=[2, 2],
            scale_transpose=paddle.zeros([9, 2], dtype="float32"),
        )

        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(
            weight, transpose=False
        )

        self.assertIs(fp8_weight, weight.fp8_weight_stacked)
        self.assertIs(fp8_scale, weight.fp8_scale_stacked)

    def test_transpose_rejects_indivisible_stacked_rows(self):
        # fp8_weight rows (6) not divisible by weight.shape[0] (4) -> assert.
        weight_stacked = paddle.ones([6, 2], dtype="float32")
        scale_stacked = paddle.ones([6, 2], dtype="float32")
        weight = _CachedWeight(
            weight_stacked, scale_stacked, shape=[4, 2], scale_transpose=None
        )
        with self.assertRaises(AssertionError):
            _get_fp8_weight_and_scale(weight, transpose=True)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestExpertsGroupGemmContiguousNodeConstruction(unittest.TestCase):
    def test_per_expert_id_selects_that_expert_only(self):
        custom_map = _CustomMap()
        node = ExpertsGroupGemmContiguousNode(custom_map, expert_id=1)
        # Selected expert identity (not just count) and split-mode flag.
        self.assertEqual(node.experts, ["expert-B"])
        self.assertEqual(node.expert_id, 1)
        self.assertTrue(node.use_fp8_mlp)
        self.assertTrue(
            node.is_split_group_gemm
        )  # moe_expert_fusion default off

    def test_no_expert_id_keeps_full_expert_list_in_order(self):
        custom_map = _CustomMap()
        node = ExpertsGroupGemmContiguousNode(custom_map)
        self.assertEqual(node.experts, ["expert-A", "expert-B", "expert-C"])
        self.assertIsNone(node.expert_id)

    def test_fused_non_fp8_uses_grouped_view_and_clears_split_flag(self):
        custom_map = _CustomMap()
        grouped = ExpertsGroupGemmContiguousNode(
            custom_map, use_fp8_mlp=False, moe_expert_fusion=True
        )
        self.assertEqual(grouped.grouped_gemm_experts, "grouped-view")
        self.assertFalse(grouped.is_split_group_gemm)
        self.assertFalse(grouped.use_fp8_mlp)

    def test_subbatch_alignment_contract(self):
        # Must be a positive multiple of FP8_ALIGN.
        self.assertEqual(FP8_ALIGN, 128)
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(
                _CustomMap(), moe_subbatch_token_num_after_dispatch=127
            )
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(
                _CustomMap(), moe_subbatch_token_num_after_dispatch=0
            )
        ok = ExpertsGroupGemmContiguousNode(
            _CustomMap(),
            moe_subbatch_token_num_after_dispatch=2 * FP8_ALIGN,
        )
        self.assertEqual(
            ok.moe_subbatch_token_num_after_dispatch, 2 * FP8_ALIGN
        )

    def test_unknown_activation_type_is_rejected(self):
        with self.assertRaises(ValueError):
            ExpertsGroupGemmContiguousNode(_CustomMap(), activation_type="relu")
        # A supported name is accepted and stored.
        node = ExpertsGroupGemmContiguousNode(
            _CustomMap(), activation_type="geglu"
        )
        self.assertEqual(node.activation_type, "geglu")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGenMIndices(unittest.TestCase):
    def test_list_counts_expand_per_expert_in_order(self):
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        counts = [1, 3, 2]  # expert0 x1, expert1 x3, expert2 x2
        out = node.gen_m_indices(counts).numpy().tolist()
        self.assertEqual(out, _reference_m_indices(counts))
        self.assertEqual(out, [0, 1, 1, 1, 2, 2])

    def test_zero_count_expert_is_skipped(self):
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        counts = [2, 0, 1]  # expert1 contributes nothing
        out = node.gen_m_indices(counts).numpy().tolist()
        self.assertEqual(out, _reference_m_indices(counts))
        self.assertEqual(out, [0, 0, 2])

    def test_tensor_counts_match_list_semantics(self):
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        counts = [1, 2]
        out = (
            node.gen_m_indices(paddle.to_tensor(counts, dtype="int64"))
            .numpy()
            .tolist()
        )
        self.assertEqual(out, _reference_m_indices(counts))
        self.assertEqual(out, [0, 1, 1])

    def test_empty_counts_return_empty(self):
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        out = node.gen_m_indices(paddle.to_tensor([], dtype="int64"))
        self.assertEqual(list(out.shape), [0])


if __name__ == "__main__":
    unittest.main()
