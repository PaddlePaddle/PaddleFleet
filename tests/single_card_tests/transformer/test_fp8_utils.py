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

"""Behavior tests for paddlefleet.transformer.moe.fp8_utils (CPU-only paths).

Scope. This file exercises the CPU-reachable, device-independent logic of the
MoE FP8 utilities: the config predicate, the per-expert token-padding rule, the
cached fp8-weight/scale lookup + on-the-fly transpose reshape, and the
ExpertsGroupGemmContiguousNode construction / cache-slot bookkeeping /
m-indices expansion.

Out of scope (needs a real GPU, so NOT claimed here): the FP8 GEMM numerics in
``kitchen_gemm`` (``fp8_gemm_blockwise``) and ``tilewise_quant``
(``fp8_quant_blockwise``). Those require a CUDA device and belong to the
single-card environment; they are intentionally not stubbed, because a stub
would only prove orchestration, not the kernel numerics.

Expected values below are derived by hand from the documented reshape/transpose
algorithm and encoded as literal arrays, independent of the production helper.
"""

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.moe.fp8_utils import (
        FP8_ALIGN,
        ExpertsGroupGemmContiguousNode,
        _get_fp8_weight_and_scale,
        fused_stack_quant,
        has_config,
        moe_token_padding_alignment,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest: deps absent here
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    None
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}"
)


class _CachedWeight:
    """Stand-in for an offline-quantized expert weight object.

    Reproduces exactly the attribute surface that ``_get_fp8_weight_and_scale``
    reads: the logical ``shape`` (used to infer expert count), the stacked fp8
    weight/scale tensors, and the optional pre-transposed tensors. This is the
    genuine (not-under-test) collaborator the helper consumes, filled with
    distinguishable ``arange`` content so mis-slicing is observable.
    """

    def __init__(self):
        # Logical weight shape[0] == 2 experts -> auto expert_num = 4 // 2 = 2.
        self.shape = [2, 3]
        self.fp8_weight_stacked = paddle.arange(12, dtype="float32").reshape(
            [4, 3]
        )
        self.fp8_scale_stacked = paddle.arange(8, dtype="float32").reshape(
            [4, 2]
        )
        self.fp8_weight_stacked_transpose = None
        self.fp8_scale_stacked_transpose = None


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON or "")
class TestHasConfig(unittest.TestCase):
    """has_config(config_map, key): truthy only when key present AND value truthy."""

    def test_none_map_is_false(self):
        self.assertIs(has_config(None, "enabled"), False)

    def test_missing_key_is_false(self):
        self.assertIs(has_config({"other": 1}, "enabled"), False)

    def test_present_but_falsy_value_is_false(self):
        # A present key whose value is falsy must not count as configured.
        for falsy in (0, "", [], {}, None, False):
            self.assertIs(
                has_config({"enabled": falsy}, "enabled"),
                False,
                msg=f"falsy value {falsy!r} should yield False",
            )

    def test_present_and_truthy_value_is_true(self):
        # Non-boolean truthy values must still map to True (not the raw value).
        for truthy in (1, "megatron", ["x"], {"a": 1}, 3.5):
            self.assertIs(
                has_config({"enabled": truthy}, "enabled"),
                True,
                msg=f"truthy value {truthy!r} should yield True",
            )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON or "")
class TestMoeTokenPaddingAlignment(unittest.TestCase):
    """moe_token_padding_alignment: return 1 ONLY for the accuracy-compatible,
    pure-bf16, non-grouped-gemm combination; align to FP8_ALIGN otherwise."""

    def test_only_accuracy_bf16_nongrouped_skips_padding(self):
        self.assertEqual(
            moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=False,
                use_accuracy_compatible=True,
            ),
            1,
        )

    def test_every_other_combination_aligns_to_fp8_align(self):
        # Enumerate the remaining 7 boolean combinations; each must align.
        for use_fp8_mlp in (True, False):
            for moe_grouped_gemm in (True, False):
                for use_accuracy_compatible in (True, False):
                    if (
                        use_accuracy_compatible
                        and not use_fp8_mlp
                        and not moe_grouped_gemm
                    ):
                        continue  # the single skip-padding case, tested above
                    self.assertEqual(
                        moe_token_padding_alignment(
                            use_fp8_mlp=use_fp8_mlp,
                            moe_grouped_gemm=moe_grouped_gemm,
                            use_accuracy_compatible=use_accuracy_compatible,
                        ),
                        FP8_ALIGN,
                        msg=(
                            f"fp8={use_fp8_mlp} grouped={moe_grouped_gemm} "
                            f"acc={use_accuracy_compatible}"
                        ),
                    )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON or "")
class TestGetFp8WeightAndScale(unittest.TestCase):
    """_get_fp8_weight_and_scale: cached passthrough vs on-the-fly transpose."""

    def test_no_transpose_returns_cached_objects_identically(self):
        # Must hand back the exact stacked tensors, not copies/reshapes.
        weight = _CachedWeight()
        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(weight)
        self.assertIs(fp8_weight, weight.fp8_weight_stacked)
        self.assertIs(fp8_scale, weight.fp8_scale_stacked)

    def test_precomputed_transpose_is_preferred_over_recompute(self):
        weight = _CachedWeight()
        pre_w = paddle.full([3, 4], 7.0)
        pre_s = paddle.full([2, 4], 9.0)
        weight.fp8_weight_stacked_transpose = pre_w
        weight.fp8_scale_stacked_transpose = pre_s
        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(
            weight, transpose=True
        )
        # When present, the cached transpose is returned verbatim.
        self.assertIs(fp8_weight, pre_w)
        self.assertIs(fp8_scale, pre_s)

    def test_on_the_fly_transpose_matches_hand_derived_values(self):
        # expert_num = fp8_weight.shape[0] // weight.shape[0] = 4 // 2 = 2.
        # weight = arange(12).reshape(4,3) -> reshape(2,2,3) -> transpose(0,2,1)
        #        -> reshape(-1,2). Hand-derived below.
        weight = _CachedWeight()
        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(
            weight, transpose=True
        )
        expected_weight = np.array(
            [
                [0.0, 3.0],
                [1.0, 4.0],
                [2.0, 5.0],
                [6.0, 9.0],
                [7.0, 10.0],
                [8.0, 11.0],
            ],
            dtype=np.float32,
        )
        # scale = arange(8).reshape(4,2) -> reshape(2,2,2) -> transpose(0,2,1)
        #       -> reshape(-1,2).
        expected_scale = np.array(
            [
                [0.0, 2.0],
                [1.0, 3.0],
                [4.0, 6.0],
                [5.0, 7.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(fp8_weight.numpy(), expected_weight)
        np.testing.assert_array_equal(fp8_scale.numpy(), expected_scale)

    def test_num_expert_override_changes_transpose_grouping(self):
        # Forcing num_expert=4 (vs the auto 2) must be consumed: h0 = 4//4 = 1,
        # so each row becomes its own group and the transpose is a no-op layout
        # collapse to a single column. If num_expert were ignored the shape
        # would be [6, 2] instead of [12, 1].
        weight = _CachedWeight()
        fp8_weight, fp8_scale = _get_fp8_weight_and_scale(
            weight, transpose=True, num_expert=4
        )
        self.assertEqual(fp8_weight.shape, [12, 1])
        self.assertEqual(fp8_scale.shape, [8, 1])
        np.testing.assert_array_equal(
            fp8_weight.numpy(),
            np.arange(12, dtype=np.float32).reshape(12, 1),
        )
        np.testing.assert_array_equal(
            fp8_scale.numpy(),
            np.arange(8, dtype=np.float32).reshape(8, 1),
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON or "")
class TestFusedStackQuantCachedPath(unittest.TestCase):
    """fused_stack_quant routes cached weights through _get_fp8_weight_and_scale
    and forwards transpose / num_expert. (Non-cached path needs CUDA ops.)"""

    def test_cached_no_transpose_returns_stacked_objects(self):
        weight = _CachedWeight()
        fp8_weight, fp8_scale = fused_stack_quant([weight])
        self.assertIs(fp8_weight, weight.fp8_weight_stacked)
        self.assertIs(fp8_scale, weight.fp8_scale_stacked)

    def test_cached_transpose_forwarded_to_helper(self):
        weight = _CachedWeight()
        fp8_weight, fp8_scale = fused_stack_quant([weight], transpose=True)
        expected_weight = np.array(
            [
                [0.0, 3.0],
                [1.0, 4.0],
                [2.0, 5.0],
                [6.0, 9.0],
                [7.0, 10.0],
                [8.0, 11.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(fp8_weight.numpy(), expected_weight)
        self.assertEqual(fp8_scale.shape, [4, 2])


class _CustomMap:
    """Minimal custom_map with distinguishable expert identities."""

    def __init__(self):
        self.experts = ["expert0", "expert1", "expert2"]
        self.grouped_gemm_experts = "grouped-object"


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON or "")
class TestExpertsGroupGemmContiguousNode(unittest.TestCase):
    """Construction routing, activation/deferral validation, cache slots,
    reset_state, and gen_m_indices."""

    def test_expert_id_selects_that_single_expert(self):
        node = ExpertsGroupGemmContiguousNode(
            _CustomMap(),
            expert_id=2,
            moe_subbatch_token_num_after_dispatch=FP8_ALIGN,
        )
        # Must pick exactly experts[2], not experts[0] or the whole list.
        self.assertEqual(node.experts, ["expert2"])
        self.assertEqual(node.expert_id, 2)

    def test_no_expert_id_keeps_full_expert_list(self):
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        self.assertEqual(node.experts, ["expert0", "expert1", "expert2"])

    def test_fusion_path_uses_grouped_experts_and_omits_experts_attr(self):
        node = ExpertsGroupGemmContiguousNode(
            _CustomMap(),
            use_fp8_mlp=False,
            moe_expert_fusion=True,
        )
        self.assertEqual(node.grouped_gemm_experts, "grouped-object")
        self.assertFalse(hasattr(node, "experts"))
        # is_split_group_gemm is the negation of moe_expert_fusion.
        self.assertFalse(node.is_split_group_gemm)

    def test_invalid_activation_type_is_rejected(self):
        with self.assertRaises(ValueError):
            ExpertsGroupGemmContiguousNode(
                _CustomMap(), activation_type="not_an_activation"
            )

    def test_subbatch_token_num_must_be_aligned(self):
        # Non-multiple of FP8_ALIGN must trip the constructor assertion.
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(
                _CustomMap(),
                moe_subbatch_token_num_after_dispatch=FP8_ALIGN + 1,
            )

    def test_dw_deferral_without_deep_gemm_is_rejected(self):
        # Deferral is a silent no-op without the deep_gemm wgrad path, so the
        # constructor must reject it rather than accept a dead flag.
        with self.assertRaises(ValueError):
            ExpertsGroupGemmContiguousNode(
                _CustomMap(),
                defer_expert_up_gate_dw=True,
                moe_deep_gemm=False,
            )

    def test_cached_tensors_reads_fields_in_documented_order(self):
        # Set each slot to a distinguishable sentinel, then verify the READ
        # contract maps field -> position correctly (catches a getter swap).
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        node.tokens_per_expert = "TPE"
        node.m_indices = "MIDX"
        node.input = "IN"
        node.input_fp8 = "IN_FP8"
        node.input_scale = "IN_SCALE"
        node.o1 = "O1"
        self.assertEqual(
            node.cached_tensors(),
            ["TPE", "MIDX", "IN", "IN_FP8", "IN_SCALE", "O1"],
        )

    def test_set_cached_tensors_writes_fields_in_documented_order(self):
        # Verify the WRITE contract maps position -> field (catches a setter
        # swap independently of the getter).
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        node.set_cached_tensors(["a", "b", "c", "d", "e", "f"])
        self.assertEqual(node.tokens_per_expert, "a")
        self.assertEqual(node.m_indices, "b")
        self.assertEqual(node.input, "c")
        self.assertEqual(node.input_fp8, "d")
        self.assertEqual(node.input_scale, "e")
        self.assertEqual(node.o1, "f")

    def test_clear_cached_tensors_nulls_every_slot(self):
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        node.set_cached_tensors([1, 2, 3, 4, 5, 6])
        node.clear_cached_tensors()
        self.assertEqual(node.cached_tensors(), [None] * 6)
        self.assertIsNone(node.tokens_per_expert)
        self.assertIsNone(node.o1)

    def test_reset_state_clears_indices_and_activations(self):
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        node.tokens_per_expert = paddle.to_tensor([2, 0, 1], dtype="int32")
        node.m_indices = node.gen_m_indices(node.tokens_per_expert)
        node.input = paddle.ones([1, 2])
        node.input_fp8 = paddle.ones([1, 2])
        node.input_scale = paddle.ones([1, 1])
        node.o1 = paddle.ones([1, 2])

        node.reset_state()

        self.assertIsNone(node.tokens_per_expert)
        self.assertIsNone(node.m_indices)
        self.assertIsNone(node.input)
        self.assertIsNone(node.input_fp8)
        self.assertIsNone(node.input_scale)
        self.assertIsNone(node.o1)

    def test_gen_m_indices_expands_counts_to_expert_ids(self):
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        # counts [1, 3, 2] -> expert 0 once, expert 1 thrice, expert 2 twice.
        out = node.gen_m_indices(paddle.to_tensor([1, 3, 2], dtype="int32"))
        np.testing.assert_array_equal(
            out.numpy(), np.array([0, 1, 1, 1, 2, 2], dtype=np.int32)
        )

    def test_gen_m_indices_skips_empty_experts(self):
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        # A zero-count expert (middle) must contribute no indices.
        out = node.gen_m_indices([2, 0, 1])
        np.testing.assert_array_equal(
            out.numpy(), np.array([0, 0, 2], dtype=np.int32)
        )

    def test_gen_m_indices_empty_counts_returns_empty(self):
        node = ExpertsGroupGemmContiguousNode(_CustomMap())
        out = node.gen_m_indices(paddle.empty([0], dtype="int32"))
        self.assertEqual(out.shape, [0])


if __name__ == "__main__":
    unittest.main()
