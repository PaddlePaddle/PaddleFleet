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
"""Behavior tests for ``transformer/moe/fp8_utils`` (计算优化 / MoE 模块).

These cover the CPU-executable control logic of the FP8 MoE expert path whose
correctness can be pinned by hand-derived expectations, independent of the
implementation:

* ``has_config`` -- the truthy-and-present guard used before touching optional
  fp8 fused-op config entries. Wrong short-circuit order would silently enable
  or disable a fused kernel.
* ``moe_token_padding_alignment`` -- decides whether each expert's GEMM ``M``
  dim is padded to ``FP8_ALIGN`` (128) or left unpadded (1). The exact boolean
  combination matters: only the accuracy-compatible pure-bf16 non-grouped path
  may skip padding, everything else must align, so we assert the full truth
  table against a spec-derived expected, not the code's own expression.
* ``ExpertsGroupGemmContiguousNode.gen_m_indices`` -- maps ``tokens_per_expert``
  to the per-token expert-id vector consumed by the grouped GEMM. Expected
  vectors are derived from MoE grouped-gemm semantics (token row -> owning
  expert), not by calling the method.
* the cached-tensor lifecycle (``set_cached_tensors`` / ``cached_tensors`` /
  ``clear_activation_tensors`` / ``reset_state``) -- a slot-ordering contract.
  A swapped slot or a clear that wipes the wrong field would corrupt the
  backward's cached activations, so we set distinguishable values through the
  real setter and observe exactly which slots survive each clear.
* ``_get_fp8_weight_and_scale`` -- selects the transpose vs non-transpose fp8
  weight/scale cache. We check object identity so a swapped selection is caught.
* the ``__init__`` fail-fast contracts (unknown ``activation_type`` and the
  sub-batch token-count alignment assertion).

The module imports ``paddle`` and ``paddlefleet_ops`` at module scope; when
those are unavailable (e.g. a CPU-only dev box without Paddle) every test is
skipped with an honest reason rather than reported as passing. The GPU FP8
kernels themselves (``kitchen_gemm``, ``fwd_*`` / ``bwd_*`` GEMM math) are NOT
exercised here -- those require an H20 single-card run with real tensors.
"""

import types
import unittest

try:
    import paddle

    from paddlefleet.transformer.moe.fp8_utils import (
        FP8_ALIGN,
        ExpertsGroupGemmContiguousNode,
        _get_fp8_weight_and_scale,
        has_config,
        moe_token_padding_alignment,
    )

    HAS_PADDLE = True
    IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet_ops not installed
    HAS_PADDLE = False
    IMPORT_ERROR = repr(exc)


def _make_custom_map(num_experts):
    """A minimal stand-in for the CustomMapping collaborator.

    Only ``.experts`` is read on the pure-bf16 non-fusion path; ``layer_number``
    and ``config`` fall back to ``getattr`` defaults. Distinct sentinel objects
    make an accidental expert-list reordering observable.
    """
    experts = [types.SimpleNamespace(_tag=i) for i in range(num_experts)]
    return types.SimpleNamespace(experts=experts)


def _make_node(num_experts=2, **kwargs):
    """Construct a bf16, non-fused node -- the CPU-constructible control path."""
    defaults = {"use_fp8_mlp": False, "moe_expert_fusion": False}
    defaults.update(kwargs)
    return ExpertsGroupGemmContiguousNode(
        _make_custom_map(num_experts), **defaults
    )


@unittest.skipUnless(
    HAS_PADDLE, f"paddle/paddlefleet_ops unavailable: {IMPORT_ERROR}"
)
class TestHasConfig(unittest.TestCase):
    """``has_config`` must require present AND truthy, and tolerate None map."""

    def test_present_and_truthy_is_true(self):
        self.assertIs(has_config({"a": 1, "b": 0}, "a"), True)

    def test_present_but_falsy_is_false(self):
        # key exists but value is falsy -> feature must stay off
        self.assertIs(has_config({"a": 0}, "a"), False)
        self.assertIs(has_config({"a": False}, "a"), False)
        self.assertIs(has_config({"a": None}, "a"), False)
        self.assertIs(has_config({"a": ""}, "a"), False)

    def test_missing_key_is_false(self):
        self.assertIs(has_config({"a": 1}, "b"), False)

    def test_none_map_is_false(self):
        # None must short-circuit before the ``key in config_map`` membership
        # test; otherwise this would raise instead of returning False.
        self.assertIs(has_config(None, "a"), False)

    def test_returns_plain_bool_not_truthy_value(self):
        # Contract is a bool, not the stored object -- callers use ``is True``.
        result = has_config({"a": "megatron"}, "a")
        self.assertIsInstance(result, bool)
        self.assertIs(result, True)


@unittest.skipUnless(
    HAS_PADDLE, f"paddle/paddlefleet_ops unavailable: {IMPORT_ERROR}"
)
class TestMoeTokenPaddingAlignment(unittest.TestCase):
    """Only the accuracy-compatible pure-bf16 non-grouped path skips padding."""

    def test_full_truth_table(self):
        # Independent expected: return 1 iff
        #   use_accuracy_compatible AND (not use_fp8_mlp) AND (not grouped_gemm),
        # else FP8_ALIGN. Enumerate all 8 combinations against that rule.
        self.assertEqual(FP8_ALIGN, 128)
        for use_fp8_mlp in (False, True):
            for moe_grouped_gemm in (False, True):
                for use_acc in (False, True):
                    expected = (
                        1
                        if (
                            use_acc and not use_fp8_mlp and not moe_grouped_gemm
                        )
                        else FP8_ALIGN
                    )
                    got = moe_token_padding_alignment(
                        use_fp8_mlp=use_fp8_mlp,
                        moe_grouped_gemm=moe_grouped_gemm,
                        use_accuracy_compatible=use_acc,
                    )
                    self.assertEqual(
                        got,
                        expected,
                        msg=(
                            f"fp8={use_fp8_mlp} grouped={moe_grouped_gemm} "
                            f"acc={use_acc}: expected {expected}, got {got}"
                        ),
                    )

    def test_only_skip_case_returns_one(self):
        # The single combination that must return 1.
        self.assertEqual(
            moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=False,
                use_accuracy_compatible=True,
            ),
            1,
        )

    def test_accuracy_compatible_alone_is_not_enough(self):
        # Turning on accuracy-compat but keeping fp8 (or grouped) must still pad.
        self.assertEqual(
            moe_token_padding_alignment(
                use_fp8_mlp=True,
                moe_grouped_gemm=False,
                use_accuracy_compatible=True,
            ),
            FP8_ALIGN,
        )
        self.assertEqual(
            moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=True,
                use_accuracy_compatible=True,
            ),
            FP8_ALIGN,
        )


@unittest.skipUnless(
    HAS_PADDLE, f"paddle/paddlefleet_ops unavailable: {IMPORT_ERROR}"
)
class TestGenMIndices(unittest.TestCase):
    """``gen_m_indices`` maps token counts to per-token owning-expert ids."""

    def test_list_counts_expand_to_expert_ids(self):
        node = _make_node(num_experts=3)
        # tokens_per_expert=[2,0,3]: rows 0,1 -> expert 0; expert 1 gets none;
        # rows 2,3,4 -> expert 2. Derived from grouped-gemm row ownership.
        indices = node.gen_m_indices([2, 0, 3])
        self.assertEqual(indices.tolist(), [0, 0, 2, 2, 2])
        self.assertEqual(indices.dtype, paddle.int32)

    def test_tensor_counts_expand_the_same_way(self):
        node = _make_node(num_experts=2)
        counts = paddle.to_tensor([1, 2], dtype="int64")
        indices = node.gen_m_indices(counts)
        self.assertEqual(indices.tolist(), [0, 1, 1])
        self.assertEqual(indices.dtype, paddle.int32)

    def test_empty_counts_list_returns_empty(self):
        node = _make_node(num_experts=1)
        indices = node.gen_m_indices([])
        self.assertEqual(indices.shape, [0])

    def test_all_zero_counts_returns_empty(self):
        # Distinct from the empty-list branch: here shape[0] != 0 but every
        # expert receives zero tokens, so the interleave yields no rows.
        node = _make_node(num_experts=2)
        indices = node.gen_m_indices([0, 0])
        self.assertEqual(indices.shape, [0])

    def test_single_expert_gets_all_rows(self):
        node = _make_node(num_experts=1)
        indices = node.gen_m_indices([4])
        self.assertEqual(indices.tolist(), [0, 0, 0, 0])


@unittest.skipUnless(
    HAS_PADDLE, f"paddle/paddlefleet_ops unavailable: {IMPORT_ERROR}"
)
class TestCachedTensorLifecycle(unittest.TestCase):
    """The 6-slot cache order and per-clear semantics are a real contract."""

    # Slot order per the production tuple:
    #   0 tokens_per_expert, 1 m_indices, 2 input,
    #   3 input_fp8, 4 input_scale, 5 o1
    def _distinct_values(self):
        # Six distinguishable sentinels so a swapped slot is observable.
        return [
            "tokens_per_expert",
            "m_indices",
            "input",
            "input_fp8",
            "input_scale",
            "o1",
        ]

    def test_set_then_get_preserves_slot_order(self):
        node = _make_node()
        values = self._distinct_values()
        node.set_cached_tensors(values)
        # Read back through the real getter; each named attribute must hold the
        # value placed in its slot (catches an unpacking-order swap).
        self.assertEqual(node.tokens_per_expert, "tokens_per_expert")
        self.assertEqual(node.m_indices, "m_indices")
        self.assertEqual(node.input, "input")
        self.assertEqual(node.input_fp8, "input_fp8")
        self.assertEqual(node.input_scale, "input_scale")
        self.assertEqual(node.o1, "o1")
        self.assertEqual(node.cached_tensors(), values)

    def test_clear_activation_keeps_persistent_slots(self):
        node = _make_node()
        node.set_cached_tensors(self._distinct_values())
        node.clear_activation_tensors()
        # Persistent routing state (slots 0,1) must survive; only the four
        # activation slots (2..5) are dropped.
        self.assertEqual(node.tokens_per_expert, "tokens_per_expert")
        self.assertEqual(node.m_indices, "m_indices")
        self.assertEqual(
            node.cached_tensors(),
            ["tokens_per_expert", "m_indices", None, None, None, None],
        )

    def test_reset_state_clears_all_slots(self):
        node = _make_node()
        node.set_cached_tensors(self._distinct_values())
        node.reset_state()
        # reset_state drops routing state and delegates activations to
        # clear_activation_tensors, so every slot ends up None.
        self.assertEqual(node.cached_tensors(), [None] * 6)

    def test_clear_cached_tensors_clears_all_slots(self):
        node = _make_node()
        node.set_cached_tensors(self._distinct_values())
        node.clear_cached_tensors()
        self.assertEqual(node.cached_tensors(), [None] * 6)


@unittest.skipUnless(
    HAS_PADDLE, f"paddle/paddlefleet_ops unavailable: {IMPORT_ERROR}"
)
class TestGetFp8WeightAndScale(unittest.TestCase):
    """Selects the transpose vs non-transpose fp8 weight/scale cache by field."""

    def _weight_with_caches(self):
        # Distinguishable real tensors (CPU); the helper only *selects* among
        # them, no GEMM runs, so this is CPU-safe.
        w = types.SimpleNamespace()
        w.fp8_weight_stacked = paddle.to_tensor([1.0, 2.0])
        w.fp8_scale_stacked = paddle.to_tensor([10.0])
        w.fp8_weight_stacked_transpose = paddle.to_tensor([3.0, 4.0])
        w.fp8_scale_stacked_transpose = paddle.to_tensor([20.0])
        return w

    def test_non_transpose_returns_direct_cache(self):
        w = self._weight_with_caches()
        weight, scale = _get_fp8_weight_and_scale(w, transpose=False)
        self.assertIs(weight, w.fp8_weight_stacked)
        self.assertIs(scale, w.fp8_scale_stacked)

    def test_transpose_returns_precomputed_transpose_cache(self):
        w = self._weight_with_caches()
        weight, scale = _get_fp8_weight_and_scale(w, transpose=True)
        # Must pick the *transpose* buffers, not the plain ones.
        self.assertIs(weight, w.fp8_weight_stacked_transpose)
        self.assertIs(scale, w.fp8_scale_stacked_transpose)
        self.assertEqual(weight.tolist(), [3.0, 4.0])
        self.assertEqual(scale.tolist(), [20.0])


@unittest.skipUnless(
    HAS_PADDLE, f"paddle/paddlefleet_ops unavailable: {IMPORT_ERROR}"
)
class TestNodeInitContracts(unittest.TestCase):
    """__init__ fail-fast guards must reject unusable configurations."""

    def test_unknown_activation_type_rejected(self):
        # An unrecognised activation would silently fall through to SwiGLU in
        # every downstream dispatch, so it must be rejected at construction.
        with self.assertRaises(ValueError):
            _make_node(activation_type="not_a_real_activation")

    def test_supported_activation_type_accepted(self):
        for name in ("swiglu", "geglu", "situ"):
            node = _make_node(activation_type=name)
            self.assertEqual(node.activation_type, name)

    def test_subbatch_token_num_must_be_positive(self):
        with self.assertRaises(AssertionError):
            _make_node(moe_subbatch_token_num_after_dispatch=-1)

    def test_subbatch_token_num_must_be_align_multiple(self):
        # 127 is positive but not a multiple of FP8_ALIGN (128).
        with self.assertRaises(AssertionError):
            _make_node(moe_subbatch_token_num_after_dispatch=127)

    def test_subbatch_token_num_aligned_value_accepted(self):
        node = _make_node(moe_subbatch_token_num_after_dispatch=FP8_ALIGN)
        self.assertEqual(node.moe_subbatch_token_num_after_dispatch, FP8_ALIGN)

    def test_expert_id_selects_single_expert(self):
        # With expert_id set on the non-fused path, only that expert is kept.
        node = _make_node(num_experts=3, expert_id=1)
        self.assertEqual(len(node.experts), 1)
        self.assertEqual(node.experts[0]._tag, 1)


if __name__ == "__main__":
    unittest.main()
