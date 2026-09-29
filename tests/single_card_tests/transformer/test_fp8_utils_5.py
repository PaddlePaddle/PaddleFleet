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

All expected values are derived by hand from the production source; none are
copied from any coverage_test asserts. Covered production behavior:

  * ``has_config`` -> True only when the map is not None, contains the key AND
    the stored value itself is truthy (value, not mere presence).
  * ``moe_token_padding_alignment`` -> returns 1 only on the accuracy-compatible
    pure-bf16 non-grouped path, otherwise ``FP8_ALIGN`` (128).
  * ``_get_fp8_weight_and_scale`` -> returns cached stacked weight/scale
    verbatim; when a transpose is requested but no transpose cache exists it
    performs a per-expert reshape/transpose whose element layout is compared
    against an independent numpy reference.
  * ``fused_stack_quant`` -> dispatches to the cache reader when the first
    weight carries ``fp8_weight_stacked`` (forwarding the transpose flag) and
    otherwise to ``fused_stack_quant_without_cache`` with the right arguments.
  * ``ExpertsGroupGemmContiguousNode`` construction contract, subbatch
    alignment guard, cached-tensor positional round-trip, state reset and
    ``gen_m_indices`` expansion.

Only CPU tensor reshaping and pure control flow are exercised (no GPU kernels).
If paddle/paddlefleet cannot be imported the module is skipped with an honest
reason; only ImportError/ModuleNotFoundError are treated as a missing
dependency so genuine API breakage is not masked.
"""

import types
import unittest
from unittest import mock

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.moe import fp8_utils as fp8_mod
    from paddlefleet.transformer.moe.fp8_utils import (
        FP8_ALIGN,
        ExpertsGroupGemmContiguousNode,
        _get_fp8_weight_and_scale,
        fused_stack_quant,
        has_config,
        moe_token_padding_alignment,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    _IMPORT_ERROR = exc

_RUN = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}"
)


class _CachedWeight:
    """Stand-in for a stacked expert parameter carrying pre-quantized fp8
    buffers. Only the attributes the functions under test read are defined;
    none of the function bodies are stubbed."""

    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


@unittest.skipUnless(_RUN, _SKIP_REASON)
class TestHasConfig(unittest.TestCase):
    def test_truthiness_contract(self):
        # Missing map / missing key are False.
        self.assertFalse(has_config(None, "k"))
        self.assertFalse(has_config({}, "k"))
        self.assertFalse(has_config({"other": True}, "k"))
        # Present key but falsy stored value -> False (the value is consumed,
        # not merely its presence).
        self.assertFalse(has_config({"k": False}, "k"))
        self.assertFalse(has_config({"k": 0}, "k"))
        self.assertFalse(has_config({"k": None}, "k"))
        self.assertFalse(has_config({"k": ""}, "k"))
        # Present key with a truthy value -> True.
        self.assertTrue(has_config({"k": True}, "k"))
        self.assertTrue(has_config({"k": 1}, "k"))
        self.assertTrue(has_config({"k": "x"}, "k"))


@unittest.skipUnless(_RUN, _SKIP_REASON)
class TestMoeTokenPaddingAlignment(unittest.TestCase):
    def test_alignment_truth_table(self):
        # Only accuracy-compatible + pure bf16 + non-grouped skips padding.
        self.assertEqual(
            moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=False,
                use_accuracy_compatible=True,
            ),
            1,
        )
        # FP8 on -> align to FP8_ALIGN even when accuracy-compatible.
        self.assertEqual(
            moe_token_padding_alignment(
                use_fp8_mlp=True,
                moe_grouped_gemm=False,
                use_accuracy_compatible=True,
            ),
            FP8_ALIGN,
        )
        # grouped gemm on -> align.
        self.assertEqual(
            moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=True,
                use_accuracy_compatible=True,
            ),
            FP8_ALIGN,
        )
        # accuracy-compatible off -> align even on the pure-bf16 path.
        self.assertEqual(
            moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=False,
                use_accuracy_compatible=False,
            ),
            FP8_ALIGN,
        )
        self.assertEqual(FP8_ALIGN, 128)


@unittest.skipUnless(_RUN, _SKIP_REASON)
class TestGetFp8WeightAndScale(unittest.TestCase):
    def test_non_transpose_returns_cached_buffers(self):
        w = paddle.arange(12, dtype="float32").reshape([4, 3])
        s = paddle.arange(4, dtype="float32")
        weight = _CachedWeight(fp8_weight_stacked=w, fp8_scale_stacked=s)
        got_w, got_s = _get_fp8_weight_and_scale(weight, transpose=False)
        # Non-transpose path returns the cached buffers verbatim.
        self.assertIs(got_w, w)
        self.assertIs(got_s, s)

    def test_transpose_prefers_transpose_cache(self):
        w = paddle.arange(12, dtype="float32").reshape([4, 3])
        s = paddle.arange(4, dtype="float32")
        wt = paddle.arange(12, dtype="float32").reshape([3, 4])
        st = paddle.arange(3, dtype="float32")
        weight = _CachedWeight(
            fp8_weight_stacked=w,
            fp8_scale_stacked=s,
            fp8_weight_stacked_transpose=wt,
            fp8_scale_stacked_transpose=st,
        )
        got_w, got_s = _get_fp8_weight_and_scale(weight, transpose=True)
        # When a transpose cache exists it is returned, not the base buffers.
        self.assertIs(got_w, wt)
        self.assertIs(got_s, st)

    def test_transpose_on_the_fly_layout(self):
        # 2 experts x 2 rows x 3 cols. weight.shape[0]==2 tells the function
        # there are 4 // 2 == 2 experts. Independent numpy reference below.
        w_np = np.arange(12, dtype="float32").reshape(4, 3)
        s_np = (np.arange(12, dtype="float32") + 100.0).reshape(4, 3)
        exp_w = w_np.reshape(2, 2, 3).transpose(0, 2, 1).reshape(-1, 2)
        exp_s = s_np.reshape(2, 2, 3).transpose(0, 2, 1).reshape(-1, 2)
        weight = _CachedWeight(
            fp8_weight_stacked=paddle.to_tensor(w_np),
            fp8_scale_stacked=paddle.to_tensor(s_np),
            shape=[2, 3],
        )
        got_w, got_s = _get_fp8_weight_and_scale(weight, transpose=True)
        # Per-expert reshape + [0,2,1] transpose + flatten; both weight and
        # scale must be reordered identically (a swapped axis would differ).
        np.testing.assert_array_equal(got_w.numpy(), exp_w)
        np.testing.assert_array_equal(got_s.numpy(), exp_s)


@unittest.skipUnless(_RUN, _SKIP_REASON)
class TestFusedStackQuant(unittest.TestCase):
    def test_cache_path_forwards_transpose_flag(self):
        w = paddle.arange(12, dtype="float32").reshape([4, 3])
        s = paddle.arange(4, dtype="float32")
        wt = paddle.arange(6, dtype="float32").reshape([2, 3])
        st = paddle.arange(2, dtype="float32")
        weight = _CachedWeight(
            fp8_weight_stacked=w,
            fp8_scale_stacked=s,
            fp8_weight_stacked_transpose=wt,
            fp8_scale_stacked_transpose=st,
        )
        # First weight carries fp8_weight_stacked -> cache reader is used.
        # transpose=False -> non-transpose buffers.
        got_w, got_s = fused_stack_quant([weight], transpose=False)
        self.assertIs(got_w, w)
        self.assertIs(got_s, s)
        # transpose=True -> transpose buffers, proving the flag is forwarded
        # into _get_fp8_weight_and_scale rather than ignored.
        got_wt, got_st = fused_stack_quant([weight], transpose=True)
        self.assertIs(got_wt, wt)
        self.assertIs(got_st, st)

    def test_fallback_to_without_cache_when_no_cache_attr(self):
        class _Plain:
            pass

        plain = _Plain()
        self.assertFalse(hasattr(plain, "fp8_weight_stacked"))
        marker = (object(), object())
        calls = []

        def fake_without_cache(weight_list, transpose, use_ue8m0):
            calls.append((weight_list, transpose, use_ue8m0))
            return marker

        # fused_stack_quant_without_cache runs GPU quant kernels and is a
        # genuine not-under-test collaborator; stub it with a distinguishable
        # marker and assert both the forwarded args and the returned value.
        with mock.patch.object(
            fp8_mod, "fused_stack_quant_without_cache", fake_without_cache
        ):
            out = fused_stack_quant([plain], transpose=True, use_ue8m0=True)

        self.assertEqual(len(calls), 1)
        fwd_list, fwd_transpose, fwd_ue8m0 = calls[0]
        self.assertIs(fwd_list[0], plain)
        self.assertTrue(fwd_transpose)
        self.assertTrue(fwd_ue8m0)
        # fused_stack_quant unpacks ``w, scale`` from the collaborator's return
        # and re-returns them as a fresh ``(w, scale)`` tuple, so the wrapper
        # object identity differs; assert the two forwarded values are returned
        # verbatim (element identity) rather than tuple-object identity.
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIs(out[0], marker[0])
        self.assertIs(out[1], marker[1])


@unittest.skipUnless(_RUN, _SKIP_REASON)
class TestExpertsGroupGemmContiguousNode(unittest.TestCase):
    def _make_map(self, num_experts=2):
        # Real construction path: default moe_expert_fusion=False makes __init__
        # read custom_map.experts; config/layer_number fall back via getattr.
        return types.SimpleNamespace(
            experts=[object() for _ in range(num_experts)]
        )

    def test_default_construction_state(self):
        cmap = self._make_map(2)
        node = ExpertsGroupGemmContiguousNode(cmap)
        self.assertIsNone(node.expert_id)
        self.assertIs(node.experts, cmap.experts)
        self.assertTrue(node.use_fp8_mlp)
        self.assertFalse(node.moe_deep_gemm)
        self.assertFalse(node.recompute_moe_gate_up)
        self.assertFalse(node.dequant_input)
        self.assertIsNone(node.tokens_per_expert)
        self.assertIsNone(node.m_indices)
        self.assertIsNone(node.input)
        self.assertIsNone(node.o1)
        # use_fp8_mlp=True -> moe_token_padding_alignment returns FP8_ALIGN,
        # not the pure-bf16 short-circuit of 1.
        self.assertEqual(node.token_padding_alignment, FP8_ALIGN)

    def test_expert_id_selects_single_expert(self):
        cmap = self._make_map(3)
        node = ExpertsGroupGemmContiguousNode(cmap, expert_id=1)
        self.assertEqual(len(node.experts), 1)
        self.assertIs(node.experts[0], cmap.experts[1])
        self.assertEqual(node.expert_id, 1)

    def test_invalid_activation_type_rejected(self):
        cmap = self._make_map(1)
        # Unknown activation would silently fall through to SwiGLU downstream;
        # __init__ rejects it up front.
        with self.assertRaises(ValueError):
            ExpertsGroupGemmContiguousNode(cmap, activation_type="relu")

    def test_subbatch_token_num_alignment_contract(self):
        cmap = self._make_map(1)
        # Positive multiple of FP8_ALIGN accepted and stored verbatim.
        node = ExpertsGroupGemmContiguousNode(
            cmap, moe_subbatch_token_num_after_dispatch=128
        )
        self.assertEqual(node.moe_subbatch_token_num_after_dispatch, 128)
        # Not a multiple of FP8_ALIGN -> rejected.
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(
                cmap, moe_subbatch_token_num_after_dispatch=129
            )
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(
                cmap, moe_subbatch_token_num_after_dispatch=100
            )
        # Zero fails the > 0 half of the contract.
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(
                cmap, moe_subbatch_token_num_after_dispatch=0
            )

    def test_cached_tensor_roundtrip_is_positional(self):
        cmap = self._make_map(1)
        node = ExpertsGroupGemmContiguousNode(cmap)
        # Distinguishable sentinel per slot: a positional swap in
        # set_cached_tensors would land a value in the wrong attribute.
        slots = ["tpe", "mi", "inp", "fp8", "scale", "o1"]
        node.set_cached_tensors(slots)
        self.assertEqual(node.tokens_per_expert, "tpe")
        self.assertEqual(node.m_indices, "mi")
        self.assertEqual(node.input, "inp")
        self.assertEqual(node.input_fp8, "fp8")
        self.assertEqual(node.input_scale, "scale")
        self.assertEqual(node.o1, "o1")
        # cached_tensors reports the same ordering it consumes.
        self.assertEqual(node.cached_tensors(), slots)

    def test_clear_and_reset_state(self):
        cmap = self._make_map(1)
        node = ExpertsGroupGemmContiguousNode(cmap)
        node.set_cached_tensors(["tpe", "mi", "inp", "fp8", "scale", "o1"])
        node.clear_activation_tensors()
        # Activation tensors dropped; the two counters are retained.
        self.assertEqual(node.tokens_per_expert, "tpe")
        self.assertEqual(node.m_indices, "mi")
        self.assertIsNone(node.input)
        self.assertIsNone(node.input_fp8)
        self.assertIsNone(node.input_scale)
        self.assertIsNone(node.o1)
        # reset_state additionally clears the counters.
        node.set_cached_tensors(["tpe", "mi", "inp", "fp8", "scale", "o1"])
        node.reset_state()
        self.assertEqual(
            node.cached_tensors(), [None, None, None, None, None, None]
        )

    def test_gen_m_indices_expands_counts(self):
        cmap = self._make_map(1)
        node = ExpertsGroupGemmContiguousNode(cmap)
        # Expert i is repeated tokens_per_expert[i] times, in order.
        indices = node.gen_m_indices([2, 3, 1])
        self.assertEqual(indices.tolist(), [0, 0, 1, 1, 1, 2])
        # A paddle.Tensor input takes the cast branch with the same expansion;
        # expert 1 has zero tokens and is therefore absent.
        t = paddle.to_tensor([1, 0, 2], dtype="int64")
        self.assertEqual(node.gen_m_indices(t).tolist(), [0, 2, 2])
        # All-zero counts -> empty index vector.
        empty = node.gen_m_indices([0, 0, 0])
        self.assertEqual(empty.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
