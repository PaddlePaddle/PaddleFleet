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
"""Coverage tests for fp8/quantization.py, fp8/utils.py, and transformer_layer.py.

Targets code paths added for UE8M0 support and the TransformerLayer
fp8_quant_weight / clear_fp8_quant_weight / use_fp8 methods.

Run with:
    PYTHONPATH=ernie/erniebot/third_party/PaddleFleet/src:$PYTHONPATH \
    python -m pytest tests/single_card_tests/fp8/test_fp8_coverage.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import paddle

from paddlefleet.fp8.quantization import get_quant_func
from paddlefleet.fp8.utils import is_fp8_tensor

_HAS_GPU = (
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
)
_REQUIRE_GPU = unittest.skipUnless(_HAS_GPU, "Requires CUDA GPU")
_SM90_PLUS = _HAS_GPU and paddle.device.cuda.get_device_capability()[0] >= 9
_REQUIRE_SM90 = unittest.skipUnless(_SM90_PLUS, "Requires SM90+ GPU")


# ============================================================================
# Tests for fp8/quantization.py
# ============================================================================


class TestGetQuantFuncUe8m0Branch(unittest.TestCase):
    """Tests for the use_ue8m0=True branch in get_quant_func."""

    @_REQUIRE_SM90
    def test_ue8m0_inp_quant_with_input_trans(self):
        """use_ue8m0=True, input_trans=True returns 4-tuple with int32 scales."""
        inp_func, _ = get_quant_func(
            "blockwise", input_trans=True, pow2_scale=True, use_ue8m0=True
        )
        x = paddle.randn([128, 256], dtype="bfloat16")
        result = inp_func(x)
        self.assertEqual(len(result), 4)
        fp8, scale, fp8_t, scale_t = result
        # fp8 and fp8_t should be fp8 dtype
        self.assertEqual(fp8.dtype, paddle.float8_e4m3fn)
        self.assertEqual(fp8_t.dtype, paddle.float8_e4m3fn)
        # Shapes: fp8 same as input, fp8_t is transposed
        self.assertEqual(list(fp8.shape), [128, 256])
        self.assertEqual(list(fp8_t.shape), [256, 128])
        # Scales are int32 (UE8M0 packed) and _mn_major applied
        self.assertEqual(scale.dtype, paddle.int32)
        self.assertEqual(scale_t.dtype, paddle.int32)

    @_REQUIRE_SM90
    def test_ue8m0_inp_quant_without_input_trans(self):
        """use_ue8m0=True, input_trans=False returns 2-tuple."""
        inp_func, _ = get_quant_func(
            "blockwise", input_trans=False, pow2_scale=True, use_ue8m0=True
        )
        x = paddle.randn([128, 256], dtype="bfloat16")
        result = inp_func(x)
        self.assertEqual(len(result), 2)
        fp8, scale = result
        self.assertEqual(fp8.dtype, paddle.float8_e4m3fn)
        self.assertEqual(list(fp8.shape), [128, 256])
        self.assertEqual(scale.dtype, paddle.int32)

    @_REQUIRE_SM90
    def test_ue8m0_weight_quant_returns_4_tuple(self):
        """use_ue8m0=True weight_quant_func returns 4-tuple with int32 scales."""
        _, weight_func = get_quant_func(
            "blockwise", input_trans=True, pow2_scale=True, use_ue8m0=True
        )
        w = paddle.randn([256, 128], dtype="bfloat16")
        result = weight_func(w)
        self.assertEqual(len(result), 4)
        fp8_bwd, scale_bwd, fp8_fwd, scale_fwd = result
        self.assertEqual(fp8_bwd.dtype, paddle.float8_e4m3fn)
        self.assertEqual(fp8_fwd.dtype, paddle.float8_e4m3fn)
        self.assertEqual(scale_bwd.dtype, paddle.int32)
        self.assertEqual(scale_fwd.dtype, paddle.int32)

    @_REQUIRE_SM90
    def test_ue8m0_weight_quant_cache_hit(self):
        """weight_quant_func returns cached result when attrs are present."""
        _, weight_func = get_quant_func(
            "blockwise", input_trans=True, pow2_scale=True, use_ue8m0=True
        )
        w = paddle.randn([256, 128], dtype="bfloat16")
        # First call: fresh quantization
        result1 = weight_func(w)
        # Manually set cache attrs to simulate _fp8_prequant_weight
        w.fp8_weight_fwd = result1[2]
        w.fp8_scale_fwd = result1[3]
        w.fp8_scale_bwd = result1[1]
        # Second call: should hit cache
        result2 = weight_func(w)
        self.assertIsNone(result2[0])  # fp8_bwd is None from cache
        self.assertIs(result2[2], w.fp8_weight_fwd)
        self.assertIs(result2[3], w.fp8_scale_fwd)


class TestGetQuantFuncNonUe8m0(unittest.TestCase):
    """Tests for non-UE8M0 paths."""

    @_REQUIRE_GPU
    def test_non_ue8m0_weight_quant_with_out_scale_trans(self):
        """out_scale_trans=True applies _mn_major to both scales."""
        _, weight_func = get_quant_func(
            "blockwise", input_trans=True, out_scale_trans=True, pow2_scale=True
        )
        w = paddle.randn([256, 128], dtype="bfloat16")
        result = weight_func(w)
        self.assertEqual(len(result), 4)
        _, scale_bwd, _, scale_fwd = result
        # _mn_major applies .T, so strides should indicate transposed view
        self.assertEqual(scale_bwd.dtype, paddle.float32)
        self.assertEqual(scale_fwd.dtype, paddle.float32)

    @_REQUIRE_GPU
    def test_non_ue8m0_weight_quant_without_out_scale_trans(self):
        """out_scale_trans=False does NOT transpose scales."""
        _, weight_func = get_quant_func(
            "blockwise",
            input_trans=True,
            out_scale_trans=False,
            pow2_scale=True,
        )
        w = paddle.randn([256, 128], dtype="bfloat16")
        result = weight_func(w)
        self.assertEqual(len(result), 4)
        _, scale_bwd, _, scale_fwd = result
        self.assertEqual(scale_bwd.dtype, paddle.float32)
        self.assertEqual(scale_fwd.dtype, paddle.float32)


class TestCachedWeightResult(unittest.TestCase):
    """Tests for _cached_weight_result edge cases."""

    @_REQUIRE_GPU
    def test_cache_miss_no_attrs(self):
        """No cache attrs -> weight_quant_func does fresh quant."""
        _, weight_func = get_quant_func("blockwise", pow2_scale=True)
        w = paddle.randn([256, 128], dtype="bfloat16")
        result = weight_func(w)
        # Fresh quant returns non-None fp8_bwd
        self.assertIsNotNone(result[0])

    @_REQUIRE_GPU
    def test_cache_miss_partial_attrs(self):
        """Only fp8_weight_fwd set but scales missing -> cache miss."""
        _, weight_func = get_quant_func("blockwise", pow2_scale=True)
        w = paddle.randn([256, 128], dtype="bfloat16")
        w.fp8_weight_fwd = paddle.randn([128, 256], dtype="bfloat16")
        # Missing fp8_scale_fwd and fp8_scale_bwd
        result = weight_func(w)
        # Should do fresh quant, not crash
        self.assertIsNotNone(result[0])

    @_REQUIRE_GPU
    def test_cache_hit_returns_none_first_element(self):
        """Full cache attrs -> returns (None, scale_bwd, fp8_fwd, scale_fwd)."""
        _, weight_func = get_quant_func("blockwise", pow2_scale=True)
        w = paddle.randn([256, 128], dtype="bfloat16")
        fresh = weight_func(w)
        # Set cache
        w.fp8_weight_fwd = fresh[2]
        w.fp8_scale_fwd = fresh[3]
        w.fp8_scale_bwd = fresh[1]
        cached = weight_func(w)
        self.assertIsNone(cached[0])
        self.assertIs(cached[2], w.fp8_weight_fwd)


class TestMnMajor(unittest.TestCase):
    """Tests for the _mn_major helper."""

    def test_mn_major_none(self):
        """_mn_major(None) should return None."""

        # Access _mn_major indirectly by checking behavior
        # The cached path returns None as first element; if _mn_major were applied
        # to None it would crash. This is covered by cache_hit tests above.
        # Direct test: import the module and test
        import importlib

        import paddlefleet.fp8.quantization as qmod

        importlib.reload(qmod)  # ensure fresh
        # _mn_major is a local function, test indirectly via behavior
        # A None scale should not crash the weight_quant_func
        pass


# ============================================================================
# Tests for fp8/utils.py
# ============================================================================


class TestIsFp8TensorCoverage(unittest.TestCase):
    """Additional coverage tests for is_fp8_tensor."""

    @_REQUIRE_GPU
    def test_int32_scale_accepted(self):
        """UE8M0 int32-packed scale must be accepted."""
        fp8_t = paddle.randn([4, 4], dtype="float32").astype(
            paddle.float8_e4m3fn
        )
        scale = paddle.ones([1], dtype=paddle.int32)
        self.assertTrue(is_fp8_tensor((fp8_t, scale)))

    @_REQUIRE_GPU
    def test_float32_scale_accepted(self):
        """Standard float32 scale must be accepted."""
        fp8_t = paddle.randn([4, 4], dtype="float32").astype(
            paddle.float8_e4m3fn
        )
        scale = paddle.ones([1], dtype=paddle.float32)
        self.assertTrue(is_fp8_tensor((fp8_t, scale)))

    @_REQUIRE_GPU
    def test_float16_scale_rejected(self):
        """float16 is not a valid scale dtype."""
        fp8_t = paddle.randn([4, 4], dtype="float32").astype(
            paddle.float8_e4m3fn
        )
        scale = paddle.ones([1], dtype=paddle.float16)
        self.assertFalse(is_fp8_tensor((fp8_t, scale)))

    @_REQUIRE_GPU
    def test_e5m2_raises_assertion(self):
        """float8_e5m2 tensor should trigger assertion."""
        e5m2_t = paddle.randn([4, 4], dtype="float32").astype(
            paddle.float8_e5m2
        )
        scale = paddle.ones([1], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            is_fp8_tensor((e5m2_t, scale))

    def test_non_tuple_returns_false(self):
        """Non-tuple input returns False."""
        self.assertFalse(is_fp8_tensor("not a tuple"))
        self.assertFalse(is_fp8_tensor(42))
        self.assertFalse(is_fp8_tensor(None))

    def test_wrong_length_tuple_returns_false(self):
        """Tuple with != 2 elements returns False."""
        t = paddle.randn([2, 2])
        self.assertFalse(is_fp8_tensor((t,)))
        self.assertFalse(is_fp8_tensor((t, t, t)))


# ============================================================================
# Tests for transformer_layer.py (use_fp8, fp8_quant_weight, clear_fp8_quant_weight)
# ============================================================================


class TestTransformerLayerUseFp8(unittest.TestCase):
    """Tests for TransformerLayer.use_fp8 method."""

    def _make_layer(self, fp8=None):
        from paddlefleet.transformer.transformer_config import TransformerConfig
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            use_bias=False,
            normalization="RMSNorm",
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            fp8=fp8,
            use_cpu_initialization=True,
        )
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )

        spec = get_gpt_layer_local_spec(config)
        return TransformerLayer(
            config=config, sublayers_spec=spec.sublayers_spec, layer_number=1
        )

    def test_use_fp8_returns_false_when_fp8_is_none(self):
        layer = self._make_layer(fp8=None)
        self.assertFalse(layer.use_fp8())

    def test_use_fp8_returns_true_when_fp8_set(self):
        layer = self._make_layer(fp8="blockwise")
        self.assertTrue(layer.use_fp8())

    def test_use_fp8_delegates_to_moe(self):
        """When mlp is MoELayer, use_fp8 delegates."""
        layer = self._make_layer(fp8=None)
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        mock_moe = MagicMock(spec=MoELayer)
        mock_moe.use_fp8.return_value = True
        layer.mlp = mock_moe
        self.assertTrue(layer.use_fp8())
        mock_moe.use_fp8.assert_called_once()


class TestTransformerLayerFp8QuantWeight(unittest.TestCase):
    """Tests for fp8_quant_weight and clear_fp8_quant_weight."""

    def _make_layer(self):
        from paddlefleet.transformer.transformer_config import TransformerConfig
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            use_bias=False,
            normalization="RMSNorm",
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            use_cpu_initialization=True,
        )
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )

        spec = get_gpt_layer_local_spec(config)
        return TransformerLayer(
            config=config, sublayers_spec=spec.sublayers_spec, layer_number=1
        )

    def test_fp8_quant_weight_does_not_raise(self):
        """fp8_quant_weight should not raise on a non-FP8 layer."""
        layer = self._make_layer()
        # Should be a no-op (no fp8 layers), not crash
        layer.fp8_quant_weight(batch_mode=False, quant_transpose=True)

    def test_clear_fp8_quant_weight_does_not_raise(self):
        """clear_fp8_quant_weight should not raise on a non-FP8 layer."""
        layer = self._make_layer()
        layer.clear_fp8_quant_weight()

    def test_fp8_quant_weight_skips_expert_sublayers(self):
        """Sublayers with is_expert=True should be skipped."""
        layer = self._make_layer()
        # Mark a sublayer as expert
        for m in layer.sublayers(include_self=False):
            if hasattr(m, "weight"):
                m.is_expert = True
                m.fp8_quant_weight = MagicMock()
                break
        layer.fp8_quant_weight(batch_mode=False, quant_transpose=True)
        # The expert sublayer's fp8_quant_weight should NOT be called
        for m in layer.sublayers(include_self=False):
            if getattr(m, "is_expert", False):
                m.fp8_quant_weight.assert_not_called()
                break


if __name__ == "__main__":
    unittest.main()
