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

import unittest
from types import SimpleNamespace

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.base import core

from paddlefleet.fp8.qat import fp4_indexer_fake_quant
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddlefleet.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from paddlefleet.transformer.mlp import MLPSublayersSpec
from paddlefleet.transformer.moe.fp8_utils import (
    ExpertsGroupGemmContiguousNode,
    apply_fp4_expert_fake_quant,
    fp4_expert_fake_quant,
)
from paddlefleet.transformer.moe.fusion_layer_utils import (
    FusionMoePyLayer,
    HybridEPMoePyLayer,
)
from paddlefleet.transformer.moe.moe_expert import StandardMLPExpert
from paddlefleet.transformer.transformer_config import TransformerConfig

try:
    from paddlefleet_ops import fused_mxfp4_fake_quant
except (ImportError, RuntimeError):
    fused_mxfp4_fake_quant = None


_FP4_BOUNDARIES = np.array(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32
)
_FP4_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _ceil_ue8m0(value):
    exponent = np.ceil(np.log2(value))
    return np.exp2(np.clip(exponent, -126, 127)).astype(np.float32)


def _quantize_group(group, floor):
    amax = np.max(np.abs(group), axis=-1, keepdims=True)
    scale = _ceil_ue8m0(floor(amax))
    scaled = group / scale
    indices = np.sum(
        np.abs(scaled)[..., None] > _FP4_BOUNDARIES,
        axis=-1,
    )
    return np.copysign(_FP4_GRID[indices], scaled) * scale


def _indexer_reference(value):
    shape = value.shape
    grouped = value.reshape(*shape[:-1], shape[-1] // 32, 32)
    quantized = _quantize_group(
        grouped,
        lambda amax: np.maximum(amax / 6.0, np.finfo(np.float32).tiny),
    )
    return quantized.reshape(shape)


def _expert_reference(value):
    transposed = np.transpose(value, (0, 2, 1))
    shape = transposed.shape
    grouped = transposed.reshape(*shape[:-1], shape[-1] // 32, 32)
    quantized = _quantize_group(
        grouped,
        lambda amax: np.maximum(amax / 6.0, np.finfo(np.float32).tiny),
    ).reshape(shape)
    return np.transpose(quantized, (0, 2, 1))


def _has_fp4_op():
    return core.is_compiled_with_cuda() and fused_mxfp4_fake_quant is not None


class TestFP4QATConfig(unittest.TestCase):
    def test_defaults_are_disabled(self):
        config = TransformerConfig(hidden_size=128)
        self.assertFalse(config.use_fp4_expert_qat)
        self.assertFalse(config.use_fp4_indexer_qat)

    def test_disabled_expert_helper_preserves_identity(self):
        weight = paddle.randn([2, 32, 4], dtype="float32")
        self.assertIs(apply_fp4_expert_fake_quant(weight, False), weight)
        weights = [weight]
        self.assertIs(apply_fp4_expert_fake_quant(weights, False), weights)

    def test_expert_node_uses_explicit_config(self):
        custom_map = SimpleNamespace(experts=[])
        disabled = ExpertsGroupGemmContiguousNode(custom_map, use_fp8_mlp=False)
        enabled = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp4_expert_qat=True,
            use_fp8_mlp=False,
        )

        self.assertFalse(disabled.use_fp4_expert_qat)
        self.assertTrue(enabled.use_fp4_expert_qat)

    def test_pylayer_backward_arity(self):
        class FusionNode:
            def set_cached_tensors(self, tensors):
                pass

            def backward(self, output_grad):
                return output_grad, output_grad

        fusion_ctx = SimpleNamespace(
            saved_tensor=lambda: ((None,),),
            node=FusionNode(),
            container=None,
        )
        fusion_grads = FusionMoePyLayer.backward(fusion_ctx, paddle.ones([1]))
        self.assertEqual(len(fusion_grads), 3)
        self.assertIsNone(fusion_grads[2])

        class HybridNode:
            def set_cached_tensors(self, tensors):
                pass

            def backward(self, output_grad, dispatched_probs):
                return output_grad, dispatched_probs

            def reset_state(self):
                pass

        hybrid_ctx = SimpleNamespace(
            saved_tensor=lambda: ((None, paddle.ones([1])),),
            node=HybridNode(),
            original_hidden_shape=(1,),
            original_probs_shape=(1,),
        )
        hybrid_grads = HybridEPMoePyLayer.backward(hybrid_ctx, paddle.ones([1]))
        self.assertEqual(len(hybrid_grads), 2)


@unittest.skipUnless(_has_fp4_op(), "CUDA MXFP4 custom op is required")
class TestFP4QATNumerics(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        initialize_rng_tracker(force_reset=True)
        model_parallel_cuda_manual_seed(1234)

    def test_indexer_compressor_v2_near_zero_and_dtype(self):
        values = np.array(
            [0.0, 1e-8, -1e-8, 1e-5, -1e-5, 1e-4, -1e-4] * 5,
            dtype=np.float32,
        )[:32].reshape(1, 1, 32)
        for dtype in ("float32", "float16", "bfloat16"):
            x = paddle.to_tensor(values, dtype=dtype)
            actual = fp4_indexer_fake_quant(x)
            expected = _indexer_reference(x.cast("float32").numpy())
            self.assertEqual(actual.shape, x.shape)
            self.assertEqual(actual.dtype, x.dtype)
            np.testing.assert_array_equal(
                actual.cast("float32").numpy(),
                paddle.to_tensor(expected, dtype=dtype).cast("float32").numpy(),
            )

    def test_expert_quantizes_axis_one(self):
        values = np.zeros([2, 64, 3], dtype=np.float32)
        values[:, :32, 0] = np.linspace(-6.0, 6.0, 32, dtype=np.float32)
        values[:, 32:, 0] = np.linspace(-0.1, 0.1, 32, dtype=np.float32)
        values[:, :, 1] = np.arange(64, dtype=np.float32)
        values[:, :, 2] = 1e-20
        for dtype in ("float32", "float16", "bfloat16"):
            weight = paddle.to_tensor(values, dtype=dtype)
            actual = fp4_expert_fake_quant(weight)
            expected = _expert_reference(weight.cast("float32").numpy())
            self.assertEqual(actual.shape, weight.shape)
            self.assertEqual(actual.dtype, weight.dtype)
            np.testing.assert_array_equal(
                actual.cast("float32").numpy(),
                paddle.to_tensor(expected, dtype=dtype).cast("float32").numpy(),
            )

    def test_non_fusion_expert_quantizes_both_weights_with_ste(self):
        config = TransformerConfig(
            hidden_size=32,
            intermediate_size=32,
            gated_linear_unit=True,
            hidden_act=F.silu,
            use_bias=True,
        )
        expert = StandardMLPExpert(
            config,
            moe_intermediate_size=32,
            is_expert=True,
            mlp_spec=MLPSublayersSpec(
                up_gate_proj=ColumnParallelLinear,
                down_proj=RowParallelLinear,
            ),
            use_fp4_expert_qat=False,
        )
        expert.up_gate_proj.weight.set_value(
            paddle.linspace(-5.5, 5.5, 32 * 64).reshape([32, 64])
        )
        expert.down_proj.weight.set_value(
            paddle.linspace(-4.5, 4.5, 32 * 32).reshape([32, 32])
        )
        hidden_states = paddle.linspace(-1.0, 1.0, 3 * 32).reshape([3, 32])
        hidden_states.stop_gradient = False

        for experimental in (False, True):
            config.gpt_model_use_experimental_version = experimental
            expert.use_fp4_expert_qat = False
            disabled, _ = expert(hidden_states)
            reference_disabled, _ = super(StandardMLPExpert, expert).forward(
                hidden_states
            )
            np.testing.assert_array_equal(
                disabled.numpy(), reference_disabled.numpy()
            )

            expert.use_fp4_expert_qat = True
            enabled, _ = expert(hidden_states)
            reference_enabled, _ = super(StandardMLPExpert, expert).forward(
                hidden_states,
                up_gate_weight=fp4_expert_fake_quant(
                    expert.up_gate_proj.weight
                ),
                down_weight=fp4_expert_fake_quant(expert.down_proj.weight),
            )
            np.testing.assert_array_equal(
                enabled.numpy(), reference_enabled.numpy()
            )
            self.assertFalse(np.array_equal(enabled.numpy(), disabled.numpy()))

        enabled.sum().backward()
        for weight in (
            expert.up_gate_proj.weight,
            expert.down_proj.weight,
        ):
            self.assertIsNotNone(weight.grad)
            self.assertTrue(bool(paddle.isfinite(weight.grad).all()))

    def test_fake_quant_uses_identity_ste(self):
        indexer_input = paddle.randn([2, 3, 32], dtype="float32")
        indexer_input.stop_gradient = False
        fp4_indexer_fake_quant(indexer_input).sum().backward()
        np.testing.assert_array_equal(
            indexer_input.grad.numpy(), np.ones(indexer_input.shape)
        )

        expert_weight = paddle.randn([2, 32, 4], dtype="float32")
        expert_weight.stop_gradient = False
        fp4_expert_fake_quant(expert_weight).sum().backward()
        np.testing.assert_array_equal(
            expert_weight.grad.numpy(), np.ones(expert_weight.shape)
        )

    def test_empty_inputs_preserve_shape_and_dtype(self):
        for shape, mode in (([0, 32], 1), ([0, 32, 4], 2)):
            x = paddle.empty(shape, dtype="bfloat16")
            actual = fused_mxfp4_fake_quant(x, mode)
            self.assertEqual(actual.shape, x.shape)
            self.assertEqual(actual.dtype, x.dtype)
            self.assertEqual(actual.numel(), 0)

    def test_invalid_mode_and_shapes(self):
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            fused_mxfp4_fake_quant(paddle.randn([1, 33]), 0)
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            fused_mxfp4_fake_quant(paddle.randn([1, 31, 2]), 2)
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            fused_mxfp4_fake_quant(paddle.randn([1, 32]), 2)
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            fused_mxfp4_fake_quant(paddle.randn([1, 32]), 99)


if __name__ == "__main__":
    unittest.main()
