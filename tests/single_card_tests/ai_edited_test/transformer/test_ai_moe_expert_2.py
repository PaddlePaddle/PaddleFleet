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
import inspect
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import patch

import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.moe.moe_expert import (
    GroupedMLPExpert,
    StandardMLPExpert,
    _UACExpertFp32WgradCapture,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "moe_intermediate_size": 128,
        "moe_deep_gemm": False,
        "intermediate_size": 128,
        "use_bias": False,
        "gated_linear_unit": True,
        "hidden_act": F.silu,
        "rms_norm_eps": 1e-5,
        "fp8": False,
        "recompute_granularity": None,
        "recompute_modules": None,
        "using_sonic_moe": False,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "bias_activation_fusion": False,
        "activation_func_clamp_value": None,
        "glu_linear_offset": 0.0,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestGroupedMLPExpertConstruction(unittest.TestCase):
    """Tests for GroupedMLPExpert __init__."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_construction_basic(self):
        """Test basic construction of GroupedMLPExpert."""
        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
        )
        self.assertEqual(expert.num_local_experts, 2)
        self.assertFalse(expert.moe_deep_gemm)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_construction_with_sonic_moe(self):
        """Test construction with sonic_moe enabled."""
        config = _make_config(using_sonic_moe=True)
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
        )
        self.assertEqual(expert.weight1.shape[0], 2)

    def test_construction_bias_not_supported(self):
        """Test that use_bias=True raises assertion."""
        config = _make_config(use_bias=True)
        with self.assertRaises(AssertionError):
            GroupedMLPExpert(
                num_local_experts=2,
                config=config,
                moe_deep_gemm=False,
            )

    def test_construction_invalid_activation(self):
        """Test that invalid activation with GLU config is set correctly."""
        config = _make_config(gated_linear_unit=True, hidden_act=F.relu)
        # Verify the config is set as intended
        self.assertTrue(config.gated_linear_unit)
        self.assertEqual(config.hidden_act, F.relu)

    def test_construction_no_glu(self):
        """Test construction without gated linear unit."""
        config = _make_config(gated_linear_unit=False, hidden_act=F.silu)
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
        )
        self.assertEqual(expert.weight1.shape[0], 2)


class TestGroupedMLPExpertForward(unittest.TestCase):
    """Tests for GroupedMLPExpert forward."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_forward_with_tokens(self):
        """Test forward with tokens allocated to experts."""
        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
        )
        tokens = paddle.randn([4, 64], dtype="bfloat16")
        tokens_per_expert = paddle.to_tensor([2, 2], dtype="int64")

        output, bias = expert(tokens, tokens_per_expert)
        self.assertEqual(output.shape[0], 4)
        self.assertIsNone(bias)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_forward_with_zero_tokens(self):
        """Test forward with zero tokens allocated to experts."""
        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
        )
        tokens = paddle.zeros([0, 64], dtype="bfloat16")
        tokens_per_expert = paddle.to_tensor([0, 0], dtype="int64")

        output, bias = expert(tokens, tokens_per_expert)
        self.assertIsNone(bias)

    def test_zero_token_branch_scales_activation_by_permuted_probs(self):
        src = inspect.getsource(GroupedMLPExpert.forward)
        else_idx = src.rfind(
            "assert paddle.count_nonzero(tokens_per_expert) == 0"
        )
        self.assertGreater(else_idx, 0)
        zero_branch = src[else_idx:]
        self.assertIn("if permuted_probs is not None:", zero_branch)
        self.assertIn("h * permuted_probs.unsqueeze(-1)", zero_branch)

    def test_live_forward_is_uac_tn_path_not_bmm_override(self):
        """Python binds the last def forward; the live method must be E-163."""
        src = inspect.getsource(GroupedMLPExpert.forward)
        self.assertEqual(src.count("def forward"), 1)
        self.assertIn("ieee_kernel_enabled()", src)
        self.assertIn("use_accuracy_compatible", src)
        self.assertIn("transpose_y=True", src)
        self.assertIn("permuted_probs", src)
        self.assertIn("row_owner", src)
        self.assertIn("_UACExpertFp32WgradCapture.apply", src)
        self.assertEqual(src.count("_UACExpertFp32WgradCapture.apply"), 4)
        capture_src = inspect.getsource(_UACExpertFp32WgradCapture.forward)
        self.assertIn("x.detach()", capture_src)
        self.assertNotIn("x.detach().clone()", capture_src)
        bwd_src = inspect.getsource(_UACExpertFp32WgradCapture.backward)
        self.assertIn("if weight.main_grad is None:", bwd_src)
        self.assertNotIn('hasattr(weight, "main_grad")', bwd_src)
        self.assertIn("x_seg = xb[i0:i1]", src)
        self.assertIn("self.weight1[expert_idx].t().contiguous()", src)
        self.assertNotIn("x_seg = xb[i0:i1].contiguous()", src)
        self.assertNotIn("x_blk = xb.contiguous()", src)
        self.assertNotIn("hidden_in = hidden.contiguous()", src)
        sig = inspect.signature(GroupedMLPExpert.forward)
        self.assertIn("permuted_probs", sig.parameters)
        self.assertIn("row_owner", sig.parameters)
        self.assertIn("expert_weights", sig.parameters)

        from paddlefleet.transformer.moe.moe_layer import MoELayer

        layer_src = inspect.getsource(
            MoELayer._forward_single_card_grouped_gemm_moe
        )
        self.assertIn("permuted_probs=", layer_src)
        self.assertIn("row_owner=", layer_src)
        self.assertIn("probs=None", layer_src)
        self.assertIn("ieee_kernel_enabled()", layer_src)
        self.assertIn("use_accuracy_compatible=True", layer_src)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_uac_forward_folds_probs_and_splits_row_owner(self):
        config = _make_config(use_accuracy_compatible=True)
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
        )
        tokens = paddle.randn([4, 64], dtype="bfloat16")
        tokens_per_expert = paddle.to_tensor([2, 2], dtype="int64")
        probs = paddle.ones([4], dtype="float32")
        row_owner = paddle.to_tensor([0, 0, 1, 1], dtype="int64")
        output, bias = expert(
            tokens,
            tokens_per_expert,
            permuted_probs=probs,
            row_owner=row_owner,
        )
        self.assertEqual(output.shape[0], 4)
        self.assertIsNone(bias)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_uac_capture_backward_on_token_slices_writes_fp32_main_grad(self):
        """Slice views of permuted tokens must not segfault SliceGrad (E-480)."""
        config = _make_config(use_accuracy_compatible=True)
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
        )
        tokens = paddle.randn([4, 64], dtype="bfloat16")
        tokens.stop_gradient = False
        tokens_per_expert = paddle.to_tensor([2, 2], dtype="int64")
        row_owner = paddle.to_tensor([0, 0, 1, 1], dtype="int64")
        expert.weight1.main_grad = paddle.zeros(
            expert.weight1.shape, dtype="float32"
        )
        expert.weight2.main_grad = paddle.zeros(
            expert.weight2.shape, dtype="float32"
        )
        output, _ = expert(tokens, tokens_per_expert, row_owner=row_owner)
        output.cast("float32").sum().backward()
        self.assertIsNotNone(expert.weight1.main_grad)
        self.assertIsNotNone(expert.weight2.main_grad)
        self.assertEqual(
            tuple(expert.weight1.main_grad.shape), tuple(expert.weight1.shape)
        )


class TestGroupedMLPExpertBackwardDW(unittest.TestCase):
    """Tests for GroupedMLPExpert backward_dw."""

    def test_backward_dw_is_noop(self):
        """Test backward_dw does nothing (empty implementation)."""
        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
        )
        # Should not raise
        expert.backward_dw()


class TestStandardMLPExpertConstruction(unittest.TestCase):
    """Tests for StandardMLPExpert construction."""

    def test_construction_basic(self):
        """Test basic construction of StandardMLPExpert."""
        config = _make_config()
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )

        spec = get_gpt_layer_local_spec(config)
        expert = StandardMLPExpert(
            config=config,
            moe_intermediate_size=128,
            is_expert=True,
            mlp_spec=spec.sublayers_spec.mlp.sublayers_spec,
        )
        self.assertIsInstance(expert, StandardMLPExpert)


if __name__ == "__main__":
    unittest.main()
