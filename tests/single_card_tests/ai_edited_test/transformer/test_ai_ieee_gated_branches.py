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
"""CPU IEEE-gated branches that FLAG+UAC CI does not execute.

MODEL_REPRO_IEEE_KERNEL=1 is the GLM-5.2 leaf. FLAG+UAC without it is the
Minimax / GLM-4.5 Air graph. These tests run the live functions, not source
string checks.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

# Local venv ops can lag this tree. CI builds matching ops; only fill
# symbols this tree imports that the installed facade may lack.
import paddlefleet_ops.flash_mask_facade as _flash_mask_facade

if not hasattr(_flash_mask_facade, "uses_cutedsl_backend"):
    _flash_mask_facade.uses_cutedsl_backend = lambda: False

import functools
import unittest
from unittest.mock import MagicMock, patch

import paddle
import paddle.nn.functional as F

from paddlefleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    _accuracy_compatible_cross_entropy,
)
from paddlefleet.transformer.dsa_attention import (
    _absorb_q_nope_k_up,
    _unfused_dsa_attention,
)
from paddlefleet.transformer.moe.moe_expert import GroupedMLPExpert
from paddlefleet.utils import init_method_normal, scaled_init_method_normal


def _expert_config(**overrides):
    # Avoid TransformerConfig.__post_init__ (FA3 facade import). The
    # expert constructor only reads these attributes.
    mock = MagicMock()
    mock.num_hidden_layers = 2
    mock.hidden_size = 16
    mock.num_attention_heads = 2
    mock.moe_intermediate_size = 32
    mock.moe_deep_gemm = False
    mock.intermediate_size = 32
    mock.use_bias = False
    mock.gated_linear_unit = True
    mock.hidden_act = F.silu
    mock.fp8 = False
    mock.recompute_granularity = None
    mock.recompute_modules = None
    mock.using_sonic_moe = False
    mock.init_method = init_method_normal(0.02)
    mock.output_layer_init_method = scaled_init_method_normal(0.02, 1, 2.0)
    mock.bias_activation_fusion = False
    mock.activation_func_clamp_value = None
    mock.glu_linear_offset = 0.0
    mock.sequence_parallel = False
    mock.tensor_model_parallel_size = 1
    mock.use_accuracy_compatible = True
    mock.perform_initialization = False
    mock.moe_token_dispatcher_type = None
    mock.expert_model_parallel_size = 1
    mock.params_dtype = "float32"
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


class TestAbsorbQNopeKUpIeee(unittest.TestCase):
    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_ieee_absorb_matches_bmm_not_einsum_layout(self):
        paddle.seed(0)
        qn3 = paddle.randn([2, 3, 4], dtype="float32")
        weight = paddle.randn([2, 4, 5], dtype="float32")
        out = _absorb_q_nope_k_up(qn3, weight)
        expected = paddle.bmm(qn3, weight)
        self.assertTrue(bool(paddle.equal_all(out, expected)))

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "0"})
    def test_flag_off_absorb_matches_einsum(self):
        paddle.seed(1)
        qn3 = paddle.randn([2, 3, 4], dtype="float32")
        weight = paddle.randn([2, 4, 5], dtype="float32")
        out = _absorb_q_nope_k_up(qn3, weight)
        expected = paddle.einsum(
            "hsk,hkd->hsd", qn3.cast("float32"), weight.cast("float32")
        ).cast(qn3.dtype)
        self.assertTrue(bool(paddle.equal_all(out, expected)))


class TestUnfusedDsaIeeeSoftmax(unittest.TestCase):
    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_ieee_unfused_attention_is_finite_and_shaped(self):
        paddle.seed(2)
        query = paddle.randn([1, 2, 2, 4], dtype="float32")
        key = paddle.randn([1, 2, 2, 4], dtype="float32")
        value = paddle.randn([1, 2, 2, 4], dtype="float32")
        out = _unfused_dsa_attention(query, key, value, None, 1.0)
        self.assertEqual(tuple(out.shape), (1, 2, 8))
        self.assertTrue(bool(paddle.isfinite(out).all()))

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "0"})
    def test_flag_off_unfused_attention_is_finite_and_shaped(self):
        paddle.seed(2)
        query = paddle.randn([1, 2, 2, 4], dtype="float32")
        key = paddle.randn([1, 2, 2, 4], dtype="float32")
        value = paddle.randn([1, 2, 2, 4], dtype="float32")
        out = _unfused_dsa_attention(query, key, value, None, 1.0)
        self.assertEqual(tuple(out.shape), (1, 2, 8))
        self.assertTrue(bool(paddle.isfinite(out).all()))


class TestLanguageLossIeeeCe(unittest.TestCase):
    def _config(self):
        mock = MagicMock()
        mock.parallel_output = True
        mock.loss_subbatch_sequence_length = 0
        mock.gpt_model_use_experimental_version = False
        mock.use_accuracy_compatible = True
        mock.fused_linear_ce_loss_chunk = 0
        mock.experimental_dataflow = False
        return mock

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_ieee_tp1_selects_accuracy_compatible_ce(self, _dist, _tp):
        loss_fn = LanguageLoss(config=self._config())
        self.assertIsInstance(loss_fn.loss_func, functools.partial)
        self.assertIs(
            loss_fn.loss_func.func, _accuracy_compatible_cross_entropy
        )
        logits = paddle.randn([2, 4, 8], dtype="float32")
        labels = paddle.randint(0, 8, [2, 4])
        out = loss_fn.forward_impl(logits, labels)
        self.assertTrue(paddle.isfinite(out))

    @patch.dict(
        os.environ,
        {
            "MODEL_REPRO_IEEE_KERNEL": "0",
            "FLAGS_use_accuracy_compatible_kernel": "1",
        },
    )
    @patch(
        "paddlefleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_flag_uac_without_ieee_keeps_native_ce(self, _dist, _tp):
        loss_fn = LanguageLoss(config=self._config())
        self.assertIsInstance(
            loss_fn.loss_func, paddle.nn.CrossEntropyLoss
        )
        logits = paddle.randn([2, 4, 8], dtype="float32")
        labels = paddle.randint(0, 8, [2, 4])
        out = loss_fn.forward_impl(logits, labels)
        self.assertTrue(paddle.isfinite(out))


class TestGroupedMlpIeeeMainGrad(unittest.TestCase):
    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_ieee_uac_claims_main_grad_buffer(self):
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=_expert_config(use_accuracy_compatible=True),
            moe_deep_gemm=False,
        )
        self.assertTrue(hasattr(expert.weight1, "main_grad"))
        self.assertIsNone(expert.weight1.main_grad)
        self.assertTrue(hasattr(expert.weight2, "main_grad"))
        self.assertIsNone(expert.weight2.main_grad)

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "0"})
    def test_flag_uac_without_ieee_does_not_claim_main_grad(self):
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=_expert_config(use_accuracy_compatible=True),
            moe_deep_gemm=False,
        )
        claimed = getattr(expert.weight1, "main_grad", "missing")
        self.assertEqual(claimed, "missing")


if __name__ == "__main__":
    unittest.main()
