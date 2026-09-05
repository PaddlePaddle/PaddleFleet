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
"""IEEE-gated branches. FLAG+UAC without IEEE stays the structure graph.

Isolation (not a docstring-only claim):
``ci/single_card_test.sh`` loops ``pytest -s "$test_file"`` / ``coverage run
-m pytest -s "$test_file"`` — one process per file. This module does not set
``CUDA_VISIBLE_DEVICES``, does not rewrite ``sys.path`` for ops, and does not
patch ``paddlefleet_ops`` symbols. Device is saved/restored per test.
Matching CUDA-built ops query ``get_device_capability()`` at import, so a
CUDA-hidden process cannot load them; hide GPUs only in a dedicated launch
command, not here. These tests do not claim the 90% diff-cover gate.
"""

from __future__ import annotations

import functools
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

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


def _independent_unfused_attn(query, key, value, combined_mask, softmax_scale):
    """Dense (non-MQA) unfused attention oracle. Does not call DSA helpers."""
    batch, seq, heads, qk_hd = query.shape
    v_hd = value.shape[-1]
    q = query.transpose([0, 2, 1, 3]).reshape([batch * heads, seq, qk_hd])
    k = key.transpose([0, 2, 1, 3]).reshape([batch * heads, seq, qk_hd])
    v = value.transpose([0, 2, 1, 3]).reshape([batch * heads, seq, v_hd])
    scores = paddle.matmul(q, k, transpose_y=True) * softmax_scale
    if combined_mask is not None:
        mask = combined_mask.expand([batch, heads, seq, seq]).reshape(
            [batch * heads, seq, seq]
        )
        scores = scores + mask.cast(scores.dtype)
    weights = F.softmax(scores, axis=-1)
    ctx = paddle.matmul(weights, v)
    return (
        ctx.reshape([batch, heads, seq, v_hd])
        .transpose([0, 2, 1, 3])
        .reshape([batch, seq, heads * v_hd])
    )


def _causal_mask(seq_len: int) -> paddle.Tensor:
    idx = paddle.arange(seq_len)
    allowed = idx.reshape([1, seq_len]) <= idx.reshape([seq_len, 1])
    zeros = paddle.zeros([seq_len, seq_len], dtype="float32")
    neginf = paddle.full([seq_len, seq_len], float("-inf"), dtype="float32")
    return paddle.where(allowed, zeros, neginf).reshape(
        [1, 1, seq_len, seq_len]
    )


def _expert_config(**overrides):
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


class _DeviceRestoreCase(unittest.TestCase):
    def setUp(self):
        self._saved_device = str(paddle.get_device())
        paddle.set_device("cpu")
        device = str(paddle.get_device())
        if not device.startswith("cpu"):
            self.skipTest(f"IEEE gated tests require cpu, got {device}")

    def tearDown(self):
        paddle.set_device(self._saved_device)


class TestAbsorbQNopeKUpIeee(_DeviceRestoreCase):
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


class TestUnfusedDsaIeeeOracle(_DeviceRestoreCase):
    def _qkv(self, seed):
        paddle.seed(seed)
        query = paddle.randn([1, 2, 2, 4], dtype="float32")
        key = paddle.randn([1, 2, 2, 4], dtype="float32")
        value = paddle.randn([1, 2, 2, 4], dtype="float32")
        return query, key, value

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_ieee_matches_independent_oracle_bits(self):
        query, key, value = self._qkv(2)
        scale = 0.5
        out = _unfused_dsa_attention(query, key, value, None, scale)
        oracle = _independent_unfused_attn(query, key, value, None, scale)
        self.assertEqual(tuple(out.shape), (1, 2, 8))
        self.assertFalse(bool(paddle.equal_all(out, paddle.zeros_like(out))))
        self.assertTrue(
            bool(paddle.equal_all(out, oracle)),
            "IEEE unfused forward must match the independent bmm+softmax oracle",
        )

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "0"})
    def test_flag_off_matches_independent_oracle_bits(self):
        query, key, value = self._qkv(2)
        scale = 0.5
        out = _unfused_dsa_attention(query, key, value, None, scale)
        oracle = _independent_unfused_attn(query, key, value, None, scale)
        self.assertFalse(bool(paddle.equal_all(out, paddle.zeros_like(out))))
        self.assertTrue(bool(paddle.equal_all(out, oracle)))

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_causal_mask_hides_future_value(self):
        paddle.seed(3)
        seq = 2
        query = paddle.ones([1, seq, 1, 2], dtype="float32")
        key = paddle.ones([1, seq, 1, 2], dtype="float32")
        value = paddle.to_tensor(
            [[[[10.0, 10.0], [99.0, 99.0]]]], dtype="float32"
        )
        mask = _causal_mask(seq)
        out = _unfused_dsa_attention(query, key, value, mask, 1.0)
        oracle = _independent_unfused_attn(query, key, value, mask, 1.0)
        self.assertTrue(bool(paddle.equal_all(out, oracle)))
        self.assertTrue(
            bool(paddle.equal_all(out[:, 0, :], paddle.full([1, 2], 10.0)))
        )
        self.assertFalse(bool(paddle.equal_all(out[:, 1, :], out[:, 0, :])))

        value_perturbed = value.clone()
        value_perturbed[:, 1, :, :] = 0.0
        out_perturbed = _unfused_dsa_attention(
            query, key, value_perturbed, mask, 1.0
        )
        self.assertTrue(
            bool(paddle.equal_all(out[:, 0, :], out_perturbed[:, 0, :])),
            "causal mask must keep y[0] independent of v[1]",
        )

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_causal_backward_zeros_future_value_grad(self):
        paddle.seed(4)
        seq = 2
        query = paddle.randn([1, seq, 1, 2], dtype="float32")
        key = paddle.randn([1, seq, 1, 2], dtype="float32")
        value = paddle.randn([1, seq, 1, 2], dtype="float32")
        query.stop_gradient = False
        value.stop_gradient = False
        mask = _causal_mask(seq)
        out = _unfused_dsa_attention(query, key, value, mask, 1.0)
        out[:, 0, :].sum().backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(value.grad)
        self.assertTrue(bool(paddle.isfinite(query.grad).all()))
        self.assertFalse(
            bool(paddle.equal_all(query.grad, paddle.zeros_like(query.grad)))
        )
        future_v_grad = value.grad[:, 1, :, :]
        self.assertTrue(
            bool(
                paddle.equal_all(
                    future_v_grad, paddle.zeros_like(future_v_grad)
                )
            ),
            "d y[0] / d v[1] must be zero under a causal mask",
        )
        masked_bits = future_v_grad.numpy().view("uint32")
        self.assertTrue((masked_bits == 0).all())


class TestLanguageLossIeeeCe(_DeviceRestoreCase):
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
        self.assertTrue(bool(paddle.isfinite(out)))
        self.assertGreater(float(out), 0.0)

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
        self.assertIsInstance(loss_fn.loss_func, paddle.nn.CrossEntropyLoss)
        logits = paddle.randn([2, 4, 8], dtype="float32")
        labels = paddle.randint(0, 8, [2, 4])
        out = loss_fn.forward_impl(logits, labels)
        self.assertTrue(bool(paddle.isfinite(out)))
        self.assertGreater(float(out), 0.0)


class TestGroupedMlpIeeeMainGrad(_DeviceRestoreCase):
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
