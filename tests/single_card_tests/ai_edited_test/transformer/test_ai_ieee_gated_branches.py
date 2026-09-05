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

import numpy as np

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
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    _accuracy_compatible_cross_entropy,
)
from paddlefleet.tensor_parallel.layers import Linear
from paddlefleet.transformer.dsa_attention import (
    DSAIndexerSublayersSpec,
    DSAttention,
    DSAttentionSublayersSpec,
    Indexer,
    _absorb_q_nope_k_up,
    _accuracy_compat_linear,
    _AccuracyCompatibleQKMatmul,
    _AccuracyCompatibleSoftmax,
    _align_sp_aux_to_query,
    _SteQKMatmul,
    _unfused_dsa_attention,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.moe.moe_expert import (
    GroupedMLPExpert,
    _UACExpertFp32WgradCapture,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal


def _explicit_masked_softmax(scores, valid_mask):
    """Stable exp/sum on finite positions. Do not call F.softmax on -inf.

    Paddle CPU F.softmax(-inf) can leave ~1e-28 tail mass. IEEE
    ``_AccuracyCompatibleSoftmax`` zeros non-finite slots after softmax;
    causal checks use this explicit reduction instead.
    """
    scores = scores.cast("float32")
    valid = valid_mask.cast("bool")
    neg_large = paddle.full_like(scores, -1e30)
    stable = paddle.where(valid, scores, neg_large)
    row_max = paddle.max(stable, axis=-1, keepdim=True)
    safe = paddle.where(valid, scores, row_max)
    exp = paddle.exp(safe - row_max)
    exp = paddle.where(valid, exp, paddle.zeros_like(exp))
    denom = paddle.sum(exp, axis=-1, keepdim=True)
    return exp / denom


def _independent_unfused_attn(query, key, value, combined_mask, softmax_scale):
    """Dense (non-MQA) unfused attention oracle. Does not call DSA helpers.

    Unmasked rows use F.softmax. Causal / -inf rows use explicit exp/sum
    plus a finite mask; F.softmax is not a strict masking rawbits oracle.
    """
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
        weights = _explicit_masked_softmax(scores, paddle.isfinite(scores))
    else:
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
        seq = 2
        query = paddle.ones([1, seq, 1, 2], dtype="float32")
        key = paddle.ones([1, seq, 1, 2], dtype="float32")
        # [batch, seq, heads, v_hd]. Nested [[[[10,10],[99,99]]]] is
        # [1, 1, 2, 2] and CI IndexError'd on value[:, 1].
        value = paddle.to_tensor(
            [10.0, 10.0, 99.0, 99.0], dtype="float32"
        ).reshape([1, seq, 1, 2])
        self.assertEqual(tuple(value.shape), (1, seq, 1, 2))
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
        # Row0 is softmax([s0, -inf]) = [1, 0], so y0 = v0 and dq0 is 0.
        # Use seq=3 and L = sum(y[1]): row1 has two finite keys, q1 is live,
        # v[2] is future and must be +0 rawbits.
        seq = 3
        query = paddle.to_tensor(
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype="float32"
        ).reshape([1, seq, 1, 2])
        key = paddle.to_tensor(
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype="float32"
        ).reshape([1, seq, 1, 2])
        value = paddle.to_tensor(
            [1.0, 0.0, 0.0, 2.0, 99.0, 99.0], dtype="float32"
        ).reshape([1, seq, 1, 2])
        self.assertEqual(tuple(query.shape), (1, seq, 1, 2))
        self.assertEqual(tuple(value.shape), (1, seq, 1, 2))
        query.stop_gradient = False
        value.stop_gradient = False
        q_ref = query.detach().clone()
        k_ref = key.detach().clone()
        v_ref = value.detach().clone()
        q_ref.stop_gradient = False
        v_ref.stop_gradient = False
        mask = _causal_mask(seq)

        out = _unfused_dsa_attention(query, key, value, mask, 1.0)
        out_ref = _independent_unfused_attn(q_ref, k_ref, v_ref, mask, 1.0)
        self.assertTrue(bool(paddle.equal_all(out, out_ref)))
        out[:, 1, :].sum().backward()
        out_ref[:, 1, :].sum().backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(value.grad)
        self.assertTrue(bool(paddle.isfinite(query.grad).all()))
        self.assertTrue(bool(paddle.isfinite(value.grad).all()))
        q1_grad = query.grad[:, 1, :, :]
        q1_ref = q_ref.grad[:, 1, :, :]
        self.assertFalse(
            bool(paddle.equal_all(q1_grad, paddle.zeros_like(q1_grad))),
            "d y[1] / d q[1] must be non-zero (two finite keys)",
        )
        self.assertFalse(
            bool(paddle.equal_all(q1_ref, paddle.zeros_like(q1_ref))),
            "oracle d y[1] / d q[1] must be non-zero",
        )
        # IEEE: fp32 bmm + F.softmax + isfinite-zero.
        # Oracle: fp32 matmul + explicit exp/sum on finite slots.
        # Different reductions; do not require equal_all. CPU measured
        # max |q1_ieee-q1_oracle| = 1.4901161193847656e-08 = 1 fp32 ULP
        # at |q1|≈0.1966. Budget 4 ULP. Not a C7 cross-framework IEEE
        # zero-diff gate for loss/params.
        q1_max_abs = float(np.max(np.abs(q1_grad.numpy() - q1_ref.numpy())))
        q1_mag = float(
            max(
                np.max(np.abs(q1_grad.numpy())),
                np.max(np.abs(q1_ref.numpy())),
            )
        )
        math_oracle_atol = float(4 * np.spacing(np.float32(q1_mag)))
        self.assertLessEqual(
            q1_max_abs,
            math_oracle_atol,
            "IEEE q[1] vs exp/sum oracle max abs "
            f"{q1_max_abs} exceeds 4 fp32 ULP atol {math_oracle_atol} "
            f"(mag={q1_mag}); math-unit only, not C7 bitwise",
        )
        future_v_grad = value.grad[:, 2, :, :]
        future_v_ref = v_ref.grad[:, 2, :, :]
        self.assertTrue(
            bool(
                paddle.equal_all(
                    future_v_grad, paddle.zeros_like(future_v_grad)
                )
            ),
            "d y[1] / d v[2] must be zero under a causal mask",
        )
        self.assertTrue(
            bool(
                paddle.equal_all(future_v_ref, paddle.zeros_like(future_v_ref))
            )
        )
        self.assertTrue((future_v_grad.numpy().view("uint32") == 0).all())
        self.assertTrue((future_v_ref.numpy().view("uint32") == 0).all())

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_ieee_mqa_uses_ste_qk_and_slices_v_from_key(self):
        # nhpp>1, key heads=1: V must be key[..., :v_hd], not the dummy value.
        seq = 2
        query = paddle.to_tensor(
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype="float32"
        ).reshape([1, seq, 2, 2])
        key = paddle.to_tensor([2.0, 0.0, 0.0, 3.0], dtype="float32").reshape(
            [1, seq, 1, 2]
        )
        dummy_v = paddle.full([1, seq, 1, 2], 99.0, dtype="float32")
        out = _unfused_dsa_attention(query, key, dummy_v, None, 1.0)
        v_from_key = key[:, :, :, :2]
        oracle = _independent_unfused_attn(
            query,
            key.expand([1, seq, 2, 2]),
            v_from_key.expand([1, seq, 2, 2]),
            None,
            1.0,
        )
        self.assertEqual(tuple(out.shape), (1, seq, 4))
        self.assertTrue(bool(paddle.equal_all(out, oracle)))
        self.assertFalse(
            bool(paddle.equal_all(out, paddle.full_like(out, 99.0))),
            "MQA IEEE must not attend dummy V=99",
        )

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_ste_qk_reduces_multihead_dk_to_mqa(self):
        # Deterministic 4D QK; dK = scale * sum_h (dS^T @ Q).
        q = paddle.to_tensor(
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype="float32"
        ).reshape([1, 2, 2, 2])
        k = paddle.to_tensor([1.0, 0.0, 0.0, 1.0], dtype="float32").reshape(
            [1, 1, 2, 2]
        )
        k.stop_gradient = False
        scale = paddle.full([], 0.5, dtype="float32")
        scores = _SteQKMatmul.apply(q.detach(), k, scale)
        expected = (
            paddle.matmul(q, k.transpose([0, 1, 3, 2]).expand([1, 2, 2, 2]))
            * 0.5
        )
        self.assertTrue(bool(paddle.equal_all(scores, expected)))
        scores.sum().backward()
        dS = paddle.ones_like(scores)
        gk_per_head = paddle.matmul(dS.transpose([0, 1, 3, 2]), q) * 0.5
        gk_ref = gk_per_head.sum(axis=1, keepdim=True)
        self.assertEqual(tuple(k.grad.shape), (1, 1, 2, 2))
        self.assertTrue(bool(paddle.equal_all(k.grad, gk_ref)))

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_accuracy_compatible_qk_reduces_key_grad_over_heads(self):
        q = paddle.to_tensor(
            [
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ],
            dtype="float32",
        ).reshape([1, 2, 2, 4])
        k = paddle.to_tensor(
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype="float32"
        ).reshape([1, 1, 4, 2])
        q.stop_gradient = False
        k.stop_gradient = False
        scores = _AccuracyCompatibleQKMatmul.apply(q, k)
        k_exp = k.expand([1, 2, 4, 2])
        expected = paddle.bmm(
            q.reshape([2, 2, 4]),
            k_exp.reshape([2, 4, 2]),
        ).reshape([1, 2, 2, 2])
        self.assertTrue(bool(paddle.equal_all(scores, expected)))
        scores.sum().backward()
        dS = paddle.ones_like(scores)
        gq_ref = paddle.bmm(
            dS.reshape([2, 2, 2]),
            k_exp.transpose([0, 1, 3, 2]).reshape([2, 2, 4]),
        ).reshape(q.shape)
        gk_per_head = paddle.matmul(q.transpose([0, 1, 3, 2]), dS)
        gk_ref = gk_per_head.sum(axis=1, keepdim=True)
        self.assertEqual(tuple(k.grad.shape), (1, 1, 4, 2))
        self.assertTrue(bool(paddle.equal_all(q.grad, gq_ref)))
        self.assertTrue(bool(paddle.equal_all(k.grad, gk_ref)))

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_accuracy_compatible_softmax_zeros_invalid_backward(self):
        # sum(p) is identically 1 so dL/dlogit is 0. Use unequal weights.
        logits = paddle.to_tensor([[0.0, 1.0, float("-inf")]], dtype="float32")
        logits.stop_gradient = False
        valid = paddle.isfinite(logits)
        probs = _AccuracyCompatibleSoftmax.apply(logits, valid)
        weights = paddle.to_tensor([[1.0, 3.0, 0.0]], dtype="float32")
        p_ref = _explicit_masked_softmax(logits.detach(), valid)
        self.assertTrue((probs.numpy()[0, 2].view("uint32") == 0).all())
        self.assertTrue((p_ref.numpy()[0, 2].view("uint32") == 0).all())
        (probs * weights).sum().backward()
        w = weights
        dlogit_ref = p_ref * (w - paddle.sum(p_ref * w, axis=-1, keepdim=True))
        dlogit_ref = paddle.where(
            valid, dlogit_ref, paddle.zeros_like(dlogit_ref)
        )
        self.assertTrue((logits.grad.numpy()[0, 2].view("uint32") == 0).all())
        self.assertTrue(
            bool(
                paddle.allclose(
                    logits.grad[:, :2], dlogit_ref[:, :2], rtol=0.0, atol=1e-6
                )
            )
        )
        self.assertFalse(
            bool(
                paddle.equal_all(
                    logits.grad[:, :2], paddle.zeros_like(logits.grad[:, :2])
                )
            )
        )

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_unfused_absorbed_ieee_projects_v_up(self):
        # Import-time _ACCURACY_COMPATIBLE_KERNEL is frozen False unless
        # MODEL_REPRO_IEEE_KERNEL=1 at import. Patch the live flag so the
        # IEEE QK/softmax path is the one under test.
        query = paddle.to_tensor(
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype="float32"
        ).reshape([1, 2, 2, 2])
        key = paddle.to_tensor([1.0, 0.0, 0.0, 1.0], dtype="float32").reshape(
            [1, 1, 2, 2]
        )
        value = paddle.to_tensor([1.0, 0.0, 0.0, 2.0], dtype="float32").reshape(
            [1, 1, 2, 2]
        )
        v_up = paddle.to_tensor(
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype="float32"
        ).reshape([2, 2, 2])
        import paddlefleet.transformer.dsa_attention as dsa_mod

        with patch.object(dsa_mod, "_ACCURACY_COMPATIBLE_KERNEL", True):
            out = dsa_mod._unfused_absorbed_dsa_attention(
                query, key, value, v_up, None, 1.0
            )
        q4 = query.transpose([0, 2, 1, 3]).cast("float32")
        k4 = key.transpose([0, 2, 3, 1]).cast("float32")
        scores = paddle.matmul(q4, k4)
        probs = F.softmax(scores, axis=-1)
        latent_v = value.transpose([0, 2, 1, 3])
        ctx = paddle.matmul(probs.cast(value.dtype), latent_v)
        projected = paddle.einsum("bhsr,hrd->bshd", ctx, v_up)
        oracle = projected.reshape([1, 2, 4])
        self.assertEqual(tuple(out.shape), (1, 2, 4))
        self.assertTrue(bool(paddle.equal_all(out, oracle)))


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


class _LinearProjection:
    def __init__(self, weight, bias=None, skip_bias_add=False):
        self.weight = weight
        self.bias = bias
        self.skip_bias_add = skip_bias_add


def _concat_axis0(tensor, group=None):
    return paddle.concat([tensor, tensor], axis=0)


class TestAccuracyCompatLinear(_DeviceRestoreCase):
    def test_fused_bias_matches_f_linear(self):
        paddle.seed(7)
        x = paddle.randn([2, 3], dtype="float32")
        weight = paddle.randn([3, 4], dtype="float32")
        bias = paddle.randn([4], dtype="float32")
        out, out_bias = _accuracy_compat_linear(
            _LinearProjection(weight, bias, skip_bias_add=False), x
        )
        expected = F.linear(x, weight, bias)
        self.assertTrue(bool(paddle.equal_all(out, expected)))
        self.assertIsNone(out_bias)

    def test_skip_bias_add_returns_bias_separately(self):
        paddle.seed(8)
        x = paddle.randn([2, 3], dtype="float32")
        weight = paddle.randn([3, 4], dtype="float32")
        bias = paddle.randn([4], dtype="float32")
        out, out_bias = _accuracy_compat_linear(
            _LinearProjection(weight, bias, skip_bias_add=True), x
        )
        expected = F.linear(x, weight, None)
        self.assertTrue(bool(paddle.equal_all(out, expected)))
        self.assertTrue(bool(paddle.equal_all(out_bias, bias)))
        self.assertFalse(bool(paddle.equal_all(out, expected + bias)))


class TestSteQkBf16Cast(_DeviceRestoreCase):
    def test_ste_qk_casts_dk_to_key_dtype(self):
        q = paddle.to_tensor(
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype="float32"
        ).reshape([1, 2, 2, 2])
        k = paddle.to_tensor([1.0, 0.0, 0.0, 1.0], dtype="bfloat16").reshape(
            [1, 1, 2, 2]
        )
        k.stop_gradient = False
        scale = paddle.full([], 0.5, dtype="float32")
        scores = _SteQKMatmul.apply(q.detach(), k, scale)
        scores.sum().backward()
        dS = paddle.ones_like(scores)
        gk_per_head = paddle.matmul(dS.transpose([0, 1, 3, 2]), q) * 0.5
        gk_ref = gk_per_head.sum(axis=1, keepdim=True).cast("bfloat16")
        self.assertEqual(k.grad.dtype, paddle.bfloat16)
        # CPU has no equal_all kernel for bfloat16.
        self.assertTrue(
            (
                k.grad.numpy().view("uint16") == gk_ref.numpy().view("uint16")
            ).all()
        )


class TestAlignSpAuxToQuery(_DeviceRestoreCase):
    def test_none_or_non4d_query_returns_input(self):
        query3 = paddle.zeros([1, 4, 8], dtype="float32")
        tensor = paddle.arange(8, dtype="float32").reshape([1, 4, 2])
        self.assertIsNone(_align_sp_aux_to_query(None, query3))
        out = _align_sp_aux_to_query(tensor, query3)
        self.assertTrue(bool(paddle.equal_all(out, tensor)))

    def test_2d_aux_returns_input(self):
        query = paddle.zeros([1, 4, 2, 8], dtype="float32")
        tensor = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _align_sp_aux_to_query(tensor, query)
        self.assertTrue(bool(paddle.equal_all(out, tensor)))

    def test_3d_batch_first_gather(self):
        query = paddle.zeros([1, 4, 2, 8], dtype="float32")
        tensor = paddle.arange(6, dtype="float32").reshape([1, 2, 3])
        with patch(
            "paddlefleet.transformer.dsa_attention."
            "gather_from_sequence_parallel_region",
            side_effect=_concat_axis0,
        ):
            out = _align_sp_aux_to_query(tensor, query)
        expected = paddle.concat(
            [tensor.transpose([1, 0, 2]), tensor.transpose([1, 0, 2])],
            axis=0,
        ).transpose([1, 0, 2])
        self.assertEqual(tuple(out.shape), (1, 4, 3))
        self.assertTrue(bool(paddle.equal_all(out, expected)))

    def test_3d_seq_first_gather_transposes(self):
        query = paddle.zeros([1, 4, 2, 8], dtype="float32")
        tensor = paddle.arange(6, dtype="float32").reshape([2, 1, 3])
        with patch(
            "paddlefleet.transformer.dsa_attention."
            "gather_from_sequence_parallel_region",
            side_effect=_concat_axis0,
        ):
            out = _align_sp_aux_to_query(tensor, query)
        expected = paddle.concat([tensor, tensor], axis=0).transpose([1, 0, 2])
        self.assertEqual(tuple(out.shape), (1, 4, 3))
        self.assertTrue(bool(paddle.equal_all(out, expected)))

    def test_3d_seq_first_gather_keeps_non_seq_batch_layout(self):
        query = paddle.zeros([1, 4, 2, 8], dtype="float32")
        tensor = paddle.arange(6, dtype="float32").reshape([2, 1, 3])

        def _concat_axis1(t, group=None):
            return paddle.concat([t, t], axis=1)

        with patch(
            "paddlefleet.transformer.dsa_attention."
            "gather_from_sequence_parallel_region",
            side_effect=_concat_axis1,
        ):
            out = _align_sp_aux_to_query(tensor, query)
        expected = paddle.concat([tensor, tensor], axis=1)
        self.assertEqual(tuple(out.shape), (2, 2, 3))
        self.assertTrue(bool(paddle.equal_all(out, expected)))

    def test_4d_keep_and_seq_first_transpose(self):
        query = paddle.zeros([1, 4, 2, 8], dtype="float32")
        keep = paddle.arange(16, dtype="float32").reshape([1, 4, 2, 2])
        self.assertTrue(
            bool(paddle.equal_all(_align_sp_aux_to_query(keep, query), keep))
        )
        seq_first = paddle.arange(16, dtype="float32").reshape([4, 1, 2, 2])
        out = _align_sp_aux_to_query(seq_first, query)
        self.assertTrue(
            bool(paddle.equal_all(out, seq_first.transpose([1, 0, 2, 3])))
        )

    def test_4d_batch_first_gather(self):
        query = paddle.zeros([1, 4, 2, 8], dtype="float32")
        tensor = paddle.arange(8, dtype="float32").reshape([1, 2, 1, 4])
        with patch(
            "paddlefleet.transformer.dsa_attention."
            "gather_from_sequence_parallel_region",
            side_effect=_concat_axis0,
        ):
            out = _align_sp_aux_to_query(tensor, query)
        expected = paddle.concat(
            [
                tensor.transpose([1, 0, 2, 3]),
                tensor.transpose([1, 0, 2, 3]),
            ],
            axis=0,
        ).transpose([1, 0, 2, 3])
        self.assertEqual(tuple(out.shape), (1, 4, 1, 4))
        self.assertTrue(bool(paddle.equal_all(out, expected)))

    def test_4d_seq_first_gather_transposes(self):
        query = paddle.zeros([1, 4, 2, 8], dtype="float32")
        tensor = paddle.arange(8, dtype="float32").reshape([2, 1, 1, 4])
        with patch(
            "paddlefleet.transformer.dsa_attention."
            "gather_from_sequence_parallel_region",
            side_effect=_concat_axis0,
        ):
            out = _align_sp_aux_to_query(tensor, query)
        expected = paddle.concat([tensor, tensor], axis=0).transpose(
            [1, 0, 2, 3]
        )
        self.assertEqual(tuple(out.shape), (1, 4, 1, 4))
        self.assertTrue(bool(paddle.equal_all(out, expected)))

    def test_4d_unmatched_layout_returns_input(self):
        query = paddle.zeros([1, 4, 2, 8], dtype="float32")
        tensor = paddle.arange(6, dtype="float32").reshape([2, 3, 1, 1])
        out = _align_sp_aux_to_query(tensor, query)
        self.assertTrue(bool(paddle.equal_all(out, tensor)))
        longer = paddle.arange(12, dtype="float32").reshape([1, 6, 1, 2])
        out_long = _align_sp_aux_to_query(longer, query)
        self.assertTrue(bool(paddle.equal_all(out_long, longer)))


def _indexer_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 2,
        "dsa_index_n_heads": 1,
        "dsa_index_head_dim": 16,
        "dsa_index_topk": 2,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 16,
        "dsa_indexer_loss_coeff": 0.0,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "sequence_parallel": False,
        "use_bias": False,
        "perform_initialization": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _indexer_spec():
    return DSAIndexerSublayersSpec(
        linear_wq_b=Linear,
        linear_wk=Linear,
        k_norm=paddle.nn.LayerNorm,
        linear_weights_proj=Linear,
    )


class TestIndexerIeeeLinear(_DeviceRestoreCase):
    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_ieee_indexer_gemms_match_f_linear(self):
        import paddlefleet.transformer.dsa_attention as dsa_mod

        indexer = Indexer(
            _indexer_config(), sublayers_spec=_indexer_spec(), layer_number=1
        )
        hidden = paddle.randn([2, 4, 64], dtype="float32")
        q_latent = paddle.randn([2, 4, 16], dtype="float32")
        q_lin, _ = _accuracy_compat_linear(indexer.wq_b, q_latent)
        k_lin, _ = _accuracy_compat_linear(indexer.wk, hidden)
        w_lin, _ = _accuracy_compat_linear(indexer.weights_proj, hidden)
        self.assertTrue(
            bool(
                paddle.equal_all(
                    q_lin, F.linear(q_latent, indexer.wq_b.weight, None)
                )
            )
        )
        self.assertTrue(
            bool(
                paddle.equal_all(
                    k_lin, F.linear(hidden, indexer.wk.weight, None)
                )
            )
        )
        self.assertTrue(
            bool(
                paddle.equal_all(
                    w_lin, F.linear(hidden, indexer.weights_proj.weight, None)
                )
            )
        )
        with (
            patch.object(dsa_mod, "_ACCURACY_COMPATIBLE_KERNEL", True),
            patch.object(
                indexer,
                "_apply_rope",
                side_effect=lambda x, freqs, mscale=1.0: x,
            ),
            patch.object(indexer.k_norm, "forward", side_effect=lambda x: x),
            patch.object(
                dsa_mod,
                "rotate_activation",
                side_effect=lambda x, use_fast_hadamard=False: x,
            ),
        ):
            q, k, weights = indexer.forward_before_topk(hidden, q_latent)
        self.assertTrue(
            bool(
                paddle.equal_all(
                    q, q_lin.reshape([2, 4, indexer.n_heads, indexer.head_dim])
                )
            )
        )
        self.assertTrue(bool(paddle.equal_all(k, k_lin)))
        scale = (indexer.n_heads**-0.5) * indexer.softmax_scale
        self.assertTrue(bool(paddle.equal_all(weights, w_lin * scale)))


class TestDsAttentionAbsorbedCore(_DeviceRestoreCase):
    def _model(self):
        config = _indexer_config()
        config.qk_nope_head_dim = 8
        config.qk_rope_head_dim = 4
        config.v_head_dim = 8
        spec = DSAttentionSublayersSpec(
            indexer=LayerSpec(layer=Indexer, sublayers_spec=_indexer_spec())
        )
        mock_pg = MagicMock()
        mock_pg.tp = None
        model = DSAttention(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=1.0,
            pg_collection=mock_pg,
        )
        model.eval()
        return model

    def _inputs(self):
        paddle.seed(11)
        b, s, h, nope, rope, kv, v_out = 1, 4, 2, 8, 4, 8, 8
        query = paddle.randn([b, s, h, nope + rope], dtype="float32")
        key = paddle.randn([b, s, h, nope + rope], dtype="float32")
        value = paddle.randn([b, s, h, v_out], dtype="float32")
        x = paddle.randn([b, s, 64], dtype="bfloat16")
        qr = paddle.randn([b, s, 16], dtype="bfloat16")
        kv_c = paddle.randn([b, s, kv], dtype="float32")
        k_pe = paddle.randn([b, s, rope], dtype="float32")
        k_abs = paddle.randn([h, nope, kv], dtype="float32")
        v_b = paddle.randn([h, v_out, kv], dtype="float32")
        topk = paddle.zeros([b, s, 2], dtype="int64")
        topk[:, :, 1] = 1
        return query, key, value, x, qr, kv_c, k_pe, k_abs, v_b, topk

    def _oracle_absorbed(
        self, query, kv_c, k_pe, k_abs, v_b, topk, softmax_scale, uac
    ):
        b, s, h, qk_hd = query.shape
        rope_hd = int(k_pe.shape[-1])
        nope_hd = qk_hd - rope_hd
        q_nope = query[..., :nope_hd]
        q_pe = query[..., nope_hd:]
        qn3 = q_nope.reshape([b * s, h, nope_hd]).transpose([1, 0, 2])
        q_abs_nope = _absorb_q_nope_k_up(qn3, k_abs)
        q_abs_nope = q_abs_nope.transpose([1, 0, 2]).reshape(
            [b, s, h, k_abs.shape[-1]]
        )
        q_absorbed = paddle.concat([q_abs_nope, q_pe], axis=-1)
        kv = kv_c + (kv_c * 0) if uac else kv_c
        k_latent = kv.unsqueeze(2)
        k_rope = k_pe.unsqueeze(2)
        key_abs = paddle.concat([k_latent, k_rope], axis=-1)
        dummy = paddle.zeros(k_latent.shape, dtype=k_latent.dtype)
        causal = paddle.triu(
            paddle.full([s, s], float("-inf"), dtype="float32"), diagonal=1
        )
        index_mask = paddle.full([b, s, s], float("-inf"), dtype="float32")
        zeros = paddle.zeros(topk.shape, dtype="float32")
        index_mask = paddle.put_along_axis(index_mask, topk, zeros, axis=-1)
        combined = (index_mask + causal.unsqueeze(0)).unsqueeze(1)
        latent_flat = _unfused_dsa_attention(
            q_absorbed, key_abs, dummy, combined, softmax_scale
        )
        kv_rank = kv.shape[-1]
        latent_out = latent_flat.reshape([b, s, h, kv_rank])
        if uac:
            lat = latent_out.transpose([2, 0, 1, 3]).reshape(
                [h, b * s, kv_rank]
            )
            core = (
                paddle.bmm(lat, v_b.transpose([0, 2, 1]))
                .reshape([h, b, s, -1])
                .transpose([1, 2, 0, 3])
            )
        else:
            core = paddle.einsum("bshc,hdc->bshd", latent_out, v_b)
        return core.reshape([b, s, h * core.shape[-1]])

    def _run(self, model, inputs, uac):
        query, key, value, x, qr, kv_c, k_pe, k_abs, v_b, topk = inputs
        scores = paddle.zeros([1, 4, 4], dtype="float32")
        env = {"MODEL_REPRO_IEEE_KERNEL": "1" if uac else "0"}
        with (
            patch.dict(os.environ, env),
            patch.object(model.indexer, "forward", return_value=(scores, topk)),
        ):
            return model(
                query,
                key,
                value,
                None,
                x=x,
                qr=qr,
                kv_compressed=kv_c,
                k_pos_emb=k_pe,
                k_abs_weight=k_abs,
                v_b_proj_weight=v_b,
            )

    def test_ieee_rebuilds_q_and_projects_v_with_bmm(self):
        model = self._model()
        inputs = self._inputs()
        out = self._run(model, inputs, uac=True)
        query, _, _, _, _, kv_c, k_pe, k_abs, v_b, topk = inputs
        with patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"}):
            expected = self._oracle_absorbed(
                query, kv_c, k_pe, k_abs, v_b, topk, 1.0, True
            )
        self.assertEqual(tuple(out.shape), (1, 4, 16))
        self.assertTrue(bool(paddle.equal_all(out, expected)))

    def test_flag_off_absorbed_uses_einsum_v_up(self):
        model = self._model()
        inputs = self._inputs()
        out = self._run(model, inputs, uac=False)
        query, _, _, _, _, kv_c, k_pe, k_abs, v_b, topk = inputs
        with patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "0"}):
            expected = self._oracle_absorbed(
                query, kv_c, k_pe, k_abs, v_b, topk, 1.0, False
            )
        self.assertEqual(tuple(out.shape), (1, 4, 16))
        self.assertTrue(bool(paddle.equal_all(out, expected)))


def _cpu_float32_expert_weights(expert):
    """CPU has no bf16 matmul. Keep the live Parameter API, change dtype only."""
    if not str(paddle.get_device()).startswith("cpu"):
        return
    w1 = paddle.create_parameter(
        shape=list(expert.weight1.shape),
        dtype="float32",
        default_initializer=paddle.nn.initializer.Constant(0.0),
    )
    w2 = paddle.create_parameter(
        shape=list(expert.weight2.shape),
        dtype="float32",
        default_initializer=paddle.nn.initializer.Constant(0.0),
    )
    if hasattr(expert.weight1, "main_grad"):
        w1.main_grad = None
        w2.main_grad = None
    expert.weight1 = w1
    expert.weight2 = w2


class TestUacExpertCaptureAndGemm(_DeviceRestoreCase):
    def test_empty_token_capture_returns_incoming_dy(self):
        x = paddle.zeros([0, 5], dtype="float32")
        x.stop_gradient = False
        wt = paddle.zeros([5, 4], dtype="float32")
        y = paddle.matmul(x, wt)
        weight = paddle.create_parameter(
            shape=[2, 5, 4],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )
        weight.main_grad = None
        out = _UACExpertFp32WgradCapture.apply(y, x, weight, 0)
        self.assertEqual(tuple(out.shape), (0, 4))
        if out.numel() == 0:
            paddle.autograd.backward(out, paddle.zeros_like(out))
        else:
            out.sum().backward()
        self.assertIsNone(weight.main_grad)

    def test_capture_allocates_main_grad_and_writes_xt_dy(self):
        paddle.seed(9)
        x = paddle.randn([3, 5], dtype="float32")
        x.stop_gradient = False
        wt = paddle.randn([5, 4], dtype="float32")
        y = paddle.matmul(x, wt)
        weight = paddle.create_parameter(
            shape=[2, 5, 4],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )
        weight.main_grad = None
        out = _UACExpertFp32WgradCapture.apply(y, x, weight, 1)
        out.sum().backward()
        wg = paddle.matmul(x, paddle.ones_like(y), transpose_x=True)
        self.assertIsNotNone(weight.main_grad)
        self.assertTrue(
            bool(paddle.equal_all(weight.main_grad[1], wg.cast("float32")))
        )
        self.assertTrue(
            bool(
                paddle.equal_all(
                    weight.main_grad[0], paddle.zeros_like(weight.main_grad[0])
                )
            )
        )

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_ieee_tn_gemm_matches_independent_matmul(self):
        paddle.seed(4)
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=_expert_config(use_accuracy_compatible=True),
            moe_deep_gemm=False,
        )
        _cpu_float32_expert_weights(expert)
        dtype = expert.weight1.dtype
        expert.weight1.set_value(
            paddle.randn(expert.weight1.shape, dtype=dtype)
        )
        expert.weight2.set_value(
            paddle.randn(expert.weight2.shape, dtype=dtype)
        )
        tokens = paddle.randn([4, 16], dtype=dtype)
        tokens_per_expert = paddle.to_tensor([4, 0], dtype="int64")
        row_owner = paddle.to_tensor([0, 0, 1, 1], dtype="int64")
        probs = paddle.to_tensor([1.0, 0.5, 0.25, 2.0], dtype="float32")
        out, bias = expert(
            tokens,
            tokens_per_expert,
            permuted_probs=probs,
            row_owner=row_owner,
        )
        self.assertIsNone(bias)
        x = tokens.cast("float32")
        w1 = expert.weight1[0].cast("float32")
        w2 = expert.weight2[0].cast("float32")
        parts = []
        for sl in (slice(0, 2), slice(2, 4)):
            hidden = paddle.matmul(x[sl], w1)
            gate, up = paddle.chunk(hidden, 2, axis=-1)
            hidden = F.silu(gate) * up
            hidden = hidden * probs[sl].unsqueeze(-1)
            parts.append(paddle.matmul(hidden, w2))
        expected = paddle.concat(parts, axis=0).cast(dtype)
        self.assertEqual(tuple(out.shape), tuple(expected.shape))
        self.assertTrue(bool(paddle.equal_all(out, expected)))

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_ieee_zero_tokens_scales_empty_activation(self):
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=_expert_config(use_accuracy_compatible=True),
            moe_deep_gemm=False,
        )
        _cpu_float32_expert_weights(expert)
        dtype = expert.weight1.dtype
        tokens = paddle.zeros([0, 16], dtype=dtype)
        tokens_per_expert = paddle.to_tensor([0, 0], dtype="int64")
        probs = paddle.zeros([0], dtype="float32")
        out, bias = expert(tokens, tokens_per_expert, permuted_probs=probs)
        self.assertIsNone(bias)
        self.assertEqual(out.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
