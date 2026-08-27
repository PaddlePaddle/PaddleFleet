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
"""Unit tests for VHA postmix on the MLA / MQA attention path.

Covers the low-rank cross-head postmix (I + U Vᵀ) added to
``MultiLatentAttention`` (base, ungrouped) and the extra per-branch
``sparse_vha_postmix_U/V`` created by ``MQASelfAttention`` for the
block-sparse HySparse branch.
"""

import sys
import types
import unittest
from unittest import mock

import paddle

import paddlefleet.transformer.multi_latent_attention as mla_mod
from paddlefleet.transformer.dot_product_attention import (
    DotProductAttention,
)
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
    MQASelfAttention,
)
from paddlefleet.transformer.transformer_config import (
    TransformerConfig,
)
from paddlefleet.utils import (
    init_method_normal,
    scaled_init_method_normal,
)

_SEED = 1234


class BiasedLinear(paddle.nn.Layer):
    """Linear that returns (output, bias) like ColumnParallelLinear."""

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    @property
    def weight(self):
        # Expose the wrapped Linear weight so the absorbed-MQA path
        # (kv_b_proj.weight.reshape(...)) works with this fake spec.
        return self.linear.weight

    def forward(self, x):
        return self.linear(x), self.linear.bias


class SimpleRMSNorm(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_size = kwargs.get("normalized_shape", kwargs.get("hidden_size"))
        eps = kwargs.get("norm_eps", kwargs.get("eps", 1e-5))
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


def _make_mla_config(**overrides):
    """Minimal TransformerConfig for MLA/MQA VHA testing (nh=4, v=32)."""
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "softmax_scale": None,
        "use_bias": False,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "recompute_modules": None,
        "apply_rope_fusion": False,
        "rotary_interleaved": False,
        "multi_latent_attention": True,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "rms_norm_eps": 1e-5,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "window_attn_skip_freq": None,
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "attention_dropout": 0.0,
        "softmax_type": "vanilla",
        "fa_version": None,
        "kv_lora_rank": 32,
        "q_lora_rank": 64,
        "qk_nope_head_dim": 24,
        "qk_rope_head_dim": 8,
        "v_head_dim": 32,
        "rope_type": "rope",
    }
    defaults.update(overrides)
    config = TransformerConfig(**defaults)
    config.dtype = "float32"
    return config


def _make_sublayers_spec(gate_proj_cls=None):
    return MLASelfAttentionSublayersSpec(
        q_proj=BiasedLinear,
        q_a_proj=BiasedLinear,
        q_b_proj=BiasedLinear,
        kv_a_proj_with_mqa=BiasedLinear,
        kv_b_proj=BiasedLinear,
        core_attention=DotProductAttention,
        o_proj=BiasedLinear,
        q_a_layernorm=SimpleRMSNorm,
        kv_a_layernorm=SimpleRMSNorm,
        gate_proj=gate_proj_cls,
    )


def _build_mla(gated_attention=False, layer_number=1, **config_overrides):
    config = _make_mla_config(
        gated_attention=gated_attention, **config_overrides
    )
    gate_cls = BiasedLinear if gated_attention else None
    spec = _make_sublayers_spec(gate_proj_cls=gate_cls)
    return MLASelfAttention(
        config=config, sublayers_spec=spec, layer_number=layer_number
    )


def _build_mqa(gated_attention=False, sliding_window=128, **config_overrides):
    """Build an MQASelfAttention that is a HySparse SWA layer (is_mqa=True)."""
    config = _make_mla_config(
        gated_attention=gated_attention,
        enable_hy_sparse_attention=True,
        sliding_window=sliding_window,
        window_attn_skip_freq=None,
        **config_overrides,
    )
    gate_cls = BiasedLinear if gated_attention else None
    spec = _make_sublayers_spec(gate_proj_cls=gate_cls)
    return MQASelfAttention(config=config, sublayers_spec=spec, layer_number=1)


# ===========================================================================
# MLA base postmix: __init__
# ===========================================================================
class TestMLAPostmixInit(unittest.TestCase):
    def test_disabled_no_postmix(self):
        attn = _build_mla(use_vha_attention=False)
        self.assertFalse(attn.use_vha_postmix)
        self.assertFalse(hasattr(attn, "vha_postmix_U"))

    def test_enabled_creates_params(self):
        attn = _build_mla(use_vha_attention=True)
        self.assertTrue(attn.use_vha_postmix)
        # nh=4, default rank = nh//4 = 1
        self.assertEqual(attn.vha_postmix_U.shape, [4, 1])
        self.assertEqual(attn.vha_postmix_V.shape, [4, 1])

    def test_v_is_zero_init(self):
        attn = _build_mla(use_vha_attention=True)
        self.assertTrue(
            paddle.all(attn.vha_postmix_V == 0).item(),
            "V must be zero-initialised (identity at init)",
        )

    def test_rank_default(self):
        # nh=4 -> nh//4 = 1
        attn = _build_mla(use_vha_attention=True)
        self.assertEqual(attn.vha_postmix_rank, 1)

    def test_rank_explicit(self):
        attn = _build_mla(use_vha_attention=True, vha_postmix_rank=2)
        self.assertEqual(attn.vha_postmix_rank, 2)
        self.assertEqual(attn.vha_postmix_U.shape, [4, 2])

    def test_rank_clamped_high(self):
        # rank clamped to nh=4
        attn = _build_mla(use_vha_attention=True, vha_postmix_rank=100)
        self.assertEqual(attn.vha_postmix_rank, 4)

    def test_rank_clamped_low(self):
        # rank clamped up to 1
        attn = _build_mla(use_vha_attention=True, vha_postmix_rank=0)
        self.assertEqual(attn.vha_postmix_rank, 1)

    def test_recompute_flag_off_by_default(self):
        attn = _build_mla(use_vha_attention=True)
        self.assertFalse(attn.recompute_vha_postmix)

    def test_recompute_flag_on(self):
        attn = _build_mla(
            use_vha_attention=True,
            recompute_granularity="selective",
            recompute_modules=["vha_postmix"],
        )
        self.assertTrue(attn.recompute_vha_postmix)

    def test_recompute_first_n_hit(self):
        # num_hidden_layers=2, first_n with n=1 -> only layer 0 recomputes.
        attn = _build_mla(
            layer_number=0,
            use_vha_attention=True,
            recompute_granularity="selective",
            recompute_modules=["vha_postmix"],
            recompute_method="first_n",
            recompute_num_layers=1,
        )
        self.assertTrue(attn.recompute_vha_postmix)

    def test_recompute_first_n_miss(self):
        # first_n with n=1 -> layer 1 is beyond the window -> no recompute.
        attn = _build_mla(
            layer_number=1,
            use_vha_attention=True,
            recompute_granularity="selective",
            recompute_modules=["vha_postmix"],
            recompute_method="first_n",
            recompute_num_layers=1,
        )
        self.assertFalse(attn.recompute_vha_postmix)

    def test_recompute_block_hit(self):
        # single chunk (pp=vpp=1), block n=1 -> layer 0 recomputes.
        attn = _build_mla(
            layer_number=0,
            use_vha_attention=True,
            recompute_granularity="selective",
            recompute_modules=["vha_postmix"],
            recompute_method="block",
            recompute_num_layers=1,
        )
        self.assertTrue(attn.recompute_vha_postmix)

    def test_recompute_block_miss(self):
        # block n=1 -> layer 1 falls outside the per-chunk window.
        attn = _build_mla(
            layer_number=1,
            use_vha_attention=True,
            recompute_granularity="selective",
            recompute_modules=["vha_postmix"],
            recompute_method="block",
            recompute_num_layers=1,
        )
        self.assertFalse(attn.recompute_vha_postmix)

    def test_recompute_dict_layer_count(self):
        # dict recompute_modules now selects layers per submodule: first_n with
        # n=1 covers layer 0 only.
        self.assertTrue(
            _build_mla(
                layer_number=0,
                use_vha_attention=True,
                recompute_granularity="selective",
                recompute_modules={"vha_postmix": 1},
                recompute_method="first_n",
            ).recompute_vha_postmix
        )
        self.assertFalse(
            _build_mla(
                layer_number=1,
                use_vha_attention=True,
                recompute_granularity="selective",
                recompute_modules={"vha_postmix": 1},
                recompute_method="first_n",
            ).recompute_vha_postmix
        )

    def test_recompute_dict_layer_list(self):
        # An explicit layer list needs no recompute_method.
        self.assertTrue(
            _build_mla(
                layer_number=1,
                use_vha_attention=True,
                recompute_granularity="selective",
                recompute_modules={"vha_postmix": [1]},
            ).recompute_vha_postmix
        )
        self.assertFalse(
            _build_mla(
                layer_number=0,
                use_vha_attention=True,
                recompute_granularity="selective",
                recompute_modules={"vha_postmix": [1]},
            ).recompute_vha_postmix
        )


# ===========================================================================
# MLA base postmix: _apply_vha_postmix
# ===========================================================================
class TestMLAPostmixApply(unittest.TestCase):
    def test_identity_at_init(self):
        attn = _build_mla(use_vha_attention=True, vha_postmix_rank=2)
        b, sq, nh, vd = 2, 5, 4, 32
        x = paddle.randn([b, sq, nh * vd], dtype="float32")
        out = attn._apply_vha_postmix(x)
        self.assertEqual(out.shape, [b, sq, nh * vd])
        self.assertTrue(
            paddle.allclose(out, x, atol=1e-6).item(),
            "V=0 => postmix must be the identity map",
        )

    def test_matches_manual_einsum(self):
        attn = _build_mla(use_vha_attention=True, vha_postmix_rank=2)
        # Break the identity by setting V to non-zero.
        new_v = paddle.randn(attn.vha_postmix_V.shape, dtype="float32") * 0.1
        attn.vha_postmix_V.set_value(new_v)

        b, sq, nh, vd = 2, 3, 4, 32
        x = paddle.randn([b, sq, nh * vd], dtype="float32")
        out = attn._apply_vha_postmix(x)

        U = attn.vha_postmix_U
        V = attn.vha_postmix_V
        mixed = x.reshape([b, sq, nh, vd])
        z = paddle.einsum("bthd,hr->btrd", mixed, U)
        delta = paddle.einsum("btrd,hr->bthd", z, V)
        ref = (mixed + delta).reshape([b, sq, nh * vd])
        self.assertTrue(paddle.allclose(out, ref, atol=1e-6).item())

    def test_backward_matches_manual_einsum(self):
        """The matmul rewrite must be gradient-equivalent to the einsum form
        (dx, dU, dV) — locks the optimization against silent regressions."""
        attn = _build_mla(use_vha_attention=True, vha_postmix_rank=2)
        attn.vha_postmix_V.set_value(
            paddle.randn(attn.vha_postmix_V.shape, dtype="float32") * 0.1
        )
        b, sq, nh, vd = 2, 3, 4, 32
        x0 = paddle.randn([b, sq, nh * vd], dtype="float32")

        # Module (matmul) path — squared loss so grads depend on values.
        x = x0.detach()
        x.stop_gradient = False
        out = attn._apply_vha_postmix(x)
        (out * out).sum().backward()
        gx = x.grad.clone()
        gU = attn.vha_postmix_U.grad.clone()
        gV = attn.vha_postmix_V.grad.clone()

        # einsum reference on independent leaves.
        xr = x0.detach()
        xr.stop_gradient = False
        Ur = attn.vha_postmix_U.detach()
        Ur.stop_gradient = False
        Vr = attn.vha_postmix_V.detach()
        Vr.stop_gradient = False
        mixed_r = xr.reshape([b, sq, nh, vd])
        z_r = paddle.einsum("bthd,hr->btrd", mixed_r, Ur)
        delta_r = paddle.einsum("btrd,hr->bthd", z_r, Vr)
        ref = (mixed_r + delta_r).reshape([b, sq, nh * vd])
        (ref * ref).sum().backward()

        self.assertTrue(
            paddle.allclose(gx, xr.grad, atol=1e-5).item(), "dx mismatch"
        )
        self.assertTrue(
            paddle.allclose(gU, Ur.grad, atol=1e-5).item(), "dU mismatch"
        )
        self.assertTrue(
            paddle.allclose(gV, Vr.grad, atol=1e-5).item(), "dV mismatch"
        )


# ===========================================================================
# MLA forward with postmix
# ===========================================================================
class TestMLAPostmixForward(unittest.TestCase):
    def test_forward_shape(self):
        attn = _build_mla(use_vha_attention=True)
        attn.eval()
        x = paddle.randn([2, 4, 128])
        out, bias = attn(x, attention_mask=None)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_forward_identity_at_init(self):
        """With V=0 the postmix is identity, so toggling it must not change
        the forward output (same layer, same weights)."""
        attn = _build_mla(use_vha_attention=True)
        attn.eval()
        x = paddle.randn([2, 4, 128])

        out_with, _ = attn(x, attention_mask=None)
        attn.use_vha_postmix = False  # bypass the postmix branch
        out_without, _ = attn(x, attention_mask=None)

        self.assertTrue(
            paddle.allclose(out_with, out_without, atol=1e-6).item(),
            "V=0 postmix must not alter the forward output",
        )

    def test_forward_recompute_branch(self):
        attn = _build_mla(
            use_vha_attention=True,
            recompute_granularity="selective",
            recompute_modules=["vha_postmix"],
        )
        attn.train()
        self.assertTrue(attn.recompute_vha_postmix)
        x = paddle.randn([2, 4, 128])
        x.stop_gradient = False
        out, _ = attn(x, attention_mask=None)
        self.assertEqual(out.shape, [2, 4, 128])
        out.sum().backward()
        self.assertIsNotNone(x.grad)


# ===========================================================================
# MQA sparse-branch postmix: __init__
# ===========================================================================
class TestMQAPostmixInit(unittest.TestCase):
    def test_is_mqa_layer(self):
        attn = _build_mqa(use_vha_attention=True)
        self.assertTrue(attn.is_mqa)
        self.assertTrue(attn.is_swa)

    def test_sparse_postmix_params_created(self):
        attn = _build_mqa(use_vha_attention=True, vha_postmix_rank=2)
        self.assertTrue(attn.use_vha_postmix)
        # Base (main-branch) params.
        self.assertEqual(attn.vha_postmix_U.shape, [4, 2])
        # Sparse-branch params: own set, same shape.
        self.assertTrue(hasattr(attn, "sparse_vha_postmix_U"))
        self.assertEqual(attn.sparse_vha_postmix_U.shape, [4, 2])
        self.assertEqual(attn.sparse_vha_postmix_V.shape, [4, 2])

    def test_sparse_postmix_v_zero_init(self):
        attn = _build_mqa(use_vha_attention=True)
        self.assertTrue(
            paddle.all(attn.sparse_vha_postmix_V == 0).item(),
            "sparse V must be zero-initialised (identity at init)",
        )

    def test_sparse_postmix_independent_of_main(self):
        """Sparse U/V are distinct parameter tensors from the main branch."""
        attn = _build_mqa(use_vha_attention=True)
        self.assertIsNot(attn.sparse_vha_postmix_U, attn.vha_postmix_U)
        self.assertIsNot(attn.sparse_vha_postmix_V, attn.vha_postmix_V)

    def test_disabled_no_sparse_params(self):
        attn = _build_mqa(use_vha_attention=False)
        self.assertTrue(attn.is_mqa)
        self.assertFalse(attn.use_vha_postmix)
        self.assertFalse(hasattr(attn, "sparse_vha_postmix_U"))


# ===========================================================================
# MLA base postmix: backward / gradient
# ===========================================================================
class TestMLAPostmixBackward(unittest.TestCase):
    def test_apply_param_grads(self):
        attn = _build_mla(use_vha_attention=True, vha_postmix_rank=2)
        # Randomise V (zero at init -> dL/dU would be zero) so both U and V get
        # a non-trivial gradient.
        attn.vha_postmix_V.set_value(
            paddle.randn(attn.vha_postmix_V.shape) * 0.1
        )
        b, sq, nh, vd = 2, 5, 4, 32
        x = paddle.randn([b, sq, nh * vd], dtype="float32")
        x.stop_gradient = False
        out = attn._apply_vha_postmix(x)
        out.sum().backward()
        self.assertIsNotNone(attn.vha_postmix_U.grad)
        self.assertIsNotNone(attn.vha_postmix_V.grad)
        self.assertIsNotNone(x.grad)
        self.assertGreater(attn.vha_postmix_U.grad.abs().sum().item(), 0.0)
        self.assertGreater(attn.vha_postmix_V.grad.abs().sum().item(), 0.0)

    def test_forward_param_grads(self):
        attn = _build_mla(use_vha_attention=True)
        attn.vha_postmix_V.set_value(
            paddle.randn(attn.vha_postmix_V.shape) * 0.1
        )
        attn.train()
        x = paddle.randn([2, 4, 128])
        x.stop_gradient = False
        out, _ = attn(x, attention_mask=None)
        out.sum().backward()
        self.assertIsNotNone(attn.vha_postmix_U.grad)
        self.assertIsNotNone(attn.vha_postmix_V.grad)
        self.assertIsNotNone(x.grad)


# ===========================================================================
# MQA sparse-branch postmix: apply + backward
# ===========================================================================
class TestMQASparsePostmixBackward(unittest.TestCase):
    def test_sparse_apply_matches_manual_einsum(self):
        attn = _build_mqa(use_vha_attention=True, vha_postmix_rank=2)
        U = attn.sparse_vha_postmix_U
        V = attn.sparse_vha_postmix_V
        new_v = paddle.randn(V.shape, dtype="float32") * 0.1
        V.set_value(new_v)

        b, sq, nh, vd = 2, 3, 4, 32
        x = paddle.randn([b, sq, nh * vd], dtype="float32")
        out = attn._apply_vha_postmix(x, U, V)

        mixed = x.reshape([b, sq, nh, vd])
        z = paddle.einsum("bthd,hr->btrd", mixed, U)
        delta = paddle.einsum("btrd,hr->bthd", z, V)
        ref = (mixed + delta).reshape([b, sq, nh * vd])
        self.assertTrue(paddle.allclose(out, ref, atol=1e-6).item())

    def test_sparse_param_grads(self):
        attn = _build_mqa(use_vha_attention=True, vha_postmix_rank=2)
        U = attn.sparse_vha_postmix_U
        V = attn.sparse_vha_postmix_V
        V.set_value(paddle.randn(V.shape, dtype="float32") * 0.1)

        b, sq, nh, vd = 2, 3, 4, 32
        x = paddle.randn([b, sq, nh * vd], dtype="float32")
        x.stop_gradient = False
        out = attn._apply_vha_postmix(x, U, V)
        out.sum().backward()
        self.assertIsNotNone(U.grad)
        self.assertIsNotNone(V.grad)
        self.assertGreater(U.grad.abs().sum().item(), 0.0)
        self.assertGreater(V.grad.abs().sum().item(), 0.0)
        # The main-branch params must not receive gradient from the sparse apply.
        main_u_grad = attn.vha_postmix_U.grad
        self.assertTrue(
            main_u_grad is None or main_u_grad.abs().sum().item() == 0.0
        )


# ===========================================================================
# MQA end-to-end forward/backward: drives MQASelfAttention.forward through the
# sliding-window (main) AND block-sparse branches with the TileLang / cuDNN
# kernels mocked, verifying sparse-branch parameter wiring + the merge result.
# ===========================================================================
_KLR = 32  # kv_lora_rank
_ROPE = 8  # qk_rope_head_dim
_NH = 4  # num_attention_heads
_VD = 32  # v_head_dim


def _fake_swa(
    q, k, v, valid_range, attn_sink=None, sm_scale=None, block_B=None
):
    # Windowed main branch: deterministic differentiable function of q.
    return q[..., :_KLR] * 1.5, None


def _fake_bsa(
    q,
    k,
    block_indices,
    valid_range,
    sm_scale=None,
    block_B=None,
    kv_lora_rank=None,
    attn_sink=None,
):
    # Block-sparse branch: distinct scale so the two branches are separable.
    return q[..., :kv_lora_rank] * 0.5, None


def _install_fake_kernels():
    """Inject fake TileLang MQA kernel modules into sys.modules.

    ``MQASelfAttention.forward`` imports the kernels lazily
    (``from paddlefleet.tilelang_ops.hysparse import ...``), so patching
    sys.modules at call time is sufficient and avoids the SM100/FlashMLA/cuDNN
    hardware dependency.
    """
    tilelang_ops = types.ModuleType("paddlefleet.tilelang_ops")
    hysparse = types.ModuleType("paddlefleet.tilelang_ops.hysparse")
    hysparse.sliding_window_mqa_attention = _fake_swa
    bsa_tl = types.ModuleType(
        "paddlefleet.tilelang_ops.hysparse.block_sparse_mqa_tl"
    )
    bsa_tl.block_sparse_mqa_attention_tl = _fake_bsa
    tilelang_ops.hysparse = hysparse
    hysparse.block_sparse_mqa_tl = bsa_tl
    return mock.patch.dict(
        sys.modules,
        {
            "paddlefleet.tilelang_ops": tilelang_ops,
            "paddlefleet.tilelang_ops.hysparse": hysparse,
            "paddlefleet.tilelang_ops.hysparse.block_sparse_mqa_tl": bsa_tl,
        },
    )


class TestMQAForwardSparseBranch(unittest.TestCase):
    """Real MQASelfAttention.forward regression (both branches, mocked kernels)."""

    def _build(self, rank=2):
        return _build_mqa(
            gated_attention=False,
            sliding_window=(128, 0),
            hy_sparse_block_sparse_use_tilelang=True,
            use_vha_attention=True,
            vha_postmix_rank=rank,
        )

    def _canned_qkv(self, attn, b, s, s_kv, requires_grad):
        """Return the (query, key, value) an MQA gqkv would produce (absorbed)."""
        dk = _KLR + _ROPE
        query = paddle.randn([b, s, _NH, dk], dtype="float32")
        query.stop_gradient = not requires_grad
        key = paddle.randn([b, s_kv, 1, dk], dtype="float32")
        value = paddle.randn([b, s_kv, 1, _KLR], dtype="float32")
        return query, key, value

    def _absorb(self, attn, x_klr):
        """Reference for forward's compute_absorbed_v: [b,s,nh*klr]->[b,s,nh*vd]."""
        b, s = x_klr.shape[0], x_klr.shape[1]
        w = attn.kv_b_proj.weight.reshape([_KLR, _NH, -1])[
            :, :, attn.qk_nope_head_dim :
        ]
        m = x_klr.reshape([b, s, _NH, _KLR])
        o = paddle.einsum("bshl,lhv->bshv", m, w)
        return o.reshape([b, s, _NH * _VD])

    def _postmix(self, x, U, V):
        """Reference for _apply_vha_postmix (ungrouped)."""
        b, s = x.shape[0], x.shape[1]
        m = x.reshape([b, s, _NH, _VD])
        z = paddle.einsum("bthd,hr->btrd", m, U)
        delta = paddle.einsum("btrd,hr->bthd", z, V)
        return (m + delta).reshape([b, s, _NH * _VD])

    def test_forward_merges_main_and_sparse_branches(self):
        attn = self._build(rank=2)
        attn.eval()
        # Main postmix V stays 0 (identity); make the sparse postmix non-trivial
        # so the merged output can only match if the sparse branch is wired to
        # sparse_vha_postmix_U/V (not the main params).
        attn.sparse_vha_postmix_V.set_value(
            paddle.randn(attn.sparse_vha_postmix_V.shape, dtype="float32") * 0.1
        )

        b, s, s_kv = 2, 4, 4
        query, key, value = self._canned_qkv(attn, b, s, s_kv, False)
        hidden = paddle.zeros([b, s, attn.config.hidden_size], dtype="float32")

        with self._patched(attn, query, key, value):
            out, _ = attn(
                hidden,
                attention_mask=None,
                shared_kv=[key, None],
            )

        # Reference merge: absorbed(swa) + sparse_postmix(absorbed(bsa)).
        a_swa = self._absorb(
            attn, (query[..., :_KLR] * 1.5).reshape([b, s, _NH * _KLR])
        )
        a_bsa = self._absorb(
            attn, (query[..., :_KLR] * 0.5).reshape([b, s, _NH * _KLR])
        )
        a_bsa = self._postmix(
            a_bsa, attn.sparse_vha_postmix_U, attn.sparse_vha_postmix_V
        )
        ref, _ = attn.o_proj(a_swa + a_bsa)

        self.assertEqual(out.shape, [b, s, attn.config.hidden_size])
        self.assertTrue(
            paddle.allclose(out, ref, atol=1e-5).item(),
            "MQA forward must merge main + sparse-postmix branches",
        )

    def test_backward_both_branch_postmix_params(self):
        attn = self._build(rank=2)
        attn.train()
        # Randomise both Vs so U and V both receive gradient (dL/dU=0 at V=0).
        attn.vha_postmix_V.set_value(
            paddle.randn(attn.vha_postmix_V.shape, dtype="float32") * 0.1
        )
        attn.sparse_vha_postmix_V.set_value(
            paddle.randn(attn.sparse_vha_postmix_V.shape, dtype="float32") * 0.1
        )

        b, s, s_kv = 2, 4, 4
        query, key, value = self._canned_qkv(attn, b, s, s_kv, True)
        hidden = paddle.zeros([b, s, attn.config.hidden_size], dtype="float32")

        with self._patched(attn, query, key, value):
            out, _ = attn(
                hidden,
                attention_mask=None,
                shared_kv=[key, None],
            )
            out.sum().backward()

        for name, p in [
            ("vha_postmix_U", attn.vha_postmix_U),
            ("vha_postmix_V", attn.vha_postmix_V),
            ("sparse_vha_postmix_U", attn.sparse_vha_postmix_U),
            ("sparse_vha_postmix_V", attn.sparse_vha_postmix_V),
        ]:
            self.assertIsNotNone(p.grad, f"{name} grad must exist")
            self.assertGreater(
                p.grad.abs().sum().item(), 0.0, f"{name} grad must be non-zero"
            )
        self.assertIsNotNone(query.grad)
        self.assertGreater(query.grad.abs().sum().item(), 0.0)

    def _patched(self, attn, query, key, value):
        """Context manager: fake kernels + canned gqkv + no-op valid_range."""

        class _Ctx:
            def __enter__(ctx):
                ctx._km = _install_fake_kernels()
                ctx._km.__enter__()
                ctx._vr = mock.patch.object(
                    mla_mod,
                    "build_hysparse_valid_range",
                    lambda *a, **k: None,
                )
                ctx._vr.__enter__()
                ctx._qkv = mock.patch.object(
                    attn,
                    "get_query_key_value_tensors",
                    lambda *a, **k: (query, key, value, None, None, None),
                )
                ctx._qkv.__enter__()
                return ctx

            def __exit__(ctx, *exc):
                ctx._qkv.__exit__(*exc)
                ctx._vr.__exit__(*exc)
                ctx._km.__exit__(*exc)
                return False

        return _Ctx()


if __name__ == "__main__":
    unittest.main()
