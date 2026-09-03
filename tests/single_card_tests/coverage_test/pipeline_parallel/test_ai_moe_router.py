#!/usr/bin/env python3
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
"""
Unit tests for the p2p_overlap_dw_calc deferral points across:
  1. FusedGateDetachMatmul (moe_router.py) - defer_dw=False/True branches
  2. gate_detach_matmul (moe_router.py) - passes defer_dw through
  3. TopKRouter.forward (moe_router.py) - reads config.p2p_overlap_dw_calc
  4. DeferredWeightGradLinear (dw_overlap.py) - deferred-dW stand-in for a bias-free linear
  5. MultiLatentAttention.forward - "attn_out_proj" selects DeferredWeightGradLinear
  6. MlpNode / FusionMoePyLayer - defer_dw pass-through
  7. MoELayer.__init__ - reads config.p2p_overlap_dw_calc
  8. deferrable_linear - each attention point is independently selectable

Each test prints a message confirming the branch that was exercised.

Run with:
  cd /root/paddlejob/share-storage/gpfs/system-public/wangjinheng/erniebot-fleet
  SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
  NVSHMEM_LIB="$SITE_PACKAGES/nvidia/nvshmem/lib"
  export LD_LIBRARY_PATH="$NVSHMEM_LIB:$LD_LIBRARY_PATH"
  export PYTHONPATH=./ernie5:./utils:./third_party/ernie-core/src:./third_party/ernie-core/PaddleFormers:./third_party/ernie-core/PaddleFleet/src/:./third_party/data_processor:$PYTHONPATH
  python third_party/ernie-core/PaddleFleet/tests/single_card_tests/ai_edited_test/test_ai_dw_overlap.py
"""

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

os.environ["FLAGS_cudnn_deterministic"] = "True"

from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    WeightGradStore,
)

from paddlefleet.transformer.dw_overlap import (
    DeferredWeightGradLinear,
    deferrable_linear,
    install_sonic_moe_dw_deferral,
)
from paddlefleet.transformer.moe.moe_router import (
    FusedGateDetachMatmul,
    gate_detach_matmul,
)
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)
from paddlefleet.transformer.transformer_config import (
    TransformerConfig,
    dw_overlap_enabled,
)
from paddlefleet.utils import init_method_normal, scaled_init_method_normal


# ---------------------------------------------------------------------------
# Helper: create a parameter that mimics AMP O2 main_grad pattern
# ---------------------------------------------------------------------------
def make_weight_with_main_grad(shape, dtype="float32"):
    """Create a parameter with main_grad + _apply_backward_hook."""
    w = paddle.create_parameter(
        shape=shape,
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Normal(),
    )
    w.main_grad = None
    w._hook_call_count = 0

    def _hook():
        w._hook_call_count += 1
        print(f"[_apply_backward_hook] called, count={w._hook_call_count}")

    w._apply_backward_hook = _hook
    return w


# ---------------------------------------------------------------------------
# Helpers for MLA tests
# ---------------------------------------------------------------------------
class _SimpleLinear(nn.Layer):
    """Returns (out, None) like RowParallelLinear with no bias.
    weight shape: [in_features, out_features] (F.linear convention)."""

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.weight = make_weight_with_main_grad([in_features, out_features])

    def forward(self, x):
        return F.linear(x, self.weight), None


class _BiasedLinear(nn.Layer):
    """Returns (out, bias) like ColumnParallelLinear."""

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x), self.linear.bias


class _SimpleRMSNorm(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_size = kwargs.get("normalized_shape", kwargs.get("hidden_size"))
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = 1e-5

    def forward(self, x):
        d = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d * self.weight


from paddlefleet.transformer.dot_product_attention import DotProductAttention


def _make_mla_config(**overrides):
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
        "pipeline_model_parallel_size": 2,
        "virtual_pipeline_model_parallel_size": 2,
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
    return TransformerConfig(**defaults)


def _make_mla(overlap_out_proj=False, use_bias=False, **cfg_kwargs):
    """Build an MLASelfAttention with _SimpleLinear as o_proj."""
    config = _make_mla_config(use_bias=use_bias, **cfg_kwargs)
    config.p2p_overlap_dw_calc = ["attn_out_proj"] if overlap_out_proj else None
    spec = MLASelfAttentionSublayersSpec(
        q_proj=_BiasedLinear,
        q_a_proj=_BiasedLinear,
        q_b_proj=_BiasedLinear,
        kv_a_proj_with_mqa=_BiasedLinear,
        kv_b_proj=_BiasedLinear,
        core_attention=DotProductAttention,
        o_proj=_SimpleLinear,
        q_a_layernorm=_SimpleRMSNorm,
        kv_a_layernorm=_SimpleRMSNorm,
    )
    attn = MLASelfAttention(config=config, sublayers_spec=spec, layer_number=1)
    return attn


# ============================================================
# Test 1: FusedGateDetachMatmul – no-overlap path (defer_dw=False)
# ============================================================
class TestFusedGateDetachMatmulNoOverlap(unittest.TestCase):
    """
    When defer_dw=False the backward should run the classic
    matmul_grad path and return (x_grad, w_grad) directly.

    Note: FusedGateDetachMatmul.forward applies w = w.T internally,
    so w input should have shape [n_experts, hidden] (as in TopKRouter.weight)
    and output is [B, n_experts].
    F.linear(x, w_t) in Paddle computes x @ w_t, so w_t = w.T = [hidden, n_experts]
    needs x = [B, hidden] to produce [B, n_experts].
    """

    def test_forward_output_shape(self):
        """Forward produces the expected [B, n_experts] output."""
        print("\n[FusedGateDetachMatmul no-overlap] testing forward shape")
        B, D, E = 8, 16, 4
        x = paddle.randn([B, D], dtype="float32")
        # w shape [E, D]: forward does w=w.T -> [D,E], F.linear(x,[D,E]) = x@[D,E] -> [B,E]
        w = paddle.randn([E, D], dtype="float32")
        x.stop_gradient = False
        w.stop_gradient = False

        out = FusedGateDetachMatmul.apply(x, w, False)
        self.assertEqual(
            list(out.shape),
            [B, E],
            f"Expected shape [{B},{E}], got {list(out.shape)}",
        )
        print(
            f"[FusedGateDetachMatmul no-overlap] forward shape={out.shape} OK"
        )

    def test_backward_returns_both_grads(self):
        """no-overlap backward: both x_grad and w_grad must be non-None."""
        print("\n[FusedGateDetachMatmul no-overlap] testing backward grads")
        B, D, E = 8, 16, 4
        paddle.seed(42)
        x = paddle.randn([B, D], dtype="float32")
        # w shape [E, D] matching TopKRouter.weight convention
        w = paddle.randn([E, D], dtype="float32")
        x.stop_gradient = False
        w.stop_gradient = False

        out = FusedGateDetachMatmul.apply(x, w, False)
        out.sum().backward()

        self.assertIsNotNone(x.grad, "x.grad should not be None")
        self.assertIsNotNone(w.grad, "w.grad should not be None (no-overlap)")
        self.assertEqual(list(x.grad.shape), [B, D], "x_grad shape mismatch")
        self.assertEqual(list(w.grad.shape), [E, D], "w_grad shape mismatch")
        print(
            f"[FusedGateDetachMatmul no-overlap] x_grad shape={x.grad.shape}, "
            f"w_grad shape={w.grad.shape} OK"
        )


# ============================================================
# Test 2: FusedGateDetachMatmul – overlap path (defer_dw=True)
# ============================================================
class TestFusedGateDetachMatmulOverlap(unittest.TestCase):
    """
    When defer_dw=True:
      - x_grad is computed immediately
      - w_grad is deferred to WeightGradStore.cache (returns None for w_grad)
      - After WeightGradStore.flush() + pop(), weight.main_grad is updated
    """

    def setUp(self):
        WeightGradStore.clear()

    def test_overlap_defers_w_grad_to_cache(self):
        """
        defer_dw=True: backward puts dw computation into WeightGradStore.cache.
        The weight needs main_grad because the overlap path requires it.
        w shape [E, D] matching TopKRouter.weight convention.
        """
        print(
            "\n[FusedGateDetachMatmul overlap] testing WeightGradStore deferral"
        )
        B, D, E = 8, 16, 4
        paddle.seed(42)
        x = paddle.randn([B, D], dtype="float32")
        # w shape [E, D]: forward does w=w.T -> [D,E], F.linear(x,[D,E])=x@[D,E]->[B,E]
        w = make_weight_with_main_grad([E, D])

        x.stop_gradient = False
        w.stop_gradient = False

        out = FusedGateDetachMatmul.apply(x, w, True)
        out.sum().backward()

        # x_grad must exist
        self.assertIsNotNone(x.grad, "x_grad should exist even in overlap mode")
        self.assertEqual(list(x.grad.shape), [B, D])

        # WeightGradStore.cache must have received the deferred computation
        self.assertGreater(
            len(WeightGradStore.cache),
            0,
            "WeightGradStore.cache should be non-empty after overlap backward",
        )
        print(
            "[FusedGateDetachMatmul overlap] deferred to WeightGradStore.cache OK, "
            f"cache len={len(WeightGradStore.cache)}"
        )

        # Flush and pop to actually run the deferred computation
        WeightGradStore.flush()
        WeightGradStore.pop()

        self.assertIsNotNone(w.main_grad, "main_grad should be set after pop()")
        self.assertEqual(
            list(w.main_grad.shape),
            [E, D],
            "main_grad shape must match weight shape [E, D]",
        )
        print(
            f"[FusedGateDetachMatmul overlap] main_grad shape={w.main_grad.shape} OK"
        )
        WeightGradStore.clear()

    def test_overlap_x_grad_matches_no_overlap(self):
        """
        x_grad in overlap mode must numerically match the no-overlap path.
        Both use w shape [E, D] matching TopKRouter.weight convention.
        """
        print("\n[FusedGateDetachMatmul overlap] x_grad numerical comparison")
        B, D, E = 8, 16, 4
        paddle.seed(7)
        x_np = np.random.randn(B, D).astype("float32")
        # w shape [E, D]
        w_np = np.random.randn(E, D).astype("float32")

        # No-overlap: plain Tensor (no main_grad needed)
        x1 = paddle.to_tensor(x_np)
        w1 = paddle.to_tensor(w_np)
        x1.stop_gradient = False
        w1.stop_gradient = False
        out1 = FusedGateDetachMatmul.apply(x1, w1, False)
        out1.sum().backward()
        x_grad_no_overlap = x1.grad.numpy().copy()

        # Overlap: weight needs main_grad
        WeightGradStore.clear()
        x2 = paddle.to_tensor(x_np)
        w2 = make_weight_with_main_grad([E, D])
        w2.set_value(paddle.to_tensor(w_np))
        x2.stop_gradient = False
        w2.stop_gradient = False
        out2 = FusedGateDetachMatmul.apply(x2, w2, True)
        out2.sum().backward()
        x_grad_overlap = x2.grad.numpy().copy()

        np.testing.assert_allclose(
            x_grad_overlap,
            x_grad_no_overlap,
            rtol=1e-5,
            err_msg="x_grad must match between overlap and no-overlap paths",
        )
        print(
            "[FusedGateDetachMatmul overlap] x_grad matches no-overlap path OK"
        )
        WeightGradStore.clear()


# ============================================================
# Test 3: gate_detach_matmul wrapper function
# ============================================================
class TestGateDetachMatmul(unittest.TestCase):
    """gate_detach_matmul must pass defer_dw through to FusedGateDetachMatmul."""

    def setUp(self):
        WeightGradStore.clear()

    def test_fuse_true_no_overlap(self):
        """use_fuse=True, defer_dw=False -> standard fused path, w_grad returned.
        w shape [E, D] as in TopKRouter.weight."""
        print("\n[gate_detach_matmul] fuse=True, no-overlap")
        B, D, E = 4, 8, 4
        x = paddle.randn([B, D], dtype="float32")
        w = paddle.randn(
            [E, D], dtype="float32"
        )  # [E, D] like TopKRouter.weight
        x.stop_gradient = False
        w.stop_gradient = False
        out = gate_detach_matmul(x, w, use_fuse=True, defer_dw=False)
        out.sum().backward()
        self.assertIsNotNone(
            w.grad, "w.grad must exist in no-overlap fused path"
        )
        print(
            f"[gate_detach_matmul] fuse=True no-overlap OK, w.grad shape={w.grad.shape}"
        )

    def test_fuse_true_overlap_defers(self):
        """use_fuse=True, defer_dw=True -> WeightGradStore.cache receives dw.
        w shape [E, D] as in TopKRouter.weight."""
        print("\n[gate_detach_matmul] fuse=True, overlap")
        B, D, E = 4, 8, 4
        x = paddle.randn([B, D], dtype="float32")
        w = make_weight_with_main_grad([E, D])  # [E, D] like TopKRouter.weight
        x.stop_gradient = False
        w.stop_gradient = False
        out = gate_detach_matmul(x, w, use_fuse=True, defer_dw=True)
        out.sum().backward()
        self.assertGreater(
            len(WeightGradStore.cache),
            0,
            "WeightGradStore.cache must receive deferred dw when overlap=True",
        )
        print(
            "[gate_detach_matmul] fuse=True overlap OK, WeightGradStore.cache non-empty"
        )
        WeightGradStore.flush()
        WeightGradStore.pop()
        WeightGradStore.clear()

    def test_fuse_false_path(self):
        """use_fuse=False -> plain F.linear(x, w) in Paddle = x @ w, so w [D, E]."""
        print("\n[gate_detach_matmul] fuse=False path")
        B, D, E = 4, 8, 4
        x = paddle.randn([B, D], dtype="float32")
        w = paddle.randn(
            [D, E], dtype="float32"
        )  # F.linear(x,w)=x@w needs [D,E]
        x.stop_gradient = False
        w.stop_gradient = False
        out = gate_detach_matmul(x, w, use_fuse=False, defer_dw=False)
        self.assertIsNotNone(out)
        self.assertEqual(list(out.shape), [B, E])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        print("[gate_detach_matmul] fuse=False path OK")


# ============================================================
# Test 4: TopKRouter reads defer_dw from config
# ============================================================
class TestTopKRouterDwP2POverlap(unittest.TestCase):
    """TopKRouter.forward must read config.p2p_overlap_dw_calc and pass it to gate_detach_matmul."""

    def _make_router_config(self, overlap_gate=False):
        config = TransformerConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            topk_method="greedy",
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            routed_scaling_factor_learnable=False,
            scoring_func="softmax",
            moe_router_load_balancing_type="aux_loss",
            moe_deep_gemm=False,
            moe_router_force_load_balancing=False,
            moe_router_fusion=True,
            router_z_loss_coef=0.0,
            router_aux_loss_coef=0.01,
            context_parallel_size=1,
            sequence_parallel=False,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
        )
        config.p2p_overlap_dw_calc = (
            ["moe_router_gate"] if overlap_gate else None
        )
        return config

    def setUp(self):
        WeightGradStore.clear()

    def test_router_forward_no_overlap(self):
        """Router forward with defer_dw=False should complete normally."""
        print("\n[TopKRouter] defer_dw=False forward pass")
        from paddlefleet.transformer.moe.moe_router import TopKRouter

        with patch(
            "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        ):
            config = self._make_router_config(overlap_gate=False)
            router = TopKRouter(config)
            hidden = paddle.randn([2, 4, 64])
            out = router(hidden)
            self.assertIsNotNone(out)
            print("[TopKRouter] defer_dw=False forward OK")

    def test_router_forward_overlap_defers_dw(self):
        """Router forward with defer_dw=True: gate dw deferred to WeightGradStore."""
        print("\n[TopKRouter] defer_dw=True forward + backward pass")
        from paddlefleet.transformer.moe.moe_router import TopKRouter

        with patch(
            "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        ):
            config = self._make_router_config(overlap_gate=True)
            router = TopKRouter(config)
            # Give router.weight a main_grad (required for overlap path)
            router.weight.main_grad = None

            def _hook():
                print(
                    "[TopKRouter] _apply_backward_hook called on router.weight"
                )

            router.weight._apply_backward_hook = _hook

            hidden = paddle.randn([2, 4, 64])
            out = router(hidden)
            self.assertIsNotNone(out)

            # Trigger backward to activate WeightGradStore
            top_gate = out[1]  # (capacity, top_gate, top_idx, ...)
            top_gate.sum().backward()

            # In overlap path, dw for the gate weight goes to WeightGradStore
            self.assertGreater(
                len(WeightGradStore.cache),
                0,
                "WeightGradStore.cache should be non-empty in overlap mode",
            )
            print(
                f"[TopKRouter] defer_dw=True WeightGradStore.cache len="
                f"{len(WeightGradStore.cache)} OK"
            )
        WeightGradStore.clear()


# ============================================================
# Test 5: DeferredWeightGradLinear - forward correctness
# ============================================================
class TestDeferredWeightGradLinearForward(unittest.TestCase):
    """DeferredWeightGradLinear forward must be bit-exact with F.linear(x, weight)."""

    def test_forward_matches_linear(self):
        print("\n[DeferredWeightGradLinear] forward matches F.linear")
        B, S, IN, OUT = 2, 4, 128, 64
        paddle.seed(0)
        x = paddle.randn([B, S, IN])
        weight = paddle.randn([IN, OUT])
        x.stop_gradient = False
        weight.stop_gradient = True  # no backward needed here

        out_overlap = DeferredWeightGradLinear.apply(x, weight)
        out_ref = F.linear(x.detach(), weight)

        np.testing.assert_allclose(
            out_overlap.numpy(),
            out_ref.numpy(),
            rtol=1e-5,
            err_msg="DeferredWeightGradLinear forward must match F.linear",
        )
        print(
            f"[DeferredWeightGradLinear] forward shape={out_overlap.shape} matches OK"
        )


# ============================================================
# Test 6: DeferredWeightGradLinear - backward defers dw to WeightGradStore
# ============================================================
class TestDeferredWeightGradLinearBackward(unittest.TestCase):
    """DeferredWeightGradLinear backward: dx computed immediately, dw deferred to WeightGradStore.cache."""

    def setUp(self):
        WeightGradStore.clear()

    def test_backward_defers_dw(self):
        print(
            "\n[DeferredWeightGradLinear] backward defers dw to WeightGradStore.cache"
        )
        B, S, IN, OUT = 2, 4, 128, 64
        paddle.seed(1)
        x = paddle.randn([B, S, IN])
        weight = make_weight_with_main_grad([IN, OUT])
        x.stop_gradient = False
        weight.stop_gradient = False

        out = DeferredWeightGradLinear.apply(x, weight)
        out.sum().backward()

        # dx must be computed
        self.assertIsNotNone(x.grad, "dx must be computed immediately")
        self.assertEqual(list(x.grad.shape), [B, S, IN])

        # WeightGradStore.cache must contain the deferred computation
        self.assertGreater(
            len(WeightGradStore.cache),
            0,
            "DeferredWeightGradLinear backward must push dw into WeightGradStore.cache",
        )
        print(
            f"[DeferredWeightGradLinear] dw correctly deferred to WeightGradStore.cache, "
            f"len={len(WeightGradStore.cache)}"
        )

        WeightGradStore.flush()
        WeightGradStore.pop()
        self.assertIsNotNone(weight.main_grad)
        self.assertEqual(list(weight.main_grad.shape), [IN, OUT])
        print(
            f"[DeferredWeightGradLinear] main_grad shape={weight.main_grad.shape} OK"
        )
        WeightGradStore.clear()

    def test_backward_dx_matches_reference(self):
        """dx from DeferredWeightGradLinear must match dx from plain F.linear backward."""
        print(
            "\n[DeferredWeightGradLinear] dx numerical comparison with reference"
        )
        B, S, IN, OUT = 2, 3, 32, 16
        paddle.seed(3)
        x_np = np.random.randn(B, S, IN).astype("float32")
        w_np = np.random.randn(IN, OUT).astype("float32")

        # Reference: F.linear backward
        x_ref = paddle.to_tensor(x_np)
        w_ref = paddle.to_tensor(w_np)
        x_ref.stop_gradient = False
        w_ref.stop_gradient = True
        F.linear(x_ref, w_ref).sum().backward()
        dx_ref = x_ref.grad.numpy().copy()

        # DeferredWeightGradLinear backward
        WeightGradStore.clear()
        x_ours = paddle.to_tensor(x_np)
        w_ours = make_weight_with_main_grad([IN, OUT])
        w_ours.set_value(paddle.to_tensor(w_np))
        x_ours.stop_gradient = False
        w_ours.stop_gradient = False
        DeferredWeightGradLinear.apply(x_ours, w_ours).sum().backward()
        dx_ours = x_ours.grad.numpy().copy()

        np.testing.assert_allclose(
            dx_ours,
            dx_ref,
            rtol=1e-5,
            err_msg="dx from DeferredWeightGradLinear must match reference",
        )
        print("[DeferredWeightGradLinear] dx matches reference OK")
        WeightGradStore.clear()

    def test_backward_dw_matches_reference(self):
        """dw from DeferredWeightGradLinear (after WeightGradStore.pop) must match x_2d.T @ og_2d."""
        print(
            "\n[DeferredWeightGradLinear] dw numerical comparison with reference"
        )
        B, S, IN, OUT = 2, 3, 32, 16
        paddle.seed(5)
        x_np = np.random.randn(B, S, IN).astype("float32")
        w_np = np.random.randn(IN, OUT).astype("float32")

        # Reference: compute dw = x_2d.T @ og_2d manually
        x_2d = x_np.reshape(-1, IN)
        og = np.ones(
            (B * S, OUT), dtype="float32"
        )  # sum backward => all-ones grad
        dw_ref = x_2d.T @ og  # [IN, OUT]

        # DeferredWeightGradLinear
        WeightGradStore.clear()
        x_ours = paddle.to_tensor(x_np)
        w_ours = make_weight_with_main_grad([IN, OUT])
        w_ours.set_value(paddle.to_tensor(w_np))
        x_ours.stop_gradient = False
        w_ours.stop_gradient = False
        DeferredWeightGradLinear.apply(x_ours, w_ours).sum().backward()

        WeightGradStore.flush()
        WeightGradStore.pop()
        dw_ours = w_ours.main_grad.numpy()

        np.testing.assert_allclose(
            dw_ours,
            dw_ref,
            rtol=1e-4,
            atol=1e-4,
            err_msg="dw from DeferredWeightGradLinear must match x.T@og",
        )
        print("[DeferredWeightGradLinear] dw matches reference OK")
        WeightGradStore.clear()


# ============================================================
# Test 7: MultiLatentAttention.forward – o_proj branch selection
# ============================================================
class TestMultiLatentAttentionOProj(unittest.TestCase):
    """
    "attn_out_proj" selected and use_bias=False -> DeferredWeightGradLinear branch.
    Not selected, or use_bias=True -> standard o_proj branch.
    We exercise the branch predicate straight from production
    (dw_overlap_enabled) so the test cannot drift away from it, without running
    a full MLA forward.
    """

    def _exercise_o_proj_branch(self, overlap_out_proj, use_bias):
        """Directly exercise the o_proj branching logic from MultiLatentAttention.forward."""
        attn = _make_mla(overlap_out_proj=overlap_out_proj, use_bias=use_bias)
        config = attn.config
        o_proj = attn.o_proj

        B, S, H = 1, 4, 128
        core_attn_out = paddle.randn([B, S, H])
        core_attn_out.stop_gradient = False

        selected = dw_overlap_enabled(config, "attn_out_proj")
        print(
            f"\n[MLA o_proj] overlap_out_proj={overlap_out_proj}, use_bias={use_bias}"
        )
        print(f"  config.p2p_overlap_dw_calc={config.p2p_overlap_dw_calc}")
        print(f"  config.use_bias={config.use_bias}")

        if selected and config.use_bias is False:
            print("  -> enter o_proj has overlap (DeferredWeightGradLinear)")
            output = DeferredWeightGradLinear.apply(
                core_attn_out, o_proj.weight
            )
            bias = None
        else:
            print("  -> enter o_proj no overlap (standard o_proj)")
            output, bias = o_proj(core_attn_out)

        return output, bias, selected, config.use_bias

    def test_overlap_branch_used_when_selected_no_bias(self):
        """attn_out_proj selected and use_bias=False -> DeferredWeightGradLinear branch, bias=None."""
        print("\n[MLA o_proj] overlap branch test")
        WeightGradStore.clear()
        output, bias, selected, ub = self._exercise_o_proj_branch(
            overlap_out_proj=True, use_bias=False
        )
        self.assertTrue(selected, "attn_out_proj must be selected")
        self.assertFalse(ub, "config.use_bias must be False")
        self.assertIsNotNone(output)
        self.assertIsNone(bias, "bias must be None in overlap path")
        self.assertEqual(output.shape[-1], 128)
        print(f"[MLA o_proj] overlap branch OK, output shape={output.shape}")
        WeightGradStore.clear()

    def test_standard_branch_used_when_not_selected(self):
        """attn_out_proj not selected -> standard o_proj branch."""
        print("\n[MLA o_proj] standard branch test (not selected)")
        output, bias, selected, ub = self._exercise_o_proj_branch(
            overlap_out_proj=False, use_bias=False
        )
        self.assertFalse(selected, "attn_out_proj must not be selected")
        self.assertIsNotNone(output)
        print("[MLA o_proj] standard branch OK (not selected)")

    def test_standard_branch_used_when_use_bias_true(self):
        """use_bias=True -> standard o_proj branch even when attn_out_proj is selected."""
        print("\n[MLA o_proj] standard branch test (use_bias=True)")
        output, bias, selected, ub = self._exercise_o_proj_branch(
            overlap_out_proj=True, use_bias=True
        )
        self.assertTrue(ub, "config.use_bias must be True")
        self.assertIsNotNone(output)
        print("[MLA o_proj] standard branch OK (use_bias=True)")


# ============================================================
# Test 8: MoELayer selects its deferral point from p2p_overlap_dw_calc
# ============================================================
class TestMoELayerP2POverlapInit(unittest.TestCase):
    """
    MoELayer.__init__ derives its per-node flag from
    dw_overlap_enabled(config, "moe_expert_up_gate_proj"). Test the selector
    semantics directly, without constructing a full MoELayer (which needs
    distributed init).
    """

    @staticmethod
    def _config():
        return TransformerConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
        )

    def test_disabled_by_default(self):
        """p2p_overlap_dw_calc defaults to None -> nothing is deferred."""
        config = self._config()
        self.assertIsNone(config.p2p_overlap_dw_calc)
        self.assertFalse(dw_overlap_enabled(config, "moe_expert_up_gate_proj"))

    def test_empty_list_disables(self):
        """An empty list is an explicit "off", not "all points"."""
        config = self._config()
        config.p2p_overlap_dw_calc = []
        self.assertFalse(dw_overlap_enabled(config, "moe_expert_up_gate_proj"))

    def test_selected_point_enabled(self):
        """Only the listed points are enabled; others stay off."""
        config = self._config()
        config.p2p_overlap_dw_calc = ["moe_expert_up_gate_proj"]
        self.assertTrue(dw_overlap_enabled(config, "moe_expert_up_gate_proj"))
        self.assertFalse(dw_overlap_enabled(config, "moe_router_gate"))
        self.assertFalse(dw_overlap_enabled(config, "attn_out_proj"))

    def test_moe_layer_source_uses_selector(self):
        """MoELayer.__init__ must go through the selector for both expert points."""
        import inspect
        import re

        from paddlefleet.transformer.moe.moe_layer import MoELayer

        # Strip whitespace so the assertion survives black reformatting.
        src = re.sub(r"\s+", "", inspect.getsource(MoELayer.__init__))
        for point in ("moe_expert_up_gate_proj", "moe_expert_down_proj"):
            self.assertIn(
                f'dw_overlap_enabled(config,"{point}")',
                src,
                f"MoELayer.__init__ must select {point} via dw_overlap_enabled",
            )


# ============================================================
# Test 9: FusionMoePyLayer passes defer_dw to MlpNode
# ============================================================
class TestFusionMoePyLayerDwP2POverlap(unittest.TestCase):
    """
    FusionMoePyLayer.forward passes defer_dw to MlpNode which passes it
    to ExpertsGroupGemmContiguousNode. Test both False and True paths.
    """

    @classmethod
    def setUpClass(cls):
        from paddlefleet.tensor_parallel.random import (
            model_parallel_cuda_manual_seed,
        )

        model_parallel_cuda_manual_seed(1234)
        # FP8 fused_stack_quant requires M % 128 == 0, use same dims as test_moe_subbatch
        cls.seq_len = 256
        cls.topk = 2
        cls.hidden_size = 256  # must be multiple of 128
        cls.intermediate_size = 256  # must be multiple of 128
        cls.n_routed_experts = 4

    def setUp(self):
        paddle.seed(42)
        np.random.seed(42)

    def _make_fake_moe_layer(self, tokens_per_expert):
        from paddlefleet.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        from paddlefleet.transformer.mlp import MLPSublayersSpec
        from paddlefleet.transformer.moe.moe_expert import StandardMLPExpert

        hidden_size = self.hidden_size
        intermediate_size = self.intermediate_size
        n_routed_experts = self.n_routed_experts

        config = TransformerConfig(
            hidden_size=hidden_size, gated_linear_unit=True
        )
        mlp_spec = MLPSublayersSpec(
            up_gate_proj=ColumnParallelLinear,
            down_proj=RowParallelLinear,
        )

        class FakeMOELayer(nn.Layer):
            def __init__(inner_self):
                super().__init__()
                inner_self.experts = nn.LayerList(
                    [
                        StandardMLPExpert(
                            config,
                            moe_intermediate_size=intermediate_size,
                            is_expert=True,
                            mlp_spec=mlp_spec,
                        )
                        for _ in range(n_routed_experts)
                    ]
                )
                inner_self.token_dispatcher = SimpleNamespace(
                    _comm_manager=SimpleNamespace(
                        tokens_per_expert=tokens_per_expert,
                    )
                )

            def clear_main_grad(inner_self):
                for expert in inner_self.experts:
                    expert.up_gate_proj.weight.main_grad = None
                    expert.down_proj.weight.main_grad = None

        layer = FakeMOELayer()
        layer = paddle.amp.decorate(layer, level="O2", dtype="bfloat16")
        layer.clear_main_grad()
        return layer

    def _make_dispatch_data(self):
        from paddlefleet.transformer.moe.fp8_utils import tilewise_quant

        hidden_states = paddle.randn(
            [self.seq_len, self.hidden_size], "bfloat16"
        )
        hidden_states, scale = tilewise_quant(hidden_states)
        probs = paddle.randn([self.seq_len, self.topk])
        hidden_states.stop_gradient = False
        probs.stop_gradient = False

        indices_np = np.zeros([self.seq_len, self.topk], dtype=np.int64)
        for i in range(self.seq_len):
            chosen = np.random.choice(
                self.n_routed_experts, self.topk, replace=False
            )
            indices_np[i] = np.sort(chosen)
        indices = paddle.to_tensor(indices_np)
        # tokens_per_expert must match the actual routing in indices,
        # otherwise the permuted buffers contain uninitialized regions.
        tokens_per_expert = (
            np.bincount(
                indices_np.reshape([-1]), minlength=self.n_routed_experts
            )
            .astype(np.int64)
            .tolist()
        )
        return hidden_states, scale, probs, indices, tokens_per_expert

    def _run_fusion_layer(self, defer_dw=False):
        from paddlefleet.transformer.moe.fusion_layer_utils import (
            FusionMoePyLayer,
        )

        hidden_states, scale, probs, indices, tokens_per_expert = (
            self._make_dispatch_data()
        )
        moe_layer = self._make_fake_moe_layer(tokens_per_expert)

        print(f"\n[FusionMoePyLayer] running with defer_dw={defer_dw}")
        out = FusionMoePyLayer.apply(
            hidden_states,
            probs,
            indices,
            moe_layer,
            self.topk,
            use_fp8_mlp=True,
            recompute_moe_gate_up=True,
            dequant_input=True,
            moe_expert_fusion=True,
            recompute_moe_premute=False,
            use_bf16_gemm_weight_grad=True,
            fp8_dispatched_handle={"scale": scale},
            use_auto_subbatch=False,
            defer_expert_up_gate_dw=defer_dw,
        )
        out_grad = paddle.randn_like(out)
        paddle.autograd.backward(out, out_grad)
        print(f"[FusionMoePyLayer] defer_dw={defer_dw} forward+backward OK")
        return out

    def test_no_overlap_forward_backward(self):
        """defer_dw=False: FusionMoePyLayer forward+backward completes."""
        out = self._run_fusion_layer(defer_dw=False)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape[-1], self.hidden_size)
        print(f"[FusionMoePyLayer] no-overlap output shape={out.shape} OK")

    def test_deferral_rejected_without_deep_gemm(self):
        """Requesting deferral on the non-deep_gemm path must raise.

        This fake layer runs with moe_deep_gemm=False, where
        bf16_weight_grad has no WeightGradStore branch, so honouring the
        request is impossible. Before the guard existed this combination was
        a silent no-op and this test claimed to cover "the overlap path".
        The real deferral behaviour is covered by TestBf16WeightGrad*, which
        builds the node with moe_deep_gemm=True.
        """
        print("\n[FusionMoePyLayer] deferral without deep_gemm must raise")
        with self.assertRaisesRegex(ValueError, "moe_deep_gemm=False"):
            self._run_fusion_layer(defer_dw=True)
        print("[FusionMoePyLayer] rejected as expected OK")


# ============================================================
# Test 10: FusedGateDetachMatmul overlap – w_stop_grad=True path
# ============================================================
class TestFusedGateDetachMatmulOverlapWStopGrad(unittest.TestCase):
    """
    When defer_dw=True and w.stop_gradient=True,
    backward should return (x_grad, None) immediately without
    touching WeightGradStore.
    """

    def setUp(self):
        WeightGradStore.clear()

    def test_overlap_w_stop_grad_true(self):
        """
        defer_dw=True, w.stop_gradient=True -> x_grad is computed,
        WeightGradStore.cache stays empty (w_grad skipped entirely).
        Covers moe_router.py lines 129-130.
        """
        print(
            "\n[FusedGateDetachMatmul overlap w_stop_grad=True] testing w_stop_grad branch"
        )
        B, D, E = 4, 8, 4
        paddle.seed(10)
        x = paddle.randn([B, D], dtype="float32")
        w = paddle.randn([E, D], dtype="float32")
        x.stop_gradient = False
        w.stop_gradient = True  # <-- key: stop_gradient=True

        out = FusedGateDetachMatmul.apply(x, w, True)
        out.sum().backward()

        self.assertIsNotNone(
            x.grad, "x_grad must exist even when w is stop_gradient"
        )
        self.assertEqual(list(x.grad.shape), [B, D])
        # WeightGradStore.cache must be EMPTY because w_stop_grad=True skips the put()
        self.assertEqual(
            len(WeightGradStore.cache),
            0,
            "WeightGradStore.cache must be empty when w.stop_gradient=True",
        )
        print(
            f"[FusedGateDetachMatmul overlap w_stop_grad=True] x_grad shape={x.grad.shape}, "
            f"WeightGradStore.cache empty OK"
        )
        WeightGradStore.clear()


# ============================================================
# Test 11: FusedGateDetachMatmul overlap – _compute_weight_grad with main_grad=None init
# ============================================================
class TestFusedGateDetachMatmulComputeWeightGrad(unittest.TestCase):
    """
    Exercises the _compute_weight_grad inner function with main_grad initially None
    (so the 'if weight.main_grad is None: weight.main_grad = ...' path is hit),
    and with _apply_backward_hook present.
    Covers moe_router.py lines 111-120.
    """

    def setUp(self):
        WeightGradStore.clear()

    def test_compute_weight_grad_initializes_main_grad_from_none(self):
        """
        After WeightGradStore.pop(), _compute_weight_grad must:
          1. Initialize main_grad from None to zeros (line 112-113)
          2. Add w_grad to main_grad (line 115)
          3. Call _apply_backward_hook if present (lines 119-120)
        """
        print("\n[_compute_weight_grad] main_grad=None initialization + hook")
        B, D, E = 4, 8, 4
        paddle.seed(11)
        x = paddle.randn([B, D], dtype="float32")
        w = make_weight_with_main_grad([E, D])
        # Ensure main_grad starts as None
        w.main_grad = None
        x.stop_gradient = False
        w.stop_gradient = False

        out = FusedGateDetachMatmul.apply(x, w, True)
        out.sum().backward()

        # At this point _compute_weight_grad is in cache but not yet run
        self.assertIsNone(
            w.main_grad, "main_grad should still be None before pop()"
        )
        print(
            f"[_compute_weight_grad] before pop: main_grad is None, cache len={len(WeightGradStore.cache)}"
        )

        # Now actually run the deferred computation
        WeightGradStore.flush()
        WeightGradStore.pop()

        # main_grad must have been initialized and filled
        self.assertIsNotNone(w.main_grad, "main_grad must be set after pop()")
        self.assertEqual(list(w.main_grad.shape), [E, D])
        # _apply_backward_hook should have been called
        self.assertGreater(
            w._hook_call_count, 0, "_apply_backward_hook must have been called"
        )
        print(
            f"[_compute_weight_grad] main_grad shape={w.main_grad.shape}, "
            f"hook_call_count={w._hook_call_count} OK"
        )
        WeightGradStore.clear()


# ============================================================
# Test 12: ExpertsGroupGemmContiguousNode.bf16_weight_grad – moe_deep_gemm paths
# ============================================================
class TestBf16WeightGradMoeDeepGemm(unittest.TestCase):
    """
    Directly tests ExpertsGroupGemmContiguousNode.bf16_weight_grad with
    moe_deep_gemm=True + moe_expert_fusion=True to cover lines 1638-1723
    in fp8_utils.py.
    Uses mock.patch for deep_gemm calls to avoid GEMM kernel dimension constraints.
    Tests both:
      - main_grad path (hasattr main_grad): lines 1636-1682
      - no-main_grad path: lines 1684-1724
      - defer_dw=False (direct compute) and defer_dw=True (deferred)
    """

    def setUp(self):
        paddle.seed(42)
        WeightGradStore.clear()

    def _make_node_and_weight(self, defer_dw=False):
        """Build node with moe_deep_gemm=True, mocked deep_gemm."""
        from types import SimpleNamespace

        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        H, I, n_exp = 256, 256, 4

        stacked_w = paddle.create_parameter(
            shape=[n_exp * 2 * I, H],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Normal(),
        )
        stacked_w.main_grad = None

        _hook_calls = [0]

        def _hook():
            _hook_calls[0] += 1
            print(
                f"[bf16_weight_grad grouped hook] called, count={_hook_calls[0]}"
            )

        stacked_w._apply_backward_hook = _hook
        stacked_w._hook_calls = _hook_calls

        custom_map = SimpleNamespace(grouped_gemm_experts=stacked_w)

        tokens_per_expert = [128, 128, 128, 128]
        tokens_per_expert_tensor = paddle.to_tensor(
            tokens_per_expert, dtype="int32"
        )

        node = ExpertsGroupGemmContiguousNode(
            custom_map=custom_map,
            moe_expert_fusion=True,
            moe_deep_gemm=True,
            use_fp8_mlp=True,
            use_bf16_gemm_weight_grad=True,
            defer_expert_up_gate_dw=defer_dw,
        )
        node.tokens_per_expert = tokens_per_expert
        node.tokens_per_expert_tensor = tokens_per_expert_tensor
        node.dequant_input = False
        node.input = None

        total = sum(tokens_per_expert)
        x = paddle.randn([total, H], dtype="bfloat16")
        dy = paddle.randn([total, 2 * I], dtype="bfloat16")
        return node, stacked_w, x, dy

    def _patched_k_grouped_gemm(self, x, dy, weight_grad, tpe, tpe_tensor, out):
        """Mock for deep_gemm.k_grouped_bf16_gemm_tn_contiguous - just adds zeros."""
        print(
            f"[mock k_grouped_bf16_gemm_tn_contiguous] called, x={x.shape}, dy={dy.shape}"
        )

    def test_main_grad_no_overlap(self):
        """
        moe_deep_gemm=True, main_grad present, defer_dw=False:
        _compute_weight_grad runs immediately.
        Covers fp8_utils.py lines 1636-1641, 1644-1645, 1661-1664.
        """
        print("\n[bf16_weight_grad] moe_deep_gemm=True, main_grad, no overlap")
        node, weights, x, dy = self._make_node_and_weight(defer_dw=False)
        weights.main_grad = None

        with patch(
            "paddlefleet.transformer.moe.fp8_utils.deep_gemm"
        ) as mock_dg:
            mock_dg.set_num_sms = lambda n: print(f"[mock] set_num_sms({n})")
            mock_dg.k_grouped_bf16_gemm_tn_contiguous = (
                self._patched_k_grouped_gemm
            )
            node.bf16_weight_grad(dy, x, weights, defer_dw=False)

        self.assertIsNotNone(
            weights.main_grad, "main_grad must be initialized after direct call"
        )
        self.assertEqual(list(weights.main_grad.shape), list(weights.shape))
        print(
            f"[bf16_weight_grad] main_grad shape={weights.main_grad.shape}, no_overlap OK"
        )

    def test_main_grad_with_overlap(self):
        """
        moe_deep_gemm=True, main_grad present, defer_dw=True:
        _compute_weight_grad deferred to WeightGradStore.
        Covers fp8_utils.py lines 1636-1641, 1644-1660.
        """
        print(
            "\n[bf16_weight_grad] moe_deep_gemm=True, main_grad, defer_dw=True"
        )
        node, weights, x, dy = self._make_node_and_weight(defer_dw=True)
        weights.main_grad = None

        with patch(
            "paddlefleet.transformer.moe.fp8_utils.deep_gemm"
        ) as mock_dg:
            mock_dg.set_num_sms = lambda n: print(f"[mock] set_num_sms({n})")
            mock_dg.k_grouped_bf16_gemm_tn_contiguous = (
                self._patched_k_grouped_gemm
            )
            node.bf16_weight_grad(dy, x, weights, defer_dw=True)

        self.assertGreater(
            len(WeightGradStore.cache),
            0,
            "WeightGradStore must have deferred computation",
        )
        print(
            f"[bf16_weight_grad] cache len={len(WeightGradStore.cache)} OK, flushing..."
        )

        with patch(
            "paddlefleet.transformer.moe.fp8_utils.deep_gemm"
        ) as mock_dg:
            mock_dg.set_num_sms = lambda n: None
            mock_dg.k_grouped_bf16_gemm_tn_contiguous = (
                self._patched_k_grouped_gemm
            )
            WeightGradStore.flush()
            WeightGradStore.pop()

        self.assertIsNotNone(
            weights.main_grad,
            "main_grad must be set after WeightGradStore.pop()",
        )
        print("[bf16_weight_grad] main_grad set after pop(), overlap OK")
        WeightGradStore.clear()

    def test_no_main_grad_no_overlap(self):
        """
        moe_deep_gemm=True, NO main_grad (weights.grad path), defer_dw=False:
        _compute_weight_grad runs immediately using weights.grad.
        Covers fp8_utils.py lines 1684-1710.
        """
        print(
            "\n[bf16_weight_grad] moe_deep_gemm=True, no main_grad, no overlap"
        )
        node, weights, x, dy = self._make_node_and_weight(defer_dw=False)
        if hasattr(weights, "main_grad"):
            del weights.main_grad

        with patch(
            "paddlefleet.transformer.moe.fp8_utils.deep_gemm"
        ) as mock_dg:
            mock_dg.set_num_sms = lambda n: None
            mock_dg.k_grouped_bf16_gemm_tn_contiguous = (
                self._patched_k_grouped_gemm
            )
            node.bf16_weight_grad(dy, x, weights, defer_dw=False)

        self.assertIsNotNone(
            weights.grad, "weights.grad must be set after direct computation"
        )
        print("[bf16_weight_grad] weights.grad set, no_main_grad no_overlap OK")

    def test_no_main_grad_with_overlap(self):
        """
        moe_deep_gemm=True, NO main_grad, defer_dw=True:
        _compute_weight_grad deferred using weights.grad.
        Covers fp8_utils.py lines 1684-1706.
        """
        print(
            "\n[bf16_weight_grad] moe_deep_gemm=True, no main_grad, defer_dw=True"
        )
        node, weights, x, dy = self._make_node_and_weight(defer_dw=True)
        if hasattr(weights, "main_grad"):
            del weights.main_grad

        with patch(
            "paddlefleet.transformer.moe.fp8_utils.deep_gemm"
        ) as mock_dg:
            mock_dg.set_num_sms = lambda n: None
            mock_dg.k_grouped_bf16_gemm_tn_contiguous = (
                self._patched_k_grouped_gemm
            )
            node.bf16_weight_grad(dy, x, weights, defer_dw=True)

        self.assertGreater(
            len(WeightGradStore.cache),
            0,
            "WeightGradStore must have deferred computation",
        )
        print(
            f"[bf16_weight_grad] cache len={len(WeightGradStore.cache)} OK, flushing..."
        )

        with patch(
            "paddlefleet.transformer.moe.fp8_utils.deep_gemm"
        ) as mock_dg:
            mock_dg.set_num_sms = lambda n: None
            mock_dg.k_grouped_bf16_gemm_tn_contiguous = (
                self._patched_k_grouped_gemm
            )
            WeightGradStore.flush()
            WeightGradStore.pop()

        self.assertIsNotNone(
            weights.grad, "weights.grad must be set after pop()"
        )
        print(
            "[bf16_weight_grad] weights.grad set after pop(), no_main_grad overlap OK"
        )
        WeightGradStore.clear()


# ============================================================
# Test 13: ExpertsGroupGemmContiguousNode.bf16_weight_grad – per-expert loop
# ============================================================
class TestBf16WeightGradPerExpertLoop(unittest.TestCase):
    """
    Tests bf16_weight_grad with moe_expert_fusion=False so it falls into the
    per-expert loop (else branch at line 1732), covering lines 1733-1759.
    Uses tensors with correct alignment: each expert gets tokens multiple of
    FP8_ALIGN=128.
    """

    def setUp(self):
        paddle.seed(42)

    def _make_node_and_weights(self, use_main_grad=True):
        from types import SimpleNamespace

        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        n_experts = 2  # 2 experts
        H = 256  # hidden size
        I = 256  # intermediate size
        # Each expert gets 128 tokens (multiple of FP8_ALIGN=128)
        tokens_per_expert = [128, 128]
        total = sum(tokens_per_expert)

        # Create per-expert weight list
        per_expert_weights = []
        for i in range(n_experts):
            # Weight shape [H, 2*I] for up_gate_proj (F.linear convention: [in, out])
            w = paddle.create_parameter(
                shape=[H, 2 * I],
                dtype="bfloat16",
                default_initializer=paddle.nn.initializer.Normal(),
            )
            if use_main_grad:
                w.main_grad = None
            elif hasattr(w, "main_grad"):
                del w.main_grad

            hook_count = [0]

            def _hook(hc=hook_count):
                hc[0] += 1
                print(f"[per-expert hook] expert called, count={hc[0]}")

            w._apply_backward_hook = _hook
            w._hook_count = hook_count
            per_expert_weights.append(w)

        custom_map = SimpleNamespace(experts=per_expert_weights)

        node = ExpertsGroupGemmContiguousNode(
            custom_map=custom_map,
            moe_expert_fusion=False,
            moe_deep_gemm=False,
            use_fp8_mlp=False,
            use_bf16_gemm_weight_grad=True,
        )
        node.tokens_per_expert = tokens_per_expert
        node.tokens_per_expert_tensor = paddle.to_tensor(
            tokens_per_expert, dtype="int32"
        )
        node.dequant_input = False
        node.input = None

        # x: [total, H], dy: [total, 2*I]  (inputs to fused_linear_param_grad_add)
        x = paddle.randn([total, H], dtype="bfloat16")
        dy = paddle.randn([total, 2 * I], dtype="bfloat16")
        return node, per_expert_weights, x, dy

    def test_per_expert_loop_with_main_grad(self):
        """
        moe_expert_fusion=False falls into per-expert loop.
        Exercises lines 1733-1759 with main_grad initialization + fused_linear_param_grad_add.
        """
        print("\n[bf16_weight_grad per-expert] with main_grad")
        node, weights, x, dy = self._make_node_and_weights(use_main_grad=True)

        node.bf16_weight_grad(dy, x, weights, defer_dw=False)

        for i, w in enumerate(weights):
            self.assertIsNotNone(
                w.main_grad, f"expert {i} main_grad must be set after loop"
            )
            print(
                f"[bf16_weight_grad per-expert] expert {i} main_grad shape={w.main_grad.shape}"
            )
        print(
            f"[bf16_weight_grad per-expert] main_grad path OK for {len(weights)} experts"
        )

    def test_per_expert_loop_without_main_grad(self):
        """
        moe_expert_fusion=False falls into per-expert loop.
        Exercises lines 1741-1759 with weights.grad initialization + fused_linear_param_grad_add.
        """
        print(
            "\n[bf16_weight_grad per-expert] without main_grad (weights.grad)"
        )
        node, weights, x, dy = self._make_node_and_weights(use_main_grad=False)

        node.bf16_weight_grad(dy, x, weights, defer_dw=False)

        for i, w in enumerate(weights):
            self.assertIsNotNone(
                w.grad, f"expert {i} grad must be set after loop"
            )
            print(
                f"[bf16_weight_grad per-expert] expert {i} grad shape={w.grad.shape}"
            )
        print(
            f"[bf16_weight_grad per-expert] grad path OK for {len(weights)} experts"
        )


# ============================================================
# Test 10b: expert dW deferral preconditions fail fast
# ============================================================
class TestExpertDwDeferralGuard(unittest.TestCase):
    """`_check_expert_dw_deferral_supported` must reject impossible requests.

    Both points need the deep_gemm weight-grad path (the only branch with a
    WeightGradStore call); down_proj additionally needs a non-subbatch
    backward, because the subbatch loop returns out_grad itself as dx and a
    deferred dw2 still reads out_grad. Silently dropping the request used to
    hide a real misconfiguration, so these are hard errors now.
    """

    def _check(
        self,
        up_gate,
        down,
        deep_gemm,
        subbatch,
        use_fp8_mlp=False,
        use_bf16_gemm_weight_grad=True,
    ):
        from paddlefleet.transformer.moe.fp8_utils import (
            _check_expert_dw_deferral_supported,
        )

        _check_expert_dw_deferral_supported(
            up_gate,
            down,
            deep_gemm,
            subbatch,
            use_fp8_mlp,
            use_bf16_gemm_weight_grad,
        )

    def test_nothing_requested_is_always_fine(self):
        """No point selected -> unsupported settings must not raise."""
        self._check(False, False, deep_gemm=False, subbatch=512)
        print("\n[dw guard] no request + unsupported settings OK")

    def test_supported_combination_passes(self):
        self._check(True, True, deep_gemm=True, subbatch=None)
        print("[dw guard] deep_gemm + no subbatch accepts both points OK")

    def test_up_gate_needs_deep_gemm(self):
        with self.assertRaisesRegex(ValueError, "moe_deep_gemm=False"):
            self._check(True, False, deep_gemm=False, subbatch=None)
        print("[dw guard] up_gate without deep_gemm raises OK")

    def test_down_needs_deep_gemm(self):
        with self.assertRaisesRegex(ValueError, "moe_deep_gemm=False"):
            self._check(False, True, deep_gemm=False, subbatch=None)
        print("[dw guard] down without deep_gemm raises OK")

    def test_down_rejects_subbatch(self):
        with self.assertRaisesRegex(
            ValueError, "moe_subbatch_token_num_after_dispatch=512"
        ):
            self._check(False, True, deep_gemm=True, subbatch=512)
        print("[dw guard] down with subbatch raises OK")

    def test_default_fp8_wgrad_rejects_expert_deferral(self):
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        with self.assertRaisesRegex(ValueError, "fp8_wgrad=True"):
            ExpertsGroupGemmContiguousNode(
                SimpleNamespace(experts=[]),
                use_fp8_mlp=True,
                moe_deep_gemm=True,
                defer_expert_up_gate_dw=True,
            )
        print("[dw guard] default native FP8 wgrad rejects deferral OK")

    def test_fp8_mlp_with_bf16_wgrad_allows_expert_deferral(self):
        self._check(
            True,
            True,
            deep_gemm=True,
            subbatch=None,
            use_fp8_mlp=True,
            use_bf16_gemm_weight_grad=True,
        )
        print("[dw guard] FP8 MLP with BF16 wgrad accepts deferral OK")

    def test_down_rejects_subbatch_under_python_optimize(self):
        code = """
from paddlefleet.transformer.moe.fp8_utils import (
    _check_expert_dw_deferral_supported,
)

try:
    _check_expert_dw_deferral_supported(False, True, True, 512, True, True)
except ValueError:
    pass
else:
    raise RuntimeError("unsupported down_proj deferral was accepted")
"""
        result = subprocess.run(
            [sys.executable, "-O", "-c", code],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        print("[dw guard] down with subbatch raises under python -O OK")

    def test_up_gate_allows_subbatch(self):
        """dw1 reads do1, not out_grad: the subbatch alias does not apply."""
        self._check(True, False, deep_gemm=True, subbatch=512)
        print("[dw guard] up_gate with subbatch OK")


# ============================================================
# Test 11: deferrable_linear routes each attention point
# ============================================================
class _PlainLinear(paddle.nn.Layer):
    """Stand-in for a bias-free projection layer returning (out, bias)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = make_weight_with_main_grad([in_dim, out_dim])
        self.bias = None

    def forward(self, x):
        return F.linear(x, self.weight), None


class _SelectorConfig:
    tensor_model_parallel_size = 1
    pipeline_model_parallel_size = 2
    virtual_pipeline_model_parallel_size = 2
    use_bias = False

    def __init__(self, selected):
        self.p2p_overlap_dw_calc = selected


class TestDeferrableLinearRouting(unittest.TestCase):
    """deferrable_linear must defer exactly the selected attention points."""

    ATTENTION_POINTS = (
        "attn_q_proj",
        "attn_kv_proj",
        "attn_out_proj",
        "attn_gate_proj",
        "attn_indexer_q_proj",
        "attn_indexer_k_proj",
        "attn_indexer_weights_proj",
    )

    def setUp(self):
        WeightGradStore.clear()

    def tearDown(self):
        WeightGradStore.clear()

    def _run(self, config, point):
        layer = _PlainLinear(32, 16)
        x = paddle.randn([2, 3, 32])
        x.stop_gradient = False
        out, bias = deferrable_linear(config, point, layer, x)
        out.sum().backward()
        self.assertIsNone(bias)
        self.assertIsNotNone(x.grad, "dx must always be computed eagerly")
        return len(WeightGradStore.cache)

    def test_each_point_is_independently_selectable(self):
        for point in self.ATTENTION_POINTS:
            with self.subTest(point=point):
                WeightGradStore.clear()
                self.assertEqual(
                    self._run(_SelectorConfig([point]), point),
                    1,
                    f"{point} selected must defer its dW",
                )
                WeightGradStore.clear()
                other = next(p for p in self.ATTENTION_POINTS if p != point)
                self.assertEqual(
                    self._run(_SelectorConfig([other]), point),
                    0,
                    f"{point} must stay inline when only {other} is selected",
                )
        print("[deferrable_linear] every attention point is selectable OK")

    def test_disabled_and_tp_fallback(self):
        for selected in (None, []):
            WeightGradStore.clear()
            self.assertEqual(
                self._run(_SelectorConfig(selected), "attn_q_proj"),
                0,
                f"p2p_overlap_dw_calc={selected!r} must disable deferral",
            )
        WeightGradStore.clear()
        config = _SelectorConfig(["attn_q_proj"])
        config.tensor_model_parallel_size = 2
        self.assertEqual(
            self._run(config, "attn_q_proj"),
            0,
            "the stand-in has no collective, so tp>1 must fall back",
        )
        for pp, vpp in ((1, None), (2, None)):
            with self.subTest(pp=pp, vpp=vpp):
                WeightGradStore.clear()
                config = _SelectorConfig(["attn_q_proj"])
                config.pipeline_model_parallel_size = pp
                config.virtual_pipeline_model_parallel_size = vpp
                self.assertEqual(
                    self._run(config, "attn_q_proj"),
                    0,
                    f"PP={pp}, VPP={vpp} must use inline dW",
                )
        print("[deferrable_linear] disabled / tp>1 fallback OK")

    def test_ordinary_pp_and_pp1_preserve_inline_weight_grad(self):
        expected = np.matmul(
            np.ones([6, 32], dtype="float32").T,
            np.ones([6, 16], dtype="float32"),
        )
        for pp, vpp in ((1, None), (2, None)):
            with self.subTest(pp=pp, vpp=vpp):
                WeightGradStore.clear()
                config = _SelectorConfig(["attn_q_proj"])
                config.pipeline_model_parallel_size = pp
                config.virtual_pipeline_model_parallel_size = vpp
                layer = _PlainLinear(32, 16)
                x = paddle.ones([2, 3, 32], dtype="float32")
                x.stop_gradient = False
                out, _ = deferrable_linear(config, "attn_q_proj", layer, x)
                out.backward(paddle.ones_like(out))
                self.assertEqual(len(WeightGradStore.cache), 0)
                np.testing.assert_allclose(
                    (
                        layer.weight.main_grad
                        if layer.weight.main_grad is not None
                        else layer.weight.grad
                    ).numpy(),
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                )


# ============================================================
# Test 12: the deferral helpers must be imported where they are used
# ============================================================
class TestDeferralHelpersAreImported(unittest.TestCase):
    """Guard against NameError at layer-construction time.

    `import module` does not execute a class body's __init__, so a module that
    calls dw_overlap_enabled / deferrable_linear without importing it imports
    fine and only blows up when the model is built. That happened once
    (MoELayer.__init__). Check the binding is reachable in each namespace.
    """

    EXPECTED = {
        "paddlefleet.transformer.moe.moe_layer": (
            "dw_overlap_enabled",
            "install_sonic_moe_dw_deferral",
        ),
        "paddlefleet.transformer.moe.moe_router": ("dw_overlap_enabled",),
        "paddlefleet.transformer.multi_latent_attention": (
            "deferrable_linear",
        ),
        "paddlefleet.transformer.dsv4_hybrid_attention": ("deferrable_linear",),
        # The two sparse-attention indexers; both feed attn_indexer_* points.
        "paddlefleet.transformer.dsa_attention": ("deferrable_linear",),
        "paddlefleet.transformer.csa_attention": ("deferrable_linear",),
    }

    def test_helpers_resolvable_in_each_module(self):
        import importlib

        for mod_name, attrs in self.EXPECTED.items():
            mod = importlib.import_module(mod_name)
            for attr in attrs:
                with self.subTest(module=mod_name, attr=attr):
                    self.assertTrue(
                        hasattr(mod, attr),
                        f"{mod_name} calls {attr} but does not import it; "
                        "layer construction would raise NameError",
                    )
        print("[imports] deferral helpers resolvable in every caller OK")

    def test_no_watched_name_used_without_binding(self):
        """Static sweep: every touched file must bind the names it loads."""
        import ast
        import builtins
        import pathlib

        import paddlefleet

        root = pathlib.Path(paddlefleet.__file__).parent
        watched = {
            "dw_overlap_enabled",
            "deferrable_linear",
            "DeferredWeightGradLinear",
            "P2P_OVERLAP_DW_CALC_CHOICES",
            "install_sonic_moe_dw_deferral",
            "install_recompute_p2p_overlap",
            "RecomputeStore",
        }
        offenders = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            bound = set(dir(builtins))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    bound |= {a.asname or a.name for a in node.names}
                elif isinstance(node, ast.Import):
                    bound |= {
                        (a.asname or a.name).split(".")[0] for a in node.names
                    }
                elif isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    bound.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(
                    node.ctx, ast.Store
                ):
                    bound.add(node.id)
                elif isinstance(node, ast.arg):
                    bound.add(node.arg)
            loaded = {
                n.id
                for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            missing = (loaded & watched) - bound
            if missing:
                offenders.append(f"{path.relative_to(root)}: {sorted(missing)}")
        self.assertEqual(
            offenders, [], "used without import: " + str(offenders)
        )
        print("[imports] static sweep clean")


# ============================================================
# Test 13: grouped matmul (linear_o_group_proj) deferred dW
# ============================================================
class TestDeferredGroupedDW(unittest.TestCase):
    """fused_grouped_matmul's dW must be deferrable and numerically identical.

    Unlike the plain linears this one goes through a Triton grouped GEMM whose
    dW is a separate kernel, and its weight is a 2-D parameter viewed as
    [G, R, D]. The deferred path therefore takes the *leaf* parameter and writes
    main_grad itself.
    """

    G, R, D, M = 4, 32, 64, 128

    @staticmethod
    def _f32(t):
        # bf16 .numpy() hands back raw bits; cast inside paddle first.
        return t.astype("float32").numpy()

    def _build(self, x_np, w_np):
        x = paddle.to_tensor(x_np).cast("bfloat16")
        x.stop_gradient = False
        param = paddle.create_parameter(
            [self.G * self.R, self.D],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )
        param.set_value(paddle.to_tensor(w_np).cast("bfloat16"))
        param.main_grad = None
        return x, param

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_deferred_matches_inline(self):
        from paddlefleet.transformer.dw_overlap import (
            deferred_grouped_dw_accumulator,
        )
        from paddlefleet.triton_ops import fused_grouped_matmul

        G, R, D, M = self.G, self.R, self.D, self.M
        np.random.seed(0)
        x_np = np.random.randn(2, M // 2, G, D).astype("float32")
        w_np = np.random.randn(G * R, D).astype("float32")
        group_shape = [G, R, D]

        x1, p1 = self._build(x_np, w_np)
        out1 = fused_grouped_matmul(x1, p1.reshape(group_shape))
        out1.sum().backward()
        ref_dx, ref_dw = self._f32(x1.grad), self._f32(p1.grad)

        WeightGradStore.clear()
        config = _SelectorConfig(["attn_o_group_proj"])
        x2, p2 = self._build(x_np, w_np)
        acc = deferred_grouped_dw_accumulator(config, "attn_o_group_proj", p2)
        self.assertIsNotNone(acc, "selected point must produce an accumulator")
        out2 = fused_grouped_matmul(
            x2, p2, dw_accumulator=acc, group_shape=group_shape
        )
        np.testing.assert_allclose(self._f32(out2), self._f32(out1), rtol=1e-6)
        out2.sum().backward()

        self.assertIsNone(p2.main_grad, "dW must not run before pop()")
        self.assertEqual(len(WeightGradStore.cache), 1)
        np.testing.assert_allclose(
            self._f32(x2.grad), ref_dx, rtol=1e-2, atol=1e-2
        )

        WeightGradStore.flush()
        WeightGradStore.pop()
        self.assertEqual(
            list(p2.main_grad.shape),
            [G * R, D],
            "main_grad must match the 2-D parameter, not the [G,R,D] view",
        )
        np.testing.assert_allclose(
            p2.main_grad.numpy(), ref_dw, rtol=3e-2, atol=3e-2
        )
        WeightGradStore.clear()
        print("[grouped dW] deferred result matches inline OK")

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_not_selected_stays_inline(self):
        from paddlefleet.transformer.dw_overlap import (
            deferred_grouped_dw_accumulator,
        )
        from paddlefleet.triton_ops import fused_grouped_matmul

        G, R, D, M = self.G, self.R, self.D, self.M
        np.random.seed(0)
        x_np = np.random.randn(2, M // 2, G, D).astype("float32")
        w_np = np.random.randn(G * R, D).astype("float32")

        config = _SelectorConfig(["attn_out_proj"])  # a different point
        self.assertIsNone(
            deferred_grouped_dw_accumulator(config, "attn_o_group_proj", None),
            "unselected point must not produce an accumulator",
        )
        WeightGradStore.clear()
        x, p = self._build(x_np, w_np)
        out = fused_grouped_matmul(x, p.reshape([G, R, D]))
        out.sum().backward()
        self.assertEqual(len(WeightGradStore.cache), 0, "nothing may be queued")
        self.assertIsNotNone(p.grad, "inline path must fill .grad")
        print("[grouped dW] unselected stays inline OK")


# ============================================================
# Test 14: SonicMoE expert wgrad deferral hook
# ============================================================
class _RecordingWeight:
    """Weight stand-in that records when sharding's grad-ready hook fires."""

    stop_gradient = False

    def __init__(self):
        self.hook_calls = 0

    def _apply_backward_hook(self):
        self.hook_calls += 1


def _sonicmoe_supported():
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        major, _ = paddle.device.cuda.get_device_capability()
        if major < 10:
            return False
    except (AttributeError, RuntimeError):
        return False
    try:
        import paddlefleet_ops.sonicmoe.functional  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    return True


@unittest.skipUnless(
    _sonicmoe_supported(), "SonicMoE is unavailable for this GPU"
)
class TestSonicMoeWgradDeferral(unittest.TestCase):
    """The hook must queue the selected point's wgrad and nothing else.

    SonicMoE runs the dW GEMM itself; we only take it over. The order that
    matters is: the GEMM runs on pop, and the sharding grad-ready hook fires
    only after it -- firing earlier would reduce a partial main_grad.
    """

    POINTS = (
        "moe_sonic_expert_up_gate_proj",
        "moe_sonic_expert_down_proj",
    )

    def setUp(self):
        WeightGradStore.clear()

    def tearDown(self):
        WeightGradStore.clear()
        from paddlefleet_ops.sonicmoe.functional import (
            set_wgrad_deferral_hook,
        )

        set_wgrad_deferral_hook(None)

    def _hook_for(self, selected):
        from paddlefleet_ops.sonicmoe import functional as sonic_functional

        install_sonic_moe_dw_deferral(_SelectorConfig(selected))
        return sonic_functional._WGRAD_DEFERRAL_HOOK

    def test_sonicmoe_exposes_the_setter(self):
        from paddlefleet_ops.sonicmoe.functional import (
            set_wgrad_deferral_hook,
        )

        self.assertTrue(callable(set_wgrad_deferral_hook))
        print("[sonic dW] set_wgrad_deferral_hook present OK")

    def test_nothing_installed_when_unselected(self):
        self.assertIsNone(
            self._hook_for(["attn_q_proj"]),
            "no sonic point selected must leave the hook uninstalled",
        )
        print("[sonic dW] unselected installs nothing OK")

    def test_nothing_installed_for_ordinary_pp_or_pp1(self):
        from paddlefleet_ops.sonicmoe import functional as sonic_functional

        for pp, vpp in ((1, None), (2, None)):
            with self.subTest(pp=pp, vpp=vpp):
                config = _SelectorConfig(["moe_sonic_expert_up_gate_proj"])
                config.pipeline_model_parallel_size = pp
                config.virtual_pipeline_model_parallel_size = vpp
                install_sonic_moe_dw_deferral(config)
                self.assertIsNone(
                    sonic_functional._WGRAD_DEFERRAL_HOOK,
                    f"PP={pp}, VPP={vpp} must keep SonicMoE dW inline",
                )
        print("[sonic dW] ordinary PP and PP=1 stay inline OK")

    def test_selected_point_defers_until_pop(self):
        for point in self.POINTS:
            with self.subTest(point=point):
                WeightGradStore.clear()
                hook = self._hook_for([point])
                self.assertIsNotNone(hook)

                weight = _RecordingWeight()
                ran = []
                taken = hook(point, weight, lambda: ran.append(point))

                self.assertTrue(taken, "hook must claim the selected point")
                self.assertEqual(ran, [], "the GEMM must not run inline")
                self.assertEqual(
                    weight.hook_calls, 0, "grad is not complete yet"
                )
                self.assertEqual(len(WeightGradStore.cache), 1)

                WeightGradStore.flush()
                WeightGradStore.pop()
                self.assertEqual(ran, [point], "pop must run the GEMM")
                self.assertEqual(
                    weight.hook_calls,
                    1,
                    "grad-ready hook must fire after the GEMM, exactly once",
                )
        print("[sonic dW] selected point defers to pop OK")

    def test_unselected_point_stays_inline(self):
        # Only w1 selected: the hook is installed, but it must refuse w2 so
        # SonicMoE runs that GEMM inline.
        hook = self._hook_for(["moe_sonic_expert_up_gate_proj"])
        weight = _RecordingWeight()
        ran = []
        taken = hook(
            "moe_sonic_expert_down_proj", weight, lambda: ran.append("w2")
        )
        self.assertFalse(taken, "unselected point must be declined")
        self.assertEqual(len(WeightGradStore.cache), 0, "nothing may be queued")
        self.assertEqual(ran, [], "declining must not run the GEMM either")
        print("[sonic dW] unselected point declined OK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
