# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
Unit tests for TeraMoE integration in MoELayer.

Test structure:
  - TestTeraMoEConfig: Config field validation (no GPU/fleet needed)
  - TestTeraMoEExpertWeightLayout: Weight layout conversion correctness
  - TestTeraMoEExpertForward: Forward call with mock buffer
  - TestTeraMoEMoELayerInstantiation: MoELayer creates TeraMoEExpert
  - TestTeraMoEMoELayerForward: Full forward/backward with mock buffer
  - TestTeraMoEVsSonicMoEWeightEquivalence: Weight parity with SonicMoE

Run with: python -m paddle.distributed.launch --gpus=0 <this_file>
"""

import unittest
from unittest.mock import MagicMock, patch

import paddle
import paddle.nn.functional as F
import paddlefleet_ops
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils

import paddlefleet.parallel_state as ps
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.moe.moe_expert import SonicMoEExpert, TeraMoEExpert
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig


def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
    """Cosine-distance-based diff metric (same as SonicMoE tests)."""
    x, y = x.astype("float64"), y.astype("float64")
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


# ── Module-level fleet initialization (single card, via launch) ──────
_strategy = fleet.DistributedStrategy()
_strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": 1,
    "pp_degree": 1,
    "sharding_degree": 1,
    "sep_degree": 1,
    "cp_degree": 1,
    "ep_degree": 1,
    "moe_sharding_degree": 1,
    "order": [
        "sharding",
        "moe_sharding",
        "pp",
        "sep",
        "cp",
        "dp",
        "ep",
        "mp",
    ],
}
fleet.init(is_collective=True, strategy=_strategy)
_hcg = fleet.get_hybrid_communicate_group()
ps.initialize_model_parallel(_hcg)


# ═══════════════════════════════════════════════════════════════════════
# Test 1: TransformerConfig
# ═══════════════════════════════════════════════════════════════════════


@unittest.skipUnless(
    paddlefleet_ops.is_teramoe_available(),
    "TeraMoE not available",
)
class TestTeraMoEConfig(unittest.TestCase):
    """TransformerConfig correctly stores TeraMoE parameters."""

    def test_default_config_teramoe_disabled(self):
        cfg = TransformerConfig(hidden_size=256, num_attention_heads=8)
        self.assertFalse(cfg.using_teramoe)

    def test_config_teramoe_enabled_with_custom_values(self):
        cfg = TransformerConfig(
            hidden_size=256,
            num_attention_heads=8,
            using_teramoe=True,
            teramoe_dispatch_sms=32,
            teramoe_combine_sms=32,
            teramoe_compute_batch_size=2048,
            teramoe_combine_start_percent=60,
        )
        self.assertTrue(cfg.using_teramoe)
        self.assertEqual(cfg.teramoe_dispatch_sms, 32)
        self.assertEqual(cfg.teramoe_combine_sms, 32)
        self.assertEqual(cfg.teramoe_compute_batch_size, 2048)
        self.assertEqual(cfg.teramoe_combine_start_percent, 60)

    def test_config_teramoe_defaults(self):
        cfg = TransformerConfig(
            hidden_size=256,
            num_attention_heads=8,
            using_teramoe=True,
        )
        self.assertEqual(cfg.teramoe_dispatch_sms, 48)
        self.assertEqual(cfg.teramoe_combine_sms, 48)
        self.assertEqual(cfg.teramoe_compute_batch_size, 4096)
        self.assertEqual(cfg.teramoe_combine_start_percent, 70)


# ═══════════════════════════════════════════════════════════════════════
# Test 2: TeraMoEExpert weight layout conversion
# ═══════════════════════════════════════════════════════════════════════


@unittest.skipUnless(
    paddlefleet_ops.is_teramoe_available(),
    "TeraMoE not available",
)
class TestTeraMoEExpertWeightLayout(unittest.TestCase):
    """TeraMoEExpert weight layout round-trip correctness."""

    def setUp(self):
        self.H, self.I, self.E, self.K = 256, 512, 4, 2
        self.cfg = TransformerConfig(
            hidden_size=self.H,
            num_attention_heads=8,
            intermediate_size=self.I,
            moe_intermediate_size=self.I,
            n_routed_experts=self.E,
            num_experts_per_tok=self.K,
            using_teramoe=True,
            gated_linear_unit=True,
            use_bias=False,
            tensor_model_parallel_size=1,
            params_dtype=paddle.bfloat16,
        )

    def test_inheritance(self):
        self.assertTrue(issubclass(TeraMoEExpert, SonicMoEExpert))

    def test_initial_grouped_layout(self):
        expert = TeraMoEExpert(self.E, self.K, self.cfg)
        self.assertEqual(expert._weights_layout, "grouped")
        self.assertEqual(
            list(expert.weight1.shape), [self.E, self.H, 2 * self.I]
        )
        self.assertEqual(list(expert.weight2.shape), [self.E, self.I, self.H])

    def test_sonic_layout_shapes(self):
        expert = TeraMoEExpert(self.E, self.K, self.cfg)
        expert.convert_weights_to_sonic_layout()
        self.assertEqual(expert._weights_layout, "sonic")
        self.assertEqual(
            list(expert.weight1.shape), [self.E, 2 * self.I, self.H]
        )
        self.assertEqual(list(expert.weight2.shape), [self.E, self.H, self.I])

    def test_round_trip(self):
        expert = TeraMoEExpert(self.E, self.K, self.cfg)
        w1_orig = expert.weight1.clone()
        w2_orig = expert.weight2.clone()
        expert.convert_weights_to_sonic_layout()
        expert.flush_to_grouped_layout()
        self.assertLess(calc_diff(expert.weight1, w1_orig), 1e-6)
        self.assertLess(calc_diff(expert.weight2, w2_orig), 1e-6)

    def test_idempotent_convert(self):
        expert = TeraMoEExpert(self.E, self.K, self.cfg)
        expert.convert_weights_to_sonic_layout()
        w1 = expert.weight1.clone()
        expert.convert_weights_to_sonic_layout()
        self.assertLess(calc_diff(expert.weight1, w1), 1e-10)


# ═══════════════════════════════════════════════════════════════════════
# Test 3: TeraMoEExpert forward with mock buffer
# ═══════════════════════════════════════════════════════════════════════


@unittest.skipUnless(
    paddlefleet_ops.is_teramoe_available(),
    "TeraMoE not available",
)
class TestTeraMoEExpertForward(unittest.TestCase):
    """TeraMoEExpert.forward() converts layout and calls buffer.teramoe_autograd."""

    def setUp(self):
        self.H, self.I, self.E, self.K, self.B = 256, 512, 4, 2, 16
        self.cfg = TransformerConfig(
            hidden_size=self.H,
            num_attention_heads=8,
            intermediate_size=self.I,
            moe_intermediate_size=self.I,
            n_routed_experts=self.E,
            num_experts_per_tok=self.K,
            using_teramoe=True,
            gated_linear_unit=True,
            use_bias=False,
            tensor_model_parallel_size=1,
            params_dtype=paddle.bfloat16,
        )
        self.expert = TeraMoEExpert(self.E, self.K, self.cfg)

    def test_forward_calls_buffer(self):
        mock_buffer = MagicMock()
        fake_out = paddle.randn([self.B, self.H], dtype=paddle.bfloat16)
        mock_buffer.teramoe_autograd.return_value = fake_out

        x = paddle.randn([self.B, self.H], dtype=paddle.bfloat16)
        idx = paddle.randint(0, self.E, [self.B, self.K])
        scores = F.softmax(paddle.randn([self.B, self.K]), axis=-1)

        out = self.expert(
            x,
            idx,
            scores,
            self.E,
            mock_buffer,
            num_dispatch_sms=48,
            num_combine_sms=48,
        )

        mock_buffer.teramoe_autograd.assert_called_once()
        args = mock_buffer.teramoe_autograd.call_args[0]
        self.assertIs(args[0], x)
        self.assertIs(args[1], idx)
        self.assertIs(args[2], scores)
        # sonic layout shapes
        self.assertEqual(list(args[3].shape), [self.E, 2 * self.I, self.H])
        self.assertEqual(list(args[4].shape), [self.E, self.H, self.I])
        self.assertEqual(args[5], self.E)
        # kwargs
        self.assertEqual(
            mock_buffer.teramoe_autograd.call_args[1]["num_dispatch_sms"], 48
        )
        self.assertTrue(
            paddle.equal_all(out.astype("float32"), fake_out.astype("float32"))
        )
        self.assertEqual(self.expert._weights_layout, "sonic")

    def test_weight_ptr_preserved(self):
        mock_buffer = MagicMock()
        mock_buffer.teramoe_autograd.return_value = paddle.randn(
            [self.B, self.H], dtype=paddle.bfloat16
        )
        w1_ptr = self.expert.weight1.data_ptr()
        w2_ptr = self.expert.weight2.data_ptr()

        x = paddle.randn([self.B, self.H], dtype=paddle.bfloat16)
        idx = paddle.randint(0, self.E, [self.B, self.K])
        scores = F.softmax(paddle.randn([self.B, self.K]), axis=-1)
        self.expert(x, idx, scores, self.E, mock_buffer)

        self.assertEqual(self.expert.weight1.data_ptr(), w1_ptr)
        self.assertEqual(self.expert.weight2.data_ptr(), w2_ptr)

    def test_forward_3d_input_reshape(self):
        """3D input [B, S, H] is flattened to 2D before the kernel and the
        output is reshaped back to 3D (covers the ndim==3 branch)."""
        B, S = 2, 8
        mock_buffer = MagicMock()
        fake_out = paddle.randn([B * S, self.H], dtype=paddle.bfloat16)
        mock_buffer.teramoe_autograd.return_value = fake_out

        x = paddle.randn([B, S, self.H], dtype=paddle.bfloat16)
        idx = paddle.randint(0, self.E, [B * S, self.K])
        scores = F.softmax(paddle.randn([B * S, self.K]), axis=-1)

        out = self.expert(x, idx, scores, self.E, mock_buffer)

        # hidden_states handed to the kernel is the flattened 2D view.
        called_hs = mock_buffer.teramoe_autograd.call_args[0][0]
        self.assertEqual(list(called_hs.shape), [B * S, self.H])
        # output is reshaped back to the original 3D layout.
        self.assertEqual(list(out.shape), [B, S, self.H])


# ═══════════════════════════════════════════════════════════════════════
# Test 4: MoELayer instantiation with TeraMoE
# ═══════════════════════════════════════════════════════════════════════


@unittest.skipUnless(
    paddlefleet_ops.is_teramoe_available(),
    "TeraMoE not available",
)
class TestTeraMoEMoELayerInstantiation(unittest.TestCase):
    """MoELayer correctly instantiates TeraMoEExpert."""

    def setUp(self):
        self.seed = 42
        self.H = 512
        self.E = 8
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    @staticmethod
    def _init(tensor):
        paddle.nn.initializer.Uniform(-0.001, 0.001)(tensor)

    def _build(self, using_teramoe=False):
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        cfg = TransformerConfig(
            hidden_size=self.H,
            num_attention_heads=4,
            n_routed_experts=self.E,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=1024,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            moe_expert_fusion=True,
            bias_activation_fusion=True,
            moe_token_dispatcher_type="alltoall",
            moe_use_fusion_node=True,
            using_teramoe=using_teramoe,
            using_sonic_moe=False,
            fp8=None,
            use_bias=False,
            init_method=self._init,
            output_layer_init_method=self._init,
        )
        spec = get_gpt_layer_local_spec(cfg, num_experts=self.E)
        return MoELayer(
            cfg,
            spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            self.pg_collection,
        )

    def test_expert_type_teramoe(self):
        layer = self._build(using_teramoe=True)
        self.assertIsInstance(layer.grouped_gemm_experts, TeraMoEExpert)
        self.assertTrue(layer.using_teramoe)

    def test_config_propagation(self):
        layer = self._build(using_teramoe=True)
        self.assertEqual(layer.teramoe_dispatch_sms, 48)
        self.assertEqual(layer.teramoe_combine_sms, 48)
        self.assertEqual(layer.teramoe_compute_batch_size, 4096)
        self.assertEqual(layer.teramoe_combine_start_percent, 70)

    def test_baseline_not_teramoe(self):
        layer = self._build(using_teramoe=False)
        self.assertNotIsInstance(layer.grouped_gemm_experts, TeraMoEExpert)
        self.assertFalse(layer.using_teramoe)

    def test_weight_shapes(self):
        layer = self._build(using_teramoe=True)
        e = layer.grouped_gemm_experts
        self.assertEqual(list(e.weight1.shape), [self.E, self.H, 2048])
        self.assertEqual(list(e.weight2.shape), [self.E, 1024, self.H])


# ═══════════════════════════════════════════════════════════════════════
# Test 5: MoELayer forward/backward with mock buffer
# ═══════════════════════════════════════════════════════════════════════


@unittest.skipUnless(
    paddlefleet_ops.is_teramoe_available(),
    "TeraMoE not available",
)
class TestTeraMoEMoELayerForward(unittest.TestCase):
    """MoELayer forward routes through TeraMoE path (mock buffer)."""

    def setUp(self):
        self.seed = 42
        self.H, self.E, self.K = 512, 8, 2
        self.B, self.S = 2, 64
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    @staticmethod
    def _init(tensor):
        paddle.nn.initializer.Uniform(-0.001, 0.001)(tensor)

    def _build(self):
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        cfg = TransformerConfig(
            hidden_size=self.H,
            num_attention_heads=4,
            n_routed_experts=self.E,
            use_cpu_initialization=False,
            num_experts_per_tok=self.K,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=1024,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            moe_expert_fusion=True,
            bias_activation_fusion=True,
            moe_token_dispatcher_type="alltoall",
            moe_use_fusion_node=True,
            using_teramoe=True,
            using_sonic_moe=False,
            fp8=None,
            use_bias=False,
            init_method=self._init,
            output_layer_init_method=self._init,
        )
        spec = get_gpt_layer_local_spec(cfg, num_experts=self.E)
        layer = MoELayer(
            cfg,
            spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            self.pg_collection,
        )
        mix_precision_utils.MixPrecisionLayer(layer, dtype="bfloat16")
        for p in layer.parameters():
            if hasattr(p, "main_grad") and p.main_grad is None:
                p.main_grad = paddle.zeros_like(p, dtype=paddle.float32)
        return layer

    def _mock_buffer(self):
        buf = MagicMock()

        def fake_autograd(x, idx, scores, w1, w2, ne, **kw):
            return x * 1.0  # identity pass-through

        buf.teramoe_autograd.side_effect = fake_autograd
        return buf

    @patch("paddlefleet.transformer.moe.moe_layer.get_teramoe_buffer")
    def test_forward_calls_buffer(self, mock_get):
        layer = self._build()
        buf = self._mock_buffer()
        mock_get.return_value = buf

        x = paddle.randn([self.B, self.S, self.H], dtype=paddle.bfloat16)
        x.stop_gradient = False
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            out, _ = layer(x)

        mock_get.assert_called_once()
        buf.teramoe_autograd.assert_called_once()
        self.assertEqual(list(out.shape), [self.B, self.S, self.H])

    @patch("paddlefleet.transformer.moe.moe_layer.get_teramoe_buffer")
    def test_output_no_nan(self, mock_get):
        layer = self._build()
        mock_get.return_value = self._mock_buffer()
        x = paddle.randn([self.B, self.S, self.H], dtype=paddle.bfloat16)
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            out, _ = layer(x)
        self.assertFalse(paddle.isnan(out).any().item())

    @patch("paddlefleet.transformer.moe.moe_layer.get_teramoe_buffer")
    def test_backward_produces_grad(self, mock_get):
        layer = self._build()
        mock_get.return_value = self._mock_buffer()
        x = paddle.randn([self.B, self.S, self.H], dtype=paddle.bfloat16)
        x.stop_gradient = False
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            out, _ = layer(x)
            loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(paddle.isnan(x.grad).any().item())

    @patch("paddlefleet.transformer.moe.moe_layer.get_teramoe_buffer")
    def test_buffer_kwargs(self, mock_get):
        layer = self._build()
        buf = self._mock_buffer()
        mock_get.return_value = buf
        x = paddle.randn([self.B, self.S, self.H], dtype=paddle.bfloat16)
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            layer(x)
        kw = buf.teramoe_autograd.call_args[1]
        self.assertEqual(kw["num_dispatch_sms"], 48)
        self.assertEqual(kw["num_combine_sms"], 48)
        self.assertEqual(kw["compute_batch_size"], 4096)
        self.assertEqual(kw["combine_start_head_percent"], 70)

    @patch("paddlefleet.transformer.moe.moe_layer.get_teramoe_buffer")
    def test_teramoe_forward_method(self, mock_get):
        """teramoe_forward (the expert-parallel routing path) runs
        _project_to_latent + _run_teramoe_forward end to end. forward() with
        ep=1 never reaches it, so exercise it directly."""
        layer = self._build()
        buf = self._mock_buffer()
        mock_get.return_value = buf

        T = self.B * self.S
        x = paddle.randn([T, self.H], dtype=paddle.bfloat16)
        topk_w = F.softmax(paddle.randn([T, self.K]), axis=-1).astype(
            paddle.bfloat16
        )
        topk_i = paddle.randint(0, self.E, [T, self.K])

        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            out = layer.teramoe_forward(
                x, topk_weights=topk_w, topk_indices=topk_i
            )

        mock_get.assert_called_once()
        buf.teramoe_autograd.assert_called_once()
        self.assertEqual(list(out.shape), [T, self.H])

    @patch("paddlefleet.transformer.moe.moe_layer.get_teramoe_buffer")
    def test_run_teramoe_forward_passes_config_kwargs(self, mock_get):
        """_run_teramoe_forward fetches the buffer for the moe group and
        forwards the four teramoe tuning knobs to the expert."""
        layer = self._build()
        buf = self._mock_buffer()
        mock_get.return_value = buf

        T = self.B * self.S
        x = paddle.randn([T, self.H], dtype=paddle.bfloat16)
        topk_w = F.softmax(paddle.randn([T, self.K]), axis=-1).astype(
            paddle.bfloat16
        )
        topk_i = paddle.randint(0, self.E, [T, self.K])

        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            layer._run_teramoe_forward(x, topk_w, topk_i)

        mock_get.assert_called_once_with(
            layer.moe_group, layer.teramoe_dispatch_sms
        )
        kw = buf.teramoe_autograd.call_args[1]
        self.assertEqual(kw["num_dispatch_sms"], layer.teramoe_dispatch_sms)
        self.assertEqual(kw["num_combine_sms"], layer.teramoe_combine_sms)
        self.assertEqual(
            kw["compute_batch_size"], layer.teramoe_compute_batch_size
        )
        self.assertEqual(
            kw["combine_start_head_percent"],
            layer.teramoe_combine_start_percent,
        )


# ═══════════════════════════════════════════════════════════════════════
# Test 6: TeraMoE vs SonicMoE weight equivalence
# ═══════════════════════════════════════════════════════════════════════


@unittest.skipUnless(
    paddlefleet_ops.is_teramoe_available(),
    "TeraMoE not available",
)
class TestTeraMoEVsSonicMoEWeightEquivalence(unittest.TestCase):
    """TeraMoEExpert and SonicMoEExpert produce identical sonic-layout weights."""

    def setUp(self):
        self.H, self.I, self.E, self.K = 512, 1024, 8, 2
        self.seed = 123

    def _cfg(self, **kw):
        return TransformerConfig(
            hidden_size=self.H,
            num_attention_heads=4,
            intermediate_size=self.I,
            moe_intermediate_size=self.I,
            n_routed_experts=self.E,
            num_experts_per_tok=self.K,
            gated_linear_unit=True,
            use_bias=False,
            tensor_model_parallel_size=1,
            params_dtype=paddle.bfloat16,
            **kw,
        )

    def test_grouped_weights_identical(self):
        paddle.seed(self.seed)
        tera = TeraMoEExpert(self.E, self.K, self._cfg(using_teramoe=True))
        paddle.seed(self.seed)
        sonic = SonicMoEExpert(self.E, self.K, self._cfg(using_sonic_moe=True))
        self.assertLess(calc_diff(tera.weight1, sonic.weight1), 1e-10)
        self.assertLess(calc_diff(tera.weight2, sonic.weight2), 1e-10)

    def test_sonic_layout_weights_identical(self):
        paddle.seed(self.seed)
        tera = TeraMoEExpert(self.E, self.K, self._cfg(using_teramoe=True))
        paddle.seed(self.seed)
        sonic = SonicMoEExpert(self.E, self.K, self._cfg(using_sonic_moe=True))
        tera.convert_weights_to_sonic_layout()
        sonic.convert_weights_to_sonic_layout()
        self.assertLess(calc_diff(tera.weight1, sonic.weight1), 1e-10)
        self.assertLess(calc_diff(tera.weight2, sonic.weight2), 1e-10)


# ═══════════════════════════════════════════════════════════════════════
# Test 7: get_teramoe_buffer caching in fused_a2a
# ═══════════════════════════════════════════════════════════════════════


@unittest.skipUnless(
    paddlefleet_ops.is_teramoe_available(),
    "TeraMoE not available",
)
class TestGetTeraMoEBuffer(unittest.TestCase):
    """get_teramoe_buffer creates one Buffer per process group and reuses it."""

    def test_buffer_cached_and_rebuilt_per_group(self):
        from paddlefleet.transformer.moe import fused_a2a

        def make_buffer(group, **kwargs):
            buf = MagicMock()
            buf.group = group
            return buf

        group_a = object()
        group_b = object()
        saved = fused_a2a._teramoe_buffer
        fused_a2a._teramoe_buffer = None
        try:
            with patch.object(fused_a2a, "teramoe") as mock_tera:
                mock_tera.Buffer.side_effect = make_buffer

                b1 = fused_a2a.get_teramoe_buffer(group_a, 48)
                b2 = fused_a2a.get_teramoe_buffer(group_a, 48)
                # Same group -> buffer reused, Buffer constructed once.
                self.assertIs(b1, b2)
                self.assertEqual(mock_tera.Buffer.call_count, 1)

                b3 = fused_a2a.get_teramoe_buffer(group_b, 48)
                # Different group -> a fresh buffer is built.
                self.assertIsNot(b3, b1)
                self.assertEqual(mock_tera.Buffer.call_count, 2)
        finally:
            fused_a2a._teramoe_buffer = saved


if __name__ == "__main__":
    unittest.main()
