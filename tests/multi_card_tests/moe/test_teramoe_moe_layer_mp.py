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
"""Multi-card (EP>1) TeraMoE tests.

TeraMoE fuses dispatch+compute+combine into one NVSHMEM-backed persistent
kernel that needs a real expert-parallel process group, so it cannot run --
nor be meaningfully mocked -- on a single card (EP<=1 is rejected at
construction). These tests build a real ep=2 MoELayer and exercise the actual
TeraMoE forward/backward path (no mocked buffer).

Run with:
  python -m paddle.distributed.launch --gpus=0,1 \
      tests/multi_card_tests/moe/test_teramoe_moe_layer_mp.py
"""

import unittest

import paddle
import paddle.nn.functional as F
import paddlefleet_ops
from paddle.distributed import fleet

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe.moe_expert import TeraMoEExpert
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig

_fleet_initialised = False
_pg_collection = None


def _ensure_fleet():
    global _fleet_initialised, _pg_collection
    if _fleet_initialised:
        return _pg_collection
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 2,
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
    initialize_fleet(strategy=strategy)
    _pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    _fleet_initialised = True
    return _pg_collection


def _init_weight(tensor):
    paddle.nn.initializer.Uniform(-0.001, 0.001)(tensor)


@unittest.skipUnless(
    paddlefleet_ops.is_teramoe_available(),
    "TeraMoE not available",
)
class TestTeraMoEMoELayerEP(unittest.TestCase):
    """Real ep=2 TeraMoE MoELayer: construction + forward/backward.

    Uses the actual TeraMoE Buffer (no mock), so it runs only where TeraMoE is
    available (Blackwell SM100+, CUDA>=12.9, Python>=3.12) with >=2 GPUs.
    """

    SEED = 42
    H, E, K = 512, 8, 2
    B, S = 2, 64
    MOE_I = 1024

    def _build(self):
        pg = _ensure_fleet()
        paddle.seed(self.SEED)
        model_parallel_cuda_manual_seed(self.SEED)
        cfg = TransformerConfig(
            hidden_size=self.H,
            num_attention_heads=4,
            n_routed_experts=self.E,
            use_cpu_initialization=False,
            num_experts_per_tok=self.K,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=2,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=self.MOE_I,
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
            init_method=_init_weight,
            output_layer_init_method=_init_weight,
        )
        spec = get_gpt_layer_local_spec(cfg, num_experts=self.E)
        return MoELayer(
            cfg,
            spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            pg,
        )

    def test_expert_type_and_ep(self):
        layer = self._build()
        self.assertTrue(layer.using_teramoe)
        self.assertIsInstance(layer.grouped_gemm_experts, TeraMoEExpert)
        self.assertGreater(layer.expert_model_parallel_size, 1)

    def test_config_propagation(self):
        layer = self._build()
        self.assertEqual(layer.teramoe_dispatch_sms, 48)
        self.assertEqual(layer.teramoe_combine_sms, 48)
        self.assertEqual(layer.teramoe_compute_batch_size, 4096)
        self.assertEqual(layer.teramoe_combine_start_percent, 70)

    def test_forward_backward(self):
        layer = self._build()
        x = paddle.randn([self.B, self.S, self.H], dtype=paddle.bfloat16)
        x.stop_gradient = False
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            out, _ = layer(x)
            loss = out.sum()
        loss.backward()
        self.assertEqual(list(out.shape), [self.B, self.S, self.H])
        self.assertFalse(paddle.isnan(out).any().item())
        self.assertIsNotNone(x.grad)

    def test_mini_training_loop_decreases_loss(self):
        """A few optimizer steps on fixed data drive the MSE loss down,
        exercising the real TeraMoE fwd+bwd+weight-update path (migrated from
        the former single-card mock-buffer e2e script)."""
        layer = self._build()
        optimizer = paddle.optimizer.AdamW(
            learning_rate=1e-2,
            parameters=layer.parameters(),
            multi_precision=False,
        )
        paddle.seed(self.SEED)
        x = paddle.randn([self.B, self.S, self.H], dtype=paddle.bfloat16)
        target = paddle.zeros_like(x)

        losses = []
        for _ in range(10):
            x.stop_gradient = False
            with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
                out, _ = layer(x)
                loss = ((out - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses.append(loss.item())

        self.assertFalse(
            any(paddle.isnan(paddle.to_tensor(v)).item() for v in losses)
        )
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
