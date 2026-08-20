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

import os
import random
import time
import unittest
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet

from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe import fused_a2a
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig

_pg_collection = None


def _ensure_fleet():
    """Initialize fleet once per process and return the process groups."""
    global _pg_collection
    if _pg_collection is not None:
        return _pg_collection

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 4,
        "pp_degree": 1,
        "sharding_degree": 2,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 4,
        "moe_sharding_degree": 2,
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
    return _pg_collection


class TestFusionBF16ExpertParallel(unittest.TestCase):
    def setUp(self):
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        paddle.manual_seed(seed)

        self.pg_collection = _ensure_fleet()
        model_parallel_cuda_manual_seed(seed)

    def test_moe_expert_fusion(self):
        n_routed_experts = 64
        hidden_size = 256
        transformer_config = TransformerConfig(
            hidden_size=hidden_size,
            num_attention_heads=4,
            n_routed_experts=n_routed_experts,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=128,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            moe_expert_fusion=True,
            moe_token_dispatcher_type="deepep",
            bias_activation_fusion=True,
        )

        transformer_layer_spec = get_gpt_layer_local_spec(
            transformer_config, num_experts=n_routed_experts
        )

        moe_layer = MoELayer(
            transformer_config,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            self.pg_collection,
        )

        input_data = paddle.randn(4, 256, hidden_size, dtype=paddle.bfloat16)

        output_moe_deep_gemm_true = moe_layer(input_data)[0]

        moe_layer.moe_deep_gemm = False

        output_moe_deep_gemm_false = moe_layer(input_data)[0]

        np.testing.assert_allclose(
            output_moe_deep_gemm_true.detach().cpu().float().numpy(),
            output_moe_deep_gemm_false.detach().cpu().float().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    def tearDown(self):
        pass


class TestStreamOrderedFenceEP(unittest.TestCase):
    """Unit tests for fused_a2a._stream_ordered_fence_ep on a real EP group."""

    @classmethod
    def setUpClass(cls):
        cls.pg_collection = _ensure_fleet()

    def setUp(self):
        self.ep_group = self.pg_collection.ep
        self._reset_fence_state()

    def tearDown(self):
        self._reset_fence_state()

    def _reset_fence_state(self):
        fused_a2a._EP_BARRIER_ASYNC = None
        fused_a2a._ep_fence_tensors.clear()

    def test_fence_tensor_is_cached_per_group(self):
        first = fused_a2a._ep_fence_tensor(self.ep_group)
        second = fused_a2a._ep_fence_tensor(self.ep_group)
        self.assertIs(first, second)
        self.assertEqual(first.shape, [1])
        self.assertEqual(first.dtype, paddle.int32)
        self.assertEqual(
            list(fused_a2a._ep_fence_tensors.keys()), [self.ep_group.id]
        )

    def test_repeated_fences_keep_tensor_zero(self):
        """The cached tensor must stay reusable across steps."""
        for _ in range(4):
            fused_a2a._stream_ordered_fence_ep(self.ep_group)
        tensor = fused_a2a._ep_fence_tensor(self.ep_group)
        self.assertEqual(int(tensor.numpy()[0]), 0)

    def test_fence_rendezvous_orders_group_traffic(self):
        """A fence between two collectives must not disturb their results."""
        ep_rank = paddle.distributed.get_rank(self.ep_group)
        expected = float(sum(range(self.ep_group.nranks)))

        before = paddle.full([4], float(ep_rank), dtype="float32")
        paddle.distributed.all_reduce(before, group=self.ep_group)

        fused_a2a._stream_ordered_fence_ep(self.ep_group)

        after = paddle.full([4], float(ep_rank), dtype="float32")
        paddle.distributed.all_reduce(after, group=self.ep_group)

        np.testing.assert_allclose(
            before.numpy(), np.full([4], expected, dtype="float32")
        )
        np.testing.assert_allclose(
            after.numpy(), np.full([4], expected, dtype="float32")
        )

    def _run_deepep_rounds(self, async_barrier):
        ep_rank = paddle.distributed.get_rank(self.ep_group)
        num_experts = self.ep_group.nranks * 2
        num_tokens, hidden_size, topk = 64, 128, 2
        outputs = []

        with patch.dict(
            os.environ,
            {"FLEET_MOE_EP_BARRIER_ASYNC": "1" if async_barrier else "0"},
        ):
            fused_a2a._EP_BARRIER_ASYNC = None
            for round_id in range(2):
                values = np.arange(num_tokens * hidden_size, dtype="float32")
                values = values.reshape(num_tokens, hidden_size)
                values = values / (num_tokens * hidden_size)
                values += ep_rank * 10 + round_id
                x = paddle.to_tensor(values, dtype=paddle.bfloat16)
                token_indices = paddle.to_tensor(
                    [
                        [
                            (token_id + ep_rank + round_id) % num_experts,
                            (
                                token_id
                                + ep_rank
                                + round_id
                                + self.ep_group.nranks
                            )
                            % num_experts,
                        ]
                        for token_id in range(num_tokens)
                    ],
                    dtype="int64",
                )
                token_probs = paddle.full(
                    [num_tokens, topk], 1.0 / topk, dtype="float32"
                )

                dispatched_x, _, states, _ = fused_a2a.fused_dispatch(
                    x,
                    token_indices,
                    token_probs,
                    num_experts,
                    self.ep_group,
                )
                if round_id == 0 and ep_rank == self.ep_group.nranks - 1:
                    time.sleep(0.5)
                outputs.append(
                    fused_a2a.fused_combine(
                        dispatched_x,
                        self.ep_group,
                        states["handle"],
                    )
                )

        return outputs

    @unittest.skipUnless(
        fused_a2a.HAVE_DEEP_EP, "DeepEP is required for stream-order testing"
    )
    def test_fence_orders_deepep_buffer_reuse_across_rounds(self):
        """The async fence must preserve DeepEP buffer reuse ordering."""
        baseline = self._run_deepep_rounds(async_barrier=False)
        paddle.distributed.barrier(self.ep_group)
        actual = self._run_deepep_rounds(async_barrier=True)

        for round_id, (expected, result) in enumerate(zip(baseline, actual)):
            np.testing.assert_array_equal(
                expected.float().numpy(),
                result.float().numpy(),
                err_msg=f"DeepEP output differs in round {round_id}",
            )

        completed_rounds = paddle.full([1], len(actual), dtype="int32")
        paddle.distributed.all_reduce(completed_rounds, group=self.ep_group)
        self.assertEqual(
            int(completed_rounds.numpy()[0]), 2 * self.ep_group.nranks
        )

    def test_barrier_ep_uses_device_barrier_by_default(self):
        fused_a2a._EP_BARRIER_ASYNC = False
        with (
            patch.object(paddle.distributed, "barrier") as device_barrier,
            patch.object(fused_a2a, "_stream_ordered_fence_ep") as fence,
        ):
            fused_a2a.barrier_ep(self.ep_group)
        device_barrier.assert_called_once_with(self.ep_group)
        fence.assert_not_called()

    def test_barrier_ep_uses_fence_when_enabled(self):
        fused_a2a._EP_BARRIER_ASYNC = True
        with (
            patch.object(paddle.distributed, "barrier") as device_barrier,
            patch.object(fused_a2a, "_stream_ordered_fence_ep") as fence,
        ):
            fused_a2a.barrier_ep(self.ep_group)
        fence.assert_called_once_with(self.ep_group)
        device_barrier.assert_not_called()

    def test_async_flag_reads_env_once(self):
        with patch.dict(os.environ, {"FLEET_MOE_EP_BARRIER_ASYNC": "1"}):
            self.assertTrue(fused_a2a._ep_barrier_async_enabled())
        # Cached: clearing the env afterwards must not flip the switch.
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(fused_a2a._ep_barrier_async_enabled())

        fused_a2a._EP_BARRIER_ASYNC = None
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(fused_a2a._ep_barrier_async_enabled())


if __name__ == "__main__":
    unittest.main()
