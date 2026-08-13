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

import os
import unittest

os.environ["MOONEP_MEM_HANDLE_TYPE"] = "fd"

import paddle
import paddle.distributed as dist
from paddlefleet_ops import is_moonep_available


class TestMoonEP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_moonep_available():
            raise unittest.SkipTest("MoonEP is not available")
        if not dist.is_initialized():
            dist.init_parallel_env()
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        if cls.world_size != 2:
            dist.destroy_process_group()
            raise unittest.SkipTest("MoonEP smoke test requires two ranks")

        from paddlefleet_ops.moonep import Buffer, MoonEPCommPlan

        cls.Buffer = Buffer
        cls.MoonEPCommPlan = MoonEPCommPlan

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_dispatch_combine_round_trip(self):
        S, H, K = 128, 128, 1
        E = self.world_size * 2
        buffer = self.Buffer(
            S=S,
            H=H,
            K=K,
            E=E,
            num_ep_ranks=self.world_size,
            num_sms=8,
        )
        try:
            hidden = paddle.arange(S * H, dtype="float32").reshape([S, H])
            hidden = (hidden / 1024 + self.rank).astype(paddle.bfloat16)
            route_weights = (
                paddle.arange(S, dtype="float32").reshape([S, K]) / S
            )

            remote_rank = (self.rank + 1) % self.world_size
            topk_experts = (
                paddle.arange(S, dtype="int32") % 2 + remote_rank * 2
            ).reshape([S, K])
            tokens_per_expert = paddle.bincount(
                topk_experts.reshape([-1]), minlength=E
            ).astype("int32")

            dispatched, dispatched_weights, cu_seqlens, plan = buffer.dispatch(
                hidden,
                route_weights,
                topk_experts,
                tokens_per_expert,
            )
            self.assertIsInstance(plan, self.MoonEPCommPlan)
            self.assertEqual(list(plan.dst.shape), [S * K])
            self.assertEqual(list(cu_seqlens.shape), [E + E // self.world_size])

            combined, combined_weights, _ = buffer.combine(
                plan=plan,
                hidden_nvsh=dispatched,
                route_weights_nvs=dispatched_weights,
            )
            paddle.device.synchronize()

            self.assertTrue(
                bool(
                    paddle.equal_all(
                        combined.astype("float32"),
                        hidden.astype("float32"),
                    )
                )
            )
            self.assertTrue(
                bool(paddle.equal_all(combined_weights, route_weights))
            )
        finally:
            buffer.destroy()


if __name__ == "__main__":
    unittest.main()
