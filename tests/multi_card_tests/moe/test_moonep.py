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
import re
import subprocess
import unittest

import paddle
import paddle.distributed as dist
from paddlefleet_ops import is_moonep_available


def _get_nvlink_fabric_status():
    selected_gpu = os.environ.get("FLAGS_selected_gpus") or os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""
    )
    device_id = selected_gpu.split(",", maxsplit=1)[0]
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-q", "-i", device_id],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return device_id or "unknown", "unknown", "unknown", "unknown"

    fabric_match = re.search(
        r"(?ms)^\s*Fabric\s*$\n"
        r"\s*State\s*:\s*(\S+)\s*$\n"
        r"\s*Status\s*:\s*(\S+)\s*$\n"
        r"\s*CliqueId\s*:\s*(\S+)\s*$",
        output,
    )
    if fabric_match is None:
        return device_id, "unknown", "unknown", "unknown"
    return device_id, *fabric_match.groups()


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

        fabric_statuses = []
        dist.all_gather_object(fabric_statuses, _get_nvlink_fabric_status())
        ready = all(
            state == "Completed" and status == "Success"
            for _, state, status, _ in fabric_statuses
        )
        cliques = {clique for _, _, _, clique in fabric_statuses}
        if not ready or len(cliques) != 1:
            details = "; ".join(str(item) for item in fabric_statuses)
            dist.destroy_process_group()
            raise unittest.SkipTest(
                f"MoonEP requires a ready NVSwitch fabric: {details}"
            )

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
