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

import unittest

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

import fleet.core.pipeline_parallel.schedules as schedule
from fleet.training.initialize import initialize_fleet


class TestParallelState(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 2,
            "pp_degree": 1,
        }
        initialize_fleet(strategy=strategy)

    def test_no_pipeline(self):
        def forward_step_func(data_iterator, model):
            dummy_data = paddle.ones(1, 4)
            rank = dist.get_rank()

            def loss_func(output_tensor):
                return output_tensor.mean(), {"loss_reduced": rank}

            return model(dummy_data), loss_func

        model = paddle.nn.Linear(4, 1)
        forward_backward_func = schedule.get_forward_backward_func()
        assert schedule.get_forward_backward_func() == schedule.no_pipeline

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=range(0, 100),
            model=model,
            num_microbatches=4,
        )
        rank = dist.get_rank()
        loss_reduced_expected = [
            {"loss_reduced": rank},
            {"loss_reduced": rank},
            {"loss_reduced": rank},
            {"loss_reduced": rank},
        ]

        for i, j in zip(losses_reduced, loss_reduced_expected):
            assert i["loss_reduced"] == j["loss_reduced"]
            print(f"i[loss_reduced]:{i['loss_reduced']}")


if __name__ == "__main__":
    unittest.main()
