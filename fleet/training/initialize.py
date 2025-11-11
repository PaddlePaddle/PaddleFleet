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

import paddle.distributed as dist
from paddle.distributed import fleet

from fleet.core import parallel_state as ps


def initialize_fleet(strategy: fleet.DistributedStrategy):
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    ps.initialize_model_parallel(hcg)
    print(f"fleet initialize successfully: {rank=} {world_size=}")
