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
import paddle.distributed.fleet as fleet

import numpy as np

from fleet.core.elastic_utils import (
    get_valid_expert_num,
    get_pp_balance_elastic_layer_manager,
    set_pp_balance_elastic_layer_manager,
    PPBalanceElasticLayerManager,
)
from fleet.core.vpp_simulator import VPPSimulator

from fleet.core.global_random import random_manager


def _get_valid_expert_num_multi_times(
    min_val: int,
    moe_num_experts: int,
    moe_world_size: int,
    max_prob: float,
    elastic_candidates_interval: int = None,
    num_iterations: int = 50
) -> tuple:
    """
    Helper function to test expert selection multiple times.
    
    Args:
        min_val: Minimum number of experts
        moe_num_experts: Maximum number of experts
        moe_world_size: Number of expert partitions
        max_prob: Probability of selecting maximum experts
        elastic_candidates_interval: Expert selection interval
        num_iterations: Number of test iterations
        
    Returns:
        tuple: Lists of expert numbers, rank lists, and gate masks
    """
    multi_new_expert_num = []
    multi_new_expert_rank = []
    multi_global_gate_mask = []

    for _ in range(num_iterations):
        new_expert_num, new_expert_rank_list, global_gate_mask = get_valid_expert_num(
            min_val, moe_num_experts, moe_world_size, max_prob, elastic_candidates_interval
        )
        multi_new_expert_num.append(new_expert_num)
        multi_new_expert_rank.append(new_expert_rank_list)
        multi_global_gate_mask.append(global_gate_mask)

    return multi_new_expert_num, multi_new_expert_rank, multi_global_gate_mask


def _assert_optional_equal(value1, value2, assert_func):
    """_assert_optional_equal"""
    if value1 is None and value2 is None:
        return
    if value1 is not None and value2 is not None:
        assert_func(value1, value2)
        return
    raise AssertionError(f"One value is None and the other is not: {value1} vs {value2}")


def _check_result_multi_times(result1, result2):
    """_check_result_multi_times"""
    (multi_new_expert_num1, multi_new_expert_rank1, multi_global_gate_mask1) = result1
    (multi_new_expert_num2, multi_new_expert_rank2, multi_global_gate_mask2) = result2

    # 检查所有结果对
    for i, (new_expert_num1, new_expert_num2) in enumerate(zip(multi_new_expert_num1, multi_new_expert_num2)):
        np.testing.assert_equal(new_expert_num1, new_expert_num2, err_msg=f"Expert num mismatch at iteration {i}")

    for i, (new_expert_rank1, new_expert_rank2) in enumerate(zip(multi_new_expert_rank1, multi_new_expert_rank2)):
        _assert_optional_equal(
            new_expert_rank1,
            new_expert_rank2,
            lambda x, y: np.testing.assert_equal(x, y, err_msg=f"Expert rank mismatch at iteration {i}"),
        )

    for i, (global_gate_mask1, global_gate_mask2) in enumerate(zip(multi_global_gate_mask1, multi_global_gate_mask2)):
        _assert_optional_equal(
            global_gate_mask1,
            global_gate_mask2,
            lambda x, y: np.testing.assert_equal(
                x._md5sum(), y._md5sum(), err_msg=f"Global gate mask mismatch at iteration {i}"
            ),
        )


class TestElasticUtils(unittest.TestCase):
    def test_valid_expert_num(self):
        min_val = 16
        moe_num_experts = 384
        moe_world_size = 64
        max_prob = 0.5

        random_manager.global_random.seed(42)
        o1 = _get_valid_expert_num_multi_times(min_val, moe_num_experts, moe_world_size, max_prob)

        elastic_candidates_interval = moe_world_size
        random_manager.global_random.seed(42)
        o2 = _get_valid_expert_num_multi_times(
            min_val, moe_num_experts, moe_world_size, max_prob, elastic_candidates_interval
        )

        _check_result_multi_times(o1, o2)

        random_manager.global_random.seed(42)
        new_moe_world_size = 32
        multi_new_expert_num, _, _ = _get_valid_expert_num_multi_times(
            min_val, moe_num_experts, new_moe_world_size, max_prob, elastic_candidates_interval=64
        )

        for new_expert_num in multi_new_expert_num:
            assert (
                new_expert_num % elastic_candidates_interval == 0
            ), f"new_expert_num {new_expert_num} is not divisible by elastic_candidates_interval {elastic_candidates_interval}"

    def test_vpp_simulator(self):
        pp_degree = 4
        vpp_degree = 2
        num_acc_steps = 16

        vpp_simulator = VPPSimulator(pp_degree, vpp_degree, num_acc_steps)
        bubble_rate = vpp_simulator.compute_bubble_rate()
        expected_bubble_rate = 1.0 * (pp_degree - 1) / (vpp_degree * num_acc_steps + (pp_degree - 1))
        # print(f"bubble_rate: {bubble_rate}, expected_bubble_rate: {expected_bubble_rate}")
        assert bubble_rate == expected_bubble_rate, f"Expected bubble rate {expected_bubble_rate}, got {bubble_rate}"

    def test_pp_balance_elastic_layer_manager(self):
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "mp_degree": 1,
            "dp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

        pp_degree = 4
        vpp_degree = 4
        num_acc_steps = 16
        no_elastic_acc_step = 12
        num_hidden_layers = pp_degree * vpp_degree
        remove_head_layers = 1
        remove_tail_layers = 1
        retain_layer_prob = 0.75

        pp_balance_elastic_layer_manager = PPBalanceElasticLayerManager(
            pp_degree,
            vpp_degree,
            num_acc_steps,
            no_elastic_acc_step,
            num_hidden_layers,
            remove_head_layers,
            remove_tail_layers,
            retain_layer_prob,
        )
        set_pp_balance_elastic_layer_manager(pp_balance_elastic_layer_manager)

        no_elastic_acc_mask = np.zeros(shape=(no_elastic_acc_step, num_hidden_layers), dtype=bool)
        elastic_acc_mask = np.array(
            [
                [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                ],
                [
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
            ]
        )

        mask = np.vstack((no_elastic_acc_mask, elastic_acc_mask))

        for i in range(num_acc_steps):
            for j in range(num_hidden_layers):
                is_elastic_layer = get_pp_balance_elastic_layer_manager().is_elastic_layer(j)
                # print(f"{i}, {j}, {mask[i, j]}, {is_elastic_layer}")
                assert is_elastic_layer == mask[i, j], f"unexpected result: i: {i}, j: {j}, {mask[i, j]}\n"
                f"mask:\n{get_pp_balance_elastic_layer_manager().elastic_layer_mask}"


if __name__ == "__main__":
    unittest.main()
