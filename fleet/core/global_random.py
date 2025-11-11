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

"""Global random number management module.

This module provides the RandomManager class for managing random number generation 
in distributed training, ensuring synchronization and consistency of random numbers 
across different processes.
"""

import logging
import random
import paddle.distributed.fleet as fleet
import paddle.distributed as dist

logger = logging.getLogger(__name__)

try:
    from paddle.distributed.fleet.recompute import custom_state_manager
except (ImportError, ModuleNotFoundError):
    custom_state_manager = None
    logger.warning("The random states may be wrong when using recompute. Please check your paddle version.")


class RandomManager(object):
    """
    Manages random number generation in distributed training environments.
    
    Ensures consistent random number generation across different parallel processes
    (data parallel, sharding parallel, pipeline parallel, etc.).
    """
    
    def __init__(self):
        """
        Initialize RandomManager with default random seed.
        """
        self.global_random = random.Random(11)  # Temporary seed before proper initialization

    def init_random(self):
        """
        Initialize random number generators with proper seeds based on parallel topology.
        
        Seeds are constructed using a combination of:
        - Pipeline parallel rank
        - Data parallel rank  
        - Sharding parallel rank
        - MoE sharding rank
        """
        hcg = fleet.get_hybrid_communicate_group()
        if hasattr(hcg, "get_context_parallel_world_size"):
            with_context_parallel = hcg.get_context_parallel_world_size() > 1
            self.sharding_size = hcg.get_sharding_parallel_world_size(with_context_parallel=with_context_parallel)
            self.sharding_rank = hcg.get_sharding_parallel_rank(with_context_parallel=with_context_parallel)
        else:
            self.sharding_size = hcg.get_sharding_parallel_world_size()
            self.sharding_rank = hcg.get_sharding_parallel_rank()

        self.dp_size = hcg.get_data_parallel_world_size()
        self.dp_rank = hcg.get_data_parallel_rank()

        self.pp_rank = hcg.get_stage_id()
        self.pp_size = hcg.get_pipe_parallel_world_size()
        if hasattr(hcg, "get_moe_sharding_parallel_rank"):
            self.moe_sharding_rank = hcg.get_moe_sharding_parallel_rank()
        else:
            self.moe_sharding_rank = 0

        seed = self.pp_rank * 1000000000 + self.dp_rank * 100000000 + self.sharding_rank * 10000000
        self.global_random = random.Random(seed)

        ep_seed = self.pp_rank * 1000000000 + self.moe_sharding_rank * 100000000
        self.ep_random = random.Random(ep_seed)

        if custom_state_manager is not None:
            custom_state_manager.set_custom_get_state_func(self.get_states)
            custom_state_manager.set_custom_set_state_func(self.set_states)

    def get_states(self) -> tuple:
        """
        Get current random states for checkpointing/recompute.
        
        Returns:
            tuple: (global_random_state, ep_random_state) - 
                  Internal states of both random number generators
        """
        return (self.global_random.getstate(), self.ep_random.getstate())

    def set_states(self, packed_states: tuple):
        """
        Restore random states from checkpoint for recompute.
        
        Args:
            packed_states: Tuple containing (global_random_state, ep_random_state)
            
        Raises:
            AssertionError: If input format is invalid
        """
        assert isinstance(packed_states, tuple), "States must be packed in a tuple"
        assert len(packed_states) == 2, "States tuple must contain exactly 2 elements"
        self.global_random.setstate(packed_states[0])
        self.ep_random.setstate(packed_states[1])

    def seed_random(self, global_step: int):
        """
        Re-seed random number generators with current training step.
        
        Ensures deterministic random number sequences across different training runs.
        
        Args:
            global_step: Current training step number, used to vary the seed
                        while maintaining determinism
        """
        seed = self.pp_rank * 1000000000 + self.dp_rank * 100000000 + self.sharding_rank * 10000000 + global_step
        self.global_random.seed(seed)

        ep_seed = self.pp_rank * 1000000000 + self.moe_sharding_rank * 100000000 + global_step
        self.ep_random = random.Random(ep_seed)


random_manager = RandomManager()
