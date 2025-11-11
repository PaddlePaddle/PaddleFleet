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

import math
import random
import numpy as np

import paddle
import paddle.distributed.fleet as fleet
from .global_random import random_manager
from .vpp_simulator import ChunkType, VPPSimulator

def get_elastic_num(
    elastic_model_list: list, 
    elastic_model_ratio_list: list,
    data_accumulation_steps: int,
    gradient_accumulation_steps: int
) -> tuple:
    """
    Calculate elastic training configuration parameters.
    
    Args:
        elastic_model_list: List of model configurations for different elastic modes
        elastic_model_ratio_list: Ratio of each elastic mode in training steps
        data_accumulation_steps: Total data accumulation steps
        gradient_accumulation_steps: Gradient accumulation steps per batch
        
    Returns:
        tuple: (need_step_list, max_step, is_full_model, full_model_batch_size, elastic_batch_size)
        
    Raises:
        AssertionError: If inputs are invalid
    """
    # Validate input lengths match
    if len(elastic_model_list) != len(elastic_model_ratio_list):
        raise ValueError("elastic_model_list and elastic_model_ratio must have the same length")
    need_step_list = []
    is_full_model = []
    full_model_batch_size = elastic_model_list[0]
    elastic_batch_size = elastic_model_list[1]
    for i, mode_ratio in enumerate(elastic_model_ratio_list):
        model_num = mode_ratio * gradient_accumulation_steps
        assert model_num % 1 == 0, f"need_data must be an integer, but got {model_num}"
        need_step = [elastic_model_list[i]] * int(model_num)
        if i == 0:
            is_full = [True] * int(model_num)
        else:
            is_full = [False] * int(model_num)
        is_full_model.extend(is_full)
        need_step_list.extend(need_step)
    sum_need_step = sum(need_step_list)
    max_step = max(need_step_list)
    assert sum_need_step == data_accumulation_steps, (
        f"sum_need_data must be equal to data_accumulation_steps, "
        f"but got {sum_need_step} and {data_accumulation_steps}"
    )
    # print(f"elastic info need_step_list: {need_step_list}, max_step: {max_step}, is_full_model: {is_full_model}")
    return need_step_list, max_step, is_full_model, full_model_batch_size, elastic_batch_size


def padding_input_for_eb5(input_dict, max_batch_size):
    """
    Pad specified keys in input_dict ("input_ids", "attention_mask",
    "position_ids", "inbatch_pack_offset") along batch dimension (dimension 0) to max_batch_size
    """
    padded_dict = {}
    pad_keys = {
        "input_ids",
        "position_ids",
        "labels",
        "task_id",
        "startend_row_indices",
        "image_infinity_scale",
        "video_infinity_scale",
    }

    for key, value in input_dict.items():
        if key in pad_keys and value is not None:
            # Get current batch_size
            current_batch_size = value.shape[0]

            # If current batch_size already equals max_batch_size, no padding needed
            if current_batch_size == max_batch_size:
                padded_dict[key] = value
                continue

            # Calculate padding size needed
            pad_size = max_batch_size - current_batch_size

            # Construct padding tensor with same shape as original tensor except for dimension 0 which is pad_size
            pad_shape = [pad_size] + list(value.shape[1:])
            if key == "labels":
                pad_tensor = paddle.ones(pad_shape, dtype=value.dtype) * -100  # ignore_index
            else:
                pad_tensor = paddle.zeros(pad_shape, dtype=value.dtype)

            # Concatenate padded tensor
            padded_dict[key] = paddle.concat([value, pad_tensor], axis=0)
        else:
            # Other keys keep original value
            padded_dict[key] = value

    return padded_dict


def padding_input(input_dict: dict, max_batch_size: int) -> dict:
    """
    Pad input tensors to specified batch size for elastic training.
    
    Handles padding for standard NLP inputs:
    - input_ids
    - attention_mask  
    - position_ids
    - inbatch_pack_offset
    
    Args:
        input_dict: Dictionary containing input tensors
        max_batch_size: Target batch size after padding
        
    Returns:
        dict: Dictionary with padded tensors
    """
    padded_dict = {}
    pad_keys = {"input_ids", "attention_mask", "position_ids", "inbatch_pack_offset"}

    for key, value in input_dict.items():
        if key in pad_keys and value is not None:
            # Get current batch_size
            current_batch_size = value.shape[0]

            # If current batch_size already equals max_batch_size, no padding needed
            if current_batch_size == max_batch_size:
                padded_dict[key] = value
                continue

            # Calculate padding size needed
            pad_size = max_batch_size - current_batch_size

            # Construct padding tensor with same shape as original tensor except for dimension 0 which is pad_size
            pad_shape = [pad_size] + list(value.shape[1:])
            pad_tensor = paddle.zeros(pad_shape, dtype=value.dtype)

            # Concatenate padded tensor
            padded_dict[key] = paddle.concat([value, pad_tensor], axis=0)
        else:
            # Other keys keep original value
            padded_dict[key] = value

    return padded_dict


def merge_inputs(elastic_inputs, inputs):
    """
    Merge new inputs with existing elastic_inputs. If elastic_inputs is None, initialize it with new inputs;
    otherwise, concatenate new inputs with existing elastic_inputs.

    Args:
        elastic_inputs (dict, optional): Inputs stored as dictionary, may contain None values. Defaults to None.
        inputs (dict): New inputs stored as dictionary, must contain all keys for new inputs.

    Returns:
        dict: Returns merged dictionary containing all inputs, including new inputs and existing elastic_inputs.
    """
    if elastic_inputs is None:
        # Initialize elastic_inputs, but preserve None values
        elastic_inputs = {key: (value.clone() if value is not None else None) for key, value in inputs.items()}
    else:
        # Merge inputs
        for key, value in inputs.items():
            if key in elastic_inputs:
                if value is None or elastic_inputs[key] is None:
                    # If either is None, do not modify, keep original value
                    continue
                # Both are not None, perform concatenation
                if key in [
                    "image_labels",
                    "image_features",
                    "video_labels",
                    "video_features",
                    "image_full_features",
                    "video_full_features",
                    "video_features_history",
                    "video_loss_reweight",
                    "audio_ids",
                    "audio_labels",
                    "audio_spk_embs",
                ]:
                    # EB5, the above attributes don't have bsz dimension, need special handling
                    pad_value = -100 if "labels" in key else -1  # default pad to -100 for labels (ignore_index)
                    # Find first pad_value
                    first_pad_index_in_ori_value = paddle.where(
                        elastic_inputs[key].reshape([elastic_inputs[key].shape[0], -1])[:, 0] == pad_value
                    )[0][0]
                    first_pad_index_in_new_value = paddle.where(value.reshape([value.shape[0], -1])[:, 0] == pad_value)[
                        0
                    ][0]
                    if first_pad_index_in_new_value.item() != 0:
                        # Fill new values
                        elastic_inputs[key][
                            first_pad_index_in_ori_value : first_pad_index_in_ori_value + first_pad_index_in_new_value
                        ] = value[:first_pad_index_in_new_value]

            else:
                # Direct assignment (preserve None case)
                elastic_inputs[key] = value.clone() if value is not None else None

    return elastic_inputs


def generate_choice_lists(layer_num, all_size, choice_size):
    """
    Generate a 2D list where each element is a list of length all_size, and each element in that list is a list of length layer_num.
    This function randomly selects all_size indices and sets the elements at those indices to 1, with all other elements set to 0.

    Args:
        layer_num (int): The first dimension length of the 2D list, i.e., the length of each list in the list.
        all_size (int): The second dimension length of the 2D list, i.e., the length of each element in the lists.
        choice_size (int): The number of indices to randomly select.

    Returns:
        list of list of int: Returns a 2D list where each element is a list of length all_size, and each element in that list is a list of length layer_num,
                              containing randomly selected indices with elements set to 1, and all other elements set to 0.
    """
    result = [[0] * all_size for _ in range(layer_num)]  # Pre-allocate 0
    for i in range(layer_num):
        indices = random_manager.global_random.sample(range(all_size), choice_size)  # Select choice_size indices
        for idx in indices:
            result[i][idx] = 1  # Set to 1
    return result


def generate_full_lists(layer_num, all_size, choice_size):
    """
    Generate a 2D list where each row is a vector of length all_size filled with 0s, but the first choice_size elements are set to 1.

    Args:
        layer_num (int): The number of rows for the 2D list to generate.
        all_size (int): The size of each row in the 2D list, i.e., the total number of elements.
        choice_size (int): The number of elements to set to 1.

    Returns:
        numpy.ndarray, shape=(layer_num, all_size): Returns a 2D list where each row is a vector of length all_size,
        with the first choice_size elements set to 1 and all other elements set to 0.
    """
    result = np.zeros((layer_num, all_size), dtype=int)
    for i in range(layer_num):
        result[i, :choice_size] = 1

    return result


def get_topk_random_uniform(topk_avg, data_num, max_topk, elastic_topk_prob=1.0):
    """
    Generate a list of random topk values based on given average, data count, and maximum value.

    Args:
        topk_avg (int): Average topk number.
        data_num (int): Number of samples in the dataset.
        max_topk (int, optional): Maximum topk value.
        elastic_topk_prob (float, optional): Probability of using elastic TOPK

    Returns:
        list of int: A list containing `data_num` elements, each representing the topk value for the corresponding sample, in range [1, max_topk].
        If `data_num` equals 1, returns a list of length 1 with element min(total, max_topk), where total is topk_avg * data_num.

    Raises:
        None
    """
    p = random_manager.global_random.random()
    total = topk_avg * data_num  # Total
    if p < (1 - elastic_topk_prob):
        return [topk_avg] * data_num
    # **Special case: data_num=1, return directly**
    if data_num == 1:
        return [min(total, max_topk)]  # Ensure not exceeding max_topk

    while True:
        # 1. Generate `data_num-1` split points, ensuring each number is greater than 0
        split_points = sorted(random_manager.global_random.sample(range(1, total), data_num - 1))
        parts = (
            [split_points[0]]
            + [split_points[i] - split_points[i - 1] for i in range(1, len(split_points))]
            + [total - split_points[-1]]
        )
        # 2. Check if all numbers are ≤ max_topk
        if all(1 <= x <= max_topk for x in parts):
            break  # Only exit loop when requirements are met

    return parts


def get_topk_random(topk, max_topk, min_topk=1, elastic_topk_prob=1.0):
    """
    Generate a random topk value based on given parameters.

    Args:
        topk (int): topk number.
        max_topk (int): Maximum topk value.
        min_topk (int, optional): Minimum topk value.
        elastic_topk_prob (float, optional): Probability of using elastic TOPK

    Returns:
        Returns random topk value, and a boolean indicating if it's different from the original topk

    Raises:
        None
    """
    p = random_manager.global_random.random()
    if p < (1 - elastic_topk_prob):
        return topk, False
    # random_topk = random_manager.global_random.randint(min_topk, max_topk)
    choice_topk_list = [x for x in range(min_topk, max_topk + 1) if x != topk]
    random_topk = random_manager.global_random.choice(choice_topk_list)
    return random_topk, True


def get_valid_expert_num(
    min_val: int,
    moe_num_experts: int, 
    moe_world_size: int,
    max_prob: float,
    elastic_candidates_interval: int = None
) -> tuple:
    """
    Dynamic expert selection for Mixture-of-Experts training.
    
    Implements elastic expert selection with:
    - Random expert selection within constraints
    - Probability-based maximum expert selection
    - Distributed-aware expert allocation
    
    Args:
        min_val: Minimum number of experts to use (must be > 0)
        moe_num_experts: Total available experts in model
        moe_world_size: Number of expert parallel partitions
        max_prob: Probability [0-1] of using max experts
        elastic_candidates_interval: Interval for expert number candidates
        
    Returns:
        tuple: (selected_expert_num, expert_rank_list, gate_mask)
        
    Raises:
        ValueError: If min_val exceeds moe_num_experts
    """
    if min_val > moe_num_experts:
        raise ValueError(f"min_val ({min_val}) cannot be greater than moe_num_expert ({moe_num_experts})")
    # print(f"yang debug min_val is {min_val}, moe_num_experts is {moe_num_experts},
    # moe_world_size is {moe_world_size}, max_prob is {max_prob}")

    # max_prob probability selects maximum expert number
    p = random_manager.global_random.random()
    if p < max_prob:
        new_expert_num = moe_num_experts
    else:
        max_val = moe_num_experts
        if elastic_candidates_interval is None:
            elastic_candidates_interval = moe_world_size
        candidates = [
            x for x in range(min_val, max_val - elastic_candidates_interval + 1) if x % elastic_candidates_interval == 0
        ]
        new_expert_num = random_manager.global_random.choice(candidates) if candidates else moe_num_experts

    if new_expert_num == moe_num_experts:
        new_expert_rank_list, global_gate_mask = None, None
    else:
        # Calculate how many experts each rank needs
        experts_per_rank = moe_num_experts // moe_world_size
        need_experts_per_rank = new_expert_num // moe_world_size

        new_expert_rank_list = []
        selected_indices = []
        # Generate new_expert_rank_list, ensuring each rank selects experts randomly
        for moe_rank in range(moe_world_size):
            one_rank_experts = sorted(
                random_manager.global_random.sample(range(experts_per_rank), need_experts_per_rank)
            )
            new_expert_rank_list.append(one_rank_experts)
            selected_indices.extend([moe_rank * experts_per_rank + i for i in one_rank_experts])

        selected_indices_cpu = paddle.to_tensor(selected_indices, place="cpu", dtype="int64")
        selected_indices_gpu = selected_indices_cpu.to("gpu", blocking=False)  # using CudaMemcpyAsync

        # Create bool mask, selected ones are True, others are False
        mask = paddle.zeros([moe_num_experts], dtype="bool")
        mask[selected_indices_gpu] = True

        global_gate_mask = paddle.where(
            mask,
            paddle.zeros_like(mask, dtype="float32"),  # If mask is True, set to 0.0
            paddle.full_like(mask, float("-inf"), dtype="float32"),  # Otherwise set to -inf
        )
        global_gate_mask.stop_gradient = True

    return new_expert_num, new_expert_rank_list, global_gate_mask


def get_valid_eval_expert_num(min_val, moe_num_experts, moe_world_size, max_prob):
    """
    Get valid expert number and expert index list
    :param min_val: Minimum number of experts
    :param moe_num_experts: Maximum number of experts
    :param moe_world_size: Number of expert partitions
    :param max_prob: Probability of selecting maximum number of experts
    :return:
    Returns new expert number, local expert list per rank, and global expert gate mask.
    """
    # Force set expert number during evaluation phase
    new_expert_num = 8

    if new_expert_num == moe_num_experts:
        new_expert_rank_list, global_gate_mask = None, None
    else:
        # Calculate how many experts each rank needs
        experts_per_rank = moe_num_experts // moe_world_size
        need_experts_per_rank = new_expert_num // moe_world_size

        new_expert_rank_list = []
        selected_indices = []
        # Generate new_expert_rank_list, ensuring each rank selects experts randomly
        for moe_rank in range(moe_world_size):
            one_rank_experts = sorted(
                random_manager.global_random.sample(range(experts_per_rank), need_experts_per_rank)
            )
            new_expert_rank_list.append(one_rank_experts)
            selected_indices.extend([moe_rank * experts_per_rank + i for i in one_rank_experts])

        # Create bool mask, selected ones are True, others are False
        mask = paddle.zeros([moe_num_experts], dtype="bool")
        mask[paddle.to_tensor(selected_indices, dtype="int64")] = True

        global_gate_mask = paddle.where(
            mask,
            paddle.zeros_like(mask, dtype="float32"),  # If mask is True, set to 0.0
            paddle.full_like(mask, float("-inf"), dtype="float32"),  # Otherwise set to -inf
        )
        global_gate_mask.stop_gradient = True
    print(
        f"yang debug new_expert_num is {new_expert_num}, new_expert_rank_list is {new_expert_rank_list}, "
        f"global_gate_mask is {global_gate_mask}"
    )
    return new_expert_num, new_expert_rank_list, global_gate_mask


def set_pp_balance_elastic_layer_manager(value):
    global _pp_balance_elastic_layer_manager
    _pp_balance_elastic_layer_manager = value


def get_pp_balance_elastic_layer_manager():
    global _pp_balance_elastic_layer_manager
    return _pp_balance_elastic_layer_manager


class PPBalanceElasticLayerManager:
    def __init__(
        self,
        pp_degree,
        vpp_degree,
        num_acc_steps,
        no_elastic_acc_step,
        num_hidden_layers,
        remove_head_layer,
        remove_tail_layer,
        retain_layer_prob,
    ):
        # Record communication group information
        self.hcg = fleet.get_hybrid_communicate_group()
        self.use_expert_group = hasattr(self.hcg, "get_expert_parallel_group")
        self.mp_group = self.hcg.get_model_parallel_group()
        self.pp_group = self.hcg.get_pipe_parallel_group()
        self.pp_rank = self.hcg.get_stage_id()
        self.mp_src_rank = self.hcg.get_model_parallel_group_src_rank()
        self.cur_rank = paddle.distributed.get_rank()
        if self.use_expert_group:
            self.expert_group = self.hcg.get_expert_parallel_group()
            self.expert_src_rank = self.hcg.get_expert_parallel_group_src_rank()

        # Record model and training configuration
        self.pp_degree = pp_degree
        self.vpp_degree = vpp_degree
        self.num_acc_steps = num_acc_steps
        self.no_elastic_acc_step = no_elastic_acc_step
        self.elastic_acc_step = num_acc_steps - no_elastic_acc_step
        self.remove_head_layer = remove_head_layer
        self.remove_tail_layer = remove_tail_layer
        self.retain_layer_prob = retain_layer_prob
        self.num_hidden_layers = num_hidden_layers - remove_head_layer - remove_tail_layer
        self.elastic_layer_ids = range(remove_head_layer, num_hidden_layers - remove_tail_layer)

        # Initialize intermediate variables
        self.acc_stamp = [0] * self.num_hidden_layers  # Records the acc_step when is_elastic_layer is called for each layer
        self.elastic_micro_step = self._get_elastic_micro_step()

        # Initialize elastic layer mask
        self.step()

    def _generate_elastic_layer_mask(self):
        """
        Generate elastic layer mask to control which layers participate in training.
        If current iteration step is selected, eligible layers will skip training.
        Returns:
            np.ndarray: 2D boolean array with shape (elastic_acc_step, num_hidden_layers)
        """
        start_step_to_acc_and_layer_ids = self.elastic_micro_step
        acc_to_elastic_layer = {}

        for acc_and_layer_ids in start_step_to_acc_and_layer_ids.values():
            if random_manager.global_random.random() > self.retain_layer_prob:
                for (acc_step, layer_index) in acc_and_layer_ids:
                    if acc_step not in acc_to_elastic_layer:
                        acc_to_elastic_layer[acc_step] = []
                    acc_to_elastic_layer[acc_step].append(layer_index)

        elastic_layer_mask = np.zeros((self.elastic_acc_step, self.num_hidden_layers), dtype=bool)
        for acc_step in acc_to_elastic_layer:
            elastic_layer_mask[acc_step - self.no_elastic_acc_step, acc_to_elastic_layer[acc_step]] = True

        return elastic_layer_mask

    def _get_elastic_micro_step(self):
        """
        Get candidate layers for PP balanced elastic layer skipping strategy.
        These are layers in BACKWARD phase with same iteration step,
        and both acc_step and layer_id are in elastic range.
        """
        vpp_simulator = VPPSimulator(
            pp_degree=self.pp_degree, vpp_degree=self.vpp_degree, num_acc_steps=self.num_acc_steps
        )
        schedule_table = vpp_simulator.schedule()

        max_micro_step = schedule_table[0][-1].end
        start_step_to_acc_and_layer_ids = {}
        elastic_acc_steps = range(self.no_elastic_acc_step, self.num_acc_steps)
        for schedule in schedule_table:
            for chunk in schedule:
                if (
                    chunk.chunk_type == ChunkType.BACKWARD
                    and chunk.acc_step in elastic_acc_steps
                    and chunk.layer_id in self.elastic_layer_ids
                ):
                    start_step = chunk.start
                    if start_step_to_acc_and_layer_ids.get(start_step, None) is None:
                        start_step_to_acc_and_layer_ids[start_step] = []
                    start_step_to_acc_and_layer_ids[start_step].append(
                        (chunk.acc_step, chunk.layer_id - self.remove_head_layer)
                    )
        return start_step_to_acc_and_layer_ids

    def step(self):
        """
        After a training step, update acc_stamp and generate new elastic_layer_mask,
        then broadcast to other ranks.
        """
        self.acc_stamp = [0] * self.num_hidden_layers

        # To ensure random consistency within mp group, all cards need to advance random state. But finally rank0 state will be broadcast
        elastic_layer_mask = [self._generate_elastic_layer_mask()]

        if self.use_expert_group:
            # expert_group broadcast
            if self.expert_group.nranks > 1 and self.pp_rank == 0:
                paddle.distributed.broadcast_object_list(
                    elastic_layer_mask,
                    src=self.expert_src_rank,
                    group=self.expert_group,
                )
        else:
            # mp_group broadcast
            if self.mp_group.nranks > 1 and self.pp_rank == 0:
                paddle.distributed.broadcast_object_list(
                    elastic_layer_mask,
                    src=self.mp_src_rank,
                    group=self.mp_group,
                )
        # pp_group broadcast
        if self.pp_group.nranks > 1:
            paddle.distributed.broadcast_object_list(
                elastic_layer_mask,
                src=self.pp_group.ranks[0],
                group=self.pp_group,
            )
        self.elastic_layer_mask = elastic_layer_mask[0]

    def is_elastic_layer(self, layer_id):
        """
        Determine if current layer can be skipped
        """
        if layer_id >= (self.num_hidden_layers + self.remove_head_layer) or layer_id < self.remove_head_layer:
            return False

        acc_step = self.acc_stamp[layer_id - self.remove_head_layer]

        # Increment execution count of current layer by 1
        self.acc_stamp[layer_id - self.remove_head_layer] += 1

        if acc_step < self.no_elastic_acc_step:
            return False

        return self.elastic_layer_mask[acc_step - self.no_elastic_acc_step][layer_id - self.remove_head_layer]

