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
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from enum import Enum

import paddle
from paddle import framework, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    HybridParallelOptimizer,
)
from paddle.distributed.fleet.meta_parallel import (
    PipelineDatasetPreprocessor as PaddlePipelineDatasetPreprocessor,
)
from paddle.distributed.fleet.utils import timer_helper as timer
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    broadcast_dp_parameters,
    broadcast_moe_sharding_parameters,
    broadcast_mp_parameters,
    broadcast_sep_parameters,
    broadcast_sharding_parameters,
)
from paddle.distributed.fleet.utils.log_util import logger
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
    assign_group_by_size,
)

from .pipeline_hooks import (
    PipelineHook,
)
from .pp_layers import PipelineLayer
from .pp_utils.utils import (
    dict_to_tuple_helper,
    profile_pipeline_details,
    tuple_to_dict_helper,
)

g_profile_pipeline_details_steps = int(
    os.getenv("FLAGS_profile_pipeline_details_steps", "0")
)

_use_four_directions = os.environ.get(
    "PADDLE_USE_FOUR_DIRECTIONS_P2P", paddle.base.core.is_compiled_with_xpu()
)
_use_four_directions = False  # xpu use the same p2p method as gpu
if _use_four_directions:
    from .pp_utils import four_directions_p2p_communication as p2p
else:
    from .pp_utils import p2p_communication as p2p


def get_action(is_dp, shard_split_param=False):
    if is_dp:
        return HOOK_ACTION.ALL_REDUCE
    if shard_split_param:
        return HOOK_ACTION.REDUCE_SCATTER
    return HOOK_ACTION.REDUCE


def _get_align_mode_scale():
    hcg = fleet.get_hybrid_communicate_group()
    data_parallel_world_size = hcg.get_data_parallel_world_size()
    sharding_parallel_world_size = hcg.get_sharding_parallel_world_size()
    return max(data_parallel_world_size, 1) * max(
        sharding_parallel_world_size, 1
    )


class PipelineDatasetPreprocessor:
    def __init__(self, function):
        self.function = function

    def __call__(self):
        return self.function()


class PipelineParallelMicroStepLocations(Enum):
    FORWARD_BEGIN = "forward_begin"
    FORWARD_END = "forward_end"
    BACKWARD_BEGIN = "backward_begin"
    BACKWARD_END = "backward_end"


class PipelineParallelMicroStepCallback:
    def __init__(self):
        # Initializes a dictionary to store hooks for each micro-step location in the pipeline.
        self.hooks: dict[PipelineParallelMicroStepLocations, list[Callable]] = {
            PipelineParallelMicroStepLocations.FORWARD_BEGIN: [],
            PipelineParallelMicroStepLocations.FORWARD_END: [],
            PipelineParallelMicroStepLocations.BACKWARD_BEGIN: [],
            PipelineParallelMicroStepLocations.BACKWARD_END: [],
        }

    def register_hook(
        self, location: PipelineParallelMicroStepLocations, hook: Callable
    ):
        """
        Registers a hook function to be called at a specified pipeline parallel micro-step location.

        Args:
            location (PipelineParallelMicroStepLocations): The micro-step location where the hook should be registered.
            hook (Callable): The hook function to be registered. The function should accept the following optional keyword arguments:
                - input_tensor (paddle.Tensor): The input tensor to the current micro-step.
                - output_tensor (paddle.Tensor): The output tensor from the current micro-step.
                - input_tensor_grad (paddle.Tensor): The gradient of the input tensor.
                - output_tensor_grad (paddle.Tensor): The gradient of the output tensor.
                - step_id (paddle.Tensor): An identifier for the current step in the pipeline.

        Raises:
            AssertionError: If the specified location is not a valid micro-step location.
        """
        assert location in self.hooks, (
            f"Invalid location '{location}'. Valid locations are 'forward_begin', 'forward_end', 'backward_begin', or 'backward_end'."
        )
        self.hooks[location].append(hook)

    def on_location(
        self, location: PipelineParallelMicroStepLocations, **kwargs
    ):
        """
        Triggers all registered hooks at a specified pipeline parallel micro-step location.

        Args:
            location (PipelineParallelMicroStepLocations): The micro-step location where the hooks should be triggered.
            kwargs: Additional keyword arguments to be passed to the hook functions.

        Raises:
            AssertionError: If the specified location is not a valid micro-step location.
        """
        assert location in self.hooks, (
            f"Invalid location '{location}'. Valid locations are 'forward_begin', 'forward_end', 'backward_begin', or 'backward_end'."
        )
        for hook in self.hooks[location]:
            hook(**kwargs)


pipeline_parallel_callbacks_ = PipelineParallelMicroStepCallback()


# assume only the first stage and last stage need data, and data consumption is ordered
# to be replaced by real micro dataset from reader
class FakeMicroDataset:
    def __init__(
        self,
        data,
        is_first_stage,
        is_last_stage,
        acc_steps,
        micro_batch_size,
    ):
        self._data = data
        self._index = 0
        self._acc_steps = acc_steps
        self._is_first_stage = is_first_stage
        self._is_last_stage = is_last_stage
        self._micro_batch_size = micro_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= self._acc_steps:
            raise StopIteration
        assert self._is_first_stage or self._is_last_stage
        micro_batch_data = self._load_micro_batch(self._index)
        self._index += 1
        return micro_batch_data

    def _load_micro_batch(self, micro_step):
        inputs = self._data
        data = None
        label = None
        if self._is_first_stage:
            assert len(inputs) == 2, "length of input should be 2"
            data = self._load_micro_batch_impl(inputs[0], micro_step)

        if self._is_last_stage:
            assert len(inputs) == 2, "length of input should be 2"
            label = self._load_micro_batch_impl(inputs[1], micro_step)
        return (data, label)

    def _load_micro_batch_impl(self, inputs, micro_step):
        begin = micro_step * self._micro_batch_size
        end = begin + self._micro_batch_size

        if isinstance(inputs, tuple):
            output = []
            for data in inputs:
                if isinstance(data, list):
                    assert len(data) == self._acc_steps, (
                        f"length of data should be {self._acc_steps}, but it is {len(data)}"
                    )
                    output.append(
                        data[micro_step].detach()
                        if data[micro_step] is not None
                        else None
                    )
                elif data is not None:
                    self._check_data_valid(data)
                    output.append(data[begin:end, :].detach())
                else:
                    output.append(None)
            return tuple(output)
        elif isinstance(inputs, dict):
            output_dict = {}
            for key, data in inputs.items():
                if isinstance(data, list):
                    assert len(data) == self._acc_steps, (
                        f"length of data should be {self._acc_steps}, but it is {len(data)}"
                    )
                    output_dict[key] = (
                        data[micro_step].detach()
                        if data[micro_step] is not None
                        else None
                    )
                elif data is not None:
                    self._check_data_valid(data)
                    output_dict[key] = data[begin:end, :].detach()
                else:
                    output_dict[key] = None
            return output_dict
        elif isinstance(inputs, list):
            assert len(inputs) == self._acc_steps, (
                f"length of data should be {self._acc_steps}, but it is {len(inputs)}"
            )
            return inputs[micro_step].detach()
        elif inputs is not None:
            self._check_data_valid(inputs)
            return inputs[begin:end, :].detach()
        else:
            return None

    def _check_data_valid(self, data):
        batch_size = data.shape[0]
        assert self._micro_batch_size * self._acc_steps == batch_size, (
            "batch_size needs to be divisible by micro_batch_size. Currently, "
            f"batch_size = {batch_size}, micro_batch_size = {self._micro_batch_size}, accumulate_steps = {self._acc_steps}."
        )


class ParallelBase(ABC):
    @abstractmethod
    def forward_backward_pipeline(
        self,
        data,
        scaler=None,
        return_micro_batch_loss=False,
    ):
        pass


class NoPipelineParallel(nn.Layer, ParallelBase):
    def __init__(self, layers, strategy):
        assert isinstance(layers, PipelineLayer)
        super().__init__()
        self._layers = layers
        self._strategy = strategy
        self.micro_batch_size = self._strategy.pipeline_configs[
            "micro_batch_size"
        ]
        self.accumulate_steps = self._strategy.pipeline_configs[
            "accumulate_steps"
        ]
        self._delay_scale_loss = self._strategy.hybrid_configs[
            "pp_configs"
        ].delay_scale_loss
        self._dp_comm_overlap = False
        self._sharding_comm_overlap = False

        # store total loss of entire batch. It contains the loss of each micro batch in a list, then contains many loss_fn's list in total_loss.
        self.total_loss = None

        # default loss function index
        self.loss_fn_idx = 0

    def _check_micro_batch_data_valid(self, micro_batch_data):
        if isinstance(micro_batch_data, (tuple, list)):
            for data in micro_batch_data:
                self._check_micro_batch_data_valid(data)
        elif isinstance(micro_batch_data, dict):
            for value in micro_batch_data.values():
                self._check_micro_batch_data_valid(value)
        elif micro_batch_data is not None:
            assert isinstance(micro_batch_data, paddle.Tensor)

    def _prepare_training(self, data, optimizer, lr_scheduler):
        assert framework._dygraph_tracer()._has_grad, (
            "Please enable the generation of gradients."
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._layers.train()
        return data

    def _optimizer_step(self):
        if self._delay_scale_loss:
            for p in self._layers.parameters():
                if hasattr(p, "main_grad") and p.main_grad is not None:
                    assert p.grad is None
                    p.main_grad = p.main_grad.scale(1.0 / self.accumulate_steps)
                elif p.grad is not None:
                    p.grad = p.grad.scale(1.0 / self.accumulate_steps)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.clear_grad()

        if self.lr_scheduler:
            self.lr_scheduler.step()

    def forward_backward_pipeline(
        self,
        data,
        scaler=None,
        return_micro_batch_loss=False,
    ):
        self.scaler = scaler
        self.total_loss = None

        if isinstance(
            data,
            (PipelineDatasetPreprocessor, PaddlePipelineDatasetPreprocessor),
        ):
            data = data()

        if (not isinstance(data, tuple)) and (not isinstance(data, list)):
            micro_dataset = data
        else:
            micro_dataset = FakeMicroDataset(
                data,
                True,
                True,
                self.accumulate_steps,
                self.micro_batch_size,
            )

        loss_list = []
        for _ in range(self.accumulate_steps):
            # data prepare
            data_iter = next(micro_dataset)
            input_tensor = data_iter[0]
            label = data_iter[1]
            self._check_micro_batch_data_valid(input_tensor)
            self._check_micro_batch_data_valid(label)

            # forward
            output_tensor = self._layers.forward(input_tensor)

            # loss is loss_fn[loss_fn_idx]'s result
            loss = None
            # cal loss
            for idx, loss_fn in enumerate(self._layers._loss_fn):
                loss_tensor = loss_fn(output_tensor, label)
                assert isinstance(loss_tensor, paddle.Tensor), (
                    "Currently, loss_fn should obtain Paddle.Tensor dtype"
                )
                with paddle.amp.auto_cast(enable=False):
                    if self.accumulate_steps > 1 and not self._delay_scale_loss:
                        loss_tensor = loss_tensor / self.accumulate_steps
                if self.total_loss is None:
                    self.total_loss = []
                # when self.total_loss length is less than idx, append a new tensor
                if len(self.total_loss) <= idx:
                    self.total_loss.append([])

                self.total_loss[idx].append(loss_tensor.detach())

                if idx == self.loss_fn_idx:
                    loss = loss_tensor

            # backward
            with paddle.amp.auto_cast(enable=False):
                if self.scaler:
                    paddle.autograd.backward(self.scaler.scale(loss))
                else:
                    paddle.autograd.backward(loss)

            assert self.total_loss is not None, (
                "train_batch() in last stage should obtain valid loss"
            )

        losses = []
        with paddle.amp.auto_cast(enable=False):
            for idx in range(len(self._layers._loss_fn)):
                self.total_loss[idx] = paddle.to_tensor(self.total_loss[idx])
                if not return_micro_batch_loss:
                    # TODO(shenliang03): it will use mean/sum to calculate loss
                    tmp = paddle.zeros_like(self.total_loss[idx][0])
                    for loss in self.total_loss[idx]:
                        tmp += loss.detach()
                    if not self._delay_scale_loss:
                        losses.append(tmp)
                    else:
                        losses.append(tmp / self.accumulate_steps)
                else:
                    losses.append(self.total_loss[idx].detach())
        return losses[0] if len(losses) == 1 else losses

    def train_batch(
        self,
        data,
        optimizer,
        lr_scheduler=None,
        scaler=None,
        loss_fn_idx=0,
        return_micro_batch_loss=False,
    ):
        data = self._prepare_training(data, optimizer, lr_scheduler)

        # check loss_fn_idx is valid and loss_fn exists
        assert (
            loss_fn_idx in range(len(self._layers._loss_fn))
            and self._layers._loss_fn[loss_fn_idx] is not None
        ), f"loss function {loss_fn_idx} should exist to compute loss"
        self.loss_fn_idx = loss_fn_idx

        # no pipeline parallel
        train_loss = self.forward_backward_pipeline(
            data, scaler, return_micro_batch_loss=return_micro_batch_loss
        )

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss


class PipelineParallel(nn.Layer, ParallelBase):
    def __init__(self, layers, hcg, strategy):
        assert isinstance(layers, PipelineLayer)
        super().__init__()
        self._layers = layers
        self._strategy = strategy
        self._hcg = hcg

        if not isinstance(layers, PipelineLayer):
            raise TypeError(
                "The Layer should be a derived class of PipelineLayer."
            )
        self.use_data_parallel = self._hcg.get_data_parallel_world_size() > 1
        self.use_model_parallel = self._hcg.get_model_parallel_world_size() > 1
        self.use_sep_parallel = self._hcg.get_sep_parallel_world_size() > 1
        self.use_sharding_parallel = (
            self._hcg.get_sharding_parallel_world_size() > 1
        )
        self.use_moe_sharding_parallel = (
            self._hcg.get_moe_sharding_parallel_world_size() > 1
        )

        self.use_dict_in_pp = True

        self.total_loss = None

        self.micro_batch_size = self._strategy.pipeline_configs[
            "micro_batch_size"
        ]
        self.accumulate_steps = self._strategy.pipeline_configs[
            "accumulate_steps"
        ]
        # If sent tensor are not the same from different hosts,
        # they shouldn't been sent partially and then concatenated as a whole tensor.
        self._enable_partial_send_recv = self._strategy.pipeline_configs[
            "enable_partial_send_recv"
        ]
        self._using_cache = self._strategy.pipeline_configs["p2p_cache_shape"]

        self.num_stages = self._hcg.get_pipe_parallel_world_size()
        self.stage_id = self._hcg.get_stage_id()
        self.global_rank = self._hcg.get_global_rank()
        self.pp_group = self._hcg.get_pipe_parallel_group()

        self.dp_group = self._hcg.get_data_parallel_group()

        # fused sep and dp
        if self.use_sep_parallel:
            self.dp_group = self._hcg.get_dp_sep_parallel_group()

        self.sharding_group = self._hcg.get_sharding_parallel_group()

        self._virtual_pp_world_size = None
        self._virtual_pp_rank = None
        self._real_pp_world_size = self.num_stages
        self._real_pp_rank = self.stage_id

        self._delay_scale_loss = self._strategy.hybrid_configs[
            "pp_configs"
        ].delay_scale_loss
        # TODO(PP Dev): support dp_comm_overlap without use_main_grad training.
        # This combination will trigger inplace check error during `reshape_` in function `_split_tensors`.
        self._dp_comm_overlap = self._strategy.hybrid_configs[
            "pp_configs"
        ].dp_comm_overlap
        self._sharding_comm_overlap = self._strategy.hybrid_configs[
            "pp_configs"
        ].sharding_comm_overlap
        self._enable_timer = self._strategy.hybrid_configs[
            "pp_configs"
        ].enable_timer
        self._release_gradients = self._strategy.hybrid_configs[
            "pp_configs"
        ].release_gradients

        self._sharding_split_param = self._strategy.hybrid_configs[
            "sharding_configs"
        ].split_param

        self._overlap_p2p_comm = self._strategy.hybrid_configs[
            "pp_configs"
        ].overlap_p2p_comm

        self._clear_every_step_cache = self._strategy.hybrid_configs[
            "pp_configs"
        ].clear_every_step_cache

        self._use_batch_p2p_comm = self._strategy.hybrid_configs[
            "pp_configs"
        ].use_batch_p2p_comm

        self._dynamic_shape = self._strategy.hybrid_configs[
            "pp_configs"
        ].enable_dynamic_shape
        logger.info(
            f"Pipeline scheduler is in dynamic_shape mode={self._dynamic_shape}"
        )

        if self._use_batch_p2p_comm and self._overlap_p2p_comm:
            warnings.warn(
                "non_batch_p2p_comm should be enabled when overlap_p2p_comm is activated, setting non_batch_p2p_comm=True."
            )
            self._use_batch_p2p_comm = False

        logger.info(
            f"dp_comm_overlap {self._dp_comm_overlap}; \
            sharding_comm_overlap {self._sharding_comm_overlap}; \
            sharding_split_param {self._sharding_split_param};"
        )

        if self._dp_comm_overlap:
            assert self.use_data_parallel and self.num_stages > 1

        if self._sharding_comm_overlap:
            assert self.use_sharding_parallel and self.num_stages > 1

        assert not (self._dp_comm_overlap and self._sharding_comm_overlap), (
            "Cannot use dp pp overlap and sharding pp overlap at the same time."
        )

        self._chunk_2_comm_buffers = defaultdict(list)
        self._comm_overlap = (
            self._dp_comm_overlap or self._sharding_comm_overlap
        )

        if self._enable_timer:
            if not timer.is_timer_initialized():
                timer.set_timers()
            self.timers = timer.get_timers()

        p2p.initialize_p2p_groups(
            hcg,
            self._enable_partial_send_recv,
            self._enable_timer,
        )

        # construct pipeline meta info
        self._p2p_helper = p2p.P2pHelper(
            self._using_cache, dynamic_shape=self._dynamic_shape
        )

        self.global_rank = self._hcg.get_global_rank()
        self.micro_batch_id = 0

        # default loss function index
        self.loss_fn_idx = 0

        self._compute_loss = True
        self._return_host_tensor = False
        self.callbacks = pipeline_parallel_callbacks_

        logger.info(
            f"Pipeline Info -- num_stages: {self.num_stages}, stage_id: {self.stage_id}"
        )

        if self.use_model_parallel:
            logger.info("start broadcast mp parameters")
            broadcast_mp_parameters(self._layers, self._hcg)

        if self.use_sep_parallel:
            logger.info("start broadcast sep parameters")
            broadcast_sep_parameters(self._layers, self._hcg)

        if self.use_sharding_parallel:
            logger.info("start broadcast sharding parameters")
            broadcast_sharding_parameters(self._layers, self._hcg)

        if self.use_data_parallel:
            logger.info("start broadcast dp parameters")
            broadcast_dp_parameters(self._layers, self._hcg)

        if self.use_moe_sharding_parallel:
            logger.info("start broadcast moe_sharding parameters")
            broadcast_moe_sharding_parameters(self._layers, self._hcg)

        if self._dp_comm_overlap:
            self.register_allreduce_overlap_hook(
                self._layers, self.dp_group, self.accumulate_steps, True
            )

        self.processed_steps = 0

        self._init_user_hooks()
        # only support user hooks during training
        self.user_hooks_enabled = True

    def register_hook(
        self, location: PipelineParallelMicroStepLocations, hook: Callable
    ):
        self.callbacks.register_hook(location, hook)

    def _init_user_hooks(self):
        self._init_user_forward_backward_hooks()
        self._init_user_bubble_hooks()

    def _init_user_forward_backward_hooks(self):
        # initialize forward hooks
        self.forward_hooks = PipelineHook()
        self.forward_hooks.set_hooks_capacity(
            (
                self._virtual_pp_world_size
                if self._virtual_pp_world_size is not None
                else 1
            )
            * self.accumulate_steps
        )

        # initialize backward hooks
        self.backward_hooks = PipelineHook()
        self.backward_hooks.set_hooks_capacity(
            (
                self._virtual_pp_world_size
                if self._virtual_pp_world_size is not None
                else 1
            )
            * self.accumulate_steps
        )

    def _init_user_bubble_hooks(self):
        # (TODO:gexiao) support bubble hooks if needed
        # Bubble hooks are required for advanced pipeline parallelism features, such as custom communication or computation overlap during pipeline bubbles.
        # Planned for implementation in future releases after design review.
        self.bubble_hooks = None
        # self.bubble_hooks = PipelineHook()
        # self.bubble_hooks.set_hooks_capacity(2 * self.num_stages - 2)

    def _reset_user_hooks_status(self):
        if self.bubble_hooks:
            self.bubble_hooks.reset_current_id()
        if self.forward_hooks:
            self.forward_hooks.reset_current_id()
        if self.backward_hooks:
            self.backward_hooks.reset_current_id()

    def _check_user_hooks_status_at_step_end(self):
        if not self.user_hooks_enabled:
            return
        expected_bubble_step = 2 * self.num_stages - 2
        expected_forward_step = (
            self._virtual_pp_world_size
            if self._virtual_pp_world_size is not None
            else 1
        ) * self.accumulate_steps
        expected_backward_step = (
            self._virtual_pp_world_size
            if self._virtual_pp_world_size is not None
            else 1
        ) * self.accumulate_steps

        if self.bubble_hooks:
            assert (self.bubble_hooks.current_id) == expected_bubble_step, (
                f"bubble hooks status is not correct, current id is {self.bubble_hooks.current_id}, expected id is {expected_bubble_step}"
            )
        if self.forward_hooks:
            assert (self.forward_hooks.current_id) == expected_forward_step, (
                f"forward hooks status is not correct, current id is {self.forward_hooks.current_id}, expected id is {expected_forward_step}"
            )
        if self.backward_hooks:
            assert (self.backward_hooks.current_id) == expected_backward_step, (
                f"backward hooks status is not correct, current id is {self.backward_hooks.current_id}, expected id is {expected_backward_step}"
            )

    def bw_hook_func(self, buffer, param):
        @paddle.autograd.no_grad()
        def fused_allreduce(*_):
            buffer.add_grad(param)

        return fused_allreduce

    def register_allreduce_overlap_hook(
        self, model, comm_group, acc_steps, dp, group_size=128 * 1024 * 1024
    ):
        # register hook
        self.fused_gradient(model, comm_group, acc_steps, dp, group_size)
        for _, buffers in self._chunk_2_comm_buffers.items():
            for buffer in buffers:
                for param in buffer._params:
                    param._register_backward_hook(
                        self.bw_hook_func(buffer, param)
                    )

    def timer_printer(self):
        if not self._enable_timer:
            return
        all_flag_names = self.timers.timers.keys()
        self.timers.log(all_flag_names)

    def register_sharding_comm_overlap_hook(self, optimizer):
        """for delayed hook register until we get optimizer"""
        assert isinstance(optimizer, HybridParallelOptimizer), (
            "optimizer should be HybridParallelOptimizer subclass."
        )
        self.optimizer = optimizer
        if self._sharding_comm_overlap and len(self._chunk_2_comm_buffers) == 0:
            self.register_allreduce_overlap_hook(
                self._layers, self.sharding_group, self.accumulate_steps, False
            )

    def _prepare_training(self, data, optimizer, lr_scheduler):
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        assert isinstance(optimizer, HybridParallelOptimizer), (
            "optimizer should be HybridParallelOptimizer subclass."
        )

        assert framework._dygraph_tracer()._has_grad, (
            "Please enable the generation of gradients."
        )

        if self.is_pipeline_first_stage(
            ignore_virtual=True
        ) or self.is_pipeline_last_stage(ignore_virtual=True):
            assert data is not None, (
                "For the first and the last stage, the data must be set."
            )
        else:
            data = None

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._layers.train()
        self.register_sharding_comm_overlap_hook(optimizer)

        return data

    def _wrap_data(self, data):
        """
        for backward compatibility, wrap data to Fake FakeMicroDataset if it is of type list or tuple
        """
        if isinstance(
            data,
            (PipelineDatasetPreprocessor, PaddlePipelineDatasetPreprocessor),
        ):
            data = data()

        if (not isinstance(data, tuple)) and (not isinstance(data, list)):
            return data

        micro_dataset = FakeMicroDataset(
            data,
            self.is_pipeline_first_stage(ignore_virtual=True),
            self.is_pipeline_last_stage(ignore_virtual=True),
            self.accumulate_steps,
            self.micro_batch_size,
        )
        return micro_dataset

    def train_batch(
        self,
        data,
        optimizer,
        lr_scheduler=None,
        scaler=None,
        loss_fn_idx=0,
        return_micro_batch_loss=False,
    ):
        data = self._prepare_training(data, optimizer, lr_scheduler)

        # check loss_fn_idx is valid and loss_fn exists
        assert (
            loss_fn_idx in range(len(self._layers._loss_fn))
            and self._layers._loss_fn[loss_fn_idx] is not None
        ), f"loss function {loss_fn_idx} should exist to compute loss"
        self.loss_fn_idx = loss_fn_idx

        # 1f1b scheduler for pipeline parallel
        train_loss = self.forward_backward_pipeline(
            data, scaler, return_micro_batch_loss=return_micro_batch_loss
        )

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss

    def eval_batch(
        self, data, compute_loss=False, loss_fn_idx=0, return_host_tensor=False
    ):
        self.user_hooks_enabled = False
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        self._layers.eval()
        origin_compute_loss = self._compute_loss
        self._compute_loss = compute_loss
        origin_return_host_tensor = self._return_host_tensor
        self._return_host_tensor = return_host_tensor

        # store data id for micro_batch
        self.micro_batch_id = 0

        # store total loss of entire batch
        self.total_loss = None

        # check loss_fn_idx is valid and loss_fn exists
        assert (
            loss_fn_idx in range(len(self._layers._loss_fn))
            and self._layers._loss_fn[loss_fn_idx] is not None
        ), f"loss function {loss_fn_idx} should exist to compute loss"
        self.loss_fn_idx = loss_fn_idx

        startup_steps = self.num_stages - self.stage_id - 1
        startup_steps = min(startup_steps, self.accumulate_steps)
        steady_steps = self.accumulate_steps - startup_steps

        output_buffers = []

        # convert to micro dataset
        micro_dataset = self._wrap_data(data)

        for step_id in range(startup_steps):
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            output_tensor, _, _ = self._forward_step(
                input_tensor, micro_dataset, step_id=None
            )
            self._p2p_helper.send_forward(
                output_tensor,
                self.is_pipeline_last_stage(),
                skip_check_meta=True,
                batch_p2p_comm=self._use_batch_p2p_comm,
            )
            if not self.is_pipeline_last_stage():
                self._release_output(output_tensor)
            else:
                self._offload_tensors(output_tensor)

            output_buffers.append(output_tensor)

        if steady_steps > 0:
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

        for i in range(steady_steps):
            last_iter = i == (steady_steps - 1)

            output_tensor, _, _ = self._forward_step(
                input_tensor, micro_dataset, step_id=None
            )
            self._p2p_helper.send_forward(
                output_tensor,
                self.is_pipeline_last_stage(),
                skip_check_meta=True,
                batch_p2p_comm=self._use_batch_p2p_comm,
            )
            if not self.is_pipeline_last_stage():
                self._release_output(output_tensor)
            else:
                self._offload_tensors(output_tensor)

            output_buffers.append(output_tensor)

            if not last_iter:
                input_tensor = self._p2p_helper.recv_forward(
                    self.is_pipeline_first_stage(),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )

        if self._compute_loss:
            train_loss = self._broadcast_final_loss()
        else:
            train_loss = output_buffers

        self._compute_loss = origin_compute_loss
        self._return_host_tensor = origin_return_host_tensor
        return train_loss

    def register_bubble_pipeline_parallel_hook(
        self, location: int, hook: Callable
    ):
        """
        Registering bubble hooks for pipeline parallelism.
        """
        if not self.bubble_hooks:
            raise ValueError("Bubble hooks are not supported yet.")
        self.bubble_hooks.register_hook(location, hook)

    def register_forward_pipeline_parallel_hook(
        self, location: int, hook: Callable
    ):
        """
        Registering forward hooks for pipeline parallelism.
        """
        if not self.forward_hooks:
            raise ValueError("Forward hooks are not supported yet.")
        self.forward_hooks.register_hook(location, hook)

    def register_backward_pipeline_parallel_hook(
        self, location: int, hook: Callable
    ):
        """
        Registering backward hooks for pipeline parallelism.
        """
        if not self.backward_hooks:
            raise ValueError("Backward hooks are not supported yet.")
        self.backward_hooks.register_hook(location, hook)

    @property
    def bubble_pipeline_parallel_hook_capacity(self):
        capacity = 0
        if self.bubble_hooks:
            capacity = self.bubble_hooks.hooks_capacity
        return capacity

    @property
    def forward_pipeline_parallel_hook_capacity(self):
        capacity = 0
        if self.forward_hooks:
            capacity = self.forward_hooks.hooks_capacity
        return capacity

    @property
    def backward_pipeline_parallel_hook_capacity(self):
        capacity = 0
        if self.backward_hooks:
            capacity = self.backward_hooks.hooks_capacity
        return capacity

    def is_pipeline_first_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self._virtual_pp_world_size is not None:
                assert self._virtual_pp_rank is not None
                if self._virtual_pp_rank != 0:
                    return False
        assert self._real_pp_rank is not None
        return self._real_pp_rank == 0

    def is_pipeline_last_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self._virtual_pp_world_size is not None:
                assert self._virtual_pp_rank is not None
                if self._virtual_pp_rank != (self._virtual_pp_world_size - 1):
                    return False
        assert self._real_pp_rank is not None
        assert self._real_pp_world_size is not None
        return self._real_pp_rank == (self._real_pp_world_size - 1)

    def set_virtual_pipeline_rank(self, rank):
        self._virtual_pp_rank = rank

    def fused_gradient(
        self, model, comm_group, acc_steps, dp, group_size=128 * 1024 * 1024
    ):
        if model.get_num_virtual_stages() > 1:
            models = model.get_model_chunks()
        else:
            models = [model]

        act = get_action(dp, self._sharding_split_param)

        if act == HOOK_ACTION.REDUCE:
            assert hasattr(self, "optimizer")
            assert hasattr(self.optimizer, "_param2rank")
            _param2rank = self.optimizer._param2rank

        for chunk_idx, model in enumerate(models):
            # For virtual pipeline. Will separate parameters in different chunk into
            # different groups to get the best performance.

            fused_parameter_group = {}
            parameter_list = [
                p for p in model.parameters() if not p.stop_gradient
            ]
            if len(parameter_list) < 1:
                return

            if act == HOOK_ACTION.REDUCE:
                # Sort parameters for sharding, since they have different dst rank
                for p in parameter_list:
                    assert p.name in _param2rank
                    dst_rank = _param2rank[p.name]
                    if dst_rank in fused_parameter_group:
                        fused_parameter_group[dst_rank].append(p)
                    else:
                        fused_parameter_group[dst_rank] = [p]
            else:
                fused_parameter_group[-1] = parameter_list

            for dst in fused_parameter_group:
                parameter_list = fused_parameter_group[dst]
                if act == HOOK_ACTION.REDUCE:
                    # parse the relative dst rank to absolute dst rank for sharding
                    dst = comm_group.ranks[dst]
                var_groups = assign_group_by_size(parameter_list, group_size)

                for group_idx, parameters in var_groups.items():
                    buffer = FusedCommBuffer(
                        group_idx,
                        parameters,
                        comm_group,
                        acc_steps,
                        act,
                        dst,
                        release_grads=self._release_gradients,
                    )
                    self._chunk_2_comm_buffers[chunk_idx].append(buffer)

        return self._chunk_2_comm_buffers

    def forward_backward_pipeline(
        self,
        data,
        scaler=None,
        return_micro_batch_loss=False,
    ):
        # use the 1f1b scheduling strategy.
        # this strategy is inspired by:
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/schedules.py
        self._reset_user_hooks_status()
        # no _forward_only mode
        self.user_hooks_enabled = True

        if self.processed_steps < g_profile_pipeline_details_steps:
            profile_pipeline_details(
                "[Pipeline details] Start_forward_backward_pipeline"
            )

        self.scaler = scaler

        # store total loss of entire batch
        self.total_loss = None

        # store data id for micro_batch
        self.micro_batch_id = 0

        startup_steps = self.num_stages - self.stage_id - 1
        startup_steps = min(startup_steps, self.accumulate_steps)
        steady_steps = self.accumulate_steps - startup_steps

        input_buffers = []
        output_buffers = []

        micro_dataset = self._wrap_data(data)

        for step_id in range(startup_steps):
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            input_tensor_dict, use_dict = tuple_to_dict_helper(input_tensor)

            output_tensor, _, _ = self._forward_step(
                input_tensor=input_tensor_dict if use_dict else input_tensor,
                micro_dataset=micro_dataset,
                step_id=step_id,
            )

            # convert dict to tuple whose tensor element has a key attribution
            output_tensor_tuple = dict_to_tuple_helper(output_tensor)

            # fwd output dict -> send tuple
            self._p2p_helper.send_forward(
                output_tensor=output_tensor_tuple,
                pp_last_stage=self.is_pipeline_last_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor_tuple)

            if not self.is_pipeline_last_stage():
                self._release_output(output_tensor_tuple)

        if steady_steps > 0:
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

        for i in range(steady_steps):
            last_iter = i == (steady_steps - 1)

            input_tensor_dict, use_dict = tuple_to_dict_helper(input_tensor)

            output_tensor, _, _ = self._forward_step(
                input_tensor=input_tensor_dict if use_dict else input_tensor,
                micro_dataset=micro_dataset,
                step_id=startup_steps + i,
            )

            output_tensor_tuple = dict_to_tuple_helper(output_tensor)
            # NOTE: `send_forward_recv_backward` is intentionally unused to
            # prevent hanging bugs in dynamic shape mode.
            self._p2p_helper.send_forward(
                output_tensor_tuple,
                self.is_pipeline_last_stage(ignore_virtual=True),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            output_tensor_grad = self._p2p_helper.recv_backward(
                self.is_pipeline_last_stage(ignore_virtual=True),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor_tuple)

            if not self.is_pipeline_last_stage():
                self._release_output(output_tensor_tuple)

            input_tensor, output_tensor = (
                input_buffers.pop(0),
                output_buffers.pop(0),
            )

            input_tensor_grad = self._backward_step(
                input_tensor, output_tensor, output_tensor_grad, step_id=i
            )

            if last_iter:
                input_tensor = None
                self._p2p_helper.send_backward(
                    input_tensor_grad,
                    self.is_pipeline_first_stage(),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
            else:
                # NOTE: `send_backward_recv_forward` is intentionally unused to
                # prevent hanging bugs in dynamic shape mode.
                input_tensor = self._p2p_helper.recv_forward(
                    self.is_pipeline_first_stage(ignore_virtual=True),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )

                self._p2p_helper.send_backward(
                    input_tensor_grad,
                    self.is_pipeline_first_stage(ignore_virtual=True),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )

        for i in range(startup_steps):
            input_tensor = input_buffers.pop(0)
            output_tensor = output_buffers.pop(0)

            output_tensor_grad = self._p2p_helper.recv_backward(
                self.is_pipeline_last_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

            input_tensor_grad = self._backward_step(
                input_tensor,
                output_tensor,
                output_tensor_grad,
                step_id=steady_steps + i,
            )

            self._p2p_helper.send_backward(
                input_tensor_grad,
                self.is_pipeline_first_stage(),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )

        if self._comm_overlap:
            assert len(self._chunk_2_comm_buffers) > 0, (
                "comm buffers should be created"
            )
            for _, buffers in self._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer.scale_grads()

        if self._enable_timer:
            self.timers("allreduce_shared_weight_gradients").start()
        self._layers.allreduce_shared_weight_gradients()
        if self._enable_timer:
            self.timers("allreduce_shared_weight_gradients").stop()
            self.timers("broadcast_final_loss").start()
        with paddle.amp.auto_cast(enable=False):
            train_loss = self._broadcast_final_loss(return_micro_batch_loss)
        if self._enable_timer:
            self.timers("broadcast_final_loss").stop()

        if self._clear_every_step_cache:
            self._p2p_helper.clear_meta_cache()

        self.timer_printer()

        if self.processed_steps < g_profile_pipeline_details_steps:
            profile_pipeline_details(
                "[Pipeline details] End_forward_backward_pipeline"
            )
        self.processed_steps += 1
        self._check_user_hooks_status_at_step_end()
        return train_loss

    def _maybe_loss_compute(
        self, output_tensor, micro_dataset, overlap_schedule_mode=False
    ):
        backward_loss_tensor = None
        backward_loss_fn_node = None
        loss_fn_node = None

        if self.is_pipeline_last_stage():
            # train calculate loss for train
            if self._compute_loss:
                assert self._layers._loss_fn[self.loss_fn_idx] is not None, (
                    "loss function should exist to compute loss"
                )
                labels = next(micro_dataset)[1]
                self._check_micro_batch_data_valid(labels)
                for idx, loss_fn in enumerate(self._layers._loss_fn):
                    if overlap_schedule_mode:
                        loss_fn_node = loss_fn.build_schedule_node()
                        loss_fn_node.labels = labels
                        if (
                            self.accumulate_steps > 1
                            and not self._delay_scale_loss
                        ):
                            loss_fn_node.scale_loss_factor = (
                                self.accumulate_steps
                            )
                        loss_tensor = loss_fn_node.forward(output_tensor)
                    else:
                        loss_tensor = loss_fn(output_tensor, labels)
                        assert isinstance(loss_tensor, paddle.Tensor), (
                            "Currently, loss_fn should obtain Paddle.Tensor dtype"
                        )

                        with paddle.amp.auto_cast(enable=False):
                            if (
                                self.accumulate_steps > 1
                                and not self._delay_scale_loss
                            ):
                                loss_tensor = (
                                    loss_tensor / self.accumulate_steps
                                )

                    if self.total_loss is None:
                        self.total_loss = []
                    # when self.total_loss length is less than idx, append a new tensor
                    if len(self.total_loss) <= idx:
                        self.total_loss.append([])
                    self.total_loss[idx].append(loss_tensor.detach())

                    if idx == self.loss_fn_idx:
                        backward_loss_tensor = loss_tensor
                        backward_loss_fn_node = loss_fn_node
        return backward_loss_tensor, backward_loss_fn_node

    def _forward_step(
        self,
        input_tensor,
        micro_dataset,
        chunk_id=None,
        step_id=None,
        overlap_schedule_mode=False,
    ):
        if self.user_hooks_enabled:
            self.forward_hooks.run_hook()
        if self.processed_steps < g_profile_pipeline_details_steps:
            profile_pipeline_details(
                f"[Pipeline details] Before_forward_step_chunk_{chunk_id}_step_{step_id}"
            )
        if self._enable_timer:
            self.timers("forward_step").start()
        if self.is_pipeline_first_stage():
            input_tensor = next(micro_dataset)[0]
            self._check_micro_batch_data_valid(input_tensor)

        assert chunk_id is None or isinstance(chunk_id, int)

        self.callbacks.on_location(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN,
            input_tensor=input_tensor,
            step_id=step_id,
        )

        schedule_chunk = None
        if overlap_schedule_mode:
            schedule_chunk = self._layers.get_schedule_chunk(chunk_id=chunk_id)
            output_tensor = schedule_chunk.forward(input_tensor)
        else:
            output_tensor = self._layers.forward(
                input_tensor, chunk_id=chunk_id
            )

        self.callbacks.on_location(
            PipelineParallelMicroStepLocations.FORWARD_END,
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            step_id=step_id,
        )

        backward_loss_tensor, backward_loss_fn_node = self._maybe_loss_compute(
            output_tensor, micro_dataset, overlap_schedule_mode
        )

        if self.is_pipeline_first_stage() or self.is_pipeline_last_stage():
            # Only increase micro batch id at virtual first/last pp stage.
            # The micro batch id is used to load data, therefore, only increase it when load data.
            self.micro_batch_id += 1
        if self._enable_timer:
            self.timers("forward_step").stop()
        if self.processed_steps < g_profile_pipeline_details_steps:
            profile_pipeline_details(
                f"[Pipeline details] After_forward_step_chunk_{chunk_id}_step_{step_id}"
            )
        if self.is_pipeline_last_stage() and self._compute_loss:
            return backward_loss_tensor, schedule_chunk, backward_loss_fn_node
        return output_tensor, schedule_chunk, backward_loss_fn_node

    def _backward_step(
        self,
        input_tensor,
        output_tensor,
        output_tensor_grad,
        chunk_id=None,
        step_id=None,
        overlap_schedule_mode=False,
        schedule_chunk=None,
        loss_fn_node=None,
    ):
        if self.user_hooks_enabled:
            self.backward_hooks.run_hook()
        if self._enable_timer:
            self.timers("backward_step").start()
        if self.processed_steps < g_profile_pipeline_details_steps:
            profile_pipeline_details(
                f"[Pipeline details] Before_backward_step_chunk_{chunk_id}_step_{step_id}"
            )
        with paddle.amp.auto_cast(enable=False):
            self.callbacks.on_location(
                PipelineParallelMicroStepLocations.BACKWARD_BEGIN,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                output_tensor_grad=output_tensor_grad,
                step_id=step_id,
            )
            if self.is_pipeline_last_stage():
                assert output_tensor_grad is None
                if overlap_schedule_mode:
                    assert (
                        loss_fn_node is not None and schedule_chunk is not None
                    ), (
                        "loss_fn_node and schedule_chunk should not be None in overlap_schedule_mode"
                    )
                    input_tensor_grad = loss_fn_node.backward(
                        scaler=self.scaler
                    )
                    input_tensor_grad = schedule_chunk.backward(
                        input_tensor_grad
                    )
                else:
                    # In align mode, we scale the grad directly after forward
                    if paddle.distributed.in_auto_parallel_align_mode():
                        output_tensor = output_tensor / _get_align_mode_scale()
                    if self.scaler:
                        paddle.autograd.backward(
                            self.scaler.scale(output_tensor)
                        )
                    else:
                        paddle.autograd.backward(output_tensor)
            else:
                if isinstance(output_tensor, tuple):
                    outputs = [t for t in output_tensor if not t.stop_gradient]
                    assert len(outputs) == len(output_tensor_grad)
                    grad_tensors = list(output_tensor_grad)
                else:
                    outputs = [output_tensor]
                    grad_tensors = [output_tensor_grad]

                if overlap_schedule_mode:
                    assert schedule_chunk is not None, (
                        "schedule_chunk should not be None in overlap_schedule_mode"
                    )
                    input_tensor_grad = schedule_chunk.backward(grad_tensors)
                else:
                    paddle.autograd.backward(
                        tensors=outputs,
                        grad_tensors=grad_tensors,
                    )

            if not overlap_schedule_mode:
                # Extract input_tensor_grad from the input tensor. In overlap_schedule_mode,
                # the input_tensor_grad is extracted inside the schedule_chunk.
                input_tensor_grad = None
                if input_tensor is not None:
                    if isinstance(input_tensor, tuple):
                        input_tensor_grad = tuple(
                            [
                                t.grad
                                for t in input_tensor
                                if not t.stop_gradient
                            ]
                        )
                    else:
                        input_tensor_grad = input_tensor.grad
            if self._enable_timer:
                self.timers("backward_step").stop()
            self.callbacks.on_location(
                PipelineParallelMicroStepLocations.BACKWARD_END,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                input_tensor_grad=input_tensor_grad,
                output_tensor_grad=output_tensor_grad,
                step_id=step_id,
            )

            if self.processed_steps < g_profile_pipeline_details_steps:
                profile_pipeline_details(
                    f"[Pipeline details] After_backward_step_chunk_{chunk_id}_step_{step_id}"
                )
            return input_tensor_grad

    def _check_micro_batch_data_valid(self, micro_batch_data):
        if isinstance(micro_batch_data, (tuple, list)):
            for data in micro_batch_data:
                self._check_micro_batch_data_valid(data)
        elif isinstance(micro_batch_data, dict):
            for value in micro_batch_data.values():
                self._check_micro_batch_data_valid(value)
        elif micro_batch_data is not None:
            assert isinstance(micro_batch_data, paddle.Tensor)

    def _broadcast_final_loss(self, return_micro_batch_loss=False):
        # Since the last backward run in interleave will set the virtual rank to 0,
        # here we need to check last stage ignoring virtual stage.
        if self.is_pipeline_last_stage(ignore_virtual=True):
            assert self.total_loss is not None, (
                "train_batch() in last stage should obtain valid loss"
            )
            losses = []
            for idx in range(len(self._layers._loss_fn)):
                self.total_loss[idx] = paddle.to_tensor(self.total_loss[idx])
                if not return_micro_batch_loss:
                    # TODO(shenliang03): it will use mean/sum to calculate loss
                    tmp = paddle.zeros_like(self.total_loss[idx][0])
                    for loss in self.total_loss[idx]:
                        tmp += loss.detach()
                    if not self._delay_scale_loss:
                        losses.append(tmp)
                    else:
                        losses.append(tmp / self.accumulate_steps)
                else:
                    losses.append(self.total_loss[idx].detach())

            for idx in range(len(self._layers._loss_fn)):
                is_fp32 = (
                    paddle.full([], 1, "int64")
                    if losses[idx].dtype == paddle.float32
                    else paddle.full([], 0, "int64")
                )
                paddle.distributed.broadcast(
                    is_fp32,
                    src=self.global_rank,
                    sync_op=True,
                    group=self.pp_group,
                )
                paddle.distributed.broadcast(
                    losses[idx],
                    src=self.global_rank,
                    sync_op=True,
                    group=self.pp_group,
                )
        else:
            losses = []
            for idx in range(len(self._layers._loss_fn)):
                is_fp32 = paddle.full([], 1, "int64")
                paddle.distributed.broadcast(
                    is_fp32,
                    src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                    sync_op=True,
                    group=self.pp_group,
                )
                if return_micro_batch_loss:
                    loss_shape = [self.accumulate_steps]
                else:
                    loss_shape = [1]
                losses.append(
                    paddle.zeros(shape=loss_shape, dtype="float32")
                    if is_fp32.item()
                    else paddle.zeros(shape=loss_shape, dtype="float16")
                )
                paddle.distributed.broadcast(
                    losses[idx],
                    src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                    sync_op=True,
                    group=self.pp_group,
                )
        return losses[0] if len(losses) == 1 else losses

    def _optimizer_step(self):
        if self._delay_scale_loss:
            for p in self._layers.parameters():
                if hasattr(p, "main_grad") and p.main_grad is not None:
                    assert p.grad is None
                    p.main_grad = p.main_grad.scale(1.0 / self.accumulate_steps)
                elif p.grad is not None:
                    p.grad = p.grad.scale(1.0 / self.accumulate_steps)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self._release_gradients:
            self.optimizer.clear_grad(set_to_zero=False)
            for _, buffers in self._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer._clear_grad_storage()
        else:
            self.optimizer.clear_grad()

        if self.lr_scheduler:
            self.lr_scheduler.step()

    def _offload_tensors(self, output_tensor):
        if not self._return_host_tensor:
            return
        if isinstance(output_tensor, (tuple, list)):
            for t in output_tensor:
                if not isinstance(t, paddle.Tensor):
                    continue
                host_tensor = (
                    t.pin_memory() if hasattr(t, "pin_memory") else t.cpu()
                )
                host_tensor._share_buffer_to(t)
        else:
            if not isinstance(output_tensor, paddle.Tensor):
                return
            host_tensor = (
                output_tensor.pin_memory()
                if hasattr(output_tensor, "pin_memory")
                else output_tensor.cpu()
            )
            host_tensor._share_buffer_to(output_tensor)

    def _release_output(self, output):
        def can_free(t):
            return (
                t is not None
                and isinstance(t, paddle.Tensor)
                and t._is_initialized()
                and (t.inplace_version == 0 or getattr(t, "pp_can_free", False))
            )

        if isinstance(output, (tuple, list)):
            for t in output:
                if can_free(t):
                    t._clear_dataptr()

        elif can_free(output):
            output._clear_dataptr()
