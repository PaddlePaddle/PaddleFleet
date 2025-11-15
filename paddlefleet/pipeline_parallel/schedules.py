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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import paddle
from paddle.distributed import fleet

from paddlefleet import parallel_state


def pipeline_dualpipev():
    # DualPipeVParallel
    pass


def pipeline_1f1b():
    # PipelineParallel
    pass


def pipeline_1f1b_with_interleave():
    # PipelineParallelWithInterleave
    pass


def pipeline_fthenb_with_interleave_in_balanced_memory():
    # VPPFhenBInBalancedMemory
    pass


def pipeline_fthenb_with_interleave():
    # PipelineParallelWithInterleaveFthenB
    pass


def no_pipeline(
    forward_step_func,
    data_iterator: Iterator | list[Iterator],
    model: paddle.nn.Layer | list[paddle.nn.Layer],
    num_microbatches: int,
):
    """Run forward and backward passes with no pipeline parallelism"""
    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    for i in range(num_microbatches):
        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
        )
        backward_step(input_tensor, output_tensor, output_tensor_grad)
    return forward_data_store


def backward_step(input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True

    paddle.autograd.backward(
        tensors=output_tensor,
        grad_tensors=output_tensor_grad,
    )

    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]
    return input_tensor_grad


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
):
    """Forward step for passed-in model.
     Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptable as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output.It return a tuple of reduced loss and some other data.
                    Note that in this case the first argument is divided by the number of global microbatches,
                    assuming it is a loss, so that the loss is stable as a function of the number of devices
                    the step is split across.
        data_iterator (iterator):
            The data iterator.
        model (nn.Layer):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data.

    Returns:
        Tensor or list[Tensor]: The output object or objects from the forward step.
    """

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    output_tensor, loss_func = forward_step_func(data_iterator, model)

    output_tensor = forward_step_calc_loss(
        output_tensor,
        loss_func,
        num_microbatches,
        forward_data_store,
    )

    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def forward_step_calc_loss(
    output_tensor,
    loss_func,
    num_microbatches,
    forward_data_store,
):
    """Calculate the loss and number of tokens for forward_step()"""
    outputs = loss_func(output_tensor)
    output_tensor, loss_reduced = outputs
    output_tensor /= num_microbatches
    forward_data_store.append(loss_reduced)
    return output_tensor


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    This function performs all forward and backward computations of the model,
    and its selection depends on whether pipeline parallelism is enabled,
    the pipeline model parallel world size, the virtual pipeline model parallel world size,
    and the accumulate_steps and best_unbalanced_scheduler parameters in the training strategy.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one paddle.Tensor and return a paddle.Tensor of loss and a
        dictionary of string -> paddle.Tensor.

        For example:
        def loss_func(output_tensor):
            loss = output_tensor.mean()
            return loss, {"loss_reduced": loss}
        def forward_step_func(data_iterator, model):
            data = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, output)

        forward_backward_func(forward_step_func=forward_step, ...)

    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism.

    num_microbatches (int, required):
        The number of microbatches to go through
    """
    pp_degree = parallel_state.get_pipeline_model_parallel_world_size()
    if pp_degree == 1:
        forward_backward_func = no_pipeline
    else:
        fleet_env = fleet.fleet
        strategy = fleet_env._user_defined_strategy
        if strategy.hybrid_configs["pp_configs"].use_dualpipev:
            forward_backward_func = pipeline_dualpipev
        elif (
            parallel_state.get_virtual_pipeline_model_parallel_world_size() == 1
        ):
            forward_backward_func = pipeline_1f1b
        else:
            accumulate_steps = strategy.pipeline_configs["accumulate_steps"]
            if accumulate_steps >= 2 * pp_degree:
                forward_backward_func = pipeline_1f1b_with_interleave
            elif pp_degree <= accumulate_steps < 2 * pp_degree:
                if strategy.hybrid_configs[
                    "pp_configs"
                ].best_unbalanced_scheduler:
                    forward_backward_func = (
                        pipeline_fthenb_with_interleave_in_balanced_memory
                    )
                else:
                    forward_backward_func = pipeline_fthenb_with_interleave
    return forward_backward_func
