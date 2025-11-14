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
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import functools
import inspect
import operator
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import logging
import math
import warnings

import paddle

from fleet.core import parallel_state

try:
    from packaging.version import Version as PkgVersion

    HAVE_PACKAGING = True
except ImportError:
    HAVE_PACKAGING = False

try:
    import nvtx

    HAVE_NVTX = True
except ImportError:
    HAVE_NVTX = False

try:
    _paddle_version = PkgVersion(paddle.__version__)
except Exception:
    # This is a WAR for building docs, where paddle is not actually imported
    _paddle_version = PkgVersion("0.0.0") if HAVE_PACKAGING else "0.0.0"


class WrappedTensor:
    """
    A wrapper for tensors that enables caller functions to pass an indirect reference
    to callee functions. By wrapping the tensor, the caller's direct reference is removed,
    allowing the tensor to be garbage collected once the callee unwraps and frees it.
    """

    def __init__(self, tensor: paddle.Tensor):
        self._wrapper = [tensor]

    def unwrap(self):
        """
        Returns the wrapped tensor while deleting the internal reference.
        Can only be called once.
        """
        if len(self._wrapper) == 0:
            raise RuntimeError("WrappedTensor has already been unwrapped")
        return self._wrapper.pop(0)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    return functools.partial(paddle.nn.init.normal_, mean=0.0, std=sigma)


def scaled_init_method_normal(sigma, num_layers, multiplier=2.0):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(multiplier * num_layers)

    return functools.partial(paddle.nn.init.normal_, mean=0.0, std=std)


def get_pg_size(group=None):
    """Get world size for a distributed group.

    Args:
        group: Process group to get world size for. If None, uses default group.

    Returns:
        int: World size (1 if distributed not initialized or group is None, else group.size())
    """
    if not paddle.distributed.is_initialized() or group is None:
        return 1
    return group.size()


def get_pg_rank(group=None):
    """Get rank for a distributed group.

    Args:
        group: Process group to get rank for. If None, uses default group.

    Returns:
        int: Rank (0 if distributed not initialized or group is None, else group.rank())
    """
    if not paddle.distributed.is_initialized() or group is None:
        return 0
    return group.rank()


def log_single_rank(
    logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any
):
    """If paddle distributed is initialized, write log on only one rank

    Args:
        logger (logging.Logger): The logger to write the logs

        args (Tuple[Any]): All logging.Logger.log positional arguments

        rank (int, optional): The rank to write on. Defaults to 0.

        kwargs (Dict[str, Any]): All logging.Logger.log keyword arguments
    """
    if paddle.distributed.is_initialized():
        if paddle.distributed.get_rank() == rank:
            logger.log(*args, **kwargs)
    else:
        logger.log(*args, **kwargs)


def get_paddle_version():
    """Get paddle version from __version__."""

    global _paddle_version
    return _paddle_version


def is_paddle_min_version(version, check_equality=True):
    """Check if minimum version of `paddle` is installed."""
    if not HAVE_PACKAGING:
        raise ImportError(
            "packaging is not installed. Please install it with `pip install packaging`."
        )
    if check_equality:
        return get_paddle_version() >= PkgVersion(version)
    return get_paddle_version() > PkgVersion(version)


######################
### NVTX profiling ###
######################
_nvtx_enabled: bool = False  # Whether NVTX range profiling is enabled
_nvtx_range_messages: list[
    str
] = []  # Messages associated with active NVTX ranges


def _nvtx_range_get_func_path():
    """Get the path of a function. Assumes being called from nvtx_range_push/pop.

    Returns:
        str: Module path and function name joined by a dot
    """
    # Get the caller's caller frame (go back 2 frames)
    frame = inspect.currentframe().f_back.f_back
    caller_func = inspect.getframeinfo(frame).function
    module = inspect.getmodule(frame)

    return f"{module.__name__}.{caller_func}"


def nvtx_range_push(msg=None, suffix=None) -> None:
    """Push NVTX range onto stack. If msg is not provided, use the calling function's path.

    Args:
        msg (str, optional): Message to associate with range
        suffix (str, optional): Suffix to append to the message
    """
    if not _nvtx_enabled:
        return

    if msg is None:
        msg = _nvtx_range_get_func_path()
    if suffix is not None:
        msg = f"{msg}.{suffix}"

    # Track messages to ensure consistency when popping
    _nvtx_range_messages.append(msg)

    # Push NVTX range
    paddle.base.core.nvprof_nvtx_push(msg)


def nvtx_range_pop(msg=None, suffix=None) -> None:
    """Pop NVTX range from stack. If msg is not provided, use the calling function's path.

    Args:
        msg (str, optional): Message to associate with range
        suffix (str, optional): Suffix to append to the message
    """
    if not _nvtx_enabled:
        return

    if msg is None:
        msg = _nvtx_range_get_func_path()
    if suffix is not None:
        msg = f"{msg}.{suffix}"

    # Update list of NVTX range messages and check for consistency
    if not _nvtx_range_messages:
        raise RuntimeError("Attempted to pop NVTX range from empty stack")
    last_msg = _nvtx_range_messages.pop()
    if msg is not None and msg != last_msg:
        raise ValueError(
            f"Attempted to pop NVTX range from stack with msg={msg}, "
            f"but last range has msg={last_msg}"
        )

    # Pop NVTX range
    paddle.base.core.nvprof_nvtx_pop()


@functools.cache
def _nvtx_decorator_get_func_path(func):
    """Get the path of a function.

    Args:
        func (Callable): Function to get path for.

    Returns:
        str: Module path and function name joined by a dot
    """
    caller_func = func.__name__
    module = inspect.getmodule(func)

    return f"{module.__name__}.{caller_func}"


def nvtx_decorator(message: str | None = None, color: str | None = None):
    """Decorator to add NVTX range to a function.

    Args:
        message (str, optional): Custom message for the NVTX range. If None, uses function path
        color (str, optional): Color for the NVTX range. Defaults to None

    Returns:
        Callable: Decorated function with NVTX profiling if enabled

    Example:
        @nvtx_decorator()
        def my_function():
            pass

        @nvtx_decorator(message="Custom Range", color="blue")
        def another_function():
            pass
    """

    def decorator(func: Callable) -> Callable:
        if _nvtx_enabled:
            return nvtx.annotate(
                message=message or _nvtx_decorator_get_func_path(func),
                color=color,
            )(func)
        return func

    return decorator


def get_tensor_model_parallel_group_if_none(
    tp_group, is_expert=False, check_initialized=True
):
    """Issue a deprecation warning if tp_group is None and return the default tp group."""
    if not paddle.distributed.is_initialized():
        return None

    if tp_group is None:
        if (
            paddle.distributed.is_initialized()
            and paddle.distributed.get_rank() == 0
        ):
            warnings.warn(
                "Warning: tp_group is None, using default tp group. "
                "Passing tp_group will be mandatory soon",
                DeprecationWarning,
                stacklevel=2,
            )
        if is_expert:
            tp_group = parallel_state.get_expert_tensor_parallel_group(
                check_initialized=check_initialized
            )
        else:
            tp_group = parallel_state.get_tensor_model_parallel_group(
                check_initialized=check_initialized
            )
    return tp_group


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(
        self,
        tensor_shape,
        dtype,
        name,
        mem_alloc_context: Callable | None = None,
    ):
        """
        Returns (potentially) a sub-tensor from the self.buffer for the given shape.
        """
        required_len = functools.reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            mem_alloc_context = (
                mem_alloc_context if mem_alloc_context else nullcontext
            )
            with mem_alloc_context():
                self.buffer[(name, dtype)] = paddle.empty(
                    required_len,
                    dtype=dtype,
                    requires_grad=False,
                )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def _kernel_make_viewless_tensor(inp, requires_grad):
    """Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    """
    out = paddle.empty(
        (1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad
    )
    out.data = inp.data
    return out


class MakeViewlessTensor(paddle.autograd.PyLayer):
    """
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    """

    @staticmethod
    def forward(ctx, inp, requires_grad):
        """Runs the fwd pass of _kernel_make_viewless_tensor"""
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        """No-op"""
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    """
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    """

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)
