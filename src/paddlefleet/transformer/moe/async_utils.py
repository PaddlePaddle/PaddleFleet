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

import queue

import paddle
from paddle import Tensor

# Hold pinned tensors until their async copies complete.
_pinned_tensor_queue = queue.deque()


def to_tensor(data, dtype=None, place=None) -> Tensor:
    """
    Copy host data to device asynchronously, safely, and without GPU bubbles.

    Note:
    1. This function uses pinned memory for asynchronous copy. The pinned memory
       is kept alive until the copy finishes to ensure data validity.
    2. This function is more efficient than `tensor.to("gpu", non_blocking=True)`
       because it uses events instead of CUDA host callbacks to control garbage
       collection, avoiding GPU bubbles.
    3. The copy is launched on the current stream, so later kernels on the same
       stream naturally depend on it and need no extra synchronization.

    Args:
        data: the host data to copy, which can be anything accepted by
            paddle.to_tensor, i.e. list, ndarray, cpu tensor, etc.
        dtype: the desired data type.

    Returns:
        data: the device tensor.
    """
    if place is None:
        place = paddle.get_device()
    assert paddle.device(place).is_gpu_place(), (
        f"to_device only supports gpu place as destination, got {place}"
    )

    pin_data = paddle.to_tensor(
        data, dtype=dtype, place=paddle.CUDAPinnedPlace()
    )
    gpu_data = paddle.empty_like(pin_data)

    cudart = paddle.cuda.cudart()

    err = cudart.cudaMemcpyAsync(
        gpu_data.data_ptr(),
        pin_data.data_ptr(),
        pin_data.size * pin_data.itemsize,
        cudart.cudaMemcpyHostToDevice,
        paddle.device.current_stream().stream_base.cuda_stream,
    )
    assert err == cudart.cudaError.success, f"cudaMemcpyAsync failed: {err}"

    event = paddle.device.Event()
    event.record()
    _pinned_tensor_queue.append((pin_data, event))

    while _pinned_tensor_queue:
        _, event = _pinned_tensor_queue[0]
        if event.query():
            _pinned_tensor_queue.popleft()
        else:
            break

    return gpu_data
