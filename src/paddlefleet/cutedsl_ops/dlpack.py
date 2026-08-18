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

"""Paddle-to-CuTe DLPack bridge.

This adapter is deliberately kept independent of cuDNN Frontend.  It exposes
Paddle's explicit capsule API through the Python DLPack protocol expected by
CuTe DSL.  Conversion is zero-copy.
"""


class _PaddleDLPackAdapter:
    """Expose Paddle's capsule API through the Python DLPack protocol."""

    def __init__(self, tensor):
        self._tensor = tensor

    def __dlpack__(self, stream=None):
        import paddle

        # Paddle's protocol implementation accepts the consumer CUDA stream
        # and inserts the required producer/consumer dependency. Calling
        # ``to_dlpack`` directly would discard that stream argument.
        exporter = getattr(self._tensor, "__dlpack__", None)
        if exporter is not None:
            if stream is None:
                return exporter()
            return exporter(stream=stream)
        return paddle.utils.dlpack.to_dlpack(self._tensor)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def current_paddle_stream_ptr() -> int:
    """Return the raw pointer of Paddle's current CUDA stream."""
    import paddle

    stream = paddle.device.current_stream()
    base = getattr(stream, "stream_base", stream)
    for name in ("cuda_stream", "raw_stream"):
        value = getattr(base, name, None)
        if value is not None:
            return int(value)
    raise RuntimeError("cannot obtain the current Paddle CUDA stream pointer")


def current_cu_stream():
    """Return Paddle's current stream as ``cuda.bindings.driver.CUstream``."""
    import cuda.bindings.driver as cuda

    return cuda.CUstream(current_paddle_stream_ptr())


def paddle_to_cute_tensor(
    tensor,
    *,
    assumed_align: int,
    leading_dim: int | None = None,
    stream_ptr: int | None = None,
):
    """Create a dynamic-layout CuTe tensor sharing a Paddle CUDA allocation.

    No copy is made.  ``assumed_align`` must describe the actual allocation:
    use 16 for contiguous Q/KV floating-point tensors and 4 for contiguous
    int32 metadata/output tensors.
    """
    from cutlass.cute.runtime import from_dlpack

    # CuTe DSL expects an object implementing the Python DLPack protocol,
    # whereas Paddle's explicit API returns a raw capsule.
    del stream_ptr
    result = from_dlpack(
        _PaddleDLPackAdapter(tensor),
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    )
    if leading_dim is None:
        return result.mark_layout_dynamic()
    return result.mark_layout_dynamic(leading_dim=leading_dim)
