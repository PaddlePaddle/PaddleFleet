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

"""Patch Triton's NVIDIA backend to use PaddlePaddle instead of PyTorch.

Triton >= 3.x detects an active GPU driver by calling
``CudaDriver.is_active()``, which returns ``torch.cuda.is_available()``.
In a PaddlePaddle-only environment (no PyTorch installed) this resolves to
``False``, leaving 0 active drivers and raising a RuntimeError at the first
kernel benchmark.

This conftest runs before any test and replaces the torch-dependent methods
with paddle-based equivalents so that Triton can compile and autotune kernels
normally.
"""

import paddle

# ---------------------------------------------------------------------------
# Helper: a minimal device-interface object expected by triton.testing.do_bench
# ---------------------------------------------------------------------------


class _PaddleCudaInterface:
    """Adapter that exposes the torch.cuda-like API consumed by triton's benchmarker."""

    @staticmethod
    def synchronize():
        paddle.device.synchronize()

    @staticmethod
    def Event(enable_timing=False):
        return paddle.device.cuda.Event(enable_timing=enable_timing)


# ---------------------------------------------------------------------------
# Patch triton.backends.nvidia.driver.CudaDriver
# ---------------------------------------------------------------------------

try:
    from triton.backends.driver import GPUDriver
    from triton.backends.nvidia import driver as _nvidia_driver

    # 1. is_active(): use paddle instead of torch
    @staticmethod  # type: ignore[misc]
    def _cuda_is_active():
        try:
            import paddle as _pd

            return _pd.is_compiled_with_cuda()
        except Exception:
            return False

    _nvidia_driver.CudaDriver.is_active = _cuda_is_active

    # 2. GPUDriver.__init__(): wire up device/stream helpers via paddle
    def _gpudriver_init(self):
        import paddle as _pd

        self.get_device_capability = _pd.device.cuda.get_device_capability

        def _get_current_stream(idx):
            stream = _pd.device.current_stream(f"gpu:{idx}")
            return stream.stream_base.cuda_stream

        self.get_current_stream = _get_current_stream
        self.get_current_device = lambda: int(
            _pd.device.get_device().split(":")[-1]
        )
        self.set_current_device = lambda idx: _pd.device.set_device(
            f"gpu:{idx}"
        )

    GPUDriver.__init__ = _gpudriver_init  # type: ignore[method-assign]

    # 3. get_device_interface(): return a paddle-based adapter
    _nvidia_driver.CudaDriver.get_device_interface = (  # type: ignore[attr-defined]
        lambda self: _PaddleCudaInterface()
    )

    # 4. get_active_torch_device(): not needed for benchmarking; return None
    _nvidia_driver.CudaDriver.get_active_torch_device = (  # type: ignore[attr-defined]
        lambda self: None
    )

    # 5. get_empty_cache_for_benchmark(): use paddle instead of torch
    def _get_empty_cache(self):
        cache_size = 256 * 1024 * 1024
        return paddle.empty([int(cache_size // 4)], dtype="int32").cuda()

    _nvidia_driver.CudaDriver.get_empty_cache_for_benchmark = _get_empty_cache  # type: ignore[attr-defined]

    # 6. Reset cached driver so the next access picks up the patched is_active
    from triton.runtime.driver import driver as _triton_driver

    _triton_driver._default = None
    _triton_driver._active = None

except ImportError:
    # triton is not installed; tests will be skipped via HAS_TRITON guard
    pass
