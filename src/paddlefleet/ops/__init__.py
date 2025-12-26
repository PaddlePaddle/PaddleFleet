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

import ctypes
import functools
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import paddle

from .utils import (
    ModuleContext,
    get_nvshmem_host_lib_path,
    import_custom_ops,
    patch_module_namespace,
)

logger = logging.getLogger(__name__)
_device_capability = paddle.cuda.get_device_capability()


HINT_MESSAGE = """For developers:
1. Imports of these modules must be guarded by `if is_deep_gemm_or_deep_ep_available():`.
2. Direct calls to `paddlefleet.ops.deep_gemm.xxx` should be conditional based on their enabling flags.

For users:
1. To avoid using these ops, set `use_deep_gemm` to False and ensure `moe_token_dispatcher_type` is not "deepep".
2. Alternatively, use a device with compute capability >= 9.0 to enable deep_gemm and deep_ep.
"""


@functools.cache
def is_deep_gemm_or_deep_ep_available():
    """Check whether deep GEMM or deep EP kernels are available on the current GPU."""
    return (
        paddle.is_compiled_with_cuda()
        and paddle.cuda.get_device_capability()[0] >= 9
    )


class HardwareIncompatibleBlocker(importlib.abc.MetaPathFinder):
    def __init__(self, capability: tuple[int, int]):
        self.capability = capability

    def find_spec(self, fullname, path, target=None):
        if fullname in ["paddlefleet.ops.deep_gemm", "paddlefleet.ops.deep_ep"]:
            raise RuntimeError(
                f"The module '{fullname}' requires GPU compute capability >= 9.0 (Hopper architecture), "
                f"but your device capability is {self.capability[0]}.{self.capability[1]}. \n{HINT_MESSAGE}"
            )


def _try_load_nvshmem(ops_dir: Path):
    third_party_temp_dir = ops_dir.parent.parent / "_third_party_install_temp"
    if third_party_temp_dir.exists():
        try:
            nvshmem_spec = importlib.util.find_spec("nvidia.nvshmem")
            if nvshmem_spec and nvshmem_spec.submodule_search_locations:
                nvshmem_dir = nvshmem_spec.submodule_search_locations[0]
                nvshmem_host_lib_path = get_nvshmem_host_lib_path(nvshmem_dir)
                logger.info(
                    f"Pre-loading NVSHMEM library from: {nvshmem_host_lib_path}"
                )
                ctypes.CDLL(str(nvshmem_host_lib_path), mode=ctypes.RTLD_GLOBAL)
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during NVSHMEM pre-loading: {e}"
            ) from e


def _safe_load_ecosystem_lib(
    lib_name: str, ops_dir: Path, module_globals: dict[str, Any]
):
    with ModuleContext(lib_name, ops_dir):
        try:
            module = importlib.import_module(lib_name)
            patch_module_namespace(lib_name, "paddlefleet.ops.")
            module_globals[lib_name] = module
            logger.info(f"Successfully loaded ecosystem library: {lib_name}")
        except ImportError as e:
            logger.warning(f"Ecosystem library '{lib_name}' not found: {e}")


import_custom_ops(
    package="paddlefleet._extensions", module_name=".ops", global_ns=globals()
)

if is_deep_gemm_or_deep_ep_available():
    try:
        paddle.compat.enable_torch_proxy(
            scope={"deep_gemm", "triton", "deep_ep"}
        )
        # paddle.compat.enable_torch_proxy(scope={"triton"}) enables the torch proxy
        # specifically for the 'triton' module. This means `import torch` inside 'triton'
        # will actually import paddle's compatibility layer (acting as torch).
        #
        # 'scope' acts as an allowlist. To add other modules, you can do:
        # paddle.compat.enable_torch_proxy(scope={"triton", "new_module"})
        #
        # Note: Ensure that any torch APIs used in 'new_module' are already implemented in Paddle.
        ops_dir = Path(__file__).parent
        # Loading libnvshmem_host.so.* first when use editable install
        _try_load_nvshmem(ops_dir)
        _safe_load_ecosystem_lib("deep_gemm", ops_dir, globals())
        _safe_load_ecosystem_lib("deep_ep", ops_dir, globals())
    finally:
        paddle.compat.disable_torch_proxy()
else:
    # Explicit error message when `import paddlefleet.ops.deep_gemm` and `from paddlefleet.ops.deep_gemm import xxx`
    sys.meta_path.insert(0, HardwareIncompatibleBlocker(_device_capability))
    logger.warning(
        f"The capability for your device is {_device_capability[0]}.{_device_capability[1]}, which is less than 9.0. Please don't call anything in 'paddlefleet.ops.deep_gemm' and 'paddlefleet.ops.deep_ep' which is unsupported in your device."
    )


# Explicit error message when call `paddlefleet.ops.deep_gemm` and `from paddlefleet.ops import deep_gemm`
def __getattr__(name):
    if name in ["deep_gemm", "deep_ep"]:
        if not is_deep_gemm_or_deep_ep_available():
            raise RuntimeError(
                f"The module 'paddlefleet.ops.{name}' requires GPU compute capability >= 9.0 (Hopper architecture), "
                f"but your device capability is {_device_capability[0]}.{_device_capability[1]}. \n{HINT_MESSAGE}"
            )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
