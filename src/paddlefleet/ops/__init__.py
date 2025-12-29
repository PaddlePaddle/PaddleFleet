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
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import paddle

from .utils import (
    HardwareIncompatibleBlocker,
    ModuleContext,
    get_nvshmem_host_lib_path,
    import_custom_ops,
    patch_module_namespace,
)

# paddle.compat.enable_torch_proxy(scope={"triton"}) enables the torch proxy
# specifically for the 'triton' module. This means `import torch` inside 'triton'
# will actually import paddle's compatibility layer (acting as torch).
#
# 'scope' acts as an allowlist. To add other modules, you can do:
# paddle.compat.enable_torch_proxy(scope={"triton", "new_module"})
#
# Note: Ensure that any torch APIs used in 'new_module' are already implemented in Paddle.

logger = logging.getLogger(__name__)


def is_deep_gemm_or_deep_ep_available():
    return paddle.cuda.get_device_capability()[0] >= 9


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
    with ModuleContext([lib_name], ops_dir):
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
        ops_dir = Path(__file__).parent
        # Loading libnvshmem_host.so.* first when use editable install
        _try_load_nvshmem(ops_dir)
        _safe_load_ecosystem_lib("deep_gemm", ops_dir, globals())
        _safe_load_ecosystem_lib("deep_ep", ops_dir, globals())
    finally:
        paddle.compat.disable_torch_proxy()
else:
    capability = paddle.cuda.get_device_capability()
    sys.meta_path.insert(0, HardwareIncompatibleBlocker(capability))
    logger.warning(
        f"The capability for your device is {capability[0]}.{capability[1]}, which is less than 9.0. Please don't call anything in 'paddle.ops.deep_gemm' and 'paddle.ops.deep_op' which is unsupported in your device"
    )

try:
    paddle.compat.enable_torch_proxy(scope={"triton"})
    from .._extensions.flashmask import (
        rr_attn_estimate_triton_func,  # noqa: F401
    )
finally:
    paddle.compat.disable_torch_proxy()
