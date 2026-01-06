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
    clean_module_namespace,
    get_cuda_version,
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
ops_dir = Path(__file__).parent
cuda_capability = (
    paddle.cuda.get_device_capability()
    if paddle.is_compiled_with_cuda()
    else None
)

_python_version = sys.version_info
_python_version_str = ".".join(map(str, _python_version[:3]))
_cuda_version = get_cuda_version()
_cuda_version_str = ".".join(map(str, _cuda_version))
_capability_str = (
    f"{cuda_capability[0]}.{cuda_capability[1]}"
    if cuda_capability
    else "unavailable"
)

DEEP_GEMM_HINT = (
    "For developers: guard imports with `is_deep_gemm_available()` and only call `paddlefleet.ops.deep_gemm` when flag branch enabled.\n"
    "For users: set `use_deep_gemm=False` if you want to skip it, or use a GPU with compute capability >= 9.0 to enable."
)

DEEP_EP_HINT = (
    "For developers: guard imports with `is_deep_ep_available()` and only call `paddlefleet.ops.deep_ep` when flag branch enabled.\n"
    "For users: avoid `moe_token_dispatcher_type='deepep'` or use a GPU with compute capability >= 9.0 to enable."
)

SONIC_MOE_HINT = (
    "For developers: guard imports with `is_sonicmoe_available()` and only call `paddlefleet.ops.sonicmoe` when flag branch enabled.\n"
    "For users: avoid `moe_token_dispatcher_type='deepep'` or upgrade to Python >= 3.12, CUDA >= 12.9, and a GPU with compute capability >= 9.0 to enable."
)


def _build_notice(
    lib_module: str, reason: str, hint_for_error: str | None = None
) -> tuple[str, str]:
    """Compose warning/error messages; only errors carry extra hints."""
    warning = f"{lib_module} not supported: {reason}"
    error_reason = f"{reason} \n{hint_for_error}" if hint_for_error else reason
    error = f"{lib_module} not supported: {error_reason}"
    return warning, error


def _hopper_requirement(
    lib_module: str, hint: str | None = None
) -> tuple[str, str]:
    reason = (
        f"{lib_module} requires GPU compute capability >= 9.0 (Hopper). "
        f"Current capability: {_capability_str}."
    )
    return _build_notice(lib_module, reason, hint_for_error=hint)


def _sonic_moe_requirement(
    lib_module: str, hint: str | None = None
) -> tuple[str, str]:
    reasons = []
    if sys.version_info < (3, 12):
        reasons.append(
            f"Python >= 3.12 required (current {_python_version_str})"
        )
    if _cuda_version < (12, 9):
        reasons.append(f"CUDA >= 12.9 required (current {_cuda_version_str})")
    if not cuda_capability or cuda_capability[0] < 9:
        reasons.append(
            f"GPU compute capability equal to 9.x required (current {_capability_str})"
        )
    reason = "; ".join(reasons) if reasons else "Runtime requirements not met."
    return _build_notice(lib_module, reason, hint_for_error=hint)


_DEEP_GEMM_AVAILABLE = False
_DEEP_EP_AVAILABLE = False
_SONIC_MOE_AVAILABLE = False

if paddle.is_compiled_with_cuda():
    if paddle.cuda.get_device_capability()[0] >= 9:
        _DEEP_GEMM_AVAILABLE = True
        _DEEP_EP_AVAILABLE = True
    if (
        sys.version_info >= (3, 12)
        and paddle.cuda.get_device_capability()[0] == 9
        and _cuda_version >= (12, 9)
    ):
        _SONIC_MOE_AVAILABLE = True


def is_deep_gemm_available():
    return _DEEP_GEMM_AVAILABLE


def is_deep_ep_available():
    return _DEEP_EP_AVAILABLE


def is_sonic_moe_available():
    return _SONIC_MOE_AVAILABLE


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
    lib_name: str,
    ops_dir: Path,
    module_globals: dict[str, Any],
    extra_libs_name: list[str] | None = None,
):
    lib_names = [lib_name]
    if extra_libs_name:
        lib_names += extra_libs_name
    with ModuleContext(lib_names, ops_dir):
        try:
            module = importlib.import_module(lib_name)
            patch_module_namespace(lib_name, "paddlefleet.ops.")
            if extra_libs_name:
                for extra_lib in extra_libs_name:
                    clean_module_namespace(extra_lib)
            module_globals[lib_name] = module
            logger.info(f"Successfully loaded ecosystem library: {lib_name}")
        except ImportError as e:
            logger.warning(f"Ecosystem library '{lib_name}' not found: {e}")


import_custom_ops(
    package="paddlefleet._extensions", module_name=".ops", global_ns=globals()
)

blocked_import_messages: dict[str, str] = {}

if is_deep_gemm_available():
    paddle.compat.enable_torch_proxy(scope={"deep_gemm", "triton"}, silent=True)
    _safe_load_ecosystem_lib("deep_gemm", ops_dir, globals())
else:
    warning, error = _hopper_requirement(
        "paddlefleet.ops.deep_gemm", hint=DEEP_GEMM_HINT
    )
    logger.warning(warning)
    blocked_import_messages["paddlefleet.ops.deep_gemm"] = error

if is_deep_ep_available():
    paddle.compat.enable_torch_proxy(scope={"deep_ep"}, silent=True)
    # Loading libnvshmem_host.so.* first when use editable install
    _try_load_nvshmem(ops_dir)
    _safe_load_ecosystem_lib("deep_ep", ops_dir, globals())
else:
    warning, error = _hopper_requirement(
        "paddlefleet.ops.deep_ep", hint=DEEP_EP_HINT
    )
    logger.warning(warning)
    blocked_import_messages["paddlefleet.ops.deep_ep"] = error

if is_sonic_moe_available():
    paddle.compat.enable_torch_proxy(
        scope={"sonicmoe", "quack", "triton"}, silent=True
    )
    _safe_load_ecosystem_lib("sonicmoe", ops_dir, globals(), ["quack"])
else:
    warning, error = _sonic_moe_requirement(
        "paddlefleet.ops.sonicmoe", hint=SONIC_MOE_HINT
    )
    logger.warning(warning)
    blocked_import_messages["paddlefleet.ops.sonicmoe"] = error


if blocked_import_messages:
    sys.meta_path.insert(
        0, HardwareIncompatibleBlocker(blocked_import_messages)
    )

try:
    paddle.compat.enable_torch_proxy(scope={"triton"}, silent=True)
    from .._extensions.flashmask import (
        rr_attn_estimate_triton_func,  # noqa: F401
    )
finally:
    paddle.compat.disable_torch_proxy()


def __getattr__(name):
    module_name = f"paddlefleet.ops.{name}"

    if module_name in blocked_import_messages:
        raise RuntimeError(blocked_import_messages[module_name])

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
