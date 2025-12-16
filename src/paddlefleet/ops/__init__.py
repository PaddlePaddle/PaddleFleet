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

import logging
import sys
from pathlib import Path
from typing import Any

import paddle

from .utils import import_custom_ops

paddle.compat.enable_torch_proxy(scope={"deep_gemm", "triton"})

# paddle.compat.enable_torch_proxy(scope={"triton"}) enables the torch proxy
# specifically for the 'triton' module. This means `import torch` inside 'triton'
# will actually import paddle's compatibility layer (acting as torch).
#
# 'scope' acts as an allowlist. To add other modules, you can do:
# paddle.compat.enable_torch_proxy(scope={"triton", "new_module"})
#
# Note: Ensure that any torch APIs used in 'new_module' are already implemented in Paddle.

logger = logging.getLogger(__name__)

import_custom_ops(
    package="paddlefleet._extensions", module_name=".ops", global_ns=globals()
)


class ModuleContext:
    """
    Manages the context for loading a module, including:
    1. Stashing existing modules to prevent conflicts.
    2. Managing sys.path.
    3. Restoring stashed modules upon exit.
    """

    def __init__(self, module_names: list[str], path: Path):
        self.module_names = module_names
        self.path = str(path)
        self._stash: dict[str, Any] = {}

    def _stash_modules(self):
        """Moves modules matching module_name from sys.modules to stash."""
        for name in list(sys.modules.keys()):
            for module_name in self.module_names:
                if name == module_name or name.startswith(module_name + "."):
                    self._stash[name] = sys.modules.pop(name)

    def _restore_modules(self):
        """Restores stashed modules to sys.modules."""
        sys.modules.update(self._stash)
        self._stash.clear()

    def __enter__(self):
        self._stash_modules()
        sys.path.insert(0, self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path in sys.path:
            sys.path.remove(self.path)
        self._restore_modules()


def patch_module_namespace(source_name: str, target_prefix: str):
    """
    Moves loaded modules from source_name to target_prefix + source_name.
    Effectively 'installs' the module into the new namespace.
    """
    for name in list(sys.modules.keys()):
        if name == source_name or name.startswith(source_name + "."):
            module = sys.modules.pop(name)
            new_name = target_prefix + name
            sys.modules[new_name] = module


ops_dir = Path(__file__).parent

with ModuleContext(["deep_gemm"], ops_dir):
    try:
        import deep_gemm  # noqa: F401

        patch_module_namespace("deep_gemm", "paddlefleet.ops.")
        logger.info("Successfully loaded ecosystem library: deep_gemm")
    except ImportError as e:
        logger.warning(f"Ecosystem library 'deep_gemm' not found: {e}")
