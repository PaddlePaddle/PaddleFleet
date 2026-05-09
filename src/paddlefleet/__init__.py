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

from . import (
    ops as ops,
    parallel_state as parallel_state,
    training as training,
    version as version,
)
from .package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __version__,
)
from .timers import Timers

mpu = parallel_state

__all__ = [
    "ops",
    "training",
    "parallel_state",
    "Timers",
    "__contact_emails__",
    "__contact_names__",
    "__description__",
    "__download_url__",
    "__homepage__",
    "__keywords__",
    "__license__",
    "__package_name__",
    "__repository_url__",
    "__version__",
    "__ops_version__",
]

# Add paddlefleet_ops version tracking
try:
    import paddlefleet_ops

    __ops_version__ = paddlefleet_ops.__version__

    # Parse the ops version from pyproject.toml or package metadata
    try:
        import importlib.metadata

        required_ops_version = importlib.metadata.version("paddlefleet-ops")
    except importlib.metadata.PackageNotFoundError:
        # Fallback to hardcoded version if package metadata not available
        required_ops_version = "0.3.0.dev1"

    # Check if versions match (strip .post and .dev suffixes for comparison)
    def strip_version_suffix(v):
        # Remove .post and .dev suffixes
        parts = v.split(".")
        result = []
        for i, part in enumerate(parts):
            if part.startswith(("post", "dev")) and i >= 3:
                break
            result.append(part)
        return ".".join(result)

    base_installed = strip_version_suffix(__ops_version__)
    base_required = strip_version_suffix(required_ops_version)

    if base_installed != base_required:
        # Detect CUDA version to determine the correct index URL
        cuda_version = None
        try:
            import paddle

            cuda_version_str = paddle.version.cuda_version()
            # Map CUDA version to PyPI index URL suffix
            # e.g., "12.9" -> "cu129", "13.0" -> "cu130"
            cuda_major_minor = cuda_version_str.replace(".", "")
            if cuda_major_minor in ["126", "129", "130"]:
                cuda_index = f"cu{cuda_major_minor}"
            else:
                # Fallback to common versions
                if cuda_version_str.startswith("12"):
                    cuda_index = "cu129"
                elif cuda_version_str.startswith("13"):
                    cuda_index = "cu130"
                else:
                    cuda_index = "cu129"
        except Exception:
            # Fallback to default if cannot detect CUDA version
            cuda_index = "cu129"

        index_url = (
            f"https://www.paddlepaddle.org.cn/packages/nightly/{cuda_index}/"
        )
        error_msg = (
            f"paddlefleet_ops version mismatch! "
            f"Required: {required_ops_version}, Installed: {__ops_version__}.\n"
            f"Please install paddlefleet-ops=={required_ops_version} "
            f"with: pip install paddlefleet-ops=={required_ops_version} "
            f"--index-url={index_url}"
        )
        raise ImportError(error_msg)

except ImportError as e:
    __ops_version__ = "unknown"
