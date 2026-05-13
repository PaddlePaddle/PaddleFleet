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

# Check paddlefleet_ops version consistency at runtime

import paddlefleet_ops

__ops_version__ = paddlefleet_ops.__version__
__ops_required_version__ = version.__ops_required_version__
if __ops_version__ != __ops_required_version__:
    cuda_index = "cu129"
    try:
        import paddle

        cuda_major_minor = paddle.version.cuda_version().replace(".", "")
        if cuda_major_minor in ("126", "129", "130"):
            cuda_index = f"cu{cuda_major_minor}"
    except Exception:
        pass

    index_url = (
        f"https://www.paddlepaddle.org.cn/packages/nightly/{cuda_index}/"
    )
    raise ImportError(
        f"paddlefleet_ops version mismatch! "
        f"Required: {__ops_required_version__}, Installed: {__ops_version__}.\n"
        f"Please install paddlefleet-ops=={__ops_required_version__} "
        f"with: pip install paddlefleet-ops=={__ops_required_version__} "
        f"--index-url={index_url}"
    )
