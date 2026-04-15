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

import re
import subprocess


def _check_cuda_version_compatible():
    """Check that the runtime CUDA version matches the version this wheel was built for.

    The CUDA major version is embedded in the wheel version string as '+cuXYZ'
    (e.g. '0.3.0.post20260415+cu129.abc'). If the runtime CUDA major version
    differs, raise a clear error instead of a cryptic 'libcudart.so.N not found'.
    """

    m = re.search(r"\+cu(\d+)\.", __version__)
    if m is None:
        return  # version string has no cuda tag, skip check

    built_cuda_major = int(m.group(1)[:2])  # e.g. "129" -> 12, "130" -> 13

    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.DEVNULL
        ).decode()
        runtime_m = re.search(r"release (\d+)\.(\d+)", out)
        if runtime_m is None:
            return
        runtime_cuda_major = int(runtime_m.group(1))
    except Exception:
        return  # nvcc not found, skip check

    if runtime_cuda_major != built_cuda_major:
        built_tag = m.group(1)  # e.g. "129"
        raise RuntimeError(
            f"paddlefleet-ops {__version__} was built for CUDA {built_cuda_major}.x "
            f"(+cu{built_tag}), but the current environment has CUDA {runtime_cuda_major}.x.\n"
            f"Please install the matching wheel:\n"
            f"  pip install paddlefleet-ops  --index-url https://www.paddlepaddle.org.cn/packages/nightly/cu{runtime_cuda_major}XX/"
        )


_check_cuda_version_compatible()

from paddlefleet_ops import ops  # noqa: F401
from paddlefleet_ops.version import __version__
