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
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import subprocess


def get_version_from_txt():
    version_file = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_file, "r") as f:
        version = f.read().strip()
    return version


def custom_version_scheme(version):
    base_version = get_version_from_txt()
    date_str = (
        subprocess.check_output(
            ["git", "log", "-1", "--format=%cd", "--date=format:%Y%m%d"]
        )
        .decode()
        .strip()
    )
    return f"{base_version}.dev{date_str}"


def no_local_scheme(version):
    return ""


def change_pwd():
    """change_pwd"""
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)


def setup_ops_extension():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    from build_utils import get_cuda_version

    # 定义 NVCC 编译参数
    nvcc_args = [
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-maxrregcount=32",
        "-lineinfo",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-gencode=arch=compute_100,code=sm_100",
        "-DNDEBUG",
    ]
    cuda_major, cuda_minor = get_cuda_version()
    if cuda_major < 12:
        raise ValueError(
            f"CUDA version must be >= 12. Detected version: {cuda_major}.{cuda_minor}"
        )
    if cuda_major == 12 and cuda_minor < 8:
        nvcc_args = [arg for arg in nvcc_args if "compute_100" not in arg]

    ext_module = CUDAExtension(
        sources=[
            # cpp files
            # cuda files
            "./src/paddlefleet/_extensions/tokens_stable_unzip.cu",
            "./src/paddlefleet/_extensions/tokens_unzip_gather.cu",
            "./src/paddlefleet/_extensions/tokens_zip_unique_add.cu",
            "./src/paddlefleet/_extensions/tokens_zip_prob.cu",
            "./src/paddlefleet/_extensions/merge_subbatch_cast.cu",
            "./src/paddlefleet/_extensions/tokens_unzip_slice.cu",
            "./src/paddlefleet/_extensions/fuse_swiglu_scale.cu",
            "./src/paddlefleet/_extensions/swiglu_kernel.cu",
        ],
        include_dirs=[
            os.path.join(os.getcwd(), "src/paddlefleet/_extensions"),
        ],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-w",
                "-Wno-abi",
                "-fPIC",
                "-std=c++17",
            ],
            "nvcc": nvcc_args,
        },
    )

    change_pwd()
    setup(
        name="paddlefleet._extensions.ops",
        ext_modules=[ext_module],
        use_scm_version={
            "version_scheme": custom_version_scheme,
            "local_scheme": no_local_scheme,
        },
    )


setup_ops_extension()
