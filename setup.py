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


def change_pwd():
    """change_pwd"""
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)


def setup_ops_extension():
    """setup_ops_extension"""
    from paddle.utils.cpp_extension import CUDAExtension, setup

    change_pwd()
    setup(
        name="paddlefleet.extensions.ops",
        ext_modules=CUDAExtension(
            sources=[
                "./src/paddlefleet/extensions/tokens_stable_unzip.cu",
            ],
            include_dirs=[
                os.path.join(os.getcwd(), "src/paddlefleet/extensions"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-w", "-Wno-abi", "-fPIC", "-std=c++17"],
                "nvcc": [
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
                    "-gencode=arch=compute_90a,code=sm_90a",
                    "-DNDEBUG",
                ],
            },
        ),
    )


setup_ops_extension()
