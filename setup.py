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
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import multiprocessing
import os


def run(func):
    """run"""
    p = multiprocessing.Process(target=func)
    p.start()
    p.join()


def change_pwd():
    """change_pwd"""
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)


def setup_moe_ops():
    """setup_moe_op"""
    from paddle.utils.cpp_extension import CUDAExtension, setup

    change_pwd()
    cutlass_include_dir = os.path.join(
        os.getcwd(), "third_party/cutlass/include"
    )
    setup(
        name="paddlefleet.extentions.ops",
        ext_modules=CUDAExtension(
            sources=[
                "./paddlefleet/extentions/moe_ops_fp8.cu",
                "./paddlefleet/extentions/tokens_stable_unzip.cu",
            ],
            include_dirs=[
                cutlass_include_dir,
                os.path.join(os.getcwd(), "paddlefleet/extentions"),
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


setup_moe_ops()
