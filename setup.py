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

import logging
import os
import shutil
from pathlib import Path

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import backends
from build_utils import get_special_build_deps


def change_pwd():
    """change_pwd"""
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)


common_dependencies = [
    "colorlog>=6.10.1",
]


def get_special_setup_deps():
    if backends.IS_NVIDIA:
        deps = [
            "triton",  # for deep_gemm, flashmask
            "nvidia-cutlass-dsl==4.2.1",  # for sonic_moe
            "filelock",  # for sonic_moe
        ]
        return deps
    elif backends.IS_XPU:
        deps = []
        return deps
    else:
        return []


class CustomBdistWheel(_bdist_wheel):
    """Custom bdist_wheel that removes .o files from wheel before packaging."""

    def _is_all_o_files(self, dir_path):
        """Check if directory contains only .o files recursively."""
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if not file.endswith(".o"):
                    return False
        return True

    def _clean_build_dir(self, wheel_dir):
        """Remove build directory if it contains only .o files."""
        build_dir = os.path.join(wheel_dir, "build")

        if not os.path.exists(build_dir):
            logging.debug(f"No build directory found at: {build_dir}")
            return

        if not self._is_all_o_files(build_dir):
            logging.info(
                f"Skipping removal of {build_dir} (contains non-.o files)"
            )
            return

        try:
            shutil.rmtree(build_dir)
            logging.info(f"Removed build directory (all .o files): {build_dir}")
        except Exception as e:
            logging.warning(f"Failed to remove directory {build_dir}: {e}")

    def write_wheelfile(self, wheelfile_base, generator=None):
        """Override to clean .o files before writing wheel."""

        if hasattr(self, "bdist_dir") and self.bdist_dir:
            self._clean_build_dir(self.bdist_dir)
            extensions_path = (
                Path(self.bdist_dir) / "paddlefleet" / "_extensions"
            )
            for ext in (".cu", ".h", ".txt"):
                for file in extensions_path.glob(f"*{ext}"):
                    try:
                        os.remove(file)
                    except Exception:
                        pass

        if generator is not None:
            super().write_wheelfile(wheelfile_base, generator=generator)
        else:
            super().write_wheelfile(wheelfile_base)


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

    if cuda_major == 12 and cuda_minor < 8:
        nvcc_args = [arg for arg in nvcc_args if "compute_100" not in arg]

    ext_module = CUDAExtension(
        sources=[
            # cpp files
            # cuda files
            "./src/paddlefleet/_extensions/fuse_transpose_split_fp8_quant.cu",
            "./src/paddlefleet/_extensions/tokens_stable_unzip.cu",
            "./src/paddlefleet/_extensions/tokens_unzip_gather.cu",
            "./src/paddlefleet/_extensions/tokens_zip_unique_add.cu",
            "./src/paddlefleet/_extensions/tokens_zip_prob.cu",
            "./src/paddlefleet/_extensions/merge_subbatch_cast.cu",
            "./src/paddlefleet/_extensions/tokens_unzip_slice.cu",
            "./src/paddlefleet/_extensions/fuse_swiglu_scale.cu",
            "./src/paddlefleet/_extensions/swiglu_kernel.cu",
            "./src/paddlefleet/_extensions/fuse_weighted_swiglu_fp8_quant.cu",
            "./src/paddlefleet/_extensions/router_metadata.cu",
            "./src/paddlefleet/_extensions/count_cumsum.cu",
            "./src/paddlefleet/_extensions/filter_scores.cu",
            "./src/paddlefleet/_extensions/fuse_stack_transpose_fp8_quant.cu",
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
        cmdclass={"bdist_wheel": CustomBdistWheel},
        install_requires=dependencies,
    )


# This func is for no extension ops backends
def setup_install_no_extension():
    from setuptools import setup

    setup(
        name="paddlefleet",
        install_requires=dependencies,
    )


dependencies = (
    common_dependencies + get_special_build_deps() + get_special_setup_deps()
)
if backends.IS_NVIDIA:
    setup_ops_extension()
elif backends.IS_XPU:
    setup_install_no_extension()
else:
    logging.error("\033[31m Error: Do not support this backend now.\033[0m\n")
