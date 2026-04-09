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
import os
import shutil
from pathlib import Path

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# backends.py and build_utils.py live alongside this file in packages/paddlefleet_ops/
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
            "nvidia-cutlass-dsl==4.4.1",  # for sonic_moe and flash_attention
            "filelock",  # for sonic_moe
        ]
        return deps
    elif backends.IS_XPU:
        return []
    else:
        return []


class CustomBdistWheel(_bdist_wheel):
    """Custom bdist_wheel that removes .o files from wheel before packaging."""

    def _is_all_o_files(self, dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if not file.endswith(".o"):
                    return False
        return True

    def _clean_build_dir(self, wheel_dir):
        build_dir = os.path.join(wheel_dir, "build")
        if not os.path.exists(build_dir):
            return
        if not self._is_all_o_files(build_dir):
            return
        try:
            shutil.rmtree(build_dir)
        except Exception as e:
            logging.warning(f"Failed to remove directory {build_dir}: {e}")

    def write_wheelfile(self, wheelfile_base, generator=None):
        # Only strip source files from the wheel bdist staging dir.
        # NOTE: we intentionally keep the build/ directory (.o files) on disk
        # so that setuptools can use incremental compilation on the next build.
        if hasattr(self, "bdist_dir") and self.bdist_dir:
            extensions_path = (
                Path(self.bdist_dir) / "paddlefleet_ops" / "_extensions"
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
    # import paddle.core
    from paddle.base.core import is_compiled_with_onednn

    # paddle_compiled_with_onednn = is_compiled_with_onednn()
    paddle_compiled_with_onednn = False

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

    if paddle_compiled_with_onednn:
        nvcc_args.append("-DPADDLE_WITH_DNNL")

    cuda_major, cuda_minor = get_cuda_version()
    if cuda_major == 12 and cuda_minor < 8:
        nvcc_args = [arg for arg in nvcc_args if "compute_100" not in arg]

    # change_pwd() MUST be called before CUDAExtension() so that the
    # os.path.abspath() calls Paddle makes internally use _pkg_dir as the
    # base.  After that, a plain relpath() (no explicit start=) is enough
    # because os.getcwd() is already _pkg_dir.
    change_pwd()
    _pkg_dir = os.getcwd()

    # setup() requires paths relative to the setup.py directory
    _ext_rel = "src/paddlefleet_ops/_extensions"

    ext_module = CUDAExtension(
        sources=[
            f"{_ext_rel}/fuse_transpose_split_fp8_quant.cu",
            f"{_ext_rel}/tokens_stable_unzip.cu",
            f"{_ext_rel}/tokens_unzip_gather.cu",
            f"{_ext_rel}/tokens_zip_unique_add.cu",
            f"{_ext_rel}/tokens_zip_prob.cu",
            f"{_ext_rel}/merge_subbatch_cast.cu",
            f"{_ext_rel}/tokens_unzip_slice.cu",
            f"{_ext_rel}/fuse_swiglu_scale.cu",
            f"{_ext_rel}/swiglu_kernel.cu",
            f"{_ext_rel}/fuse_weighted_swiglu_fp8_quant.cu",
            f"{_ext_rel}/router_metadata.cu",
            f"{_ext_rel}/count_cumsum.cu",
            f"{_ext_rel}/filter_scores.cu",
            f"{_ext_rel}/fuse_stack_transpose_fp8_quant.cu",
            f"{_ext_rel}/fuse_apply_rotary_pos_emb_vision.cu",
        ],
        include_dirs=[str(Path(__file__).parent / _ext_rel)],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-w",
                "-Wno-abi",
                "-fPIC",
                "-std=c++17",
            ] + (["-DPADDLE_WITH_DNNL"] if paddle_compiled_with_onednn else []),
            "nvcc": nvcc_args,
        },
    )

    # Paddle's CUDAExtension (and its setup()) re-converts sources to absolute
    # paths internally via os.path.abspath(), which setuptools then rejects.
    # We wrap the extension in a subclass that intercepts the `sources`
    # attribute: writes always store the raw value (absolute paths are fine for
    # the compiler), but reads always return paths relative to _pkg_dir so
    # setuptools' validation passes.
    class _RelativeSourcesExt(ext_module.__class__):
        @property
        def sources(self):
            return [
                os.path.relpath(s, _pkg_dir) if os.path.isabs(s) else s
                for s in self._sources
            ]

        @sources.setter
        def sources(self, value):
            self._sources = value if value is not None else []

    # Read sources via the old class (plain list attribute) BEFORE switching,
    # then hand them to the new setter so _sources is initialised correctly.
    _saved_sources = list(ext_module.sources)
    ext_module.__class__ = _RelativeSourcesExt
    ext_module.sources = _saved_sources

    setup(
        name="paddlefleet_ops._extensions.ops",
        ext_modules=[ext_module],
        cmdclass={"bdist_wheel": CustomBdistWheel},
        install_requires=dependencies,
    )


def setup_install_no_extension():
    from setuptools import setup

    setup(
        name="paddlefleet-ops",
        install_requires=dependencies,
    )


try:
    dependencies = (
        common_dependencies
        + get_special_build_deps()
        + get_special_setup_deps()
    )
except Exception as e:
    raise Exception(
        f"Failed to resolve special dependencies: {e}, using common dependencies only"
    ) from e

if backends.IS_NVIDIA:
    setup_ops_extension()
elif backends.IS_XPU:
    setup_install_no_extension()
else:
    logging.error("\033[31m Error: Do not support this backend now.\033[0m\n")
