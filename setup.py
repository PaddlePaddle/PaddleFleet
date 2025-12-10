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

import glob
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from setuptools.command.build_py import build_py

current_dir = Path(__file__).parent
third_party_dir = current_dir / "third_party"
deep_gemm_dir = third_party_dir / "DeepGEMM"

is_whl = False

if "bdist_wheel" in sys.argv:
    is_whl = True


def change_pwd():
    """change_pwd"""
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)


def copy_external_assets(source_dir, base_dir):
    target_dir = base_dir

    print(f"[Custom] Copying {source_dir} -> {target_dir}")
    if os.path.exists(source_dir):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)


from contextlib import contextmanager


@contextmanager
def pushd(path):
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def extract_whl(source_dir, output_dir):
    whl_files = glob.glob(os.path.join(source_dir, "*.whl"))

    if not whl_files:
        print(f"在 {source_dir} 没有找到 .whl 文件")
        return

    target_whl = whl_files[0]
    print(f"正在解压: {target_whl}")

    with zipfile.ZipFile(target_whl, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"解压完成，位置: {output_dir}")


def setup_deep_gemm():
    print("----------------------")
    print(sys.argv)
    if "egg_info" in sys.argv:
        return
    print("----------------------")

    with pushd(deep_gemm_dir):
        print("Cleaning up...")

        for dir_name in ["build", "dist"]:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(f"Removed {dir_name}")

        for egg_dir in glob.glob("*.egg-info"):
            if os.path.isdir(egg_dir):
                shutil.rmtree(egg_dir)
                print(f"Removed {egg_dir}")
        if is_whl:
            print("Building wheel...")
            try:
                subprocess.run(
                    [sys.executable, "setup.py", "bdist_wheel"], check=True
                )
                print("Build success!")
            except subprocess.CalledProcessError:
                print("Build failed!")
                sys.exit(1)
            extract_whl("./dist", "./dist/.temp_unpack")
        else:
            print("Begin editable install")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    ".",
                    "--no-build-isolation",
                    "-v",
                ],
                check=True,
            )


def setup_ops_extension():
    setup_deep_gemm()
    from paddle.utils.cpp_extension import CUDAExtension, setup

    try:
        from wheel.bdist_wheel import bdist_wheel
    except ImportError:
        bdist_wheel = None

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise FileNotFoundError(
            "nvcc command not found. Please make sure CUDA toolkit is installed and nvcc is in PATH."
        )

    result = subprocess.run(
        ["nvcc", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    version_output = result.stdout

    match = re.search(r"release (\d+)\.(\d+)", version_output)
    if not match:
        raise ValueError(
            f"Cannot parse CUDA version from nvcc output:\n{version_output}"
        )
    cuda_major = int(match.group(1))
    cuda_minor = int(match.group(2))

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
        "-DPADDLE_NO_PYTHON",
        # Limited API Macro for NVCC
        "-DPy_LIMITED_API=0x030A0000",
    ]
    if cuda_major < 12:
        raise ValueError(
            f"CUDA version must be >= 12. Detected version: {cuda_major}.{cuda_minor}"
        )
    if cuda_major == 12 and cuda_minor < 8:
        nvcc_args = [arg for arg in nvcc_args if "compute_100" not in arg]

    ext_module = CUDAExtension(
        sources=[
            # cpp files
            "./src/paddlefleet/extensions/matmul_bwd.cc",
            # cuda files
            "./src/paddlefleet/extensions/tokens_stable_unzip.cu",
            "./src/paddlefleet/extensions/tokens_unzip_gather.cu",
            "./src/paddlefleet/extensions/tokens_zip_unique_add.cu",
            "./src/paddlefleet/extensions/tokens_zip_prob.cu",
            "./src/paddlefleet/extensions/merge_subbatch_cast.cu",
            "./src/paddlefleet/extensions/tokens_unzip_slice.cu",
        ],
        include_dirs=[
            os.path.join(os.getcwd(), "src/paddlefleet/extensions"),
        ],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-w",
                "-Wno-abi",
                "-fPIC",
                "-std=c++17",
                "-DPADDLE_NO_PYTHON",
                "-DPy_LIMITED_API=0x030A0000",
            ],
            "nvcc": nvcc_args,
        },
        py_limited_api=True,
    )

    ext_module.py_limited_api = True

    cmdclass = {}
    if bdist_wheel:

        class ABI3Wheel(bdist_wheel):
            def get_tag(self):
                python, abi, plat = super().get_tag()
                if python.startswith("cp"):
                    return python, "abi3", plat
                return python, abi, plat

        cmdclass["bdist_wheel"] = ABI3Wheel

    class CustomBuildPy(build_py):
        def run(self):
            build_py.run(self)
            if is_whl:
                copy_external_assets(
                    deep_gemm_dir / "dist" / ".temp_unpack" / "deep_gemm",
                    f"{self.build_lib}/deep_gemm",
                )
                copy_external_assets(
                    deep_gemm_dir / "dist" / ".temp_unpack" / "deep_gemm_cpp",
                    f"{self.build_lib}/deep_gemm_cpp",
                )

    cmdclass["build_py"] = CustomBuildPy

    change_pwd()
    setup(
        name="paddlefleet.extensions.ops",
        ext_modules=[ext_module],
        cmdclass=cmdclass,
    )


setup_ops_extension()
