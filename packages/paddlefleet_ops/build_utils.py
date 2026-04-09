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

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import backends

logger = logging.getLogger(__name__)

# packages/paddlefleet_ops/
PKG_ROOT = Path(__file__).parent.resolve()
# workspace root (packages/paddlefleet_ops/ → packages/ → workspace root)
ROOT_DIR = PKG_ROOT.parent.parent.resolve()

OPS_DIR = PKG_ROOT / "src" / "paddlefleet_ops" / "ops"
THIRD_PARTY_INSTALL_TEMP = PKG_ROOT / "src" / "_third_party_install_temp"


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def create_symlink(src: Path, dst: Path) -> None:
    remove_path(dst)
    logger.info(f"Symlinking {src} -> {dst}")
    dst.symlink_to(src, target_is_directory=src.is_dir())


@dataclass
class Artifact:
    source_rel_path: str
    target_name: str


class EcosystemLibrary:
    def __init__(
        self,
        name: str,
        source_rel_path: str,
        artifacts: list[Artifact],
        extra_env: dict[str, str] | None = None,
    ):
        self.name = name
        # source_rel_path is relative to workspace root (where third_party/ lives)
        self.source_dir = ROOT_DIR / source_rel_path
        self.install_dir = THIRD_PARTY_INSTALL_TEMP / name
        self.artifacts = artifacts
        self._extra_env = extra_env or {}

    def build(self) -> None:
        logger.info(f"Building ecosystem library: {self.name}")
        self.install_dir.mkdir(parents=True, exist_ok=True)

        if self.name.lower() == "deepgemm":
            cutlass_root = (
                self.source_dir / "third-party" / "cutlass" / "include"
            )
            target_include_dir = self.source_dir / "deep_gemm" / "include"
            target_include_dir.mkdir(parents=True, exist_ok=True)
            links = {
                cutlass_root / "cutlass": target_include_dir / "cutlass",
                cutlass_root / "cute": target_include_dir / "cute",
            }
            for src, dst in links.items():
                create_symlink(src, dst)

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            ".",
            "--target",
            str(self.install_dir),
            "--no-deps",
            "--no-build-isolation",
            "--no-compile",
            "--upgrade",
            "-v",
        ]
        try:
            _env = os.environ.copy()
            _env.update(self._extra_env)
            subprocess.check_call(cmd, cwd=self.source_dir, env=_env)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build {self.name}: {e}")
            raise

    def install(self, use_symlinks: bool = False) -> None:
        for artifact in self.artifacts:
            src = self.install_dir / artifact.source_rel_path
            dst = OPS_DIR / artifact.target_name

            if use_symlinks:
                create_symlink(src, dst)
            else:
                remove_path(dst)
                logger.info(f"Copying {src} -> {dst}")
                if src.is_dir():
                    shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)
                else:
                    shutil.copy(src, dst)

            if artifact.target_name == "deep_ep_cpp.so":
                cmd = [
                    "patchelf",
                    "--add-rpath",
                    "$ORIGIN/../../nvidia/nvshmem/lib",
                    str(dst),
                ]
                try:
                    subprocess.check_call(cmd)
                except subprocess.CalledProcessError as e:
                    cmd_str = " ".join(cmd)
                    logger.error(f"Failed to run {cmd_str}.")
                    raise


def check_submodule_updated():
    if backends.IS_NVIDIA:
        missing = not all([
            (ROOT_DIR / "third_party" / "DeepGEMM" / ".git").exists(),
            (ROOT_DIR / "third_party" / "DeepEP" / ".git").exists(),
            (ROOT_DIR / "third_party" / "quack" / ".git").exists(),
            (ROOT_DIR / "third_party" / "sonic-moe" / ".git").exists(),
            (ROOT_DIR / "third_party" / "flash-attention" / ".git").exists(),
        ])
        if missing:
            logger.error(
                "\033[91m Found uninitialized submodules. Please use "
                "'git submodule update --init --recursive' to fix!\033[0m"
            )
            sys.exit(1)
    elif backends.IS_XPU:
        pass


def check_patchelf_exists():
    if shutil.which("patchelf") is None:
        logger.error(
            "\033[31m Error: 'patchelf' not found in PATH.\033[0m\n"
            "\033[31m Please install 'patchelf' before proceeding.\033[0m"
        )
        sys.exit(1)


def get_cuda_version():
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise FileNotFoundError(
            "nvcc not found. Please ensure CUDA toolkit is installed."
        )
    result = subprocess.run(
        ["nvcc", "--version"], capture_output=True, text=True, check=True
    )
    match = re.search(r"release (\d+)\.(\d+)", result.stdout)
    if not match:
        raise ValueError(
            f"Cannot parse CUDA version from nvcc output:\n{result.stdout}"
        )
    cuda_major, cuda_minor = int(match.group(1)), int(match.group(2))
    if cuda_major < 12:
        raise ValueError(
            f"CUDA version must be >= 12. Detected: {cuda_major}.{cuda_minor}"
        )
    return cuda_major, cuda_minor


def get_special_build_deps():
    if backends.IS_NVIDIA:
        cuda_major, cuda_minor = get_cuda_version()
        deps = ["paddlepaddle-gpu==3.3.1.post20260403+ef0820a64e9"]
        if cuda_major == 12:
            if cuda_minor > 6:
                deps.append("paddle-nvidia-nvshmem-cu12>=3.3.9,<3.5")
            else:
                deps.append("nvidia-nvshmem-cu12>=3.3.9,<3.5")
        elif cuda_major == 13:
            deps.append("paddle-nvidia-nvshmem-cu13>=3.3.9,<3.5")
        else:
            raise ValueError(f"Unsupported CUDA version: {cuda_major}.{cuda_minor}.")
        return deps
    elif backends.IS_XPU:
        return ["paddlepaddle-xpu>=3.3.0"]
    else:
        return []


def get_libs():
    cuda_major, cuda_minor = get_cuda_version()

    LIBRARIES: list[EcosystemLibrary] = [
        EcosystemLibrary(
            name="DeepGEMM",
            source_rel_path="third_party/DeepGEMM",
            artifacts=[
                Artifact("deep_gemm", "deep_gemm"),
                Artifact("deep_gemm_cpp", "deep_gemm_cpp"),
            ],
        ),
        EcosystemLibrary(
            name="DeepEP",
            source_rel_path="third_party/DeepEP",
            artifacts=[
                Artifact("deep_ep", "deep_ep"),
                Artifact("deep_ep_cpp.so", "deep_ep_cpp.so"),
            ],
            extra_env={"PADDLE_CUDA_ARCH_LIST": "9.0"}
            if (cuda_major == 12 and cuda_minor < 8)
            else {"PADDLE_CUDA_ARCH_LIST": "9.0;10.0;10.3"},
        ),
        EcosystemLibrary(
            name="flash-attention",
            source_rel_path="third_party/flash-attention/flashmask",
            artifacts=[Artifact("flash_mask", "flash_mask")],
            extra_env={"FLASHMASK_BUILD": "fa4"},
        ),
    ]
    if sys.version_info >= (3, 12):
        LIBRARIES.append(
            EcosystemLibrary(
                name="quack",
                source_rel_path="third_party/quack",
                artifacts=[Artifact("quack", "quack")],
            )
        )
        LIBRARIES.append(
            EcosystemLibrary(
                name="sonic-moe",
                source_rel_path="third_party/sonic-moe",
                artifacts=[Artifact("sonicmoe", "sonicmoe")],
            )
        )
    return LIBRARIES
