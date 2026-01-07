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

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).parent.resolve()
OPS_DIR = ROOT_DIR / "src" / "paddlefleet" / "ops"
THIRD_PARTY_INSTALL_TEMP = ROOT_DIR / "src" / "_third_party_install_temp"


def remove_path(path: Path) -> None:
    """Removes a path (file, directory, or symlink) if it exists."""
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def create_symlink(src: Path, dst: Path) -> None:
    """Creates a symlink from src to dst, overwriting dst if it exists."""
    remove_path(dst)
    logger.info(f"Symlinking {src} -> {dst}")
    dst.symlink_to(src, target_is_directory=src.is_dir())


@dataclass
class Artifact:
    """
    Defines a mapping from a path in the installation directory to a target name in the ops directory.

    source_rel_path: Relative path from the library's installation directory (e.g., 'deep_gemm').
    target_name: Name of the symlink/directory to create in 'src/paddlefleet/ops' (e.g., 'deep_gemm').
    """

    source_rel_path: str
    target_name: str


class EcosystemLibrary:
    """
    Represents an external ecosystem operator library.
    Encapsulates the logic for building and installing the library.
    """

    def __init__(
        self,
        name: str,
        source_rel_path: str,
        artifacts: list[Artifact],
        extra_env: dict[str, str] | None = None,
    ):
        self.name = name
        self.source_dir = ROOT_DIR / source_rel_path
        # Install into a subdirectory named after the library
        self.install_dir = THIRD_PARTY_INSTALL_TEMP / name
        self.artifacts = artifacts
        self._extra_env = extra_env or {}

    def build(self) -> None:
        """Builds the library."""
        logger.info(f"Building ecosystem library: {self.name}")
        self.install_dir.mkdir(parents=True, exist_ok=True)

        # Special pre-build step for DeepGEMM: link CUTLASS headers into deep_gemm/include
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

        # pip install . --target  <install_dir> --no-deps --no-build-isolation
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
        ]

        try:
            _env = os.environ.copy()
            _env.update(self._extra_env)
            subprocess.check_call(cmd, cwd=self.source_dir, env=_env)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build {self.name}: {e}")
            raise

    def install(self, use_symlinks: bool = False) -> None:
        """Installs artifacts to the ops directory via symlink or copy."""
        for artifact in self.artifacts:
            # Artifact source path is relative to the installation directory
            src = self.install_dir / artifact.source_rel_path
            dst = OPS_DIR / artifact.target_name

            if use_symlinks:
                create_symlink(src, dst)
            else:
                remove_path(dst)
                logger.info(f"Copying {src} -> {dst}")
                if src.is_dir():
                    shutil.copytree(
                        src, dst, symlinks=False, dirs_exist_ok=True
                    )
                else:
                    shutil.copy(src, dst)

            if artifact.target_name == "deep_ep_cpp.so":
                cmd = [
                    "patchelf",
                    "--add-rpath",
                    "$ORIGIN/../../nvidia/nvshmem/lib",
                    dst,
                ]
                try:
                    subprocess.check_call(cmd)
                except subprocess.CalledProcessError as e:
                    cmd_str = " ".join(cmd)
                    logger.error(f"Failed to run {cmd_str}.")
                    raise


def check_submodule_updated():
    if not (
        (ROOT_DIR / "third_party" / "DeepGEMM" / ".git").exists()
        and (ROOT_DIR / "third_party" / "DeepEP" / ".git").exists()
    ):
        logger.error(
            "\033[91m Found uninitialized submodules. Please use 'git submodule update --init --recursive' to fix!\033[0m"
        )
        sys.exit(1)


def get_cuda_version():
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
    return cuda_major, cuda_minor
