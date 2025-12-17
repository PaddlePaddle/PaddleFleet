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
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

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


def copy_path(src: Path, dst: Path) -> None:
    """Copies a path (file or directory) to dst, overwriting if it exists."""
    remove_path(dst)
    logger.info(f"Copying {src} -> {dst}")
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=False)
    else:
        shutil.copy(src, dst)


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
        pre_build_func: Callable[[EcosystemLibrary], None] | None = None,
        cleanup_func: Callable[[EcosystemLibrary], None] | None = None,
        extra_env: dict[str, str] | None = None,
    ):
        self.name = name
        self.source_dir = ROOT_DIR / source_rel_path
        # Install into a subdirectory named after the library
        self.install_dir = THIRD_PARTY_INSTALL_TEMP / name
        self.artifacts = artifacts
        self._pre_build_func = pre_build_func
        self._cleanup_func = cleanup_func
        self._extra_env = extra_env or {}

    def build(self) -> None:
        """Builds the library."""
        logger.info(f"Building ecosystem library: {self.name}")
        self.install_dir.mkdir(parents=True, exist_ok=True)

        if self._pre_build_func:
            logger.info(f"Running pre-build hook for {self.name}")
            self._pre_build_func(self)

        # Default build command: python setup.py install --install-lib <install_dir>
        cmd = [
            sys.executable,
            "setup.py",
            "install",
            "--install-lib",
            str(self.install_dir),
            "--no-compile",
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
        install_strategy = create_symlink if use_symlinks else copy_path
        for artifact in self.artifacts:
            # Artifact source path is relative to the installation directory
            src = self.install_dir / artifact.source_rel_path
            dst = OPS_DIR / artifact.target_name
            install_strategy(src, dst)

    def cleanup(self) -> None:
        """Executes injected cleanup logic."""
        if self._cleanup_func:
            logger.info(f"Running cleanup hook for {self.name}")
            self._cleanup_func(self)


def check_submodule_updated():
    if not (ROOT_DIR / "third_party" / "DeepGEMM" / ".git").exists():
        logger.error(
            "\033[91m Found uninitialized submodules. Please use 'git submodule update --init --recursive' to fix!\033[0m"
        )
        sys.exit(1)
