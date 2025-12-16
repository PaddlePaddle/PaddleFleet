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
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.resolve()
OPS_DIR = ROOT_DIR / "src" / "paddlefleet" / "ops"
THIRD_PARTY_INSTALL_TEMP = ROOT_DIR / "src" / "_third_party_install_temp"


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
        self, name: str, source_rel_path: str, artifacts: list[Artifact]
    ):
        self.name = name
        self.source_dir = ROOT_DIR / source_rel_path
        # Install into a subdirectory named after the library
        self.install_dir = THIRD_PARTY_INSTALL_TEMP / name
        self.artifacts = artifacts

    def build(self) -> None:
        """Builds the library."""
        logger.info(f"Building ecosystem library: {self.name}")
        self.install_dir.mkdir(parents=True, exist_ok=True)

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
            subprocess.check_call(cmd, cwd=self.source_dir)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build {self.name}: {e}")
            raise

    def clean(self) -> None:
        """Cleans up existing artifacts in the ops directory."""
        for artifact in self.artifacts:
            target_path = OPS_DIR / artifact.target_name
            if target_path.exists():
                if target_path.is_symlink():
                    target_path.unlink()
                elif target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()

    def install(self, use_symlinks: bool = False) -> None:
        """Installs artifacts to the ops directory via symlink or copy."""
        self.clean()  # Ensure clean state first

        for artifact in self.artifacts:
            # Artifact source path is relative to the installation directory
            src = self.install_dir / artifact.source_rel_path
            dst = OPS_DIR / artifact.target_name

            if not src.exists():
                logger.warning(f"Artifact source not found: {src}")
                continue

            if use_symlinks:
                logger.info(f"Symlinking {src} -> {dst}")
                dst.symlink_to(src, target_is_directory=src.is_dir())
            else:
                logger.info(f"Copying {src} -> {dst}")
                if src.is_dir():
                    shutil.copytree(
                        src, dst, symlinks=False, dirs_exist_ok=True
                    )
                else:
                    shutil.copy(src, dst)
