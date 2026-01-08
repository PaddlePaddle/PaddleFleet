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
import subprocess
from pathlib import Path

from setuptools import build_meta as orig

from build_utils import (
    Artifact,
    EcosystemLibrary,
    check_patchelf_exists,
    check_submodule_updated,
    get_cuda_version,
)

logger = logging.getLogger(__name__)
_root = Path(__file__).parent.resolve()


def is_git_repo():
    return (_root / ".git").is_dir()


def get_git_commit_hash(cwd: Path | None) -> str:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(cmd, cwd=cwd).strip().decode("utf-8")
    except Exception:
        return "unknown"


def get_git_commit_date(cwd: Path | None) -> str:
    try:
        cmd = ["git", "log", "-1", "--format=%cd", "--date=format:%Y%m%d"]
        return subprocess.check_output(cmd, cwd=cwd).strip().decode("utf-8")
    except Exception:
        return "unknown"


def _generate_version_info():
    """Generate version info file with git metadata."""
    version_file = _root / "version.txt"
    with open(version_file, "r") as f:
        version = f.read().strip()

    # Get git info
    git_commit_hash = get_git_commit_hash(_root)
    git_commit_date = get_git_commit_date(_root)

    # Create version info in the source tree
    package_dir = Path(__file__).parent / "src" / "paddlefleet"
    _version_file = package_dir / "version.py"

    # If file exists and not in git repo (installing from sdist), keep existing file
    if _version_file.exists() and not is_git_repo():
        logger.info(
            "The version.py file already exists (not in git repo), keeping it"
        )
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    final_version = f"{version}.dev{git_commit_date}"
    if os.environ.get("PADDLEFLEET_VERSION") is not None:
        final_version = os.environ["PADDLEFLEET_VERSION"]
    with open(_version_file, "w") as f:
        f.write('"""Generate version info file with git metadata."""\n')
        f.write(f'__version__ = "{final_version}"\n')
        f.write(f'commit = "{git_commit_hash}"\n')
    logger.info(f"Created version.py with version {final_version}")
    return final_version


# Generate version info as soon as this module is imported
_generate_version_info()

cuda_major, cuda_minor = get_cuda_version()
if cuda_major < 12:
    raise ValueError(
        f"CUDA version must be >= 12. Detected version: {cuda_major}.{cuda_minor}"
    )

LIBRARIES: list[EcosystemLibrary] = [
    EcosystemLibrary(
        name="DeepGEMM",
        source_rel_path="third_party/DeepGEMM",
        artifacts=[
            # Updated paths to point to the installation directory
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
        else {"PADDLE_CUDA_ARCH_LIST": "9.0;10.0"},
    ),
    EcosystemLibrary(
        name="sonic-moe",
        source_rel_path="third_party/sonic-moe",
        artifacts=[
            Artifact("sonicmoe", "sonicmoe"),
        ],
    ),
    EcosystemLibrary(
        name="quack",
        source_rel_path="third_party/quack",
        artifacts=[
            Artifact("quack", "quack"),
        ],
    ),
]


def _prepare_ecosystem(use_symlinks: bool):
    """Iterates over all registered libraries and prepares them."""
    for lib in LIBRARIES:
        lib.build()
        lib.install(use_symlinks=use_symlinks)


def get_cuda_special_build_deps(cuda_major, cuda_minor):
    deps = [
        "paddlepaddle-gpu>=3.3.0.dev",
    ]
    if cuda_major == 12:
        deps.append("nvidia-nvshmem-cu12>=3.3.9,!=3.5.*")  # for deep_ep build
    elif cuda_major == 13:
        deps.append("nvidia-nvshmem-cu13>=3.3.9,!=3.5.*")  # for deep_ep build
    else:
        raise ValueError(
            f"Unsupported CUDA version: {cuda_major}.{cuda_minor}."
        )
    return deps


def get_requires_for_build_wheel(config_settings=None):
    return get_cuda_special_build_deps(cuda_major, cuda_minor)


def get_requires_for_build_sdist(config_settings=None):
    return get_cuda_special_build_deps(cuda_major, cuda_minor)


def get_requires_for_build_editable(config_settings=None):
    return get_cuda_special_build_deps(cuda_major, cuda_minor)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return orig.prepare_metadata_for_build_wheel(
        metadata_directory, config_settings
    )


def prepare_metadata_for_build_editable(
    metadata_directory, config_settings=None
):
    return orig.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    check_patchelf_exists()  # for deep_ep_cpp.so to add-rpath
    check_submodule_updated()
    _prepare_ecosystem(use_symlinks=False)
    return orig.build_wheel(
        wheel_directory, config_settings, metadata_directory
    )


def build_editable(
    wheel_directory, config_settings=None, metadata_directory=None
):
    check_patchelf_exists()  # for deep_ep_cpp.so to add-rpath
    check_submodule_updated()
    _prepare_ecosystem(use_symlinks=True)
    return orig.build_editable(
        wheel_directory, config_settings, metadata_directory
    )


def build_sdist(sdist_directory, config_settings=None):
    check_submodule_updated()
    return orig.build_sdist(sdist_directory, config_settings)
