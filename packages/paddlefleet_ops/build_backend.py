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
import subprocess
from pathlib import Path

from setuptools import build_meta as orig

import backends
from build_utils import (
    check_patchelf_exists,
    check_submodule_updated,
    get_libs,
    get_special_build_deps,
)

backends.init_backend_type()

logger = logging.getLogger(__name__)

# Root of this sub-package (packages/paddlefleet_ops/)
_pkg_root = Path(__file__).parent.resolve()
# Workspace root (two levels up: packages/paddlefleet_ops/ → packages/ → workspace root)
_workspace_root = _pkg_root.parent.parent.resolve()


def is_git_repo():
    return (_workspace_root / ".git").is_dir()


def get_git_commit_hash(cwd: Path | None) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .strip()
            .decode("utf-8")
        )
    except Exception:
        return "unknown"


def get_git_commit_date(cwd: Path | None) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "log", "-1", "--format=%cd", "--date=format:%Y%m%d"],
                cwd=cwd,
            )
            .strip()
            .decode("utf-8")
        )
    except Exception:
        return "unknown"


def _generate_version_info():
    """Generate version info file with git metadata."""
    version_file = _workspace_root / "version.txt"
    with open(version_file) as f:
        version = f.read().strip()

    git_commit_hash = get_git_commit_hash(_workspace_root)
    git_commit_date = get_git_commit_date(_workspace_root)

    version_py = _pkg_root / "src" / "paddlefleet_ops" / "version.py"

    if version_py.exists() and not is_git_repo():
        logger.info("version.py already exists (not in git repo), keeping it")
        return version

    final_version = f"{version}.dev{git_commit_date}"
    if os.environ.get("PADDLEFLEET_VERSION") is not None:
        final_version = os.environ["PADDLEFLEET_VERSION"]
    with open(version_py, "w") as f:
        f.write('"""Auto-generated version file."""\n')
        f.write(f'__version__ = "{final_version}"\n')
        f.write(f'commit = "{git_commit_hash}"\n')
    logger.info(f"Created version.py with version {final_version}")
    return final_version


_generate_version_info()


def _prepare_ecosystem(use_symlinks: bool):
    """Iterates over all registered libraries and prepares them."""
    if backends.IS_NVIDIA:
        for lib in get_libs():
            lib.build()
            lib.install(use_symlinks=use_symlinks)
    elif backends.IS_XPU:
        pass
    elif backends.IS_ILUVATAR_GPU:
        pass
    elif backends.IS_METAX_GPU:
        pass


def _clean_egg_info():
    """Remove stale *.egg-info directories under _pkg_root.

    setuptools' egg_info command reads the SOURCES.txt it finds inside an
    existing egg-info directory.  If a previous (possibly failed) build left
    one behind that contains absolute paths, every subsequent build will fail
    with "path is not relative".  Deleting it forces setuptools to regenerate
    SOURCES.txt from scratch with correct relative paths.

    egg-info directories can appear at the top level *or* inside src/
    (e.g. src/paddlefleet_ops.egg-info/), so we use rglob to cover both.
    """
    for egg_info_dir in _pkg_root.rglob("*.egg-info"):
        if egg_info_dir.is_dir():
            logger.info(f"Removing stale egg-info: {egg_info_dir}")
            shutil.rmtree(egg_info_dir, ignore_errors=True)


def get_requires_for_build_wheel(config_settings=None):
    return get_special_build_deps()


def get_requires_for_build_sdist(config_settings=None):
    return get_special_build_deps()


def get_requires_for_build_editable(config_settings=None):
    return get_special_build_deps()


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
    _clean_egg_info()
    check_patchelf_exists()
    check_submodule_updated()
    _prepare_ecosystem(use_symlinks=False)
    return orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory, config_settings=None, metadata_directory=None
):
    _clean_egg_info()
    check_patchelf_exists()
    check_submodule_updated()
    _prepare_ecosystem(use_symlinks=True)
    return orig.build_editable(
        wheel_directory, config_settings, metadata_directory
    )


def build_sdist(sdist_directory, config_settings=None):
    _clean_egg_info()
    check_submodule_updated()
    return orig.build_sdist(sdist_directory, config_settings)
