# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Lightweight PEP 517 build backend for the root paddlefleet package.

Generates ``src/paddlefleet/_version.py`` at build time (same pattern as
``packages/paddlefleet_ops/build_backend.py``) and delegates every actual
build hook to ``setuptools.build_meta``.

At build time the ``paddlefleet-ops`` dependency is pinned to the exact
version of the ops package built in the same workspace (xFormers-style),
so the published wheel carries a precise ``Requires-Dist: paddlefleet-ops==X``
rather than an unconstrained bare name.
"""

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

from setuptools import build_meta as orig  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_pkg_root = Path(__file__).parent.resolve()


def is_git_repo() -> bool:
    return (_pkg_root / ".git").exists()


def get_git_commit_hash(cwd: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .strip()
            .decode("utf-8")
        )
    except Exception:
        return "unknown"


def _get_ops_version() -> str | None:
    """Compute the paddlefleet-ops version matching the ops build backend logic.

    Uses PADDLEFLEET_VERSION env var if set (CI builds), otherwise computes
    from version.txt (base) + ops_required_version.txt (build number) + branch.
    Falls back to reading the ops source _version.py directly when building
    both packages together in the same workspace.
    """
    if os.environ.get("PADDLEFLEET_VERSION") is not None:
        return os.environ["PADDLEFLEET_VERSION"]

    version_file = _pkg_root / "version.txt"
    ops_req_file = _pkg_root / "ops_required_version.txt"

    if not version_file.exists() or not ops_req_file.exists():
        raise RuntimeError("version.txt or ops_required_version.txt not found")
    base_version = version_file.read_text().strip()
    build_num = ops_req_file.read_text().strip()
    if not base_version or not build_num:
        raise RuntimeError("version.txt or ops_required_version.txt is empty")

    is_release_branch = False
    branch = (
        subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_pkg_root,
            stderr=subprocess.DEVNULL,
        )
        .decode("utf-8")
        .strip()
    )
    is_release_branch = branch.startswith("release/")
    suffix = ".post" if is_release_branch else ".dev"
    return f"{base_version}{suffix}{build_num}"


def _generate_version_info() -> str:
    """Generate ``src/paddlefleet/_version.py`` with git metadata."""
    version_file = _pkg_root / "version.txt"
    with open(version_file) as f:
        version = f.read().strip()

    git_commit_hash = get_git_commit_hash(_pkg_root)

    version_py = _pkg_root / "src" / "paddlefleet" / "_version.py"

    # If file exists and not in git repo (installing from sdist), keep existing
    if version_py.exists() and not is_git_repo():
        logger.info("_version.py already exists (not in git repo), keeping it")
        return version

    if os.environ.get("PADDLEFLEET_VERSION") is not None:
        final_version = os.environ["PADDLEFLEET_VERSION"]
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=_pkg_root,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        if branch.startswith("release/"):
            commit_short = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short=11", "HEAD"],
                    cwd=_pkg_root,
                )
                .strip()
                .decode("utf-8")
            )
            final_version = f"{version}.post{date_str}+{commit_short}"
        else:
            final_version = f"{version}.dev{date_str}"

    ops_version = _get_ops_version() or "unknown"

    with open(version_py, "w") as f:
        f.write(
            "# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.\n"
            "#\n"
            '# Licensed under the Apache License, Version 2.0 (the "License");\n'
            "# you may not use this file except in compliance with the License.\n"
            "# You may obtain a copy of the License at\n"
            "#\n"
            "#     http://www.apache.org/licenses/LICENSE-2.0\n"
            "#\n"
            "# Unless required by applicable law or agreed to in writing, software\n"
            '# distributed under the License is distributed on an "AS IS" BASIS,\n'
            "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
            "# See the License for the specific language governing permissions and\n"
            "# limitations under the License.\n"
            "\n"
            '"""Auto-generated version info — do not edit."""\n'
            "\n"
            f'__version__ = "{final_version}"\n'
            f'__ops_required_version__ = "{ops_version}"\n'
            f'commit = "{git_commit_hash}"\n'
        )
    logger.info(f"Created _version.py with version {final_version}")
    return final_version


def _generate_requirements_build() -> None:
    """Generate requirements-build.txt with the pinned paddlefleet-ops version.

    This file is referenced by [tool.setuptools.dynamic] in pyproject.toml.
    Setuptools reads it natively to populate install_requires, avoiding
    unreliable monkey-patching of Distribution internals.
    """
    ops_version = _get_ops_version()
    if ops_version is None:
        logger.warning(
            "paddlefleet-ops version could not be determined; "
            "falling back to unpinned 'paddlefleet-ops' dependency."
        )
        ops_dep = "paddlefleet-ops"
    else:
        ops_dep = f"paddlefleet-ops=={ops_version}"
        logger.info(f"Pinning ops dependency: {ops_dep}")

    dependencies = [
        "colorlog>=6.10.1",
        ops_dep,
    ]

    req_file = _pkg_root / "requirements-build.txt"
    req_file.write_text("\n".join(dependencies) + "\n")


# Run at import time so both wheel and editable builds pick up the patches.
_generate_version_info()
_generate_requirements_build()


# -- PEP 517 hooks (pure delegation to setuptools) --


def get_requires_for_build_wheel(config_settings=None):
    return orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return orig.get_requires_for_build_editable(config_settings)


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
    return orig.build_wheel(
        wheel_directory, config_settings, metadata_directory
    )


def build_editable(
    wheel_directory, config_settings=None, metadata_directory=None
):
    return orig.build_editable(
        wheel_directory, config_settings, metadata_directory
    )


def build_sdist(sdist_directory, config_settings=None):
    return orig.build_sdist(sdist_directory, config_settings)
