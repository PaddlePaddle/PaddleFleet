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
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .strip()
        .decode("utf-8")
    )


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
            f'commit = "{git_commit_hash}"\n'
        )
    logger.info(f"Created _version.py with version {final_version}")
    return final_version


# Run at import time so both wheel and editable builds pick up the patches.
_generate_version_info()


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
