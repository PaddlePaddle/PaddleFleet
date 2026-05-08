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

Generates ``src/paddlefleet/version.py`` at build time (same pattern as
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
from setuptools.dist import Distribution

logger = logging.getLogger(__name__)

_pkg_root = Path(__file__).parent.resolve()
_ops_version_py = (
    _pkg_root
    / "packages"
    / "paddlefleet_ops"
    / "src"
    / "paddlefleet_ops"
    / "version.py"
)


def is_git_repo() -> bool:
    return (_pkg_root / ".git").is_dir()


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
    """Read the required paddlefleet-ops version.

    Prefers ``ops_required_version.txt`` (updated by paddlefleet_ops build,
    decouples build cadence) over reading the ops source version.py directly.
    Falls back to the ops source tree when building both packages together in
    the same workspace without a pre-existing ops_required_version.txt.
    """
    ops_req_file = _pkg_root / "ops_required_version.txt"
    if ops_req_file.exists():
        version = ops_req_file.read_text().strip()
        if version:
            return version
    if not _ops_version_py.exists():
        return None
    globs: dict = {}
    exec(_ops_version_py.read_text(), globs)
    return globs.get("__version__")


def _generate_version_info() -> str:
    """Generate ``src/paddlefleet/version.py`` with git metadata."""
    version_file = _pkg_root / "version.txt"
    with open(version_file) as f:
        version = f.read().strip()

    git_commit_hash = get_git_commit_hash(_pkg_root)

    version_py = _pkg_root / "src" / "paddlefleet" / "version.py"

    # If file exists and not in git repo (installing from sdist), keep existing
    if version_py.exists() and not is_git_repo():
        logger.info("version.py already exists (not in git repo), keeping it")
        return version

    if os.environ.get("PADDLEFLEET_VERSION") is not None:
        final_version = os.environ["PADDLEFLEET_VERSION"]
    else:
        date_str = datetime.now().strftime("%Y%m%d")
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
    logger.info(f"Created version.py with version {final_version}")
    return final_version


def _pin_ops_dependency() -> None:
    """Replace the bare 'paddlefleet-ops' dependency with an exact version pin.

    Patches setuptools' Distribution.finalize_options so that by the time
    setuptools writes the wheel METADATA, install_requires already carries
    the precise version (e.g. 'paddlefleet-ops==0.3.0.dev20260415').
    Falls back silently to the bare name when the ops version is unavailable
    (e.g. building paddlefleet alone from an sdist).
    """
    ops_version = _get_ops_version()
    if ops_version is None:
        logger.warning(
            "paddlefleet-ops version.py not found; "
            "falling back to unpinned 'paddlefleet-ops' dependency."
        )
        return

    pinned = f"paddlefleet-ops=={ops_version}"
    logger.info(f"Pinning ops dependency: {pinned}")

    _orig_finalize = Distribution.finalize_options

    def _patched_finalize(self):
        _orig_finalize(self)
        if self.install_requires:
            self.install_requires = [
                pinned if req.strip() == "paddlefleet-ops" else req
                for req in self.install_requires
            ]

    Distribution.finalize_options = _patched_finalize


# Run at import time so both wheel and editable builds pick up the patches.
_generate_version_info()
_pin_ops_dependency()


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
