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

Generates ``src/paddlefleet/_version.py`` at build time and delegates
every actual build hook to ``setuptools.build_meta``.

On XPU builds the backend rewrites ``METADATA`` so the wheel carries
``paddlepaddle-xpu`` instead of ``paddlepaddle-gpu``.
On non-XPU builds it does nothing.
"""

import logging
import os
import re
import subprocess
import zipfile
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
        commit_short = (
            subprocess.check_output(
                ["git", "rev-parse", "--short=11", "HEAD"],
                cwd=_pkg_root,
            )
            .strip()
            .decode("utf-8")
        )
        date_str = (
            subprocess.check_output(
                [
                    "git",
                    "log",
                    "-1",
                    "--format=%cd",
                    "--date=format:%Y%m%d",
                    "HEAD",
                ],
                cwd=_pkg_root,
            )
            .strip()
            .decode("utf-8")
        )
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
            final_version = f"{version}.post{date_str}+{commit_short}"
        else:
            final_version = f"{version}.dev{date_str}+{commit_short}"

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


# -- XPU helpers -------------------------------------------------------------


def _is_xpu() -> bool:
    """Detect XPU via ``IS_XPU`` env var or ``xpu-smi`` command."""
    env = os.environ.get("IS_XPU")
    if env is not None:
        return env.lower() in {"1", "true", "yes", "on"}
    try:
        subprocess.run(
            ["xpu-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def _xpu_dep() -> str | None:
    """Read ``[tool.paddlefleet.paddle].xpu`` from ``pyproject.toml``."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return None

    try:
        with open(_pkg_root / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        return (
            data.get("tool", {})
            .get("paddlefleet", {})
            .get("paddle", {})
            .get("xpu")
        )
    except Exception:
        return None


def _patch_metadata(path: Path) -> None:
    """Replace ``paddlepaddle-gpu`` with ``paddlepaddle-xpu`` in a METADATA file.

    Does nothing on non-XPU builds or when the XPU spec is missing.
    """
    if not _is_xpu():
        return

    xpu = _xpu_dep()
    if not xpu:
        return

    content = path.read_text(encoding="utf-8")
    m = re.search(r"Requires-Dist: paddlepaddle-gpu(==[^\s;]+)", content)
    if not m:
        return

    new = content.replace(m.group(0), f"Requires-Dist: {xpu}", 1)
    path.write_text(new, encoding="utf-8")
    logger.info(f"Rewrote METADATA: {m.group(0)} -> Requires-Dist: {xpu}")


def _patch_wheel(path: Path) -> None:
    """Rewrite METADATA inside a ``.whl`` zip for XPU builds."""
    if not _is_xpu():
        return

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        ext = Path(td) / "ext"
        ext.mkdir()

        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(ext)

        for meta in ext.rglob("*.dist-info/METADATA"):
            _patch_metadata(meta)

        new = Path(td) / path.name
        with zipfile.ZipFile(new, "w", zipfile.ZIP_DEFLATED) as zf_out:
            for f in ext.rglob("*"):
                if f.is_file():
                    zf_out.write(f, f.relative_to(ext))

        path.write_bytes(new.read_bytes())


# -- PEP 517 hooks -----------------------------------------------------------


def get_requires_for_build_wheel(config_settings=None):
    return orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return orig.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    dist_info = orig.prepare_metadata_for_build_wheel(
        metadata_directory, config_settings
    )
    _patch_metadata(Path(metadata_directory) / dist_info / "METADATA")
    return dist_info


def prepare_metadata_for_build_editable(
    metadata_directory, config_settings=None
):
    dist_info = orig.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )
    _patch_metadata(Path(metadata_directory) / dist_info / "METADATA")
    return dist_info


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    wheel_name = orig.build_wheel(
        wheel_directory, config_settings, metadata_directory
    )
    _patch_wheel(Path(wheel_directory) / wheel_name)
    return wheel_name


def build_editable(
    wheel_directory, config_settings=None, metadata_directory=None
):
    wheel_name = orig.build_editable(
        wheel_directory, config_settings, metadata_directory
    )
    _patch_wheel(Path(wheel_directory) / wheel_name)
    return wheel_name


def build_sdist(sdist_directory, config_settings=None):
    return orig.build_sdist(sdist_directory, config_settings)
