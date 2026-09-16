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

"""Behavior unit tests for paddlefleet.package_info (pure Python, no device).

package_info is mostly package metadata, but it re-exports ``__version__``
and ``commit`` from paddlefleet.version. When the package has been built (as
in CI) a generated ``_version.py`` is present and ``version.py`` re-exports it;
``build_backend.py`` writes that file as::

    __version__ = f"{base_version}.dev{YYYYMMDD}+{commit_short}"   # dev builds
    __version__ = f"{base_version}.post{YYYYMMDD}+{commit_short}"  # release/*

where ``base_version`` is the stripped ``version.txt`` contents, ``YYYYMMDD`` is
the HEAD commit's own date, and ``commit_short`` is the 11-char prefix of the
full 40-char SHA also exported as ``commit``. When no ``_version.py`` exists,
``version.py`` falls back to ``f"{base_version}.dev{today}"`` (no ``+`` local
segment). The centerpiece test decomposes the assembled string and checks each
component against an independent source (raw ``version.txt``; the git commit
date; the exported full SHA), so a wrong separator, wrong/absent date, dropped
base version, or a mismatched short SHA would all be rejected. The remaining
tests pin the exact-value contracts consumers actually rely on (distribution
name, the Apache license *tuple*, and the full git-SHA shape of ``commit``).
"""

import importlib
import os
import re
import subprocess
import sys
import types
import unittest
from datetime import date

# Locate <repo>/src so the real production source is importable, and <repo>
# itself so we can read the genuine version.txt input.
_REPO_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
_REPO_ROOT = os.path.dirname(_REPO_SRC)
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

try:
    # Preferred path: exercise the real package import when paddle is present.
    from paddlefleet import package_info as _pi
except ImportError:
    # paddlefleet/__init__.py transitively imports paddle (via parallel_state),
    # which is absent in the no-card CPU environment. package_info.py and the
    # version.py it re-exports from are pure Python and import nothing from
    # paddle. To exercise the genuine production source without triggering the
    # heavy package __init__, register a lightweight package object whose
    # __path__ points at the real src/paddlefleet directory, then import the
    # real submodule through it (its ``from .version import ...`` relative
    # import resolves against that __path__). This runs the actual module code
    # -- constants plus the version assembly -- not a stub. The full
    # ``import paddlefleet`` chain (which needs paddle) is intentionally not
    # exercised here.
    _pkg = types.ModuleType("paddlefleet")
    _pkg.__path__ = [os.path.join(_REPO_SRC, "paddlefleet")]
    sys.modules["paddlefleet"] = _pkg
    _pi = importlib.import_module("paddlefleet.package_info")


def _read_version_txt():
    """Independently read the raw base-version input (not the prod formula)."""
    with open(os.path.join(_REPO_ROOT, "version.txt"), encoding="utf-8") as f:
        return f.read().strip()


class TestPackageInfo(unittest.TestCase):
    """Contract tests for the assembled version and metadata constants."""

    @staticmethod
    def _git_commit_date(commit):
        """Independently read HEAD's own commit date as YYYYMMDD (or None).

        This is the ground truth ``build_backend.py`` stamps into the generated
        ``_version.py`` (``--date=format:%Y%m%d`` of the commit that was built).
        Only genuine git/subprocess failures return None (so the caller can
        skip the date cross-check when git history is unavailable); this narrow
        catch does not swallow any error from the code under test.
        """
        try:
            out = subprocess.check_output(
                [
                    "git",
                    "show",
                    "-s",
                    "--format=%cd",
                    "--date=format:%Y%m%d",
                    commit,
                ],
                cwd=_REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        return out.decode("utf-8").strip()

    def test_version_assembled_from_version_txt_and_commit(self):
        # Decompose the assembled __version__ and check each component against
        # an INDEPENDENT source, never re-deriving it from the production code:
        #   * base version -> the raw version.txt input (currently "1.0.0").
        #   * When built in CI a generated _version.py is present and the string
        #     is ``f"{base}.{dev|post}{YYYYMMDD}+{commit_short}"`` where the date
        #     is HEAD's own commit date and commit_short is the 11-char prefix
        #     of the full SHA also exported as ``commit``.
        #   * With no _version.py the source fallback is ``f"{base}.dev{today}"``
        #     (no ``+`` local segment).
        # A wrong separator, wrong/absent date, dropped base version, or a
        # short SHA that is not a prefix of the exported commit all fail here.
        base = _read_version_txt()
        self.assertEqual(base, "1.0.0")  # pin the known current base version
        version = _pi.__version__

        if "+" in version:
            # Built form: "<base>.<dev|post><YYYYMMDD>+<commit_short>".
            core, local = version.split("+", 1)
            # The local segment is the short commit and must be a genuine hex
            # prefix of the full 40-char SHA exported as ``commit`` -- this ties
            # the two exports together (a swapped/stale short SHA is rejected).
            self.assertGreaterEqual(len(local), 11)
            self.assertTrue(all(c in "0123456789abcdef" for c in local))
            self.assertTrue(
                _pi.commit.startswith(local),
                f"version local segment {local!r} is not a prefix of "
                f"commit {_pi.commit!r}",
            )
            m = re.fullmatch(re.escape(base) + r"\.(dev|post)(\d{8})", core)
            self.assertIsNotNone(
                m, f"unexpected version core {core!r} for base {base!r}"
            )
            stamp = m.group(2)
            # Cross-check the date stamp against HEAD's real commit date.
            expected_stamp = self._git_commit_date(_pi.commit)
            if expected_stamp is not None:
                self.assertEqual(stamp, expected_stamp)
        else:
            # Source fallback: "<base>.dev<today>" (no local segment).
            stamp = date.today().strftime("%Y%m%d")
            self.assertEqual(version, base + ".dev" + stamp)

    def test_package_name_is_paddlefleet(self):
        # The distribution/import name every consumer depends on. Exact value.
        self.assertEqual(_pi.__package_name__, "paddlefleet")

    def test_license_is_single_element_apache_tuple(self):
        # __license__ is a 1-tuple, not a bare string -- a real structural
        # contract: setup-metadata consumers must index element [0]. Asserting
        # the exact tuple (and its length) catches an accidental change to a
        # bare string, which would silently drop the trailing comma.
        self.assertEqual(_pi.__license__, ("Apache Software License",))
        self.assertEqual(len(_pi.__license__), 1)

    def test_commit_is_full_40_char_lowercase_git_sha(self):
        # With no generated _version.py, commit comes from
        # _get_commit_from_source() -> `git rev-parse HEAD`, whose documented
        # output is the full 40-character lowercase hex SHA-1 of HEAD. If the
        # module imported at all, that resolution already succeeded, so here we
        # pin its exact structural contract (not merely "is a str"): 40 chars,
        # all lowercase hex. A short SHA, empty string, or error text fails.
        self.assertEqual(len(_pi.commit), 40)
        self.assertTrue(all(c in "0123456789abcdef" for c in _pi.commit))


if __name__ == "__main__":
    unittest.main()
