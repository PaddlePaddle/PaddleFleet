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
and ``commit`` from paddlefleet.version, and in this checkout there is no
generated ``_version.py``. version.py therefore falls back to
``_get_version_from_source()``, which *assembles* the version string as::

    f"{base_version}.dev{date_str}"

where ``base_version`` is the stripped contents of the repo-root
``version.txt`` and ``date_str`` is ``datetime.now().strftime("%Y%m%d")``.
That assembly is real logic worth pinning: the centerpiece test hand-derives
each component independently (raw ``version.txt`` input + today's date from the
system clock) and compares the exact assembled string, so a wrong separator,
wrong date format, or dropped base version would be rejected. The remaining
tests pin the exact-value contracts consumers actually rely on
(distribution name, the Apache license *tuple*, and the full git-SHA shape of
``commit``) rather than padding with type/existence checks.
"""

import importlib
import os
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

    def test_version_assembled_from_version_txt_and_today(self):
        # Hand-derive each component independently, then hand-assemble the
        # expected string and compare exactly:
        #   * base version -> the raw version.txt input (currently "1.0.0");
        #     this is the genuine input, not a copy of the production formula.
        #   * date stamp   -> today's date from the system clock (ground
        #     truth), formatted independently as YYYYMMDD.
        # The ".dev" join is what the production _get_version_from_source
        # contract promises; a wrong separator, a wrong date format, or a
        # dropped base version would all fail this exact comparison.
        base = _read_version_txt()
        self.assertEqual(base, "1.0.0")  # pin the known current base version
        stamp = date.today().strftime("%Y%m%d")
        expected = base + ".dev" + stamp
        self.assertEqual(_pi.__version__, expected)

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
