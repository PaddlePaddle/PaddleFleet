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

"""Behavior tests for ``paddlefleet_ops.utils``.

Module under test: ``paddlefleet_ops.utils`` (repo root
``packages/paddlefleet_ops/src/paddlefleet_ops/utils.py``). In the module map
this is the "计算优化" boundary (仓库根 ``packages/paddlefleet_ops/``); the
helpers here are pure-Python support code for the compiled ecosystem package:
module-namespace stashing / patching / cleaning, a meta-path import blocker for
hardware-incompatible libraries, nvshmem host-library discovery, and CUDA
version parsing. These are CPU-observable, so they are asserted with
hand-derived expectations under the no-card environment.

Import note: ``import paddlefleet_ops`` executes the package ``__init__`` which
imports ``paddle`` and performs CUDA/ecosystem initialization. Where that
package import is unavailable (e.g. ``paddle`` not installed, or the compiled
extension is absent), we fall back to loading the real ``utils.py`` source
module directly by file path. This exercises the genuine production functions;
it does NOT verify the paddle-dependent package ``__init__`` path. If neither
import succeeds, every test class skips honestly with the captured reason
rather than fake-passing.
"""

import importlib
import importlib.abc  # ensure stdlib submodule is registered before source load
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

# --- Import the module under test -------------------------------------------
_utils = None
_IMPORT_VIA = None
_IMPORT_ERROR = None

_SRC_ROOT = (
    Path(__file__).resolve().parents[3] / "packages" / "paddlefleet_ops" / "src"
)
_UTILS_FILE = _SRC_ROOT / "paddlefleet_ops" / "utils.py"

try:
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    from paddlefleet_ops import utils as _utils

    _IMPORT_VIA = "package"
except ImportError as _pkg_exc:
    # Package __init__ needs paddle / compiled extension. Fall back to loading
    # the pure-Python source file directly (no package __init__ side effects).
    try:
        _spec = importlib.util.spec_from_file_location(
            "paddlefleet_ops_utils_under_test", str(_UTILS_FILE)
        )
        _utils = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_utils)
        _IMPORT_VIA = "source-file"
    except Exception as _src_exc:
        _IMPORT_ERROR = (
            f"package import failed ({_pkg_exc!r}); source-file load of "
            f"{_UTILS_FILE} also failed ({_src_exc!r})"
        )

_AVAILABLE = _utils is not None
_SKIP_REASON = _IMPORT_ERROR or ""


def _fresh_module(name):
    """Build a distinct, identifiable module object for sys.modules fixtures."""
    mod = types.ModuleType(name)
    mod._probe_tag = name
    return mod


def _register_module(testcase, name, mod):
    """Put ``mod`` in sys.modules under ``name`` with guaranteed cleanup."""
    sys.modules[name] = mod
    testcase.addCleanup(lambda: sys.modules.pop(name, None))


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestImportCustomOps(unittest.TestCase):
    """import_custom_ops: real import, function filtering, graceful skip."""

    def _write_probe_module(self, source):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmpdir, True))
        modname = "probe_ops_src_module"
        (Path(tmpdir) / (modname + ".py")).write_text(source)
        sys.path.insert(0, tmpdir)
        self.addCleanup(lambda: sys.path.remove(tmpdir))
        self.addCleanup(lambda: sys.modules.pop(modname, None))
        return modname

    def test_collects_functions_and_filters_c_ops(self):
        modname = self._write_probe_module(
            "def add_one(x):\n"
            "    return x + 1\n"
            "def mul_two(x):\n"
            "    return x * 2\n"
            "def _helper(x):\n"
            "    return x - 1\n"
            "def _C_ops():\n"
            "    return 'sentinel'\n"
            "some_int = 5\n"
        )
        sentinel = object()
        ns = {"preexisting": sentinel}

        self._call(modname, ns)

        # Real function objects are installed and actually behave.
        self.assertEqual(ns["add_one"](3), 4)
        self.assertEqual(ns["mul_two"](3), 6)
        # Single-underscore helper is NOT filtered (only dunder / exact _C_ops).
        self.assertEqual(ns["_helper"](3), 2)
        # Exactly-named _C_ops is filtered out.
        self.assertNotIn("_C_ops", ns)
        # Non-function members (module-level int) are not collected.
        self.assertNotIn("some_int", ns)
        # Pre-existing namespace entries are preserved untouched.
        self.assertIs(ns["preexisting"], sentinel)
        # Exactly the expected new keys were added.
        self.assertEqual(
            set(ns) - {"preexisting"}, {"add_one", "mul_two", "_helper"}
        )

    def test_installed_object_is_the_real_function(self):
        modname = self._write_probe_module(
            "def add_one(x):\n    return x + 1\n"
        )
        real = importlib.import_module(modname)
        ns = {}
        self._call(modname, ns)
        self.assertIs(ns["add_one"], real.add_one)

    def test_missing_module_is_swallowed_without_mutation(self):
        sentinel = object()
        ns = {"keep": sentinel}
        # Nonexistent module: production contract logs a warning and returns,
        # leaving the namespace fully intact (no crash, no partial writes).
        self._call("no_such_probe_module_zzz", ns)
        self.assertEqual(ns, {"keep": sentinel})

    def _call(self, modname, ns):
        _utils.import_custom_ops(
            package="probe_pkg_placeholder",
            module_name=modname,
            global_ns=ns,
        )


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestModuleContext(unittest.TestCase):
    """ModuleContext: prefix-scoped stash/restore of sys.modules + sys.path."""

    def test_stash_restore_with_prefix_boundary_and_identity(self):
        pkg = _fresh_module("pctx_pkg")
        sub = _fresh_module("pctx_pkg.sub")
        other = _fresh_module("pctx_other")
        # Name shares the "pctx_pkg" prefix but WITHOUT a dot separator:
        # must NOT be treated as a submodule.
        near = _fresh_module("pctx_pkgxtra")
        _register_module(self, "pctx_pkg", pkg)
        _register_module(self, "pctx_pkg.sub", sub)
        _register_module(self, "pctx_other", other)
        _register_module(self, "pctx_pkgxtra", near)

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmpdir, True))
        self.addCleanup(lambda: tmpdir in sys.path and sys.path.remove(tmpdir))

        with _utils.ModuleContext(["pctx_pkg"], Path(tmpdir)):
            # Matching module and its dotted submodule are removed.
            self.assertNotIn("pctx_pkg", sys.modules)
            self.assertNotIn("pctx_pkg.sub", sys.modules)
            # Unrelated and prefix-without-dot modules are left in place.
            self.assertIs(sys.modules["pctx_other"], other)
            self.assertIs(sys.modules["pctx_pkgxtra"], near)
            # Path is inserted at the front of sys.path.
            self.assertEqual(sys.path[0], tmpdir)

        # On exit: path removed, stashed modules restored to same objects.
        self.assertNotIn(tmpdir, sys.path)
        self.assertIs(sys.modules["pctx_pkg"], pkg)
        self.assertIs(sys.modules["pctx_pkg.sub"], sub)
        self.assertIs(sys.modules["pctx_other"], other)

    def test_empty_names_only_manages_path(self):
        before = _fresh_module("pctx_untouched")
        _register_module(self, "pctx_untouched", before)
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmpdir, True))
        self.addCleanup(lambda: tmpdir in sys.path and sys.path.remove(tmpdir))

        with _utils.ModuleContext([], Path(tmpdir)):
            self.assertEqual(sys.path[0], tmpdir)
            # No module names to match: nothing is stashed.
            self.assertIs(sys.modules["pctx_untouched"], before)

        self.assertNotIn(tmpdir, sys.path)
        self.assertIs(sys.modules["pctx_untouched"], before)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestPatchModuleNamespace(unittest.TestCase):
    """patch_module_namespace: re-key modules under a new prefix."""

    def test_moves_module_and_submodules_preserving_identity(self):
        pkg = _fresh_module("ppatch_pkg")
        sub = _fresh_module("ppatch_pkg.sub")
        near = _fresh_module("ppatch_pkgxtra")  # prefix without dot: untouched
        _register_module(self, "ppatch_pkg", pkg)
        _register_module(self, "ppatch_pkg.sub", sub)
        _register_module(self, "ppatch_pkgxtra", near)
        self.addCleanup(lambda: sys.modules.pop("newns.ppatch_pkg", None))
        self.addCleanup(lambda: sys.modules.pop("newns.ppatch_pkg.sub", None))

        _utils.patch_module_namespace("ppatch_pkg", "newns.")

        # Old keys gone, new keys hold the SAME module objects.
        self.assertNotIn("ppatch_pkg", sys.modules)
        self.assertNotIn("ppatch_pkg.sub", sys.modules)
        self.assertIs(sys.modules["newns.ppatch_pkg"], pkg)
        self.assertIs(sys.modules["newns.ppatch_pkg.sub"], sub)
        # Prefix-without-dot sibling is not moved.
        self.assertIs(sys.modules["ppatch_pkgxtra"], near)
        self.assertNotIn("newns.ppatch_pkgxtra", sys.modules)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestCleanModuleNamespace(unittest.TestCase):
    """clean_module_namespace: EXACT-name removal only (no submodules)."""

    def test_removes_exact_name_but_keeps_submodules(self):
        pkg = _fresh_module("pclean_pkg")
        sub = _fresh_module("pclean_pkg.sub")
        _register_module(self, "pclean_pkg", pkg)
        _register_module(self, "pclean_pkg.sub", sub)

        _utils.clean_module_namespace("pclean_pkg")

        # Exact match removed; dotted submodule intentionally left behind
        # (this distinguishes clean_* from the prefix-based helpers).
        self.assertNotIn("pclean_pkg", sys.modules)
        self.assertIs(sys.modules["pclean_pkg.sub"], sub)

    def test_absent_name_leaves_others_intact(self):
        keep = _fresh_module("pclean_keep")
        _register_module(self, "pclean_keep", keep)
        _utils.clean_module_namespace("pclean_definitely_absent_zzz")
        self.assertIs(sys.modules["pclean_keep"], keep)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestHardwareIncompatibleBlocker(unittest.TestCase):
    """HardwareIncompatibleBlocker.find_spec: raise mapped msg, else None."""

    def test_blocked_exact_and_submodule_raise_mapped_message(self):
        blocker = _utils.HardwareIncompatibleBlocker(
            {"blk.alpha": "alpha reason", "blk.beta": "beta reason"}
        )

        with self.assertRaises(RuntimeError) as ctx:
            blocker.find_spec("blk.alpha", None, None)
        self.assertEqual(str(ctx.exception), "alpha reason")

        # Dotted submodule of a blocked package also raises the SAME message.
        with self.assertRaises(RuntimeError) as ctx2:
            blocker.find_spec("blk.beta.child", None, None)
        self.assertEqual(str(ctx2.exception), "beta reason")

    def test_prefix_without_dot_and_unrelated_return_none(self):
        blocker = _utils.HardwareIncompatibleBlocker(
            {"blk.alpha": "alpha reason"}
        )
        # "blk.alphaX" shares a char-prefix but is not the module nor a dotted
        # submodule -> not blocked.
        self.assertIsNone(blocker.find_spec("blk.alphaX", None, None))
        self.assertIsNone(blocker.find_spec("some.other.mod", None, None))


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestGetNvshmemHostLibPath(unittest.TestCase):
    """get_nvshmem_host_lib_path: locate libnvshmem_host.so.* under <dir>/lib."""

    def test_returns_resolved_path_of_matching_lib(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, True))
        lib = base / "lib"
        lib.mkdir()
        target = lib / "libnvshmem_host.so.3"
        target.write_bytes(b"\x00")

        result = _utils.get_nvshmem_host_lib_path(str(base))
        self.assertEqual(result, target.resolve())

    def test_finds_lib_in_nested_subdirectory(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, True))
        nested = base / "lib" / "deep"
        nested.mkdir(parents=True)
        target = nested / "libnvshmem_host.so.12"
        target.write_bytes(b"\x00")

        result = _utils.get_nvshmem_host_lib_path(str(base))
        self.assertEqual(result, target.resolve())

    def test_missing_lib_raises_with_searched_path(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, True))
        (base / "lib").mkdir()  # present but empty

        with self.assertRaises(FileNotFoundError) as ctx:
            _utils.get_nvshmem_host_lib_path(str(base))
        msg = str(ctx.exception)
        self.assertIn("libnvshmem_host.so", msg)
        self.assertIn(str(base / "lib"), msg)


@unittest.skipUnless(_AVAILABLE, _SKIP_REASON)
class TestGetCudaVersion(unittest.TestCase):
    """get_cuda_version: nvcc discovery + release-string parsing to (M, m)."""

    def test_raises_when_nvcc_absent(self):
        with mock.patch.object(_utils.shutil, "which", return_value=None):
            with self.assertRaises(FileNotFoundError):
                _utils.get_cuda_version()

    def test_parses_major_minor_as_ints(self):
        stdout = (
            "nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Cuda compilation tools, release 12.4, V12.4.131\n"
        )
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return types.SimpleNamespace(stdout=stdout)

        with (
            mock.patch.object(
                _utils.shutil, "which", return_value="/usr/bin/nvcc"
            ),
            mock.patch.object(_utils.subprocess, "run", side_effect=fake_run),
        ):
            result = _utils.get_cuda_version()

        self.assertEqual(result, (12, 4))
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], int)
        # The real nvcc query command is what gets dispatched.
        self.assertEqual(captured["cmd"], ["nvcc", "--version"])

    def test_parses_distinct_version_from_output(self):
        # A second, different string proves the value is extracted from the
        # output rather than being a constant.
        stdout = "Cuda compilation tools, release 11.8, V11.8.89\n"
        with (
            mock.patch.object(
                _utils.shutil, "which", return_value="/usr/bin/nvcc"
            ),
            mock.patch.object(
                _utils.subprocess,
                "run",
                return_value=types.SimpleNamespace(stdout=stdout),
            ),
        ):
            self.assertEqual(_utils.get_cuda_version(), (11, 8))

    def test_unparsable_output_raises_valueerror(self):
        stdout = "no recognizable version banner here\n"
        with (
            mock.patch.object(
                _utils.shutil, "which", return_value="/usr/bin/nvcc"
            ),
            mock.patch.object(
                _utils.subprocess,
                "run",
                return_value=types.SimpleNamespace(stdout=stdout),
            ),
            self.assertRaises(ValueError) as ctx,
        ):
            _utils.get_cuda_version()
        self.assertIn(stdout.strip(), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
