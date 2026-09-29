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
"""CPU-observable behavior tests for the extensions ``ops`` bootstrap.

The ``paddlefleet_ops._extensions.ops`` module is a compiled CUDA shared
library; its kernels (tokens_unzip_gather, fused_swiglu_scale, ...) require a
GPU and cannot be validated on CPU. What IS CPU-observable is the loading /
registration machinery in ``paddlefleet_ops.utils`` that assembles the ``ops``
namespace: it is invoked by ``_extensions/__init__.py`` (``from . import ops``)
and by the package ``__init__`` via ``import_custom_ops(..., module_name=".ops")``.
These tests drive that real production source with hand-derived expectations.

The full package import requires paddle (imported at package import time). This
env has no paddle, so when the package import fails we load the real
``utils.py`` source file directly (it has no paddle dependency). That path
exercises the identical production code but does not verify package import or
full startup.
"""

import importlib.util
import os
import sys
import types
import unittest

# Repo root: tests/single_card_tests/extensions/test_ops.py -> up 4 levels.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_UTILS_PATH = os.path.join(
    _REPO_ROOT,
    "packages",
    "paddlefleet_ops",
    "src",
    "paddlefleet_ops",
    "utils.py",
)


def _load_utils():
    """Return the real paddlefleet_ops.utils module, or None if unavailable.

    Prefers the installed package (full env). Falls back to loading the real
    source file directly when paddle is absent, since utils.py itself has no
    paddle dependency.
    """
    try:
        from paddlefleet_ops import utils as _utils

        return _utils
    except ImportError:
        pass
    if not os.path.exists(_UTILS_PATH):
        return None
    spec = importlib.util.spec_from_file_location(
        "_pf_ops_utils_under_test", _UTILS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_UTILS = _load_utils()


@unittest.skipUnless(
    _UTILS is not None,
    "paddlefleet_ops.utils source not importable (no paddle and no source file)",
)
class TestImportCustomOps(unittest.TestCase):
    """import_custom_ops copies real functions into a target namespace."""

    def _make_fake_module(self, name):
        mod = types.ModuleType(name)

        def foo():
            return "foo"

        def _private():
            return "priv"

        def _C_ops():
            return "cops"

        def __dunder__():
            return "dunder"

        mod.foo = foo
        mod._private = _private
        mod._C_ops = _C_ops
        mod.__dunder__ = __dunder__
        mod.NUM = 123  # non-function attribute must be ignored
        self._fns = {"foo": foo, "_private": _private}
        return mod

    def test_copies_functions_filtering_dunder_and_c_ops(self):
        name = "_fake_ops_probe_mod"
        mod = self._make_fake_module(name)
        sys.modules[name] = mod
        self.addCleanup(sys.modules.pop, name, None)

        ns = {}
        _UTILS.import_custom_ops(package=name, module_name=name, global_ns=ns)

        # Only public/single-underscore functions copied; identity preserved.
        self.assertEqual(sorted(ns.keys()), ["_private", "foo"])
        self.assertIs(ns["foo"], self._fns["foo"])
        self.assertIs(ns["_private"], self._fns["_private"])
        # Filtered out: dunder names, the _C_ops sentinel, non-functions.
        self.assertNotIn("_C_ops", ns)
        self.assertNotIn("__dunder__", ns)
        self.assertNotIn("NUM", ns)

    def test_import_failure_is_swallowed_and_namespace_untouched(self):
        # Missing/uncompiled module: must not raise, must not mutate namespace.
        ns = {"sentinel": object()}
        expected = dict(ns)
        _UTILS.import_custom_ops(
            package="missing_pkg",
            module_name="definitely_missing_ops_mod_xyz",
            global_ns=ns,
        )
        self.assertEqual(ns, expected)


@unittest.skipUnless(_UTILS is not None, "utils source not importable")
class TestModuleContext(unittest.TestCase):
    """ModuleContext stashes matching modules and manages sys.path."""

    def test_stash_restore_and_path(self):
        pkg = types.ModuleType("mctxpkg")
        sub = types.ModuleType("mctxpkg.sub")
        other = types.ModuleType("mctxpkgother")
        for k, v in [
            ("mctxpkg", pkg),
            ("mctxpkg.sub", sub),
            ("mctxpkgother", other),
        ]:
            sys.modules[k] = v
            self.addCleanup(sys.modules.pop, k, None)

        probe_path = "/tmp/_mctx_probe_path_xyz"
        self.addCleanup(
            lambda: probe_path in sys.path and sys.path.remove(probe_path)
        )

        ctx = _UTILS.ModuleContext(["mctxpkg"], probe_path)
        with ctx:
            # Exact name and dotted children are stashed away.
            self.assertNotIn("mctxpkg", sys.modules)
            self.assertNotIn("mctxpkg.sub", sys.modules)
            # A name that merely shares a prefix (no dot boundary) is untouched.
            self.assertIs(sys.modules.get("mctxpkgother"), other)
            self.assertEqual(sys.path[0], probe_path)

        # On exit: stashed modules restored by identity, path removed.
        self.assertIs(sys.modules.get("mctxpkg"), pkg)
        self.assertIs(sys.modules.get("mctxpkg.sub"), sub)
        self.assertNotIn(probe_path, sys.path)


@unittest.skipUnless(_UTILS is not None, "utils source not importable")
class TestNamespacePatching(unittest.TestCase):
    """patch_module_namespace / clean_module_namespace rewrite sys.modules."""

    def test_patch_moves_matching_modules_under_prefix(self):
        s = types.ModuleType("srcmod")
        child = types.ModuleType("srcmod.child")
        other = types.ModuleType("srcmodother")
        for k, v in [
            ("srcmod", s),
            ("srcmod.child", child),
            ("srcmodother", other),
        ]:
            sys.modules[k] = v
        for k in [
            "srcmod",
            "srcmod.child",
            "srcmodother",
            "prefix.srcmod",
            "prefix.srcmod.child",
        ]:
            self.addCleanup(sys.modules.pop, k, None)

        _UTILS.patch_module_namespace("srcmod", "prefix.")

        # Exact + dotted children renamed with prefix, identity preserved.
        self.assertNotIn("srcmod", sys.modules)
        self.assertNotIn("srcmod.child", sys.modules)
        self.assertIs(sys.modules.get("prefix.srcmod"), s)
        self.assertIs(sys.modules.get("prefix.srcmod.child"), child)
        # Prefix-only (no dot boundary) sibling untouched.
        self.assertIs(sys.modules.get("srcmodother"), other)

    def test_clean_removes_exact_name_only(self):
        exact = types.ModuleType("cleanme")
        sub = types.ModuleType("cleanme.sub")
        sys.modules["cleanme"] = exact
        sys.modules["cleanme.sub"] = sub
        self.addCleanup(sys.modules.pop, "cleanme", None)
        self.addCleanup(sys.modules.pop, "cleanme.sub", None)

        _UTILS.clean_module_namespace("cleanme")

        self.assertNotIn("cleanme", sys.modules)
        # Dotted child is NOT removed (only exact-name match is popped).
        self.assertIs(sys.modules.get("cleanme.sub"), sub)


@unittest.skipUnless(_UTILS is not None, "utils source not importable")
class TestHardwareIncompatibleBlocker(unittest.TestCase):
    """The meta-path finder raises for blocked modules, ignores the rest."""

    def test_raises_for_exact_and_dotted_child(self):
        blocker = _UTILS.HardwareIncompatibleBlocker(
            {"paddlefleet_ops.deep_gemm": "deep_gemm not supported: boom"}
        )
        with self.assertRaises(RuntimeError) as cm:
            blocker.find_spec("paddlefleet_ops.deep_gemm", None)
        self.assertEqual(str(cm.exception), "deep_gemm not supported: boom")

        with self.assertRaises(RuntimeError):
            blocker.find_spec("paddlefleet_ops.deep_gemm.submod", None)

    def test_returns_none_for_unblocked_module(self):
        blocker = _UTILS.HardwareIncompatibleBlocker(
            {"paddlefleet_ops.deep_gemm": "boom"}
        )
        # Non-matching name: finder abstains (returns None), no raise.
        self.assertIsNone(blocker.find_spec("some.other.module", None))
        # Prefix-only without dot boundary must not match.
        self.assertIsNone(blocker.find_spec("paddlefleet_ops.deep_gemmX", None))


if __name__ == "__main__":
    unittest.main()
