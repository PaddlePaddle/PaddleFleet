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

"""CPU-only behavior tests for the ``paddlefleet`` top-level package wiring
and the lazy-import mechanism that exposes it.

Two device-independent surfaces are covered:

  * ``paddlefleet.utils.lazy_import._LazyModule`` -- the attribute-resolution
    engine the top-level package installs into ``sys.modules``. Its dispatch
    order (cached ``extra_objects`` before ``import_structure`` before the
    ``AttributeError`` fallback), its ``__all__`` construction, ``__dir__`` and
    ``__reduce__`` are exercised against hand-built fixtures whose expected
    results are derived by hand, never from the class output.
  * The top-level ``paddlefleet`` package object itself -- the ``mpu``/
    ``parallel_state`` alias identity, the ``Timers`` re-export identity, and
    the ``package_info`` metadata re-export values.

Importing ``paddlefleet`` pulls in paddle + transformers transitively; on a
CPU-only host where those wheels are absent the import raises and every class
is skipped with the real error repr, never silently passed.
"""

import unittest

try:
    import paddlefleet
    from paddlefleet.utils.lazy_import import _LazyModule

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddlefleet = None
    _LazyModule = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddlefleet not importable on this CPU-only host: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestLazyModuleDispatch(unittest.TestCase):
    """``_LazyModule.__getattr__`` resolution order and metadata construction.

    Every fixture below is hand-authored; the anchor package name is chosen to
    be un-importable so the ``import_structure`` branch resolves to a
    deterministic ``ModuleNotFoundError`` rather than pulling a real module.
    """

    ANCHOR = "pf_lazy_probe_pkg_zzq"
    FILE = "/tmp/pf_lazy_probe_pkg_zzq/__init__.py"

    def _make(self):
        # "Foo" appears BOTH as an extra_object and as an import_structure
        # value on purpose, to pin the priority of extra_objects.
        structure = {"alpha": ["Foo", "Bar"], "beta": []}
        extra = {"answer": 42, "Foo": "shadow_value"}
        return (
            _LazyModule(self.ANCHOR, self.FILE, structure, extra_objects=extra),
            structure,
        )

    def test_extra_objects_take_priority_over_import_structure(self):
        lm, _ = self._make()
        # "Foo" is a class_to_module entry, but the cached object must win, so
        # no import is attempted and the literal string is handed back.
        self.assertEqual(lm.Foo, "shadow_value")
        self.assertEqual(lm.answer, 42)
        # Second access returns the same cached value (now a real attribute).
        self.assertEqual(lm.answer, 42)

    def test_import_structure_value_failure_is_reported(self):
        lm, _ = self._make()
        # "Bar" -> module "alpha"; the anchor package cannot be imported, so
        # the helper must surface a ModuleNotFoundError naming the object.
        with self.assertRaisesRegex(ModuleNotFoundError, "Bar"):
            _ = lm.Bar

    def test_bare_module_name_failure_is_reported(self):
        lm, _ = self._make()
        # "beta" is a module key with no members; resolving it still goes
        # through the (failing) relative import and names the module.
        with self.assertRaisesRegex(ModuleNotFoundError, "beta"):
            _ = lm.beta

    def test_unknown_attribute_raises_attribute_error(self):
        lm, _ = self._make()
        with self.assertRaisesRegex(AttributeError, "no attribute nope"):
            _ = lm.nope

    def test_all_is_modules_plus_flattened_members(self):
        lm, _ = self._make()
        # Hand-derived: the two module keys plus alpha's two members; beta
        # contributes nothing. Order is set-dependent, so compare as multisets.
        self.assertCountEqual(lm.__all__, ["alpha", "beta", "Foo", "Bar"])

    def test_dir_is_sorted_and_surfaces_all_entries(self):
        lm, _ = self._make()
        listing = lm.__dir__()
        self.assertEqual(listing, sorted(listing))
        for name in ("alpha", "beta", "Foo", "Bar"):
            self.assertIn(name, listing)

    def test_reduce_roundtrips_name_file_and_structure(self):
        lm, structure = self._make()
        self.assertEqual(
            lm.__reduce__(),
            (_LazyModule, (self.ANCHOR, self.FILE, structure)),
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTopLevelPackageWiring(unittest.TestCase):
    """Observable contracts of the installed ``paddlefleet`` module object."""

    def test_mpu_is_the_same_object_as_parallel_state(self):
        # The alias must be identity-equal, and it must resolve to the real
        # submodule, not a stub -- checked via its dotted module name.
        self.assertIs(paddlefleet.mpu, paddlefleet.parallel_state)
        self.assertEqual(
            paddlefleet.parallel_state.__name__, "paddlefleet.parallel_state"
        )

    def test_timers_reexport_is_the_submodule_class(self):
        # Top-level ``Timers`` must be the very object defined in the
        # ``paddlefleet.timers`` submodule, proving the re-export wiring.
        self.assertIs(paddlefleet.Timers, paddlefleet.timers.Timers)
        self.assertIsInstance(paddlefleet.Timers, type)
        self.assertEqual(paddlefleet.Timers.__name__, "Timers")

    def test_metadata_reexports_expected_constants(self):
        # Values are the hand-written specification from package_info; they are
        # NOT read back from production at assertion time.
        self.assertEqual(paddlefleet.__package_name__, "paddlefleet")
        self.assertEqual(paddlefleet.__contact_names__, "PaddlePaddle")
        # Deliberate quirk: __license__ is a 1-tuple, not a bare string.
        self.assertEqual(paddlefleet.__license__, ("Apache Software License",))

    def test_metadata_is_wired_to_package_info_submodule(self):
        # The top-level dunder must be the same value the canonical
        # ``package_info`` submodule defines (re-export wiring, not a stub).
        self.assertEqual(
            paddlefleet.__version__, paddlefleet.package_info.__version__
        )
        self.assertEqual(
            paddlefleet.__repository_url__,
            paddlefleet.package_info.__repository_url__,
        )


if __name__ == "__main__":
    unittest.main()
