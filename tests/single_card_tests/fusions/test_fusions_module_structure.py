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
"""Contract-surface tests for the ``paddlefleet.fusions`` package.

These verify the *public export surface* of the fusion modules, in the spirit
of ``TestModuleSurface`` in ``tests/formers/cli/test_ernie_fp8_utils.py``.
Because these modules define **no** ``__all__``, the contract is expressed as
the exact set of public functions/classes each module *defines locally*
(``obj.__module__ == that module``), plus the re-export / lazy-import wiring
that binds sibling kernels. Every expected value is hand-derived by reading
the sources -- no ``inspect.getsource``, no self-referential expectations.

Two structural facts are pinned deliberately (both hand-derived):
  * ``paddlefleet.fusions`` ships no ``__init__.py``; it resolves as a PEP 420
    namespace package (``__file__`` is None, ``__path__`` present).
  * ``fused_rms_norm`` imports its paddle kernel unconditionally, while
    ``fused_layer_norm`` guards the equivalent import with try/except and a
    ``HAVE_FUSED_LAYER_NORM`` flag; ``HAVE_PERSIST_LAYER_NORM`` is a hardcoded
    ``False``. The kernel re-exports are load-bearing (used in ``forward``).

paddlefleet imports paddle at import time; when paddle (or a required paddle
kernel) is genuinely absent the ImportError is captured precisely and the
whole module is skipped with an honest reason -- it is never swallowed.
"""

import inspect
import unittest

try:
    import paddle

    import paddlefleet.fusions as fusions_pkg
    from paddlefleet.fusions import (
        fused_bias_dropout,
        fused_layer_norm,
        fused_rms_norm,
        fused_softmax,
        fused_swiglu_scale,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # honest: paddle / a paddle kernel genuinely absent
    paddle = None
    fusions_pkg = None
    fused_bias_dropout = None
    fused_layer_norm = None
    fused_rms_norm = None
    fused_softmax = None
    fused_swiglu_scale = None
    _IMPORT_ERROR = exc

_HAS_PADDLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle/paddlefleet.fusions not importable: {_IMPORT_ERROR}"
    if _IMPORT_ERROR is not None
    else ""
)

# Hand-derived from the sources: the exact public (non-underscore) callables
# each module DEFINES locally. Re-exported kernels/constants live in the module
# namespace but have a foreign ``__module__`` and are checked separately below.
_EXPECTED = {
    "paddlefleet.fusions.fused_bias_dropout": {
        "functions": {"bias_dropout_add_unfused", "get_bias_dropout_add"},
        "classes": set(),
    },
    "paddlefleet.fusions.fused_softmax": {
        "functions": set(),
        "classes": {"SoftmaxOne", "FusedScaleMaskSoftmax"},
    },
    "paddlefleet.fusions.fused_swiglu_scale": {
        "functions": {
            "fused_swiglu_scale_forward",
            "fused_swiglu_scale_backward",
        },
        "classes": set(),
    },
    "paddlefleet.fusions.fused_rms_norm": {
        "functions": set(),
        "classes": {"FusedRmsNorm"},
    },
    "paddlefleet.fusions.fused_layer_norm": {
        "functions": set(),
        "classes": {"FusedLayerNorm"},
    },
}


def _modules_by_name():
    """Map qualified name -> live module object (only call when not skipped)."""
    return {
        "paddlefleet.fusions.fused_bias_dropout": fused_bias_dropout,
        "paddlefleet.fusions.fused_softmax": fused_softmax,
        "paddlefleet.fusions.fused_swiglu_scale": fused_swiglu_scale,
        "paddlefleet.fusions.fused_rms_norm": fused_rms_norm,
        "paddlefleet.fusions.fused_layer_norm": fused_layer_norm,
    }


def _local_public(mod, predicate):
    """Public names in ``mod`` satisfying ``predicate`` and DEFINED in ``mod``.

    ``__module__ == mod.__name__`` excludes imported helpers and re-exported
    kernels, isolating what the module itself contributes to the surface.
    """
    names = set()
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if predicate(obj) and getattr(obj, "__module__", None) == mod.__name__:
            names.add(name)
    return names


# APPEND_TESTS
@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestFusionsNamespacePackage(unittest.TestCase):
    """``paddlefleet.fusions`` is a PEP 420 namespace package (no __init__)."""

    def test_is_namespace_package_without_init(self):
        # A regular package would bind __file__ to its __init__.py; a namespace
        # package has __file__ is None but still exposes __path__.
        self.assertEqual(fusions_pkg.__name__, "paddlefleet.fusions")
        self.assertIsNone(getattr(fusions_pkg, "__file__", None))
        self.assertTrue(hasattr(fusions_pkg, "__path__"))


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestDefinedPublicSurface(unittest.TestCase):
    """The set of locally-defined public callables is the module contract.

    This is the ``__all__``-analog for modules that ship no ``__all__``:
    the computed live surface must equal the hand-derived expected set
    exactly (no additions, removals, or renames), split by kind so a class
    cannot silently masquerade as a function or vice versa.
    """

    def test_defined_functions_match_exactly(self):
        for qname, mod in _modules_by_name().items():
            with self.subTest(module=qname):
                self.assertEqual(mod.__name__, qname)
                got = _local_public(mod, inspect.isfunction)
                self.assertEqual(got, _EXPECTED[qname]["functions"])

    def test_defined_classes_match_exactly(self):
        for qname, mod in _modules_by_name().items():
            with self.subTest(module=qname):
                got = _local_public(mod, inspect.isclass)
                self.assertEqual(got, _EXPECTED[qname]["classes"])

    def test_each_expected_name_resolves_to_defining_module(self):
        # Identity of binding: every promised name is present, of the right
        # kind, and actually defined here (not a shadowing re-export).
        for qname, mod in _modules_by_name().items():
            spec = _EXPECTED[qname]
            for fname in spec["functions"]:
                with self.subTest(module=qname, func=fname):
                    obj = getattr(mod, fname)
                    self.assertTrue(inspect.isfunction(obj))
                    self.assertEqual(obj.__module__, qname)
            for cname in spec["classes"]:
                with self.subTest(module=qname, cls=cname):
                    obj = getattr(mod, cname)
                    self.assertTrue(inspect.isclass(obj))
                    self.assertEqual(obj.__module__, qname)

    def test_public_layer_classes_subclass_paddle_layer(self):
        # All exported classes in this scope are paddle Layers by contract.
        layer_classes = {
            fused_softmax.SoftmaxOne,
            fused_softmax.FusedScaleMaskSoftmax,
            fused_rms_norm.FusedRmsNorm,
            fused_layer_norm.FusedLayerNorm,
        }
        for cls in layer_classes:
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, paddle.nn.Layer))


# APPEND_REEXPORT
@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestReExportWiring(unittest.TestCase):
    """Re-export / lazy-import wiring must bind the exact sibling objects.

    These names have a foreign ``__module__`` (they are imported, not defined),
    yet they are load-bearing: the ``forward`` paths call through them. The
    expected targets are hand-derived from the ``import`` statements and are
    obtained via an independent import here, so this is an identity contract
    against the true defining module -- not a module-vs-itself comparison.
    """

    def test_fused_rms_norm_reexports_paddle_kernel(self):
        # `from paddle.incubate.nn.functional.fused_rms_norm import
        #  fused_rms_norm` -- unconditional; the module attribute must be that
        # exact kernel used inside FusedRmsNorm.forward.
        from paddle.incubate.nn.functional.fused_rms_norm import (
            fused_rms_norm as paddle_rms_kernel,
        )

        self.assertIs(fused_rms_norm.fused_rms_norm, paddle_rms_kernel)
        self.assertTrue(callable(fused_rms_norm.fused_rms_norm))

    def test_fused_swiglu_scale_reexports_paddle_swiglu(self):
        # `from paddle.nn.functional import swiglu` -- used in the CPU-fallback
        # forward as `out = swiglu(x)`.
        self.assertIs(fused_swiglu_scale.swiglu, paddle.nn.functional.swiglu)
        self.assertTrue(callable(fused_swiglu_scale.swiglu))

    def test_fused_layer_norm_conditional_kernel_flag_is_consistent(self):
        # HAVE_FUSED_LAYER_NORM is a bool that must agree with whether the
        # guarded kernel import succeeded and bound `fused_layer_norm`.
        have = fused_layer_norm.HAVE_FUSED_LAYER_NORM
        self.assertIsInstance(have, bool)
        bound = getattr(fused_layer_norm, "fused_layer_norm", None)
        if have:
            from paddle.incubate.nn.functional.fused_layer_norm import (
                fused_layer_norm as paddle_ln_kernel,
            )

            self.assertIs(bound, paddle_ln_kernel)
        else:
            self.assertIsNone(bound)

    def test_have_persist_layer_norm_is_hardcoded_false(self):
        # Hand-derived: both norm modules hardcode this to False, which forces
        # the persistent-kernel branch off in FusedLayerNorm.__init__.
        self.assertIs(fused_layer_norm.HAVE_PERSIST_LAYER_NORM, False)
        self.assertIs(fused_rms_norm.HAVE_PERSIST_LAYER_NORM, False)


if __name__ == "__main__":
    unittest.main()
