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
"""Behavior tests for ``paddlefleet.spec_utils``.

``spec_utils`` is a backward-compatibility shim: it re-exports ``LayerSpec``
and ``build_layer`` (an alias of Paddle's ``build_spec_layer``) so existing
imports keep working after the symbols were migrated into Paddle. The contract
this file pins down, exercised through the shim's public names against the real
Paddle collaborator that does the resolution:

* the re-exported names are bound to the exact Paddle symbols, and ``__all__``
  is the promised public surface;
* a ``LayerSpec`` wrapping a *class* resolves to an instance built with the
  spec's ``extra_kwargs`` (the constructor is really fed those kwargs, so a
  missing-wire would raise ``TypeError`` rather than silently pass);
* runtime kwargs passed to ``build_layer`` are merged into the constructor call
  alongside ``extra_kwargs``;
* a *function* layer is returned as-is, never invoked;
* a ``(module_path, attr)`` tuple is imported and the resolved class is built
  with ``extra_kwargs`` (hand-derived to ``Fraction(3, 4)``);
* a key present in both ``extra_kwargs`` and the runtime kwargs raises a
  ``UserWarning`` while still delivering the non-conflicting kwargs.

The winner of a key collision is intentionally not asserted: it is not
derivable without the (migrated) Paddle source, so only the two independently
derivable facts -- the warning fires and non-conflicting kwargs survive -- are
checked. Paddle is required to exercise the shim at all; when it is absent the
whole module skips with an honest reason rather than faking a pass.
"""

import os
import sys
import unittest
from fractions import Fraction

# Mirror the repo layout so the package imports under the ``src`` layout when it
# is not pip installed. A genuinely broken shim (Paddle present but the migrated
# names gone) is left to surface as an error, not swallowed into a skip.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import paddle
    from paddle import nn

    HAS_PADDLE = True
    _IMPORT_ERROR = None
except ImportError as exc:  # paddle unavailable in this environment
    paddle = None
    nn = None
    HAS_PADDLE = False
    _IMPORT_ERROR = exc


_SKIP_REASON = (
    "paddle is not installed in this environment; the spec_utils shim imports "
    "from paddle.distributed.fleet.meta_parallel and cannot be exercised"
)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestSpecUtilsReExports(unittest.TestCase):
    """The shim must forward the exact migrated Paddle symbols."""

    def test_reexports_bind_to_paddle_symbols(self):
        from paddle.distributed.fleet.meta_parallel import (
            LayerSpec as PaddleLayerSpec,
            build_spec_layer,
        )

        from paddlefleet import spec_utils

        self.assertIs(spec_utils.LayerSpec, PaddleLayerSpec)
        self.assertIs(spec_utils.build_layer, build_spec_layer)
        self.assertEqual(spec_utils.__all__, ["LayerSpec", "build_layer"])


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestBuildLayerResolution(unittest.TestCase):
    """Spec resolution control flow, observed via the shim's ``build_layer``."""

    def test_class_spec_instantiated_with_extra_kwargs(self):
        from paddlefleet.spec_utils import LayerSpec, build_layer

        class _TwoArgLayer(nn.Layer):
            def __init__(self, alpha, beta):
                super().__init__()
                self.alpha = alpha
                self.beta = beta

        spec = LayerSpec(
            layer=_TwoArgLayer, extra_kwargs={"alpha": 3, "beta": 7}
        )
        built = build_layer(spec)

        self.assertIsInstance(built, _TwoArgLayer)
        self.assertEqual(built.alpha, 3)
        self.assertEqual(built.beta, 7)

    def test_runtime_kwargs_merge_with_extra_kwargs(self):
        from paddlefleet.spec_utils import LayerSpec, build_layer

        class _TwoArgLayer(nn.Layer):
            def __init__(self, alpha, beta):
                super().__init__()
                self.alpha = alpha
                self.beta = beta

        spec = LayerSpec(layer=_TwoArgLayer, extra_kwargs={"alpha": 3})
        built = build_layer(spec, beta=9)

        self.assertIsInstance(built, _TwoArgLayer)
        self.assertEqual(built.alpha, 3)
        self.assertEqual(built.beta, 9)

    def test_function_layer_returned_without_invocation(self):
        from paddlefleet.spec_utils import build_layer

        calls = []

        def _factory():
            calls.append(1)
            return "sentinel"

        result = build_layer(_factory)

        self.assertIs(result, _factory)
        self.assertEqual(calls, [])

    def test_module_path_tuple_resolved_and_built(self):
        from paddlefleet.spec_utils import LayerSpec, build_layer

        spec = LayerSpec(
            layer=("fractions", "Fraction"),
            extra_kwargs={"numerator": 3, "denominator": 4},
        )
        built = build_layer(spec)

        self.assertEqual(built, Fraction(3, 4))
        self.assertEqual(float(built), 0.75)

    def test_conflicting_kwargs_warn_and_preserve_others(self):
        from paddlefleet.spec_utils import LayerSpec, build_layer

        class _TwoArgLayer(nn.Layer):
            def __init__(self, alpha, beta):
                super().__init__()
                self.alpha = alpha
                self.beta = beta

        spec = LayerSpec(
            layer=_TwoArgLayer, extra_kwargs={"alpha": 4, "beta": 7}
        )
        with self.assertWarns(UserWarning):
            built = build_layer(spec, alpha=10)

        self.assertIsInstance(built, _TwoArgLayer)
        self.assertEqual(built.beta, 7)


if __name__ == "__main__":
    unittest.main()
