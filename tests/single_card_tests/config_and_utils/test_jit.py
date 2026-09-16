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

"""Behavior unit tests for paddlefleet.jit (pure Python, no device).

paddlefleet.jit currently defines ``jit_fuser = lambda fn: fn``. The
``paddle.jit.to_static`` backend is intentionally disabled (see the TODO in
the module), so the documented behavior of ``jit_fuser`` is a pure
pass-through decorator: it must return the *same* callable object it was
given, leaving identity, call semantics, and metadata untouched. These tests
pin that contract; if someone re-enabled a wrapping backend without updating
the module, the identity/metadata assertions below would fail.
"""

import importlib.util
import os
import sys
import unittest

# Locate <repo>/src so the real production source is importable.
_REPO_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

try:
    # Preferred path: exercise the real package import when paddle is present.
    from paddlefleet.jit import jit_fuser
except ImportError:
    # paddlefleet/__init__.py transitively imports paddle, which is absent in
    # the no-card CPU environment. jit.py itself imports nothing from paddle,
    # so we load the exact production source file directly and test it for
    # real -- this is the genuine module under test, not a stub.
    _JIT_PATH = os.path.join(_REPO_SRC, "paddlefleet", "jit.py")
    _spec = importlib.util.spec_from_file_location(
        "paddlefleet_jit_under_test", _JIT_PATH
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    jit_fuser = _mod.jit_fuser


class TestJitFuser(unittest.TestCase):
    """jit_fuser is a pass-through decorator: same object, same behavior."""

    def test_returns_the_same_object(self):
        # The load-bearing contract: jit_fuser must hand back the identical
        # callable, not a wrapper. `is` (not ==) is what distinguishes a true
        # pass-through from a to_static-style wrapper that compares equal only
        # by accident.
        def original(x):
            return x + 1

        self.assertIs(jit_fuser(original), original)

    def test_executes_the_original_function_body(self):
        # Prove the returned callable runs the real body (not a stub returning
        # a canned value) by observing both hand-derived outputs and a
        # side effect recorded by the genuine function.
        seen = []

        def poly(x):
            seen.append(x)
            return x * x - 3 * x + 2

        wrapped = jit_fuser(poly)
        # Hand-derived: 5*5-15+2=12 ; 0-0+2=2 ; 4+6+2=12.
        self.assertEqual(wrapped(5), 12)
        self.assertEqual(wrapped(0), 2)
        self.assertEqual(wrapped(-2), 12)
        # The genuine function executed once per call, in order, with the
        # exact arguments passed through unchanged.
        self.assertEqual(seen, [5, 0, -2])

    def test_preserves_function_metadata(self):
        # Because the same object is returned, introspection metadata is
        # preserved verbatim. A naive `def inner(*a): return fn(*a)` wrapper
        # would report __name__ == "inner"; functools.wraps would still change
        # __wrapped__. Exact-string checks catch such a regression.
        def annotated_fn(x):
            """Docstring anchor for metadata preservation."""
            return x

        wrapped = jit_fuser(annotated_fn)
        self.assertEqual(wrapped.__name__, "annotated_fn")
        self.assertEqual(
            wrapped.__doc__, "Docstring anchor for metadata preservation."
        )
        self.assertEqual(wrapped.__module__, annotated_fn.__module__)
        self.assertFalse(hasattr(wrapped, "__wrapped__"))

    def test_preserves_bound_method(self):
        # Bound methods are callables too; pass-through must keep the binding
        # (self) intact and return the same bound-method object semantics.
        class Accumulator:
            def __init__(self, base):
                self.base = base

            def scaled(self, x):
                return self.base * 10 + x

        obj = Accumulator(base=4)
        wrapped = jit_fuser(obj.scaled)
        # Hand-derived: base(4)*10 + 7 = 47 ; 4*10 + 0 = 40.
        self.assertEqual(wrapped(7), 47)
        self.assertEqual(wrapped(0), 40)
        self.assertEqual(wrapped.__self__, obj)
        self.assertEqual(wrapped.__func__, Accumulator.scaled)

    def test_preserves_closure_binding(self):
        # A closure over a captured value must keep working after pass-through.
        def make_adder(offset):
            def add(x):
                return x + offset

            return add

        add_by_100 = make_adder(100)
        wrapped = jit_fuser(add_by_100)
        self.assertIs(wrapped, add_by_100)
        # Hand-derived: 3 + 100 = 103 ; -100 + 100 = 0.
        self.assertEqual(wrapped(3), 103)
        self.assertEqual(wrapped(-100), 0)

    def test_passthrough_preserves_signature_and_builtins(self):
        # Positional args, keyword args and defaults must reach the original
        # signature unchanged, and even non-Python builtins pass through by
        # identity.
        def combine(a, b, c=0):
            return a * 100 + b * 10 + c

        wrapped = jit_fuser(combine)
        self.assertIs(wrapped, combine)
        # Hand-derived: 1*100+2*10+3=123 ; default c=0 -> 120 ; kwargs -> 129.
        self.assertEqual(wrapped(1, 2, 3), 123)
        self.assertEqual(wrapped(1, 2), 120)
        self.assertEqual(wrapped(1, c=9, b=2), 129)

        # Builtin callable is returned as-is and still computes correctly.
        wrapped_len = jit_fuser(len)
        self.assertIs(wrapped_len, len)
        self.assertEqual(wrapped_len([10, 20, 30]), 3)


if __name__ == "__main__":
    unittest.main()
