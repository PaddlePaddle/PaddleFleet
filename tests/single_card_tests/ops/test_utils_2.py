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

"""Behavior tests for ``paddlefleet.triton_ops.utils`` (part 2).

Repository module map placement: 计算优化 / Fused Ops dispatch plumbing
(``triton_ops``). This file deliberately targets *different* symbols than the
base ``test_utils.py``: the conditional dispatch decorator ``dispatch_to``, the
compound capability gate ``is_triton_available``, and the CUDA-gated kernel
wrapper ``enable_compat_on_triton_kernel``.

These are control-flow / dispatch behaviours that do not themselves touch a GPU
kernel, so the environment-dependent collaborators they consult
(``is_torch_compat_available``, ``paddle.is_compiled_with_cuda``,
``_is_package_installed``) are pinned to specific values so each branch and the
AND-composition can be exercised deterministically; the code under test (the
branch selection, argument forwarding and wrapping) runs for real. The module
still requires an importable ``paddle`` build, so when paddle / paddlefleet are
unavailable the whole suite is skipped with an honest reason rather than faking
a pass -- nothing here can be exercised without paddle installed.
"""

import unittest
from unittest import mock

_IMPORT_ERROR = None
try:
    from paddlefleet.triton_ops import utils as tri_utils

    _DEPS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as exc:  # precise capability probe
    _DEPS_AVAILABLE = False
    _IMPORT_ERROR = repr(exc)

_SKIP_REASON = (
    "paddlefleet.triton_ops.utils not importable (needs paddle): "
    + str(_IMPORT_ERROR)
)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestDispatchTo(unittest.TestCase):
    """``dispatch_to`` chooses dispatch_fn vs fallback and forwards args."""

    def test_dispatches_when_cond_true_and_compat_available(self):
        rec = {}

        def dispatch_fn(*args, **kwargs):
            rec["d_args"], rec["d_kwargs"] = args, kwargs
            return "DISPATCHED"

        def cond(*args, **kwargs):
            rec["c_args"], rec["c_kwargs"] = args, kwargs
            return True

        with mock.patch.object(
            tri_utils, "is_torch_compat_available", return_value=True
        ):

            @tri_utils.dispatch_to(dispatch_fn, cond=cond)
            def fallback(*args, **kwargs):
                rec["fallback_ran"] = True
                return "FALLBACK"

            result = fallback(7, 8, name="k")

        # dispatch_fn output is returned; fallback body never runs.
        self.assertEqual(result, "DISPATCHED")
        self.assertNotIn("fallback_ran", rec)
        # Both cond and dispatch_fn receive the caller's exact args/kwargs.
        self.assertEqual(rec["c_args"], (7, 8))
        self.assertEqual(rec["c_kwargs"], {"name": "k"})
        self.assertEqual(rec["d_args"], (7, 8))
        self.assertEqual(rec["d_kwargs"], {"name": "k"})

    def test_falls_back_when_cond_false(self):
        rec = {}

        def dispatch_fn(*args, **kwargs):
            rec["dispatched"] = True
            return "DISPATCHED"

        def cond(*args, **kwargs):
            rec["c_args"] = args
            return False

        with mock.patch.object(
            tri_utils, "is_torch_compat_available", return_value=True
        ):

            @tri_utils.dispatch_to(dispatch_fn, cond=cond)
            def fallback(x):
                return "FALLBACK"

            result = fallback(99)

        self.assertEqual(result, "FALLBACK")
        self.assertNotIn("dispatched", rec)  # dispatch_fn must be skipped
        self.assertEqual(rec["c_args"], (99,))

    def test_falls_back_when_compat_unavailable(self):
        rec = {}

        def dispatch_fn(x):
            rec["dispatched"] = True
            return "DISPATCHED"

        def cond(x):
            rec["cond_ran"] = True
            return True  # cond passes; the compat gate must still block

        with mock.patch.object(
            tri_utils, "is_torch_compat_available", return_value=False
        ):

            @tri_utils.dispatch_to(dispatch_fn, cond=cond)
            def fallback(x):
                return "FALLBACK"

            result = fallback(5)

        self.assertEqual(result, "FALLBACK")
        self.assertTrue(rec["cond_ran"])  # cond evaluated before the gate
        self.assertNotIn("dispatched", rec)

    def test_default_cond_dispatches_with_positional_arg(self):
        # Default cond is ``lambda self, *a, **k: True`` (method-oriented), so
        # the decorated callable is invoked with a leading positional arg.
        with mock.patch.object(
            tri_utils, "is_torch_compat_available", return_value=True
        ):

            @tri_utils.dispatch_to(lambda x: "DISPATCHED")
            def fallback(x):
                return "FALLBACK"

            self.assertEqual(fallback(42), "DISPATCHED")

    def test_preserves_original_fn(self):
        with mock.patch.object(
            tri_utils, "is_torch_compat_available", return_value=True
        ):

            @tri_utils.dispatch_to(lambda x: "DISPATCHED")
            def fallback(x):
                return "FALLBACK"

            # Wrapper dispatches, but __original_fn__ is the undecorated body
            # and bypasses dispatch entirely.
            self.assertEqual(fallback(1), "DISPATCHED")
            self.assertIsNot(fallback.__original_fn__, fallback)
            self.assertEqual(fallback.__original_fn__(1), "FALLBACK")


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestIsTritonAvailable(unittest.TestCase):
    """``is_triton_available`` is the AND of three capability gates."""

    def _evaluate(self, compat, cuda, triton_installed):
        def fake_pkg(name):
            # Only the "triton" distribution is relevant to this gate.
            self.assertEqual(name, "triton")
            return triton_installed

        with (
            mock.patch.object(
                tri_utils, "is_torch_compat_available", return_value=compat
            ),
            mock.patch.object(
                tri_utils.paddle, "is_compiled_with_cuda", return_value=cuda
            ),
            mock.patch.object(
                tri_utils, "_is_package_installed", side_effect=fake_pkg
            ),
        ):
            return tri_utils.is_triton_available()

    def test_true_only_when_all_gates_true(self):
        self.assertIs(self._evaluate(True, True, True), True)

    def test_false_when_compat_missing(self):
        self.assertIs(self._evaluate(False, True, True), False)

    def test_false_when_cuda_missing(self):
        self.assertIs(self._evaluate(True, False, True), False)

    def test_false_when_triton_package_missing(self):
        self.assertIs(self._evaluate(True, True, False), False)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestEnableCompatOnTritonKernel(unittest.TestCase):
    """CUDA gate decides passthrough vs wrapping of a triton kernel."""

    def test_returns_kernel_unchanged_without_cuda(self):
        sentinel = object()
        with mock.patch.object(
            tri_utils.paddle, "is_compiled_with_cuda", return_value=False
        ):
            result = tri_utils.enable_compat_on_triton_kernel(sentinel)
        # No CUDA -> exact same kernel object flows through untouched.
        self.assertIs(result, sentinel)

    def test_wraps_kernel_with_cuda(self):
        sentinel = object()
        with mock.patch.object(
            tri_utils.paddle, "is_compiled_with_cuda", return_value=True
        ):
            result = tri_utils.enable_compat_on_triton_kernel(sentinel)
        # A distinct wrapper is produced and the original kernel is preserved.
        self.assertIsNot(result, sentinel)
        self.assertIs(result.kernel, sentinel)
        self.assertEqual(type(result).__name__, "WrappedTritonKernel")


if __name__ == "__main__":
    unittest.main()
