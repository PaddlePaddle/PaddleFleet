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

"""Behavior tests for Triton-ops dispatch utilities (variant _3).

Module under test: ``paddlefleet.triton_ops.utils``. In the repository
module map this is the "计算优化 / Fused Ops" boundary: pure control-logic
helpers that decide *whether* a Triton fast path may be used and *how* a
Triton kernel is wrapped so it launches with the correct driver.

Scope of this file (variant _3 -- deliberately exercises functions/branches
distinct from a base/_2 that would cover ``is_torch_compat_available``,
``dispatch_to``, ``_is_package_installed`` and ``swap_driver_guard`` in
isolation):

* ``is_triton_available`` -- the compound guard
  ``is_torch_compat_available() and paddle.is_compiled_with_cuda()
  and _is_package_installed("triton")``. Verified as a real three-input
  AND (each condition individually gates the result), that the package it
  probes is specifically ``"triton"``, and that Python ``and``
  short-circuits so a later collaborator is not consulted once an earlier
  condition is False.
* ``enable_compat_on_triton_kernel`` -- returns the kernel *unchanged*
  (object identity) on a non-CUDA build, and otherwise returns a wrapper
  whose ``__getitem__`` forwards the grid index to the underlying kernel
  and hands the result through ``swap_driver_guard`` so the returned
  launcher still calls through to the real kernel with the original args.

These are CPU-testable control-logic paths -- they need only that the
module imports (which requires paddle); no GPU kernel is launched, so the
environment collaborators ``paddle.is_compiled_with_cuda`` /
``is_torch_compat_available`` / ``_is_package_installed`` are patched to
drive the branches while the code under test runs for real.

Honest gating: ``paddlefleet.triton_ops.utils`` does ``import paddle`` at
module top, so when paddle (or the op module) is unimportable the whole
file cannot load. In that case every test skips with an explicit reason
rather than fake-passing.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import paddle

    from paddlefleet.triton_ops import utils as utils_mod

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # narrow: honest missing dep
    paddle = None
    utils_mod = None
    _IMPORT_ERROR = repr(exc)


def _skip_reason():
    """Return an honest skip reason, or None when the tests can really run."""
    if _IMPORT_ERROR is not None:
        return (
            "paddle / paddlefleet.triton_ops.utils import failed: "
            + _IMPORT_ERROR
        )
    return None


@unittest.skipUnless(_skip_reason() is None, _skip_reason() or "")
class TestIsTritonAvailable(unittest.TestCase):
    """``is_triton_available`` composes three conditions with a real AND."""

    def _run(self, compat, cuda, triton_installed):
        """Drive is_triton_available with the three inputs fixed."""
        with (
            patch.object(
                utils_mod, "is_torch_compat_available", return_value=compat
            ),
            patch.object(
                utils_mod.paddle, "is_compiled_with_cuda", return_value=cuda
            ),
            patch.object(
                utils_mod,
                "_is_package_installed",
                return_value=triton_installed,
            ),
        ):
            return utils_mod.is_triton_available()

    def test_requires_all_three_conditions(self):
        """True only when compat AND cuda AND triton-installed all hold.

        Hand-derived truth table: flipping any single input to False must
        flip the result to False. This catches a dropped condition or an
        ``or`` written in place of ``and``.
        """
        self.assertIs(self._run(True, True, True), True)
        self.assertIs(self._run(False, True, True), False)
        self.assertIs(self._run(True, False, True), False)
        self.assertIs(self._run(True, True, False), False)

    def test_probes_the_triton_package_specifically(self):
        """The package-existence probe is called with the name "triton"."""
        captured = []

        def fake_installed(name):
            captured.append(name)
            return True

        with (
            patch.object(
                utils_mod, "is_torch_compat_available", return_value=True
            ),
            patch.object(
                utils_mod.paddle, "is_compiled_with_cuda", return_value=True
            ),
            patch.object(
                utils_mod, "_is_package_installed", side_effect=fake_installed
            ),
        ):
            result = utils_mod.is_triton_available()

        self.assertIs(result, True)
        self.assertEqual(captured, ["triton"])

    def test_short_circuits_before_package_probe(self):
        """A False first condition must stop before the package probe runs.

        Verifies real ``and`` short-circuit semantics: with compat False the
        result is False and ``_is_package_installed`` is never consulted.
        """
        probed = []

        with (
            patch.object(
                utils_mod, "is_torch_compat_available", return_value=False
            ),
            patch.object(
                utils_mod.paddle, "is_compiled_with_cuda", return_value=True
            ),
            patch.object(
                utils_mod,
                "_is_package_installed",
                side_effect=lambda name: probed.append(name) or True,
            ),
        ):
            result = utils_mod.is_triton_available()

        self.assertIs(result, False)
        self.assertEqual(probed, [])


@unittest.skipUnless(_skip_reason() is None, _skip_reason() or "")
class TestEnableCompatOnTritonKernel(unittest.TestCase):
    """``enable_compat_on_triton_kernel`` passes through / wraps a kernel."""

    class _FakeKernel:
        """Stand-in Triton kernel; records the grid index it is subscripted with."""

        def __init__(self):
            self.indexed_with = []

        def __getitem__(self, index):
            self.indexed_with.append(index)

            def launch(*args, **kwargs):
                return ("launched", index, args, kwargs)

            return launch

    def test_returns_kernel_unchanged_without_cuda(self):
        """On a non-CUDA build the exact same kernel object is returned."""
        kernel = self._FakeKernel()
        with patch.object(
            utils_mod.paddle, "is_compiled_with_cuda", return_value=False
        ):
            result = utils_mod.enable_compat_on_triton_kernel(kernel)

        self.assertIs(result, kernel)
        self.assertEqual(kernel.indexed_with, [])  # never subscripted

    def test_wraps_and_forwards_index_and_call_with_cuda(self):
        """On a CUDA build the wrapper forwards the grid index and call-through.

        ``__getitem__`` must index the *underlying* kernel with the exact
        grid tuple and return a launcher (produced via ``swap_driver_guard``)
        that, when called, still invokes the real kernel with the original
        positional/keyword arguments and returns its result. The driver
        global is pinned to None so this test isolates the wrapping/forwarding
        behavior from swap_driver_guard's own driver-swap side effects.
        """
        kernel = self._FakeKernel()
        self.addCleanup(
            setattr, utils_mod, "_paddle_driver", utils_mod._paddle_driver
        )
        with (
            patch.object(
                utils_mod.paddle, "is_compiled_with_cuda", return_value=True
            ),
            patch.object(utils_mod, "_paddle_driver", None),
        ):
            wrapped = utils_mod.enable_compat_on_triton_kernel(kernel)

            # Not the raw kernel, but retains a handle to it.
            self.assertIsNot(wrapped, kernel)
            self.assertIs(wrapped.kernel, kernel)

            launcher = wrapped[(4, 2)]
            # The grid index reached the underlying kernel unchanged.
            self.assertEqual(kernel.indexed_with, [(4, 2)])
            self.assertTrue(callable(launcher))

            result = launcher(7, key="v")

        # The guard is a pass-through here, so the real kernel's launch runs
        # with the original args and its return value propagates back out.
        self.assertEqual(result, ("launched", (4, 2), (7,), {"key": "v"}))


if __name__ == "__main__":
    unittest.main()
