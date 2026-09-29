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

"""Behavior tests for ``paddlefleet.triton_ops.triton_compat``.

Repository module map placement: 计算优化 / triton_ops. This module is a
compat/shim layer that decides, based on the environment, whether a Triton
kernel needs Paddle's driver swapped in before launch:

* ``_is_package_installed`` reports whether a named distribution is installed.
* ``enable_compat_on_triton_kernel`` performs *fallback selection*: it returns
  the kernel unchanged when torch is absent OR when Paddle is not a CUDA build,
  and only otherwise wraps it in a driver-swapping proxy.
* ``_swap_driver_guard`` wraps a callable so the active Triton driver is set to
  the Paddle driver for the duration of the call and reset afterwards, even if
  the call raises.

The selection and driver-swap lifecycle are CPU-observable and are what these
tests assert (with genuine not-under-test collaborators -- the package probe,
``paddle.is_compiled_with_cuda`` and the Triton driver singleton -- controlled
via distinguishable doubles). No GPU is required. When paddle / the module
cannot be imported the whole suite is skipped with an honest reason rather than
faking a pass; the actual on-device kernel launch is out of scope here.
"""

import importlib.metadata as importlib_metadata
import sys
import types
import unittest
from unittest import mock

_IMPORT_ERROR = None
try:
    import paddle

    from paddlefleet.triton_ops import triton_compat

    _DEPS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as exc:  # precise capability probe
    _DEPS_AVAILABLE = False
    _IMPORT_ERROR = repr(exc)

_SKIP_REASON = (
    f"paddle / triton_compat import unavailable: {_IMPORT_ERROR}"
    if not _DEPS_AVAILABLE
    else ""
)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestIsPackageInstalled(unittest.TestCase):
    """``_is_package_installed`` True/False contract vs an independent probe."""

    def test_absent_distribution_is_false(self):
        # A name that cannot correspond to any installed distribution must hit
        # the PackageNotFoundError branch and return exactly False.
        result = triton_compat._is_package_installed(
            "paddlefleet_no_such_distribution_zzz_0123456789"
        )
        self.assertIs(result, False)

    def test_present_distribution_is_true(self):
        # Independently discover a genuinely-installed distribution via the
        # stdlib (not via the code under test), then assert the probe agrees.
        present_name = None
        for dist in importlib_metadata.distributions():
            name = dist.metadata["Name"]
            if name:
                present_name = name
                break
        if present_name is None:
            self.skipTest("no installed distribution to probe against")
        self.assertIs(triton_compat._is_package_installed(present_name), True)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestEnableCompatFallbackSelection(unittest.TestCase):
    """``enable_compat_on_triton_kernel`` fallback vs wrapping selection."""

    def test_kernel_returned_unchanged_when_torch_absent(self):
        # torch missing -> first guard returns the kernel object itself,
        # with no proxy wrapping (identity, not merely an equal object).
        kernel = object()
        with mock.patch.object(
            triton_compat, "_is_package_installed", return_value=False
        ):
            result = triton_compat.enable_compat_on_triton_kernel(kernel)
        self.assertIs(result, kernel)

    def test_kernel_returned_unchanged_when_cuda_absent(self):
        # torch present but Paddle is not a CUDA build -> second guard returns
        # the kernel unchanged. Distinguishes it from the torch-absent path.
        kernel = object()
        with (
            mock.patch.object(
                triton_compat, "_is_package_installed", return_value=True
            ),
            mock.patch.object(
                paddle, "is_compiled_with_cuda", return_value=False
            ),
        ):
            result = triton_compat.enable_compat_on_triton_kernel(kernel)
        self.assertIs(result, kernel)

    def test_kernel_wrapped_and_getitem_dispatches(self):
        # torch + CUDA both present -> a distinct proxy is returned that keeps
        # the original kernel and, on __getitem__, indexes the real kernel and
        # routes the indexed launcher through _swap_driver_guard.
        class _KernelStub:
            def __init__(self):
                self.seen_index = []
                self.marker = object()

            def __getitem__(self, index):
                self.seen_index.append(index)
                return self.marker

        kernel = _KernelStub()
        guard_out = object()
        captured = {}

        def fake_guard(fn):
            captured["fn"] = fn
            return guard_out

        with (
            mock.patch.object(
                triton_compat, "_is_package_installed", return_value=True
            ),
            mock.patch.object(
                paddle, "is_compiled_with_cuda", return_value=True
            ),
            mock.patch.object(
                triton_compat, "_swap_driver_guard", side_effect=fake_guard
            ),
        ):
            proxy = triton_compat.enable_compat_on_triton_kernel(kernel)
            # A real proxy, not the kernel itself, but retaining the kernel.
            self.assertIsNot(proxy, kernel)
            self.assertIs(proxy.kernel, kernel)

            grid = (4, 2)
            returned = proxy[grid]

        # __getitem__ indexed the underlying kernel with the exact grid,
        # handed that launcher to the guard, and returned the guard's result.
        self.assertEqual(kernel.seen_index, [grid])
        self.assertIs(captured["fn"], kernel.marker)
        self.assertIs(returned, guard_out)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestSwapDriverGuard(unittest.TestCase):
    """``_swap_driver_guard`` set/call/reset ordering and finally guarantee."""

    def _install_fake_driver(self):
        """Inject a recording Triton driver singleton and Paddle driver.

        The Triton driver module and ``paddle_driver`` are genuine
        not-under-test collaborators that require a GPU build in reality; here
        they are recording doubles so the guard's set/call/reset ordering is
        observable on CPU. Originals are restored on cleanup.
        """
        events = []

        class _Driver:
            def set_active(self, drv):
                events.append(("set_active", drv))

            def reset_active(self):
                events.append(("reset_active", None))

        driver_singleton = _Driver()

        triton_mod = types.ModuleType("triton")
        runtime_mod = types.ModuleType("triton.runtime")
        driver_mod = types.ModuleType("triton.runtime.driver")
        driver_mod.driver = driver_singleton
        runtime_mod.driver = driver_mod
        triton_mod.runtime = runtime_mod

        for name, mod in (
            ("triton", triton_mod),
            ("triton.runtime", runtime_mod),
            ("triton.runtime.driver", driver_mod),
        ):
            original = sys.modules.get(name, None)
            self.addCleanup(self._restore_module, name, original)
            sys.modules[name] = mod

        paddle_driver = object()
        had_attr = hasattr(triton_compat, "paddle_driver")
        original_pd = getattr(triton_compat, "paddle_driver", None)
        if had_attr:
            self.addCleanup(
                setattr, triton_compat, "paddle_driver", original_pd
            )
        else:
            self.addCleanup(self._del_attr, triton_compat, "paddle_driver")
        triton_compat.paddle_driver = paddle_driver

        return events, paddle_driver

    @staticmethod
    def _restore_module(name, original):
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    @staticmethod
    def _del_attr(obj, attr):
        if hasattr(obj, attr):
            delattr(obj, attr)

    def test_sets_paddle_driver_calls_then_resets(self):
        events, paddle_driver = self._install_fake_driver()
        call_log = []

        def target(a, b, *, k):
            call_log.append((a, b, k))
            return a + b + k

        wrapped = triton_compat._swap_driver_guard(target)
        result = wrapped(2, 3, k=5)

        self.assertEqual(result, 10)
        self.assertEqual(call_log, [(2, 3, 5)])
        # set_active(paddle_driver) must precede the call, reset_active follow.
        self.assertEqual(
            events,
            [("set_active", paddle_driver), ("reset_active", None)],
        )

    def test_resets_even_when_call_raises(self):
        events, paddle_driver = self._install_fake_driver()

        def boom():
            raise ValueError("kernel launch failed")

        wrapped = triton_compat._swap_driver_guard(boom)
        with self.assertRaises(ValueError):
            wrapped()

        # The finally clause must still reset the driver after set_active.
        self.assertEqual(
            events,
            [("set_active", paddle_driver), ("reset_active", None)],
        )


# __TESTS__


if __name__ == "__main__":
    unittest.main()
