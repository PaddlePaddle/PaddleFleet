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

"""Behavior tests for ``paddlefleet.triton_ops.utils``.

Repository module map placement: 计算优化 / Fused Ops (``triton_ops``). This is
the base helper module of the triton_ops set and it exposes CPU-observable
control logic rather than GPU numerics:

* ``is_torch_compat_available`` returns ``hasattr(paddle, "enable_compat")``.
* ``is_triton_available`` is the conjunction of torch-compat, a CUDA-compiled
  paddle build, and an installed ``triton`` distribution.
* ``dispatch_to`` builds a decorator that routes to the high-performance
  ``dispatch_fn`` only when ``cond(*args, **kwargs)`` *and* torch-compat are
  both true, otherwise it calls the decorated fallback; it also records the
  original function on ``wrapper.__original_fn__``.
* ``_is_package_installed`` maps a successful ``importlib.metadata.distribution``
  lookup to ``True`` and ``PackageNotFoundError`` to ``False`` (other errors
  propagate rather than being swallowed).
* ``swap_driver_guard`` wraps a fn so the paddle triton driver is set active
  before the call and reset in a ``finally`` afterwards, but only when the
  module-level ``_paddle_driver`` was initialised.
* ``enable_compat_on_triton_kernel`` returns the kernel unchanged on a non-CUDA
  build and otherwise wraps it.

Expectations below are hand-derived from that control logic. Genuine
not-under-test collaborators (``paddle``, ``importlib.metadata.distribution``,
the triton runtime driver) are stubbed with distinguishable responses so the
real orchestration is observed; the functions under test are never mocked.

Importing the module requires ``import paddle`` at module load time. In a
paddle-less / CPU-only environment the whole suite skips with an honest reason
instead of faking a pass -- none of this logic can even be imported without
paddle present.
"""

import sys
import types
import unittest
from unittest import mock

_IMPORT_ERROR = None
try:
    from paddlefleet.triton_ops import utils as triton_utils

    _DEPS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as exc:  # precise capability probe
    triton_utils = None
    _DEPS_AVAILABLE = False
    _IMPORT_ERROR = repr(exc)

_SKIP_REASON = f"paddlefleet.triton_ops.utils not importable: {_IMPORT_ERROR}"


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestIsTorchCompatAvailable(unittest.TestCase):
    """``is_torch_compat_available`` reflects the presence of enable_compat."""

    def test_true_when_paddle_exposes_enable_compat(self):
        fake_paddle = types.SimpleNamespace(enable_compat=lambda: None)
        with mock.patch.object(triton_utils, "paddle", fake_paddle):
            self.assertTrue(triton_utils.is_torch_compat_available())

    def test_false_when_paddle_lacks_enable_compat(self):
        fake_paddle = types.SimpleNamespace()  # no enable_compat attribute
        with mock.patch.object(triton_utils, "paddle", fake_paddle):
            self.assertFalse(triton_utils.is_torch_compat_available())


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestIsTritonAvailable(unittest.TestCase):
    """``is_triton_available`` is the conjunction of three gates."""

    def test_true_only_when_all_three_hold(self):
        fake_paddle = types.SimpleNamespace(is_compiled_with_cuda=lambda: True)
        with (
            mock.patch.object(triton_utils, "paddle", fake_paddle),
            mock.patch.object(
                triton_utils, "is_torch_compat_available", return_value=True
            ),
            mock.patch.object(
                triton_utils, "_is_package_installed", return_value=True
            ) as pkg,
        ):
            self.assertTrue(triton_utils.is_triton_available())
        # The triton availability gate must query the "triton" distribution.
        pkg.assert_called_with("triton")

    def test_false_when_any_single_condition_missing(self):
        # Each row flips exactly one gate off; the AND must reject all of them.
        for compat, cuda, pkg in [
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ]:
            with self.subTest(compat=compat, cuda=cuda, pkg=pkg):
                fake_paddle = types.SimpleNamespace(
                    is_compiled_with_cuda=lambda c=cuda: c
                )
                with (
                    mock.patch.object(triton_utils, "paddle", fake_paddle),
                    mock.patch.object(
                        triton_utils,
                        "is_torch_compat_available",
                        return_value=compat,
                    ),
                    mock.patch.object(
                        triton_utils,
                        "_is_package_installed",
                        return_value=pkg,
                    ),
                ):
                    self.assertFalse(triton_utils.is_triton_available())


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestDispatchTo(unittest.TestCase):
    """``dispatch_to`` routes between dispatch_fn and the fallback."""

    def _build(self, cond=None):
        events = {}

        def dispatch_fn(a, b):
            events["dispatch"] = (a, b)
            return ("dispatch", a + b)

        def fallback(a, b):
            events["fallback"] = (a, b)
            return ("fallback", a - b)

        decorated = triton_utils.dispatch_to(dispatch_fn, cond=cond)(fallback)
        return decorated, events

    def test_dispatches_when_compat_available_and_default_cond(self):
        decorated, events = self._build()
        with mock.patch.object(
            triton_utils, "is_torch_compat_available", return_value=True
        ):
            result = decorated(3, 4)
        self.assertEqual(result, ("dispatch", 7))
        self.assertEqual(events.get("dispatch"), (3, 4))
        self.assertNotIn("fallback", events)

    def test_falls_back_when_compat_unavailable(self):
        decorated, events = self._build()
        with mock.patch.object(
            triton_utils, "is_torch_compat_available", return_value=False
        ):
            result = decorated(3, 4)
        self.assertEqual(result, ("fallback", -1))
        self.assertEqual(events.get("fallback"), (3, 4))
        self.assertNotIn("dispatch", events)

    def test_cond_gates_dispatch_independently_of_compat(self):
        # cond is evaluated on the real call args; a false cond forces the
        # fallback even when torch-compat is available.
        decorated, events = self._build(cond=lambda a, b: a > 100)
        with mock.patch.object(
            triton_utils, "is_torch_compat_available", return_value=True
        ):
            self.assertEqual(decorated(3, 4), ("fallback", -1))
            self.assertNotIn("dispatch", events)
            self.assertEqual(decorated(200, 4), ("dispatch", 204))
        self.assertEqual(events.get("dispatch"), (200, 4))

    def test_forwards_keyword_arguments_to_selected_fn(self):
        def dispatch_fn(a, *, scale):
            return a * scale

        def fallback(a, *, scale):
            return a + scale

        decorated = triton_utils.dispatch_to(dispatch_fn)(fallback)
        with mock.patch.object(
            triton_utils, "is_torch_compat_available", return_value=True
        ):
            self.assertEqual(decorated(6, scale=4), 24)
        with mock.patch.object(
            triton_utils, "is_torch_compat_available", return_value=False
        ):
            self.assertEqual(decorated(6, scale=4), 10)

    def test_wrapper_exposes_original_fn(self):
        def dispatch_fn(*args, **kwargs):
            return None

        def fallback(*args, **kwargs):
            return None

        decorated = triton_utils.dispatch_to(dispatch_fn)(fallback)
        self.assertIs(decorated.__original_fn__, fallback)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestIsPackageInstalled(unittest.TestCase):
    """``_is_package_installed`` maps distribution lookup to a bool."""

    def setUp(self):
        # The helper is functools.cache'd; clear before and after so each case
        # actually re-runs the lookup logic under its own stub.
        triton_utils._is_package_installed.cache_clear()
        self.addCleanup(triton_utils._is_package_installed.cache_clear)

    def test_true_when_distribution_resolves(self):
        seen = []

        def fake_distribution(name):
            seen.append(name)
            return object()

        with mock.patch.object(
            triton_utils, "distribution", side_effect=fake_distribution
        ):
            self.assertTrue(
                triton_utils._is_package_installed("some-installed-pkg")
            )
        self.assertEqual(seen, ["some-installed-pkg"])

    def test_false_when_package_not_found(self):
        def fake_distribution(name):
            raise triton_utils.PackageNotFoundError(name)

        with mock.patch.object(
            triton_utils, "distribution", side_effect=fake_distribution
        ):
            self.assertFalse(triton_utils._is_package_installed("missing-pkg"))

    def test_other_errors_are_not_swallowed(self):
        # Only PackageNotFoundError means "absent"; anything else must surface.
        def fake_distribution(name):
            raise RuntimeError("metadata backend broken")

        with (
            mock.patch.object(
                triton_utils, "distribution", side_effect=fake_distribution
            ),
            self.assertRaises(RuntimeError),
        ):
            triton_utils._is_package_installed("weird-pkg")

    def test_real_absent_distribution_returns_false(self):
        # Exercises the genuine importlib.metadata path with no stubbing.
        self.assertFalse(
            triton_utils._is_package_installed(
                "paddlefleet-nonexistent-distribution-xyz-123"
            )
        )


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestEnableCompatOnTritonKernel(unittest.TestCase):
    """``enable_compat_on_triton_kernel`` branches on the CUDA build flag."""

    def test_returns_kernel_unchanged_without_cuda(self):
        fake_paddle = types.SimpleNamespace(is_compiled_with_cuda=lambda: False)
        sentinel = object()
        with mock.patch.object(triton_utils, "paddle", fake_paddle):
            self.assertIs(
                triton_utils.enable_compat_on_triton_kernel(sentinel), sentinel
            )

    def test_wraps_kernel_and_retains_reference_with_cuda(self):
        fake_paddle = types.SimpleNamespace(is_compiled_with_cuda=lambda: True)
        kernel = object()
        with mock.patch.object(triton_utils, "paddle", fake_paddle):
            wrapped = triton_utils.enable_compat_on_triton_kernel(kernel)
        self.assertIsNot(wrapped, kernel)
        self.assertIs(wrapped.kernel, kernel)


@unittest.skipUnless(_DEPS_AVAILABLE, _SKIP_REASON)
class TestSwapDriverGuard(unittest.TestCase):
    """``swap_driver_guard`` toggles the paddle triton driver around fn."""

    def _install_fake_triton(self, driver_obj):
        # swap_driver_guard does ``from triton.runtime.driver import driver``;
        # inject a controllable stub and restore sys.modules afterwards so the
        # real triton (if any) is untouched outside this test.
        keys = ["triton", "triton.runtime", "triton.runtime.driver"]
        saved = {k: sys.modules.get(k) for k in keys}

        def restore():
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

        self.addCleanup(restore)
        triton_mod = types.ModuleType("triton")
        runtime_mod = types.ModuleType("triton.runtime")
        driver_mod = types.ModuleType("triton.runtime.driver")
        driver_mod.driver = driver_obj
        runtime_mod.driver = driver_mod
        triton_mod.runtime = runtime_mod
        sys.modules["triton"] = triton_mod
        sys.modules["triton.runtime"] = runtime_mod
        sys.modules["triton.runtime.driver"] = driver_mod

    def test_passthrough_when_no_paddle_driver(self):
        driver_obj = mock.Mock()
        self._install_fake_triton(driver_obj)
        calls = []

        def fn(x):
            calls.append(x)
            return x * 2

        with mock.patch.object(triton_utils, "_paddle_driver", None):
            guarded = triton_utils.swap_driver_guard(fn)
            result = guarded(5)

        self.assertEqual(result, 10)
        self.assertEqual(calls, [5])
        driver_obj.set_active.assert_not_called()
        driver_obj.reset_active.assert_not_called()

    def test_sets_then_resets_active_around_fn(self):
        driver_obj = mock.Mock()
        self._install_fake_triton(driver_obj)
        sentinel_driver = object()
        order = []
        driver_obj.set_active.side_effect = lambda d: order.append(("set", d))
        driver_obj.reset_active.side_effect = lambda: order.append(("reset",))

        def fn():
            order.append(("fn",))
            return "ok"

        with mock.patch.object(triton_utils, "_paddle_driver", sentinel_driver):
            guarded = triton_utils.swap_driver_guard(fn)
            result = guarded()

        self.assertEqual(result, "ok")
        self.assertEqual(order, [("set", sentinel_driver), ("fn",), ("reset",)])

    def test_resets_active_even_when_fn_raises(self):
        driver_obj = mock.Mock()
        self._install_fake_triton(driver_obj)
        sentinel_driver = object()
        order = []
        driver_obj.set_active.side_effect = lambda d: order.append("set")
        driver_obj.reset_active.side_effect = lambda: order.append("reset")

        def fn():
            order.append("fn")
            raise ValueError("boom")

        with mock.patch.object(triton_utils, "_paddle_driver", sentinel_driver):
            guarded = triton_utils.swap_driver_guard(fn)
            with self.assertRaises(ValueError):
                guarded()

        # reset_active must run in the finally clause after the failure.
        self.assertEqual(order, ["set", "fn", "reset"])


if __name__ == "__main__":
    unittest.main()
