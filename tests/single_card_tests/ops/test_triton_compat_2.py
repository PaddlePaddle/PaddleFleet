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

"""Behavior tests for the Triton driver-swap compat shim (branch set 2).

Module under test: ``paddlefleet.triton_ops.triton_compat`` (repository
module map: "计算优化" / backend dispatch). This file deliberately targets
branches that are NOT the trivial "return the kernel unchanged" paths a base
suite would cover. It exercises the *dispatch and driver-activation
orchestration* that is pure Python and observable on CPU:

  * ``_swap_driver_guard`` -- the returned wrapper must call
    ``driver.set_active(paddle_driver)`` BEFORE the wrapped function, run the
    function, then call ``driver.reset_active()`` AFTER, forwarding positional
    and keyword arguments and passing the return value through unchanged. The
    reset must also happen when the wrapped function raises (the ``finally``).
  * ``enable_compat_on_triton_kernel`` torch+CUDA branch -- it must return a
    wrapper (not the raw kernel) whose ``__getitem__(index)`` selects
    ``kernel[index]`` for the *actual* requested index and routes it through
    ``_swap_driver_guard``.

The ``driver`` object comes from ``triton.runtime.driver`` and the concrete
``paddle_driver`` is produced by Triton's ``_create_driver``; both are genuine
not-under-test collaborators, so they are substituted with recording doubles
to observe the activation order and the value forwarded. The branch-selection
probes (``_is_package_installed`` for torch, ``paddle.is_compiled_with_cuda``)
are patched only to enter the branch under test; the wrapping/indexing/guard
code itself runs for real.

Environment note: the module imports ``paddle`` at top level and runs
import-time compat setup, and these tests patch ``triton.runtime.driver``.
When ``paddle`` or ``triton`` is not installed the whole suite is skipped
honestly (only ``ImportError`` is treated as a missing dependency; genuine
API/compile errors are allowed to surface). No GPU is required: the driver
swap and kernel dispatch are validated with CPU-side recording doubles, and
no real device numerics are claimed.
"""

import unittest
from unittest import mock

try:
    import paddle
    import triton  # noqa: F401
    import triton.runtime.driver as triton_driver_mod

    from paddlefleet.triton_ops import triton_compat as tc

    _IMPORT_OK = True
    _IMPORT_ERR = ""
except ImportError as exc:  # missing dependency -> honest skip, not swallowed
    paddle = None
    triton_driver_mod = None
    tc = None
    _IMPORT_OK = False
    _IMPORT_ERR = str(exc)


_SKIP_REASON = (
    "triton_compat driver-swap tests require paddle + triton to be importable "
    "(the module runs import-time compat setup and the tests patch "
    "triton.runtime.driver); import error: %s" % (_IMPORT_ERR or "none",)
)


class _RecordingDriver:
    """Stand-in for ``triton.runtime.driver.driver``.

    Records the ordered activation events and the exact driver object handed
    to ``set_active`` so the guard's orchestration can be asserted.
    """

    def __init__(self, events):
        self._events = events
        self.set_active_arg = "<unset>"

    def set_active(self, active_driver):
        self.set_active_arg = active_driver
        self._events.append("set_active")

    def reset_active(self):
        self._events.append("reset_active")


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestTritonCompatDriverSwap(unittest.TestCase):
    """CPU-observable driver-swap and kernel-dispatch orchestration."""

    def _install_paddle_driver(self, sentinel):
        """Set the module-global ``paddle_driver`` (only defined at import in
        the torch+CUDA branch) to a sentinel, restoring the prior state even
        on failure."""
        had = hasattr(tc, "paddle_driver")
        old = getattr(tc, "paddle_driver", None)

        def _restore():
            if had:
                tc.paddle_driver = old
            elif hasattr(tc, "paddle_driver"):
                del tc.paddle_driver

        self.addCleanup(_restore)
        tc.paddle_driver = sentinel

    def test_guard_activates_before_call_and_resets_after(self):
        events = []
        driver = _RecordingDriver(events)
        sentinel = object()
        self._install_paddle_driver(sentinel)

        received = {}

        def fn(a, b, *, scale):
            events.append("call")
            received["args"] = (a, b)
            received["scale"] = scale
            return a * 10 + b + scale

        with mock.patch.object(triton_driver_mod, "driver", driver):
            wrapped = tc._swap_driver_guard(fn)
            result = wrapped(3, 4, scale=5)

        # set_active must precede the call, reset_active must follow it.
        self.assertEqual(events, ["set_active", "call", "reset_active"])
        # The exact module paddle_driver is forwarded to set_active.
        self.assertIs(driver.set_active_arg, sentinel)
        # Positional and keyword args reach the wrapped fn untouched.
        self.assertEqual(received["args"], (3, 4))
        self.assertEqual(received["scale"], 5)
        # The wrapped fn's return value is passed straight through.
        self.assertEqual(result, 39)

    def test_guard_resets_active_even_when_fn_raises(self):
        events = []
        driver = _RecordingDriver(events)
        self._install_paddle_driver(object())

        def boom():
            events.append("call")
            raise ValueError("kernel launch failed")

        with mock.patch.object(triton_driver_mod, "driver", driver):
            wrapped = tc._swap_driver_guard(boom)
            with self.assertRaises(ValueError):
                wrapped()

        # The finally-clause reset must still run after the exception.
        self.assertEqual(events, ["set_active", "call", "reset_active"])

    def test_enable_compat_wraps_and_dispatches_selected_index(self):
        events = []
        driver = _RecordingDriver(events)
        sentinel = object()
        self._install_paddle_driver(sentinel)

        launch_calls = []

        class _FakeKernel:
            def __getitem__(self, index):
                def launcher(*args, **kwargs):
                    events.append("call")
                    launch_calls.append((index, args, kwargs))
                    return ("launched", index, args)

                return launcher

        fake = _FakeKernel()

        with (
            mock.patch.object(tc, "_is_package_installed", return_value=True),
            mock.patch.object(
                paddle, "is_compiled_with_cuda", return_value=True
            ),
            mock.patch.object(triton_driver_mod, "driver", driver),
        ):
            wrapped = tc.enable_compat_on_triton_kernel(fake)
            # torch + CUDA branch returns a new wrapper, not the raw kernel.
            self.assertIsNot(wrapped, fake)
            guarded = wrapped[(8, 1, 1)]
            self.assertTrue(callable(guarded))
            out = guarded(7, meta="m")

        # __getitem__ forwarded exactly the requested grid to the real kernel.
        self.assertEqual(launch_calls, [((8, 1, 1), (7,), {"meta": "m"})])
        # The selected launcher ran inside the driver-swap guard.
        self.assertEqual(events, ["set_active", "call", "reset_active"])
        self.assertIs(driver.set_active_arg, sentinel)
        # Return value of the underlying launcher is passed through the guard.
        self.assertEqual(out, ("launched", (8, 1, 1), (7,)))

    def test_wrapped_kernel_forwards_distinct_indices(self):
        events = []
        driver = _RecordingDriver(events)
        self._install_paddle_driver(object())

        launch_calls = []

        class _FakeKernel:
            def __getitem__(self, index):
                def launcher(*args, **kwargs):
                    launch_calls.append(index)
                    return index

                return launcher

        fake = _FakeKernel()

        with (
            mock.patch.object(tc, "_is_package_installed", return_value=True),
            mock.patch.object(
                paddle, "is_compiled_with_cuda", return_value=True
            ),
            mock.patch.object(triton_driver_mod, "driver", driver),
        ):
            wrapped = tc.enable_compat_on_triton_kernel(fake)
            self.assertEqual(wrapped[0](), 0)
            self.assertEqual(wrapped[2](), 2)

        # The index is not hard-coded: each subscription reaches its own grid.
        self.assertEqual(launch_calls, [0, 2])


if __name__ == "__main__":
    unittest.main()
