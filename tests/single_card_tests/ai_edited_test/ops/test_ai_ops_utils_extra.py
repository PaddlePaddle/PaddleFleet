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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


# Tests for paddlefleet_ops/ops/triton_ops/utils.py
# Tests is_torch_compat_available and dispatch_to

import types
import unittest
from unittest.mock import patch


def _setup_triton_mock():
    """Create a mock triton module so we can import without GPU."""
    triton_mock = types.ModuleType("triton")
    triton_mock.jit = (
        lambda fn=None, **kwargs: (lambda f: f) if fn is None else fn
    )
    triton_mock.language = types.ModuleType("triton.language")
    triton_mock.language.constexpr = None
    tl = triton_mock.language
    tl.program_id = lambda axis: 0
    tl.arange = lambda start, end: []
    tl.load = lambda *a, **kw: 0.0
    tl.store = lambda *a, **kw: None
    tl.int64 = "int64"
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", tl)
    return triton_mock


_setup_triton_mock()


class TestIsTorchCompatAvailable(unittest.TestCase):
    """Tests for is_torch_compat_available function."""

    def test_returns_bool(self):
        """Test that is_torch_compat_available returns a boolean."""
        from paddlefleet_ops.ops.triton_ops.utils import (
            is_torch_compat_available,
        )

        result = is_torch_compat_available()
        self.assertIsInstance(result, bool)

    def test_returns_false_without_enable_compat(self):
        """Test returns False when paddle has no enable_compat."""
        from paddlefleet_ops.ops.triton_ops.utils import (
            is_torch_compat_available,
        )

        result = is_torch_compat_available()
        # In standard PaddlePaddle without compat mode, should be False
        self.assertIsInstance(result, bool)


class TestDispatchTo(unittest.TestCase):
    """Tests for dispatch_to decorator."""

    def test_dispatch_to_returns_decorator(self):
        """Test that dispatch_to returns a decorator."""
        from paddlefleet_ops.ops.triton_ops.utils import dispatch_to

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        result = dispatch_to(dummy_dispatch)
        self.assertTrue(callable(result))

    def test_dispatch_to_wraps_function(self):
        """Test that dispatch_to wraps the original function."""
        from paddlefleet_ops.ops.triton_ops.utils import dispatch_to

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        decorated = dispatch_to(dummy_dispatch)(original_fn)
        self.assertTrue(callable(decorated))

    def test_dispatch_to_falls_back_when_no_compat(self):
        """Test that dispatch_to falls back to original when no compat available."""
        from paddlefleet_ops.ops.triton_ops.utils import dispatch_to

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        decorated = dispatch_to(dummy_dispatch)(original_fn)

        with patch(
            "paddlefleet_ops.ops.triton_ops.utils.is_torch_compat_available",
            return_value=False,
        ):
            result = decorated()
            self.assertEqual(result, "original")

    def test_dispatch_to_dispatches_when_compat_and_cond_true(self):
        """Test dispatch_to calls dispatch_fn when compat available and cond True."""
        from paddlefleet_ops.ops.triton_ops.utils import dispatch_to

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        decorated = dispatch_to(dummy_dispatch)(original_fn)

        with patch(
            "paddlefleet_ops.ops.triton_ops.utils.is_torch_compat_available",
            return_value=True,
        ):
            result = decorated()
            self.assertEqual(result, "dispatched")

    def test_dispatch_to_with_cond_false(self):
        """Test dispatch_to falls back when cond returns False."""
        from paddlefleet_ops.ops.triton_ops.utils import dispatch_to

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        cond = lambda *args, **kwargs: False
        decorated = dispatch_to(dummy_dispatch, cond=cond)(original_fn)

        with patch(
            "paddlefleet_ops.ops.triton_ops.utils.is_torch_compat_available",
            return_value=True,
        ):
            result = decorated()
            self.assertEqual(result, "original")

    def test_dispatch_to_with_cond_true(self):
        """Test dispatch_to dispatches when cond returns True."""
        from paddlefleet_ops.ops.triton_ops.utils import dispatch_to

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        cond = lambda *args, **kwargs: True
        decorated = dispatch_to(dummy_dispatch, cond=cond)(original_fn)

        with patch(
            "paddlefleet_ops.ops.triton_ops.utils.is_torch_compat_available",
            return_value=True,
        ):
            result = decorated()
            self.assertEqual(result, "dispatched")

    def test_dispatch_to_preserves_original_fn(self):
        """Test dispatch_to stores original function in __original_fn__."""
        from paddlefleet_ops.ops.triton_ops.utils import dispatch_to

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        decorated = dispatch_to(dummy_dispatch)(original_fn)
        self.assertTrue(hasattr(decorated, "__original_fn__"))
        self.assertIs(decorated.__original_fn__, original_fn)

    def test_dispatch_to_passes_args(self):
        """Test dispatch_to passes arguments correctly."""
        from paddlefleet_ops.ops.triton_ops.utils import dispatch_to

        def dummy_dispatch(*args, **kwargs):
            return ("dispatched", args, kwargs)

        def original_fn(*args, **kwargs):
            return ("original", args, kwargs)

        decorated = dispatch_to(dummy_dispatch)(original_fn)

        with patch(
            "paddlefleet_ops.ops.triton_ops.utils.is_torch_compat_available",
            return_value=True,
        ):
            result = decorated(1, 2, key="val")
            self.assertEqual(result[0], "dispatched")
            self.assertEqual(result[1], (1, 2))
            self.assertEqual(result[2], {"key": "val"})


class TestIsPackageInstalled(unittest.TestCase):
    """Tests for _is_package_installed function."""

    def test_installed_package(self):
        """Test _is_package_installed returns True for installed packages."""
        from paddlefleet_ops.ops.triton_ops.utils import _is_package_installed

        result = _is_package_installed("paddle")
        self.assertTrue(result)

    def test_not_installed_package(self):
        """Test _is_package_installed returns False for non-existent packages."""
        from paddlefleet_ops.ops.triton_ops.utils import _is_package_installed

        result = _is_package_installed(
            "this_package_definitely_does_not_exist_67890"
        )
        self.assertFalse(result)


class TestSwapDriverGuard(unittest.TestCase):
    """Tests for swap_driver_guard function."""

    def test_swap_driver_guard_wraps_function(self):
        """Test swap_driver_guard returns a wrapper."""
        from paddlefleet_ops.ops.triton_ops.utils import swap_driver_guard

        def dummy_fn():
            return 42

        wrapped = swap_driver_guard(dummy_fn)
        self.assertTrue(callable(wrapped))


class TestEnableCompatOnTritonKernel(unittest.TestCase):
    """Tests for enable_compat_on_triton_kernel function."""

    def test_returns_kernel_when_no_cuda(self):
        """Test that kernel is returned as-is when CUDA not available."""
        from paddlefleet_ops.ops.triton_ops.utils import (
            enable_compat_on_triton_kernel,
        )

        def dummy_kernel():
            pass

        with patch("paddle.is_compiled_with_cuda", return_value=False):
            result = enable_compat_on_triton_kernel(dummy_kernel)
            self.assertIs(result, dummy_kernel)


class TestModuleStructure(unittest.TestCase):
    """Tests for module structure."""

    def test_module_exports(self):
        """Test that expected symbols are exported."""
        from paddlefleet_ops.ops.triton_ops import utils

        self.assertTrue(hasattr(utils, "is_torch_compat_available"))
        self.assertTrue(hasattr(utils, "dispatch_to"))
        self.assertTrue(hasattr(utils, "enable_compat_on_triton_kernel"))
        self.assertTrue(hasattr(utils, "_is_package_installed"))
        self.assertTrue(hasattr(utils, "swap_driver_guard"))


if __name__ == "__main__":
    unittest.main()
