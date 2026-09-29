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

"""Behavior tests for the fifth slice of paddlefleet.utils._fleet_utils.

This is the fifth concurrent slice of the paddlefleet.utils coverage;
sibling files test_utils.py / _2 / _3 / _4 cover the other helpers. This
file covers exactly the helpers the base slice explicitly deferred to a
sibling: the context-parallel batch dispatcher and the NVTX profiling
family.

Slice covered here:
  * get_batch_on_this_cp_rank - the type-dispatch and per-key routing of the
                                CP scatter: which dict keys get scattered,
                                which pass through by identity, and the two
                                error branches (list -> AssertionError,
                                other -> ValueError). The real scatter op
                                (ContextParallelScatterOp, a genuine
                                collaborator that needs a CP process group)
                                is replaced by an input-dependent marker so
                                the dispatch itself is what is observed.
  * nvtx_decorator            - the enabled/disabled switch: disabled returns
                                the ORIGINAL function object (identity), and
                                enabled forwards message/color to
                                nvtx.annotate and returns the wrapped result.
  * nvtx_range_push /
    nvtx_range_pop            - the message-stack bookkeeping: disabled is a
                                no-op, enabled appends/pops the module-level
                                message stack, applies the suffix, drives
                                paddle's nvprof push/pop, and raises on an
                                empty stack or a mismatched message.

Every expected value is hand-derived. The only mocks isolate genuine
NON-tested collaborators (the CP scatter op, the optional `nvtx` package,
and paddle's C++ nvprof entry points); the dispatch, switch and stack logic
under test are never mocked. Mutated module globals (`_nvtx_enabled`,
`_nvtx_range_messages`) are restored with patch scoping / addCleanup so no
state leaks between tests.

paddlefleet.utils._fleet_utils imports paddle unconditionally at module top,
so nothing here is reachable without paddle. Only a genuine missing
dependency (ImportError) is allowed to skip; any other error surfaces as a
real failure rather than a fake pass.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle

    import paddlefleet.utils._fleet_utils as fu

    HAS_PADDLE = True
    _IMPORT_ERROR = None
except ImportError as exc:  # genuine missing dependency only
    paddle = None
    fu = None
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)

_SKIP_REASON = (
    "paddle is not installed; paddlefleet.utils._fleet_utils is unimportable "
    f"on this CPU-only box ({_IMPORT_ERROR})"
)

_DEFAULT_MODE = "dualchunk_allgather"


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGetBatchOnThisCpRank(unittest.TestCase):
    """Dispatch/routing contract of get_batch_on_this_cp_rank.

    ContextParallelScatterOp.apply is the real scatter and needs a CP process
    group, so it is replaced by an input-dependent marker (t -> t * 10). The
    marker records every (tensor, axis, mode) it receives, which lets each
    test assert *which* inputs were scattered, with what axis/mode, and that
    non-target values are passed through by object identity.
    """

    def _install_marker(self):
        calls = []

        def marker(input_tensor, axis=0, mode=_DEFAULT_MODE):
            calls.append({"tensor": input_tensor, "axis": axis, "mode": mode})
            return input_tensor * 10.0

        patcher = mock.patch.object(
            fu.ContextParallelScatterOp, "apply", staticmethod(marker)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def test_tensor_input_scattered_with_default_mode(self):
        calls = self._install_marker()
        inp = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]], dtype="float32")

        out = fu.get_batch_on_this_cp_rank(inp)

        # Return value is exactly the scatter result (hand-derived: t * 10).
        np.testing.assert_array_equal(
            out.numpy(), np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["tensor"], inp)
        self.assertEqual(calls[0]["axis"], -1)
        self.assertEqual(calls[0]["mode"], _DEFAULT_MODE)

    def test_tensor_input_forwards_custom_mode(self):
        calls = self._install_marker()
        inp = paddle.to_tensor([[5.0, 6.0]], dtype="float32")

        out = fu.get_batch_on_this_cp_rank(inp, cp_balance_mode="contiguous")

        np.testing.assert_array_equal(
            out.numpy(), np.array([[50.0, 60.0]], dtype=np.float32)
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["axis"], -1)
        self.assertEqual(calls[0]["mode"], "contiguous")

    def test_dict_scatters_only_target_keys_and_preserves_others(self):
        calls = self._install_marker()
        # Distinct constant per key so a swapped routing would be visible.
        input_ids = paddle.full([1, 2], 1.0, dtype="float32")
        position_ids = paddle.full([1, 2], 2.0, dtype="float32")
        labels = paddle.full([1, 2], 3.0, dtype="float32")
        attention_mask = paddle.full([1, 2], 4.0, dtype="float32")
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
        }

        res = fu.get_batch_on_this_cp_rank(inputs)

        # Same keys back, none dropped or added.
        self.assertEqual(
            set(res),
            {"input_ids", "attention_mask", "position_ids", "labels"},
        )
        # The three target keys are scattered (marker -> value * 10).
        np.testing.assert_array_equal(
            res["input_ids"].numpy(), np.full([1, 2], 10.0, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            res["position_ids"].numpy(),
            np.full([1, 2], 20.0, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            res["labels"].numpy(), np.full([1, 2], 30.0, dtype=np.float32)
        )
        # Non-target key is passed through by identity, untouched.
        self.assertIs(res["attention_mask"], attention_mask)

        # Exactly the three target tensors were scattered, axis=-1, default mode.
        self.assertEqual(len(calls), 3)
        scattered = {id(c["tensor"]) for c in calls}
        self.assertEqual(
            scattered, {id(input_ids), id(position_ids), id(labels)}
        )
        self.assertNotIn(id(attention_mask), scattered)
        for c in calls:
            self.assertEqual(c["axis"], -1)
            self.assertEqual(c["mode"], _DEFAULT_MODE)

    def test_list_input_raises_assertion_error(self):
        self._install_marker()
        with self.assertRaises(AssertionError):
            fu.get_batch_on_this_cp_rank(
                [paddle.to_tensor([1.0, 2.0], dtype="float32")]
            )

    def test_unsupported_type_raises_value_error(self):
        self._install_marker()
        with self.assertRaises(ValueError):
            fu.get_batch_on_this_cp_rank("not_a_tensor_or_dict")


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestNvtxDecorator(unittest.TestCase):
    """The enable/disable switch inside nvtx_decorator."""

    def test_disabled_returns_original_function_unchanged(self):
        # _nvtx_enabled defaults to False; the decorator must be transparent.
        self.assertFalse(fu._nvtx_enabled)

        def my_function(a, b):
            return a - b

        decorated = fu.nvtx_decorator(message="ignored")(my_function)

        # Identity: the exact same object is returned, no wrapping happens.
        self.assertIs(decorated, my_function)
        self.assertEqual(decorated(7, 4), 3)

    def test_enabled_forwards_message_and_color_to_annotate(self):
        wrapped_sentinel = object()
        annotate_returned = mock.MagicMock(return_value=wrapped_sentinel)
        fake_nvtx = mock.MagicMock()
        fake_nvtx.annotate = mock.MagicMock(return_value=annotate_returned)

        def my_function():
            return 42

        with (
            mock.patch.object(fu, "_nvtx_enabled", True),
            mock.patch.object(fu, "nvtx", fake_nvtx, create=True),
        ):
            result = fu.nvtx_decorator(message="Custom Range", color="blue")(
                my_function
            )

        # message/color are forwarded verbatim, and the annotate wrapper is
        # applied to the real function; the wrapped result is returned.
        fake_nvtx.annotate.assert_called_once_with(
            message="Custom Range", color="blue"
        )
        annotate_returned.assert_called_once_with(my_function)
        self.assertIs(result, wrapped_sentinel)

    def test_enabled_default_message_is_module_qualified_func_path(self):
        annotate_returned = mock.MagicMock(return_value="wrapped")
        fake_nvtx = mock.MagicMock()
        fake_nvtx.annotate = mock.MagicMock(return_value=annotate_returned)

        def my_function():
            return None

        # Independent expectation: module-qualified name via standard dunders,
        # not via the production path helper.
        expected_msg = f"{my_function.__module__}.{my_function.__name__}"

        with (
            mock.patch.object(fu, "_nvtx_enabled", True),
            mock.patch.object(fu, "nvtx", fake_nvtx, create=True),
        ):
            fu.nvtx_decorator()(my_function)

        fake_nvtx.annotate.assert_called_once_with(
            message=expected_msg, color=None
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestNvtxRangeStack(unittest.TestCase):
    """Message-stack bookkeeping of nvtx_range_push / nvtx_range_pop.

    paddle's C++ nvprof entry points are genuine collaborators that need a
    GPU/profiler; they are replaced by recording stubs so the pure stack logic
    is what is exercised. The module-level message stack is patched to a fresh
    list per test so nothing leaks.
    """

    def test_push_is_noop_when_disabled(self):
        self.assertFalse(fu._nvtx_enabled)
        stack = []
        push_stub = mock.MagicMock()
        with (
            mock.patch.object(fu, "_nvtx_range_messages", stack),
            mock.patch(
                "paddlefleet.utils._fleet_utils.paddle.base.core.nvprof_nvtx_push",
                push_stub,
                create=True,
            ),
        ):
            fu.nvtx_range_push(msg="range_a")

        # Disabled: no message recorded and the profiler is never touched.
        self.assertEqual(stack, [])
        push_stub.assert_not_called()

    def test_enabled_push_pop_roundtrip_with_suffix(self):
        stack = []
        pushed = []
        popped = []

        with (
            mock.patch.object(fu, "_nvtx_enabled", True),
            mock.patch.object(fu, "_nvtx_range_messages", stack),
            mock.patch(
                "paddlefleet.utils._fleet_utils.paddle.base.core.nvprof_nvtx_push",
                side_effect=lambda m: pushed.append(m),
                create=True,
            ),
            mock.patch(
                "paddlefleet.utils._fleet_utils.paddle.base.core.nvprof_nvtx_pop",
                side_effect=lambda: popped.append(True),
                create=True,
            ),
        ):
            fu.nvtx_range_push(msg="outer")
            fu.nvtx_range_push(msg="inner", suffix="fwd")
            # Stack holds both, suffix joined with a dot.
            self.assertEqual(stack, ["outer", "inner.fwd"])
            self.assertEqual(pushed, ["outer", "inner.fwd"])

            fu.nvtx_range_pop(msg="inner", suffix="fwd")
            self.assertEqual(stack, ["outer"])
            fu.nvtx_range_pop(msg="outer")
            self.assertEqual(stack, [])

        # Two pushes and two pops actually drove the profiler.
        self.assertEqual(pushed, ["outer", "inner.fwd"])
        self.assertEqual(len(popped), 2)

    def test_enabled_pop_empty_stack_raises_runtime_error(self):
        stack = []
        with (
            mock.patch.object(fu, "_nvtx_enabled", True),
            mock.patch.object(fu, "_nvtx_range_messages", stack),
            mock.patch(
                "paddlefleet.utils._fleet_utils.paddle.base.core.nvprof_nvtx_pop",
                mock.MagicMock(),
                create=True,
            ),
            self.assertRaises(RuntimeError),
        ):
            fu.nvtx_range_pop(msg="anything")

    def test_enabled_pop_mismatched_message_raises_value_error(self):
        stack = []
        with (
            mock.patch.object(fu, "_nvtx_enabled", True),
            mock.patch.object(fu, "_nvtx_range_messages", stack),
            mock.patch(
                "paddlefleet.utils._fleet_utils.paddle.base.core.nvprof_nvtx_push",
                mock.MagicMock(),
                create=True,
            ),
            mock.patch(
                "paddlefleet.utils._fleet_utils.paddle.base.core.nvprof_nvtx_pop",
                mock.MagicMock(),
                create=True,
            ),
        ):
            fu.nvtx_range_push(msg="pushed_name")
            with self.assertRaises(ValueError):
                fu.nvtx_range_pop(msg="different_name")


if __name__ == "__main__":
    unittest.main()
