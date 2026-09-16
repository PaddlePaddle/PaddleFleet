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

"""No-card behavior tests for paddlefleet.transformer.moe.fused_a2a.

These exercise CPU-executable control/layout logic only:
  * get_hidden_bytes            -- per-token byte budget arithmetic
  * _normalize_fp8_scale_for_deepep -- transpose / truncate / validate
  * _ep_fence_tensor            -- per-group fence-tensor caching
  * barrier_ep                  -- env-driven barrier vs. stream fence branch
  * DispatchNode / CombineNode  -- name + reset_statue state contract

Real EP collectives (barrier / all_reduce) are genuine not-under-test
collaborators that require a real process group; they are mocked here and the
cross-rank rendezvous is NOT verified by this file. Expected values are derived
by hand from the production source, independent of any coverage_test file.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe import fused_a2a
    from paddlefleet.transformer.moe.fused_a2a import (
        CombineNode,
        DispatchNode,
        _ep_fence_tensor,
        _normalize_fp8_scale_for_deepep,
        barrier_ep,
        get_hidden_bytes,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest, precise skip
    _IMPORT_ERROR = exc


_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else f"fused_a2a import failed: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestGetHiddenBytes(unittest.TestCase):
    """get_hidden_bytes(x) == x.shape[1] * max(x.element_size(), 2)."""

    def test_uses_hidden_dim_and_element_size(self):
        # Non-square tensors: token count (shape[0]) must NOT enter the result,
        # only hidden dim (shape[1]). float32 element_size == 4.
        self.assertEqual(
            get_hidden_bytes(paddle.zeros([3, 64], "float32")), 256
        )
        self.assertEqual(
            get_hidden_bytes(paddle.zeros([5, 128], "float32")), 512
        )

    def test_float16_and_bfloat16_use_two_bytes(self):
        self.assertEqual(
            get_hidden_bytes(paddle.zeros([3, 64], "float16")), 128
        )
        self.assertEqual(
            get_hidden_bytes(paddle.zeros([3, 64], "bfloat16")), 128
        )

    def test_bool_is_floored_at_two_bytes(self):
        # bool element_size == 1, but max(.., 2) floors it -> 64 * 2, not 64 * 1.
        self.assertEqual(get_hidden_bytes(paddle.zeros([3, 64], "bool")), 128)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestNormalizeFp8Scale(unittest.TestCase):
    """_normalize_fp8_scale_for_deepep transpose/truncate/validate layout."""

    def _x(self, hidden):
        # only shape is read by the function; content/dtype are irrelevant here.
        return paddle.zeros([4, hidden], "float32")

    def test_transposes_when_rows_equal_num_scales(self):
        # x hidden=256 -> num_scales=2, num_tokens=4. scale rows==2 -> transpose.
        scale = paddle.arange(8, dtype="int32").reshape([2, 4])
        out = _normalize_fp8_scale_for_deepep(self._x(256), scale)
        np.testing.assert_array_equal(
            out.numpy(), np.array([[0, 4], [1, 5], [2, 6], [3, 7]], "int32")
        )

    def test_no_transpose_when_rows_differ_from_num_scales(self):
        # scale already [num_tokens, num_scales] = [4, 2]; must pass through as-is.
        scale = paddle.arange(8, dtype="int32").reshape([4, 2])
        out = _normalize_fp8_scale_for_deepep(self._x(256), scale)
        np.testing.assert_array_equal(
            out.numpy(), np.array([[0, 1], [2, 3], [4, 5], [6, 7]], "int32")
        )

    def test_truncates_extra_leading_rows(self):
        # rows=6 > num_tokens=4 -> keep first 4 rows, in order.
        scale = paddle.arange(12, dtype="int32").reshape([6, 2])
        out = _normalize_fp8_scale_for_deepep(self._x(256), scale)
        np.testing.assert_array_equal(
            out.numpy(), np.array([[0, 1], [2, 3], [4, 5], [6, 7]], "int32")
        )

    def test_ue8m0_flag_changes_num_scales(self):
        # hidden=1024 -> num_scales=8; with use_ue8m0 it becomes 8//4=2 so the
        # [2,4] scale transposes to a valid [4,2]. The same inputs WITHOUT the
        # flag leave num_scales=8 and must be rejected -> proves flag is consumed.
        scale = paddle.arange(8, dtype="int32").reshape([2, 4])
        out = _normalize_fp8_scale_for_deepep(
            self._x(1024), scale, use_ue8m0=True
        )
        np.testing.assert_array_equal(
            out.numpy(), np.array([[0, 4], [1, 5], [2, 6], [3, 7]], "int32")
        )
        with self.assertRaises(RuntimeError):
            _normalize_fp8_scale_for_deepep(
                self._x(1024), scale, use_ue8m0=False
            )

    def test_invalid_shape_raises(self):
        # rows=3: != num_scales(2), not > num_tokens(4) -> final shape check fails.
        scale = paddle.arange(6, dtype="int32").reshape([3, 2])
        with self.assertRaises(RuntimeError):
            _normalize_fp8_scale_for_deepep(self._x(256), scale)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestEpFenceTensor(unittest.TestCase):
    """_ep_fence_tensor caches one zero int32[1] tensor per group id."""

    def setUp(self):
        saved = dict(fused_a2a._ep_fence_tensors)

        def _restore():
            fused_a2a._ep_fence_tensors.clear()
            fused_a2a._ep_fence_tensors.update(saved)

        self.addCleanup(_restore)

    def test_content_identity_and_per_group_isolation(self):
        g1 = SimpleNamespace(id=90001)
        g2 = SimpleNamespace(id=90002)

        t1 = _ep_fence_tensor(g1)
        self.assertEqual(list(t1.shape), [1])
        self.assertEqual(t1.dtype, paddle.int32)
        np.testing.assert_array_equal(t1.numpy(), np.array([0], "int32"))

        # Same id returns the very same cached object (not a fresh alloc).
        self.assertIs(_ep_fence_tensor(g1), t1)
        # A different id gets its own distinct tensor.
        self.assertIsNot(_ep_fence_tensor(g2), t1)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestBarrierEp(unittest.TestCase):
    """barrier_ep selects real barrier vs. stream-ordered all_reduce fence.

    The env switch is cached in the module global _EP_BARRIER_ASYNC; each test
    resets it, forces re-read, and restores it plus the fence-tensor cache.
    The collectives themselves are mocked (real rendezvous NOT verified here).
    """

    def _reset_switch(self, value):
        orig_flag = fused_a2a._EP_BARRIER_ASYNC
        self.addCleanup(setattr, fused_a2a, "_EP_BARRIER_ASYNC", orig_flag)
        saved_fence = dict(fused_a2a._ep_fence_tensors)

        def _restore_fence():
            fused_a2a._ep_fence_tensors.clear()
            fused_a2a._ep_fence_tensors.update(saved_fence)

        self.addCleanup(_restore_fence)
        env = patch.dict(os.environ, {"FLEET_MOE_EP_BARRIER_ASYNC": value})
        env.start()
        self.addCleanup(env.stop)
        fused_a2a._EP_BARRIER_ASYNC = None  # force env re-read

    def test_sync_branch_calls_barrier_only(self):
        self._reset_switch("0")
        group = SimpleNamespace(id=91001)
        with (
            patch.object(paddle.distributed, "barrier") as barrier,
            patch.object(paddle.distributed, "all_reduce") as all_reduce,
        ):
            barrier_ep(group)
        barrier.assert_called_once_with(group)
        all_reduce.assert_not_called()

    def test_async_branch_all_reduces_fence_tensor_on_group(self):
        self._reset_switch("1")
        group = SimpleNamespace(id=91002)
        with (
            patch.object(paddle.distributed, "barrier") as barrier,
            patch.object(paddle.distributed, "all_reduce") as all_reduce,
        ):
            barrier_ep(group)
        barrier.assert_not_called()
        all_reduce.assert_called_once()
        args, kwargs = all_reduce.call_args
        # The reduced tensor is exactly the cached fence tensor for this group,
        # and the reduction targets this group (not some default world group).
        self.assertIs(args[0], _ep_fence_tensor(group))
        self.assertIs(kwargs["group"], group)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDispatchCombineNodeState(unittest.TestCase):
    """DispatchNode/CombineNode name + reset_statue state contract."""

    def test_dispatch_node_reset_state(self):
        node = DispatchNode()
        self.assertEqual(node.name, "dispatch")
        self.assertFalse(hasattr(node, "handle"))  # not set until reset/forward
        node.reset_statue()
        self.assertIsNone(node.handle)

    def test_combine_node_reset_state(self):
        node = CombineNode()
        self.assertEqual(node.name, "combine")
        self.assertFalse(hasattr(node, "handle"))
        node.reset_statue()
        self.assertIsNone(node.handle)


if __name__ == "__main__":
    unittest.main()
