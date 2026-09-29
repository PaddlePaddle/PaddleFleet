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
"""Behavior tests for MoE straight-through and group all-gather helpers in
``paddlefleet.transformer.moe.moe_utils``.

Every expectation is hand-derived from the documented behavior of the code
under test, independent of any coverage-oriented fixture:

* ``RandomSTE`` / ``apply_random_logits`` -- a straight-through PyLayer whose
  forward *replaces* the logits with fresh ``randn`` samples (same shape and
  dtype, content independent of the input, different on every call) and whose
  backward is a hard zero regardless of the incoming gradient. Both the forward
  replacement and the zero backward are driven through the real ``.apply``
  autograd path, not by hand-constructing a ``ctx``.
* ``all_gather_group`` / ``reduce_scatter_group`` -- only the genuine
  ``nranks == 1`` local fallback (returns a clone of the input) and the
  divisibility guard that rejects a bad shape *before* any collective runs.
  These assert nothing about real cross-rank communication; true all-gather /
  reduce-scatter numerics require a real multi-card process group.
* ``AllGatherGroupOp`` -- the ``nranks == 1`` forward/backward path exercised
  through ``.apply``. The only mocked collaborator is the distributed
  ``barrier`` sync point (not the gather/scatter math); the test records the
  argument it receives to confirm the group is propagated into ``ctx`` and
  reused on the backward, and it checks the real cloned tensor content.

Paddle is required to import the module under test. When it is unavailable the
whole suite is skipped with an honest reason rather than reported as passing.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.moe_utils import (
        AllGatherGroupOp,
        RandomSTE,
        all_gather_group,
        apply_random_logits,
        reduce_scatter_group,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # CPU env without paddle
    paddle = None
    _IMPORT_ERROR = exc

_HAS_PADDLE = paddle is not None
_SKIP_REASON = (
    "paddle / paddlefleet.transformer.moe.moe_utils is not importable in this "
    f"environment ({_IMPORT_ERROR!r}); these contracts exercise real paddle "
    "tensors and PyLayer autograd, so they cannot run without paddle."
)

_BARRIER_PATH = (
    "paddlefleet.transformer.moe.moe_utils.paddle.distributed.barrier"
)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestRandomSTE(unittest.TestCase):
    """RandomSTE: forward = fresh randn (shape/dtype kept), backward = zeros.

    Single-process ``get_world_size() == 1`` local path; no distributed init.
    """

    def test_forward_replaces_content_but_keeps_shape_and_dtype(self):
        # Constant input: if forward merely cloned/passed through, the output
        # would still be all 100.0. RandomSTE must instead emit fresh randn,
        # so the output keeps the shape/dtype yet differs in content.
        logits = paddle.full([4, 8], 100.0, dtype="float32")
        out = RandomSTE.apply(logits)
        self.assertEqual(list(out.shape), [4, 8])
        self.assertEqual(out.dtype, paddle.float32)
        self.assertFalse(
            np.allclose(out.numpy(), logits.numpy()),
            "forward must replace logits with random samples, not pass through",
        )

    def test_forward_dtype_follows_input_dtype(self):
        # randn defaults to float32; the ``.cast(x.dtype)`` must follow the
        # input dtype rather than hardcoding float32.
        logits = paddle.full([2, 3], 1.0, dtype="float64")
        out = RandomSTE.apply(logits)
        self.assertEqual(out.dtype, paddle.float64)
        self.assertEqual(list(out.shape), [2, 3])

    def test_forward_is_random_on_each_call(self):
        logits = paddle.full([5, 6], 0.0, dtype="float32")
        first = RandomSTE.apply(logits).numpy()
        second = RandomSTE.apply(logits).numpy()
        self.assertFalse(
            np.array_equal(first, second),
            "two forward calls must draw independent random samples",
        )

    def test_backward_is_zero_ignoring_upstream_gradient(self):
        logits = paddle.arange(12, dtype="float32").reshape([3, 4])
        logits.stop_gradient = False
        out = RandomSTE.apply(logits)
        # Distinguishable, non-uniform upstream gradient: a straight-through
        # (identity) or scaled backward would forward these values; RandomSTE
        # must instead return an exact zero of the input shape/dtype.
        upstream = (
            paddle.arange(12, dtype="float32").reshape([3, 4]) + 1.0
        ) * 7.0
        out.backward(upstream)
        self.assertIsNotNone(logits.grad)
        self.assertEqual(list(logits.grad.shape), [3, 4])
        self.assertEqual(logits.grad.dtype, paddle.float32)
        np.testing.assert_array_equal(
            logits.grad.numpy(), np.zeros([3, 4], dtype=np.float32)
        )


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestApplyRandomLogits(unittest.TestCase):
    """apply_random_logits is the public RandomSTE entry point."""

    def test_public_entry_keeps_shape_dtype_and_zeroes_gradient(self):
        logits = paddle.arange(8, dtype="float32").reshape([2, 4])
        logits.stop_gradient = False
        out = apply_random_logits(logits)
        self.assertTrue(paddle.is_tensor(out))
        self.assertEqual(list(out.shape), [2, 4])
        self.assertEqual(out.dtype, paddle.float32)
        # Random replacement, not a clone of the input.
        self.assertFalse(np.allclose(out.numpy(), logits.numpy()))

        upstream = paddle.full([2, 4], -3.0, dtype="float32")
        out.backward(upstream)
        np.testing.assert_array_equal(
            logits.grad.numpy(), np.zeros([2, 4], dtype=np.float32)
        )


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestGroupCollectiveLocalFallback(unittest.TestCase):
    """nranks == 1 local fallbacks and the pre-collective divisibility guard.

    These cover the genuine world-size-1 code path and an input-validation
    guard that raises *before* any collective is issued. They do NOT validate
    real cross-rank all-gather / reduce-scatter numerics, which need a real
    multi-card process group.
    """

    def test_all_gather_group_single_rank_returns_clone(self):
        group = SimpleNamespace(nranks=1)
        x = paddle.arange(6, dtype="float32").reshape([2, 3])
        out = all_gather_group(x, group=group)
        # nranks == 1 must return the input content unchanged, as a fresh
        # tensor (clone), not the same object and not a resized gather.
        self.assertEqual(list(out.shape), [2, 3])
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertIsNot(out, x)

    def test_reduce_scatter_group_single_rank_returns_clone(self):
        group = SimpleNamespace(nranks=1)
        x = paddle.arange(6, dtype="float32").reshape([2, 3])
        out = reduce_scatter_group(x, group=group)
        self.assertEqual(list(out.shape), [2, 3])
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertIsNot(out, x)

    def test_reduce_scatter_group_rejects_indivisible_leading_dim(self):
        # nranks == 2 but leading dim 3 is not divisible: the guard must raise
        # AssertionError before any collective is attempted.
        group = SimpleNamespace(nranks=2)
        x = paddle.zeros([3, 4], dtype="float32")
        with self.assertRaises(AssertionError):
            reduce_scatter_group(x, group=group)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestAllGatherGroupOp(unittest.TestCase):
    """AllGatherGroupOp nranks == 1 forward/backward via the real .apply path.

    Only ``paddle.distributed.barrier`` (a sync point, not the gather math) is
    mocked. The recorded argument confirms the group is propagated to the
    barrier and stored on ``ctx`` for reuse in the backward; tensor content is
    checked against the real nranks == 1 clone fallback.
    """

    def test_single_rank_forward_and_backward_clone_and_reuse_group(self):
        group = SimpleNamespace(nranks=1)
        captured = []

        def fake_barrier(barrier_group=None, *args, **kwargs):
            captured.append(barrier_group)

        x = paddle.arange(6, dtype="float32").reshape([2, 3])
        x.stop_gradient = False
        with mock.patch(_BARRIER_PATH, side_effect=fake_barrier):
            out = AllGatherGroupOp.apply(x, group)
            # Forward: nranks == 1 -> clone of the input content.
            self.assertEqual(list(out.shape), [2, 3])
            np.testing.assert_array_equal(out.numpy(), x.numpy())
            # barrier ran once so far, with the exact group object.
            self.assertEqual(len(captured), 1)
            self.assertIs(captured[0], group)

            # Backward: reduce_scatter_group at nranks == 1 clones the upstream
            # gradient straight through to the input.
            upstream = (
                paddle.arange(6, dtype="float32").reshape([2, 3]) + 1.0
            ) * 5.0
            out.backward(upstream)

        self.assertIsNotNone(x.grad)
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())
        # barrier ran again on the backward, again with the group carried on
        # ctx (proves ctx.group was set in forward and reused).
        self.assertEqual(len(captured), 2)
        self.assertIs(captured[1], group)


if __name__ == "__main__":
    unittest.main()
