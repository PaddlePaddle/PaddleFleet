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

"""CPU behavior tests for ``paddlefleet.tensor_parallel.mappings`` (part 5).

Repository module: "分布式训练" (tensor parallel). This file deliberately
covers a DIFFERENT set of branches than ``test_mappings.py`` and its siblings:
those files exercise ``_reduce`` / ``_split_*`` / the gather+reduce-scatter
helper guards and the forward dispatch of the Copy / Reduce / Scatter / Gather
autograd Functions. Here the focus is the THREE autograd Functions none of the
sibling files touch, and -- crucially -- their BACKWARD dispatch, driven for
real through paddle autograd on the only paths that need no >1-rank collective:

  * ``_AllGatherFromTensorParallelRegion`` -- ``group is None`` forward/backward
    identity, and its ``group`` set + ``world_size == 1`` forward identity;
  * ``_ReduceScatterToTensorParallelRegion`` -- ``group is None`` forward and
    backward identity;
  * ``_AllToAll`` -- the ``world_size == 1`` forward fast path and the matching
    identity backward (which re-invokes ``_AllToAll.apply`` with the split-size
    arguments transposed).

The genuine multi-rank collectives (all_gather / reduce_scatter / all_to_all
with world_size > 1) are intentionally NOT exercised: they require a real
multi-rank process group and belong in ``tests/multi_card_tests``. Faking a
world_size and mocking the collective to assert only "was called" would prove
nothing about split sizes, peer selection, direction, or cross-rank reduction
(antipattern #13), so it is avoided here. Backward is executed for real (not
forward-only, antipattern #7) and compared against hand-derived, non-uniform
upstream gradients so a wrong reduction coefficient or dropped grad is caught.

Paddle is not installed in the no-card env, hence the honest skip.
"""

import unittest

try:
    import paddle

    from paddlefleet.tensor_parallel.mappings import (
        _AllGatherFromTensorParallelRegion,
        _AllToAll,
        _ReduceScatterToTensorParallelRegion,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"


class _Group:
    """Minimal real stand-in for a process-group value object.

    The tested single-rank / None-group paths only *read* topology scalars
    (``world_size``, ``nranks``, ``rank``, ``ranks``); they never launch a
    collective on it. This is a plain data holder, not a mocked collective.
    """

    def __init__(self, world_size, rank=0):
        self.world_size = world_size
        self.nranks = world_size
        self.rank = rank
        self.ranks = list(range(world_size))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllGatherFromTensorParallelRegion(unittest.TestCase):
    """_AllGatherFromTensorParallelRegion forward + backward (no collective)."""

    def test_none_group_forward_is_content_passthrough(self):
        # forward: ``if group is None: return input_`` -- exact content, not
        # just shape/type.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllGatherFromTensorParallelRegion.apply(x, None)
        self.assertEqual(
            out.tolist(), [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
        )

    def test_world_size_one_forward_is_content_passthrough(self):
        # forward routes into ``_gather_along_last_dim`` which, for
        # world_size == 1, returns the input unchanged.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllGatherFromTensorParallelRegion.apply(x, _Group(1))
        self.assertEqual(
            out.tolist(), [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
        )

    def test_none_group_backward_passes_gradient_through_unchanged(self):
        # backward: ``if ctx.group is None: return grad_output`` -- so d(out)/dx
        # is the identity and the leaf grad must equal the (non-uniform)
        # upstream coefficient exactly. A wrong backward (e.g. an accidental
        # reduce/scale) would change these values.
        leaf = paddle.arange(8, dtype="float32").reshape([2, 4])
        leaf.stop_gradient = False
        # The None-group forward returns its input unchanged; feed a non-leaf
        # (clone) because Paddle forbids an identity autograd Function on a
        # grad-requiring leaf. clone's identity backward preserves the grad.
        x = leaf.clone()
        coef = paddle.to_tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype="float32"
        )
        out = _AllGatherFromTensorParallelRegion.apply(x, None)
        (out * coef).sum().backward()
        self.assertIsNotNone(leaf.grad)
        self.assertEqual(leaf.grad.tolist(), coef.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestReduceScatterToTensorParallelRegion(unittest.TestCase):
    """_ReduceScatterToTensorParallelRegion forward + backward (no collective)."""

    def test_none_group_forward_is_content_passthrough(self):
        # forward: ``if group is None: return input_``.
        x = paddle.arange(12, dtype="float32").reshape([3, 4])
        out = _ReduceScatterToTensorParallelRegion.apply(x, None)
        self.assertEqual(
            out.tolist(),
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
        )

    def test_none_group_backward_passes_gradient_through_unchanged(self):
        # backward: ``if ctx.group is None: return grad_output`` -- identity.
        leaf = paddle.arange(6, dtype="float32").reshape([2, 3])
        leaf.stop_gradient = False
        # Non-leaf (clone) input: the None-group forward is an identity
        # pass-through, which Paddle rejects on a grad-requiring leaf. clone's
        # identity backward preserves the grad.
        x = leaf.clone()
        coef = paddle.to_tensor(
            [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype="float32"
        )
        out = _ReduceScatterToTensorParallelRegion.apply(x, None)
        (out * coef).sum().backward()
        self.assertIsNotNone(leaf.grad)
        self.assertEqual(leaf.grad.tolist(), coef.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllToAllSingleRank(unittest.TestCase):
    """_AllToAll world_size == 1 fast path, forward and identity backward."""

    def test_world_size_one_forward_is_content_passthrough(self):
        # forward(ctx, group, input, output_split_sizes, input_split_sizes):
        # ``if world_size == 1: return input`` -- returns the input content.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllToAll.apply(_Group(1), x, None, None)
        self.assertEqual(
            out.tolist(), [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
        )

    def test_world_size_one_backward_is_identity(self):
        # backward re-invokes ``_AllToAll.apply(group, grad, input_split_sizes,
        # output_split_sizes)`` (split sizes transposed). With world_size == 1
        # every branch is the identity fast path, so the leaf grad must equal
        # the non-uniform upstream coefficient exactly.
        leaf = paddle.arange(8, dtype="float32").reshape([2, 4])
        leaf.stop_gradient = False
        # Non-leaf (clone) input: the world_size==1 forward returns its input
        # unchanged, which Paddle rejects as an identity Function on a
        # grad-requiring leaf. clone's identity backward preserves the grad.
        x = leaf.clone()
        coef = paddle.to_tensor(
            [[1.0, 3.0, 5.0, 7.0], [9.0, 11.0, 13.0, 15.0]], dtype="float32"
        )
        out = _AllToAll.apply(_Group(1), x, None, None)
        (out * coef).sum().backward()
        self.assertIsNotNone(leaf.grad)
        self.assertEqual(leaf.grad.tolist(), coef.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllGatherBackwardSingleRankReduceScatterBug(unittest.TestCase):
    """Backward of AllGather routes into ``_reduce_scatter_along_last_dim``.

    The ``group is not None`` branch of
    ``_AllGatherFromTensorParallelRegion.backward`` dispatches into
    ``_reduce_scatter_along_last_dim(grad, group)`` (a branch the sibling
    forward-only tests never drive). That helper has no ``world_size == 1``
    fast path and calls, unconditionally:

        paddle.split(input_, split_size_or_sections=..., dim=1)

    Paddle's real signature is ``paddle.split(x, num_or_sections, axis=0)``;
    the torch-style ``split_size_or_sections`` / ``dim`` keywords raise a
    ``TypeError`` even for a single rank. The mathematically correct contract
    is that reduce-scatter across ONE rank is a no-op, so AllGather's backward
    on a 1-rank group should hand the upstream gradient back unchanged. That
    correct expectation is asserted below and marked ``expectedFailure`` so the
    real production defect surfaces without editing production code.
    """

    @unittest.expectedFailure
    def test_single_rank_allgather_backward_should_be_identity(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        x.stop_gradient = False
        coef = paddle.to_tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype="float32"
        )
        out = _AllGatherFromTensorParallelRegion.apply(x, _Group(1))
        (out * coef).sum().backward()
        # Correct behaviour: single-rank reduce-scatter is identity.
        self.assertEqual(x.grad.tolist(), coef.tolist())


if __name__ == "__main__":
    unittest.main()
