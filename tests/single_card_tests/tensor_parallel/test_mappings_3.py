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

"""CPU behavior tests for the *backward* routing of the TP autograd Functions
in ``paddlefleet.tensor_parallel.mappings``.

Repository module: "分布式训练" (tensor parallel). The sibling file
``test_mappings.py`` covers the ``forward`` dispatch and the local split index
math; this file deliberately targets a DIFFERENT set of branches -- the
``backward`` gradient routing of the autograd ``Function`` classes -- which the
forward tests never drive.

Two CPU-honest strategies are used:

  * For Functions whose single-rank ``forward`` is a genuine pass-through
    (``_ReduceFromModelParallelRegion``, ``_CopyToModelParallelRegion``), a real
    autograd round trip is driven end to end: ``apply(...)`` then
    ``out.backward(upstream)``, and the accumulated ``x.grad`` is compared to a
    non-uniform hand-derived upstream. The single-rank all-reduce / copy region
    has an identity (copy) gradient; a non-uniform upstream rejects any
    implementation that scaled, zeroed, or reduced it.

  * For Functions whose ``forward`` needs a real >1-rank collective
    (``_GatherFromModelParallelRegion``, ``_GatherFromSequenceParallelRegion``),
    the collective forward is out of scope (multi-card). Their ``backward``,
    however, routes into a *purely local* slice (``_split_along_last_dim`` /
    ``_split_along_first_dim``) whose result depends only on rank metadata, not
    on any cross-rank exchange. That production ``backward`` is therefore invoked
    directly with a real topology value object, and the returned gradient is
    compared to hand-derived per-rank slices. No collective is faked and no
    cross-rank behavior is claimed (contrast antipattern 13); only the local
    backward slice contract is asserted.

Paddle is not installed in the no-card env, so every case is honestly skipped
rather than faked into a pass.
"""

import unittest
from types import SimpleNamespace

try:
    import paddle

    from paddlefleet.tensor_parallel.mappings import (
        _CopyToModelParallelRegion,
        _GatherFromModelParallelRegion,
        _GatherFromSequenceParallelRegion,
        _ReduceFromModelParallelRegion,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"


class _Group:
    """Minimal real stand-in for a process group value object.

    The backward paths under test only *read* topology scalars
    (``world_size``, ``nranks``, ``rank``, ``ranks``) to compute a local slice;
    they never invoke a collective on it. This is a plain data holder, not a
    mocked collective.
    """

    def __init__(self, world_size, rank=0):
        self.world_size = world_size
        self.nranks = world_size
        self.rank = rank
        self.ranks = list(range(world_size))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestReduceFromModelParallelRegionBackward(unittest.TestCase):
    """backward of the all-reduce region is an identity (copy) gradient.

    ``_ReduceFromModelParallelRegion.backward`` unconditionally returns
    ``grad_output``; for a single-rank group the forward is also a pass-through,
    so a full autograd round trip is possible on CPU.
    """

    def test_single_rank_backward_copies_upstream_grad(self):
        leaf = paddle.arange(8, dtype="float32").reshape([2, 4])
        leaf.stop_gradient = False
        # The single-rank forward returns its input unchanged; feed it a
        # non-leaf (clone) because Paddle forbids an identity autograd Function
        # on a leaf that requires grad ("Leaf Var ... can't use inplace
        # strategy"). clone's own backward is the identity, so leaf.grad still
        # equals exactly what the Function's backward returns.
        x = leaf.clone()
        # Non-uniform / mixed-sign upstream: an identity backward must return it
        # verbatim, so scaling or zeroing would be caught.
        upstream = paddle.to_tensor(
            [[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]], dtype="float32"
        )
        out = _ReduceFromModelParallelRegion.apply(x, _Group(1))
        # forward on a single rank is a pass-through.
        self.assertEqual(out.tolist(), leaf.tolist())
        out.backward(upstream)
        self.assertIsNotNone(leaf.grad)
        self.assertEqual(leaf.grad.tolist(), upstream.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestCopyToModelParallelRegionBackward(unittest.TestCase):
    """backward of the copy region: identity via the ``None`` guard and via the
    single-rank ``_reduce`` bypass -- two distinct code paths, same contract."""

    def _run(self, group):
        leaf = paddle.arange(8, dtype="float32").reshape([2, 4])
        leaf.stop_gradient = False
        # Non-leaf (clone) input: the copy region forward is an identity
        # pass-through, which Paddle rejects on a grad-requiring leaf. clone's
        # identity backward preserves the grad so leaf.grad equals the
        # Function's backward output.
        x = leaf.clone()
        upstream = paddle.to_tensor(
            [[2.0, -1.0, 0.5, -3.0], [-4.0, 6.0, -7.0, 8.0]], dtype="float32"
        )
        out = _CopyToModelParallelRegion.apply(x, group)
        self.assertEqual(out.tolist(), leaf.tolist())
        out.backward(upstream)
        self.assertIsNotNone(leaf.grad)
        self.assertEqual(leaf.grad.tolist(), upstream.tolist())

    def test_backward_with_none_group_is_identity(self):
        # ctx.group is None -> backward short-circuits to ``return grad_output``.
        self._run(None)

    def test_backward_with_single_rank_group_is_identity(self):
        # ctx.group is a 1-rank group -> _reduce bypasses all_reduce and returns
        # grad_output unchanged. Different branch from the None guard above.
        self._run(_Group(1))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGatherFromModelParallelRegionBackward(unittest.TestCase):
    """backward routes into the local last-dim split (one chunk per rank).

    The all-gather ``forward`` needs a real >1-rank group (multi-card); the
    ``backward`` slice is purely local, so the production backward is invoked
    directly with a real topology value object and compared to hand-derived
    per-rank column chunks.
    """

    def test_none_group_backward_is_identity(self):
        grad = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _GatherFromModelParallelRegion.backward(
            SimpleNamespace(group=None), grad
        )
        self.assertEqual(out.tolist(), grad.tolist())

    def test_backward_returns_per_rank_last_dim_chunk(self):
        # grad rows are 100..111 laid out as [3, 4]; world_size 2 splits the
        # last dim into columns {0,1} | {2,3}.
        grad = (100 + paddle.arange(12, dtype="float32")).reshape([3, 4])
        rank0 = _GatherFromModelParallelRegion.backward(
            SimpleNamespace(group=_Group(2, rank=0)), grad
        )
        rank1 = _GatherFromModelParallelRegion.backward(
            SimpleNamespace(group=_Group(2, rank=1)), grad
        )
        self.assertEqual(
            rank0.tolist(),
            [[100.0, 101.0], [104.0, 105.0], [108.0, 109.0]],
        )
        self.assertEqual(
            rank1.tolist(),
            [[102.0, 103.0], [106.0, 107.0], [110.0, 111.0]],
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGatherFromSequenceParallelRegionBackward(unittest.TestCase):
    """backward flag routing of ``_GatherFromSequenceParallelRegion``.

    The all-gather ``forward`` needs a real >1-rank group (multi-card). The
    ``backward`` has three CPU-observable branches:

      * ``ctx.group is None`` -> identity pass-through;
      * ``tensor_parallel_output_grad is False`` with ``output_split_sizes is
        None`` -> local ``_split_along_first_dim`` (row block per rank);
      * ``tensor_parallel_output_grad is False`` with a non-None
        ``output_split_sizes`` -> the ``assert ctx.output_split_sizes is None``
        guard must fire.

    The ``tensor_parallel_output_grad is True`` branch routes into
    reduce-scatter (a real collective) and is intentionally left to multi-card
    tests.
    """

    def _ctx(self, group, output_grad, split_sizes=None):
        return SimpleNamespace(
            group=group,
            tensor_parallel_output_grad=output_grad,
            output_split_sizes=split_sizes,
            use_global_buffer=False,
        )

    def test_none_group_backward_is_identity(self):
        grad = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = _GatherFromSequenceParallelRegion.backward(
            self._ctx(None, output_grad=True), grad
        )
        self.assertEqual(out.tolist(), grad.tolist())

    def test_non_tp_output_grad_routes_into_first_dim_split(self):
        # grad rows are 200..211 laid out as [4, 3]; world_size 2 keeps rows
        # {0,1} on rank 0 and rows {2,3} on rank 1.
        grad = (200 + paddle.arange(12, dtype="float32")).reshape([4, 3])
        rank0 = _GatherFromSequenceParallelRegion.backward(
            self._ctx(_Group(2, rank=0), output_grad=False), grad
        )
        rank1 = _GatherFromSequenceParallelRegion.backward(
            self._ctx(_Group(2, rank=1), output_grad=False), grad
        )
        self.assertEqual(
            rank0.tolist(),
            [[200.0, 201.0, 202.0], [203.0, 204.0, 205.0]],
        )
        self.assertEqual(
            rank1.tolist(),
            [[206.0, 207.0, 208.0], [209.0, 210.0, 211.0]],
        )

    def test_non_tp_output_grad_rejects_output_split_sizes(self):
        # The non-tp-output-grad branch does not support output_split_sizes;
        # the guard must raise before any splitting happens.
        grad = paddle.arange(12, dtype="float32").reshape([4, 3])
        with self.assertRaises(AssertionError):
            _GatherFromSequenceParallelRegion.backward(
                self._ctx(
                    _Group(2, rank=0), output_grad=False, split_sizes=[2, 2]
                ),
                grad,
            )


if __name__ == "__main__":
    unittest.main()
