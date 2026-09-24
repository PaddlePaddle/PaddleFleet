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

"""CPU behavior tests for ``paddlefleet.tensor_parallel.mappings`` (part 4).

Repository module: "分布式训练" (tensor parallel). This file deliberately
targets branches that the base ``test_mappings.py`` does NOT exercise, while
staying inside what a single no-card process can honestly prove:

  * the ``symbolic`` static methods of the autograd ``Function`` classes
    (their identity / split-routing decisions);
  * the ``backward`` routing decisions -- the ``ctx.group is None``
    pass-through guards, plus the ``tensor_parallel_output_grad=False`` branch
    of ``_GatherFromSequenceParallelRegion.backward`` which routes into the
    local first-dim split (distinguishable from identity at world_size 2) and
    its ``output_split_sizes is None`` assertion guard; and
  * the ``forward`` bypass paths of the classes base never reaches
    (``_AllGatherFromTensorParallelRegion``, ``_ReduceScatterToTensorParallelRegion``,
    ``_AllToAll``), asserting captured ``ctx`` state and hand-derived content.

The genuine collective paths (all_gather / reduce_scatter with world_size > 1)
are intentionally NOT exercised: they need a real multi-rank process group and
belong in ``tests/multi_card_tests``. Faking world_size and mocking the
collective to assert only "was called" would prove nothing about peer
selection, split sizes, or cross-rank reduction (see antipattern 13), so it is
avoided. Where a branch reduces to a single-rank local path, that is all the
assertion claims. Paddle is not installed in this no-card env, hence the honest
skip rather than a fake pass.
"""

import types
import unittest

try:
    import paddle

    from paddlefleet.tensor_parallel.mappings import (
        _AllGatherFromTensorParallelRegion,
        _AllToAll,
        _CopyToModelParallelRegion,
        _GatherFromModelParallelRegion,
        _GatherFromSequenceParallelRegion,
        _ReduceFromModelParallelRegion,
        _ReduceScatterToSequenceParallelRegion,
        _ReduceScatterToTensorParallelRegion,
        _ScatterToModelParallelRegion,
        _ScatterToSequenceParallelRegion,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"


class _Group:
    """Minimal real topology value object (never a mocked collective).

    The branches under test only read scalar topology fields
    (``world_size`` / ``nranks`` / ``rank`` / ``ranks``) to choose a local
    code path; no collective is ever invoked on this object.
    """

    def __init__(self, world_size, rank=0):
        self.world_size = world_size
        self.nranks = world_size
        self.rank = rank
        self.ranks = list(range(world_size))


def _ctx(**attrs):
    """A plain autograd ctx stand-in holding exactly the read/written attrs."""
    return types.SimpleNamespace(**attrs)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestSymbolicDispatch(unittest.TestCase):
    """The ``symbolic`` static methods: identity vs local-split routing.

    ``symbolic`` is a separate entry from ``forward`` (used for graph tracing);
    base only covered ``forward``. No collective is reachable here because we
    keep the group at nranks<=1 for the reduce/gather cases and only route the
    scatter cases into pure local slicing.
    """

    def test_copy_symbolic_is_identity(self):
        # _CopyToModelParallelRegion.symbolic returns input_ unconditionally.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _CopyToModelParallelRegion.symbolic(None, x, _Group(2, rank=1))
        self.assertIs(out, x)
        self.assertEqual(
            out.tolist(), [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
        )

    def test_reduce_from_symbolic_none_group_is_identity(self):
        # `if group is None or group.nranks <= 1: return input_`
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ReduceFromModelParallelRegion.symbolic(None, x, None)
        self.assertIs(out, x)

    def test_reduce_from_symbolic_single_rank_is_identity(self):
        # nranks == 1 short-circuits before any _reduce/all_reduce call.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ReduceFromModelParallelRegion.symbolic(None, x, _Group(1))
        self.assertIs(out, x)

    def test_scatter_to_mp_symbolic_routes_into_last_dim_split(self):
        # symbolic -> _split_along_last_dim; rank 1 of ws=2 keeps cols {2,3}.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ScatterToModelParallelRegion.symbolic(None, x, _Group(2, rank=1))
        self.assertEqual(out.tolist(), [[2.0, 3.0], [6.0, 7.0]])

    def test_scatter_to_sp_symbolic_routes_into_first_dim_split(self):
        # symbolic -> _split_along_first_dim; rank 0 of ws=2 keeps rows {0,1}.
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = _ScatterToSequenceParallelRegion.symbolic(
            None, x, _Group(2, rank=0)
        )
        self.assertEqual(out.tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

    def test_gather_from_mp_symbolic_world_size_one_passthrough(self):
        # symbolic -> _gather_along_last_dim; ws==1 returns the input as-is.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _GatherFromModelParallelRegion.symbolic(None, x, _Group(1))
        self.assertIs(out, x)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestBackwardNoneGroupPassthrough(unittest.TestCase):
    """``backward`` ``ctx.group is None`` guards: straight grad pass-through.

    When the group is None the backward must NOT call any gather / reduce /
    reduce-scatter -- it returns the upstream gradient unchanged. Base covered
    only forward dispatch, so these backward routing guards are new. A
    distinguishable arange gradient is used and full content is compared.
    """

    def _grad(self):
        return paddle.arange(8, dtype="float32").reshape([2, 4]) + 0.5

    def test_scatter_to_mp_backward_none_group(self):
        g = self._grad()
        out = _ScatterToModelParallelRegion.backward(_ctx(group=None), g)
        self.assertIs(out, g)

    def test_gather_from_mp_backward_none_group(self):
        g = self._grad()
        out = _GatherFromModelParallelRegion.backward(_ctx(group=None), g)
        self.assertIs(out, g)

    def test_scatter_to_sp_backward_none_group(self):
        g = self._grad()
        out = _ScatterToSequenceParallelRegion.backward(_ctx(group=None), g)
        self.assertIs(out, g)

    def test_reduce_scatter_to_sp_backward_none_group(self):
        g = self._grad()
        out = _ReduceScatterToSequenceParallelRegion.backward(
            _ctx(group=None, input_split_sizes=None, use_global_buffer=False),
            g,
        )
        self.assertIs(out, g)

    def test_all_gather_from_tp_backward_none_group(self):
        g = self._grad()
        out = _AllGatherFromTensorParallelRegion.backward(_ctx(group=None), g)
        self.assertIs(out, g)

    def test_reduce_scatter_to_tp_backward_none_group(self):
        g = self._grad()
        out = _ReduceScatterToTensorParallelRegion.backward(_ctx(group=None), g)
        self.assertIs(out, g)

    def test_gather_from_mp_backward_routes_into_last_dim_split(self):
        # Non-None group: backward routes into _split_along_last_dim. rank 0
        # of ws=2 keeps last-dim cols {0,1} -- distinguishable from identity.
        g = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _GatherFromModelParallelRegion.backward(
            _ctx(group=_Group(2, rank=0)), g
        )
        self.assertEqual(out.tolist(), [[0.0, 1.0], [4.0, 5.0]])


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGatherFromSequenceBackwardRouting(unittest.TestCase):
    """``_GatherFromSequenceParallelRegion.backward`` output-grad routing.

    The backward chooses between reduce-scatter (tensor_parallel_output_grad)
    and a plain first-dim split. The split branch and its guard are fully
    CPU-observable; the reduce-scatter branch needs a real process group and is
    left to the multi-card suite.
    """

    def test_none_group_passthrough(self):
        g = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = _GatherFromSequenceParallelRegion.backward(
            _ctx(
                group=None,
                tensor_parallel_output_grad=True,
                output_split_sizes=None,
                use_global_buffer=False,
            ),
            g,
        )
        self.assertIs(out, g)

    def test_non_tp_output_grad_routes_into_first_dim_split(self):
        # tensor_parallel_output_grad=False -> _split_along_first_dim. rank 1
        # of ws=2 keeps rows {2,3}; a wrong branch (reduce-scatter/identity)
        # would not yield this exact slice.
        g = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = _GatherFromSequenceParallelRegion.backward(
            _ctx(
                group=_Group(2, rank=1),
                tensor_parallel_output_grad=False,
                output_split_sizes=None,
                use_global_buffer=False,
            ),
            g,
        )
        self.assertEqual(out.tolist(), [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])

    def test_non_tp_output_grad_with_split_sizes_asserts(self):
        # The False branch requires output_split_sizes is None; a non-None
        # value must trip the guard rather than silently splitting.
        g = paddle.arange(12, dtype="float32").reshape([4, 3])
        with self.assertRaises(AssertionError):
            _GatherFromSequenceParallelRegion.backward(
                _ctx(
                    group=_Group(2, rank=0),
                    tensor_parallel_output_grad=False,
                    output_split_sizes=[2, 2],
                    use_global_buffer=False,
                ),
                g,
            )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestForwardBypassUncoveredClasses(unittest.TestCase):
    """forward bypass paths of classes base's dispatch test never touched."""

    def test_all_gather_from_tp_forward_none_group_passthrough(self):
        ctx = _ctx()
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllGatherFromTensorParallelRegion.forward(ctx, x, None)
        self.assertIs(out, x)
        self.assertIsNone(ctx.group)  # forward records the group on ctx

    def test_all_gather_from_tp_forward_world_size_one_passthrough(self):
        ctx = _ctx()
        grp = _Group(1)
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllGatherFromTensorParallelRegion.forward(ctx, x, grp)
        self.assertIs(out, x)  # ws==1 gather is a genuine local no-op
        self.assertIs(ctx.group, grp)

    def test_reduce_scatter_to_tp_forward_none_group_passthrough(self):
        ctx = _ctx()
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ReduceScatterToTensorParallelRegion.forward(ctx, x, None)
        self.assertIs(out, x)
        self.assertIsNone(ctx.group)

    def test_all_to_all_forward_world_size_one_captures_ctx_and_bypasses(self):
        # ws==1: returns input unchanged AND stores the split sizes on ctx so
        # the backward can invert them. Observe both the bypass and capture.
        ctx = _ctx()
        grp = _Group(1)
        x = paddle.arange(6, dtype="float32").reshape([2, 3])
        out = _AllToAll.forward(ctx, grp, x, [1, 1], [2, 2])
        self.assertIs(out, x)
        self.assertIs(ctx.group, grp)
        self.assertEqual(ctx.output_split_sizes, [1, 1])
        self.assertEqual(ctx.input_split_sizes, [2, 2])


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestReduceScatterToTPForwardWorldSizeOneBug(unittest.TestCase):
    """A real production bug reached through the public autograd forward.

    ``_ReduceScatterToTensorParallelRegion.forward`` with a world_size==1 group
    delegates to ``_reduce_scatter_along_last_dim``, which -- unlike every
    sibling primitive -- has no single-rank fast path and immediately calls
    ``paddle.split(input_, split_size_or_sections=..., dim=1)`` /
    ``paddle.concat(..., dim=0)``. Paddle's real signatures are
    ``paddle.split(x, num_or_sections, axis=0)`` and ``paddle.concat(x,
    axis=0)``; the torch-style ``split_size_or_sections`` / ``dim`` kwargs
    raise ``TypeError`` even for a single rank. The correct contract is that a
    reduce-scatter over one rank is an identity, so we assert that and mark the
    case ``expectedFailure`` -- surfacing the bug without editing production.
    """

    @unittest.expectedFailure
    def test_world_size_one_should_be_identity(self):
        ctx = _ctx()
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ReduceScatterToTensorParallelRegion.forward(ctx, x, _Group(1))
        self.assertEqual(out.tolist(), x.tolist())


if __name__ == "__main__":
    unittest.main()
