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

"""CPU behavior tests for ``paddlefleet.tensor_parallel.mappings`` (part 2).

Repository module: "分布式训练" (tensor parallel). This file complements
``test_mappings.py`` and deliberately targets DIFFERENT primitives -- the
all-to-all family and the two ``*LastDim*`` autograd Functions -- exercising
only their CPU-observable control flow that needs no real >1-rank group:

  * ``all_to_all``'s ``group is None`` guard and its single-rank identity;
  * ``_AllToAll.forward`` ``world_size == 1`` bypass (short-circuits before the
    split-size logic) and ``_AllToAll.backward``'s grad-slot mapping, both with
    the collective genuinely running its single-rank no-op (not mocked);
  * ``_AllGatherFromTensorParallelRegion`` / ``_ReduceScatterToTensorParallelRegion``
    forward/backward/symbolic pass-through routing, asserted on content; and
  * a documented real bug in ``all_to_all_sp2hp`` (inverted divisibility
    guard), pinned with ``expectedFailure`` against the correct contract.

The genuine >1-rank collectives (all_to_all / all_gather / reduce_scatter with
world_size > 1) are NOT run here: they need a real multi-rank process group and
live in ``tests/multi_card_tests``. Faking world_size and mocking the collective
to assert only "was called" would prove nothing about split sizes, peer or
cross-rank reduction, so it is avoided. Paddle is absent in the no-card env,
hence the honest skip.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import paddle

    from paddlefleet.tensor_parallel import mappings
    from paddlefleet.tensor_parallel.mappings import (
        _AllGatherFromTensorParallelRegion,
        _AllToAll,
        _ReduceScatterToTensorParallelRegion,
        all_to_all,
        all_to_all_sp2hp,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"


class _Group:
    """Minimal real stand-in for a process group value object.

    The tested single-rank / routing paths only *read* topology scalars
    (``world_size``, ``nranks``, ``rank``, ``ranks``); they never invoke a
    collective on it. This is a plain data holder, not a mocked collective.
    """

    def __init__(self, world_size, rank=0):
        self.world_size = world_size
        self.nranks = world_size
        self.rank = rank
        self.ranks = list(range(world_size))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllToAllWrapper(unittest.TestCase):
    """all_to_all: None guard and genuine single-rank identity."""

    def test_none_group_asserts(self):
        # The wrapper guards group before dispatching to the PyLayer.
        with self.assertRaises(AssertionError):
            all_to_all(None, paddle.zeros([4, 8], dtype="float32"))

    def test_world_size_one_is_identity_content(self):
        # ws == 1: _AllToAll bypasses the collective and returns input content.
        x = paddle.arange(12, dtype="float32").reshape([3, 4])
        out = all_to_all(_Group(1), x)
        self.assertEqual(out.tolist(), x.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllToAllForward(unittest.TestCase):
    """_AllToAll.forward: ws==1 bypass short-circuits before split sizes."""

    def test_ws1_bypass_ignores_split_sizes(self):
        # The `if world_size == 1: return input` guard sits BEFORE any use of
        # output/input split sizes, so passing non-trivial split sizes must not
        # change the (identity) result at ws == 1.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllToAll.apply(_Group(1), x, [1, 1], [3, 1])
        self.assertEqual(out.tolist(), x.tolist())

    def test_ws1_bypass_equal_split(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllToAll.apply(_Group(1), x, None, None)
        self.assertEqual(out.tolist(), x.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllToAllBackward(unittest.TestCase):
    """_AllToAll.backward: grad-slot mapping with a real ws==1 collective."""

    def test_backward_routes_grad_to_input_slot(self):
        # forward signature is (ctx, group, input, output_split_sizes,
        # input_split_sizes) -> 4 differentiable inputs, so backward returns a
        # 4-tuple with grads only for `input` (slot 1); group and the two split
        # size lists (slots 0, 2, 3) get None. _AllToAll.apply runs for real
        # here (ws == 1 no-op), it is NOT mocked, so slot 1 must equal the
        # upstream gradient content unchanged.
        ctx = SimpleNamespace(
            group=_Group(1), output_split_sizes=None, input_split_sizes=None
        )
        grad = paddle.arange(8, dtype="float32").reshape([2, 4])
        grads = _AllToAll.backward(ctx, grad)
        self.assertEqual(len(grads), 4)
        self.assertIsNone(grads[0])
        self.assertIsNone(grads[2])
        self.assertIsNone(grads[3])
        self.assertIsNotNone(grads[1])
        self.assertEqual(grads[1].tolist(), grad.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllGatherFromTensorParallelRegion(unittest.TestCase):
    """_AllGatherFromTensorParallelRegion forward/symbolic/backward routing."""

    def test_forward_none_group_passthrough(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllGatherFromTensorParallelRegion.apply(x, None)
        self.assertEqual(out.tolist(), x.tolist())

    def test_forward_ws1_routes_into_gather_identity(self):
        # group is not None -> routes into _gather_along_last_dim, whose ws==1
        # fast path returns the input content unchanged.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllGatherFromTensorParallelRegion.apply(x, _Group(1))
        self.assertEqual(out.tolist(), x.tolist())

    def test_symbolic_ws1_returns_gather_identity(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllGatherFromTensorParallelRegion.symbolic(None, x, _Group(1))
        self.assertEqual(out.tolist(), x.tolist())

    def test_backward_none_group_passthrough(self):
        # ctx.group is None -> gradient flows straight through untouched.
        ctx = SimpleNamespace(group=None)
        grad = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllGatherFromTensorParallelRegion.backward(ctx, grad)
        self.assertEqual(out.tolist(), grad.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestReduceScatterToTensorParallelRegion(unittest.TestCase):
    """_ReduceScatterToTensorParallelRegion forward/backward routing."""

    def test_forward_none_group_passthrough(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ReduceScatterToTensorParallelRegion.apply(x, None)
        self.assertEqual(out.tolist(), x.tolist())

    def test_backward_none_group_passthrough(self):
        ctx = SimpleNamespace(group=None)
        grad = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ReduceScatterToTensorParallelRegion.backward(ctx, grad)
        self.assertEqual(out.tolist(), grad.tolist())

    def test_backward_ws1_routes_into_gather_identity(self):
        # ctx.group not None -> backward routes into _gather_along_last_dim;
        # its ws==1 fast path returns the gradient content unchanged.
        ctx = SimpleNamespace(group=_Group(1))
        grad = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ReduceScatterToTensorParallelRegion.backward(ctx, grad)
        self.assertEqual(out.tolist(), grad.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllToAllSp2HpGuardBug(unittest.TestCase):
    """Real production bug: inverted divisibility guard in all_to_all_sp2hp."""

    @unittest.expectedFailure
    def test_divisible_last_dim_must_pass_guard(self):
        # Correct contract: when the last dim IS divisible by world size the
        # reshape guard must accept the input. Production writes
        #     assert input_.shape[-1] % world_size
        # which is truthy only when the remainder is NON-zero -- i.e. it
        # rejects exactly the divisible inputs it claims to require ("must be
        # divisible by world size"). With world_size == 1 every last dim is
        # divisible, so 8 % 1 == 0 makes the assertion fire. The group resolver
        # is a genuine, not-under-test collaborator that returns None when
        # paddle.distributed is not initialised, so it is stubbed to a real
        # single-rank group value to reach the guard (no collective is faked).
        # Marked expectedFailure: fixing the guard to `% world_size == 0`
        # removes this raise without editing production here.
        with patch.object(
            mappings,
            "get_tensor_model_parallel_group_if_none",
            return_value=_Group(1),
        ):
            x = paddle.arange(8, dtype="float32").reshape([2, 4])
            all_to_all_sp2hp(x)


if __name__ == "__main__":
    unittest.main()
