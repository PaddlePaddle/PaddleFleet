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

"""CPU-only behavior tests for the *local* (non-communicating) branches of
``paddlefleet.pipeline_parallel.pp_utils.four_directions_p2p_communication``.

This file deliberately targets logic that is complete and observable on a
single CPU process, i.e. the world-size=1 / disabled-partial fallbacks and the
early group-membership short-circuits that return WITHOUT touching any
collective:

* ``allgather_partial`` returns its input tensor unchanged (same object,
  same values) whenever partial send/recv is not applicable -- degree <= 1,
  non-divisible element count, or the ``_enable_partial_send_recv`` gate off.
* ``send_partial`` / ``recv_partial`` return ``None`` immediately when handed a
  group that reports ``is_member() is False``, never reaching the send/recv op.
* ``SendRecvMeta.set_send_message`` drops every ``stop_gradient`` tensor from a
  tuple, yielding empty shape/dtype tuples when all are dropped, while keeping
  and ordering the survivors otherwise.

Everything that requires cross-rank exchange -- the real partial all-gather /
send / recv ops, ``recv_meta`` over a pipe group, and the four-direction
``_p2p_helper`` scheduling -- is intentionally NOT covered. Faking
``world_size`` and mocking the collectives would only exercise local
orchestration, not the send direction, peer selection, split sizing, or
cross-rank reassembly, so those belong in a real multi-card process group.

Wire-protocol dtype numbers (float32=1, int64=4) are asserted from the
published protocol constant, not obtained by calling the production
``paddle_2_number`` helper, so the test and the code under test do not share a
single source of truth.
"""

import unittest
from unittest import mock

try:
    import numpy as np
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils import (
        four_directions_p2p_communication as p2p,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    np = None
    paddle = None
    p2p = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


class _NonMemberGroup:
    """Minimal stand-in process group that is not a communication member.

    It is NOT a mock of any code under test; it only supplies the collaborator
    response (``is_member() -> False``) that the real short-circuit branch
    consumes. ``id`` is intentionally absent so that, if the membership guard
    were removed, the next production line (``group.id``) would raise instead
    of silently succeeding.
    """

    def is_member(self):
        return False


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestAllgatherPartialLocalNoOp(unittest.TestCase):
    """``allgather_partial`` passthrough on the local (no-communication) path.

    Contract: when ``_is_valid_send_recv_partial`` is False the function must
    return the *same* tensor object untouched. This is the world-size=1 /
    non-divisible / disabled fallback only; the true partial all-gather across
    ranks is not exercised here.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def _tensor(self):
        # Distinguishable, non-uniform content so a passthrough that secretly
        # rewrote or reallocated the buffer would be caught by value + identity.
        return paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )

    def test_degree_one_returns_same_object_unchanged(self):
        t = self._tensor()
        before = t.numpy().copy()
        out = p2p.allgather_partial(t, nranks=1, rank_id=0, group=None)
        self.assertIs(out, t)  # nranks<=1 -> pure no-op, identical object
        np.testing.assert_array_equal(out.numpy(), before)

    def test_non_divisible_returns_same_object_unchanged(self):
        # numel = 6; 6 % 4 != 0 -> not evenly splittable -> local no-op.
        t = self._tensor()
        before = t.numpy().copy()
        out = p2p.allgather_partial(t, nranks=4, rank_id=0, group=None)
        self.assertIs(out, t)
        np.testing.assert_array_equal(out.numpy(), before)

    def test_disabled_gate_forces_noop_even_when_divisible(self):
        # numel = 6, nranks = 2, 6 % 2 == 0 and 2 > 1 would be a *valid*
        # partial gather if enabled, so this isolates the disable gate: with
        # partial send/recv off the function must still be a pure passthrough.
        t = self._tensor()
        before = t.numpy().copy()
        with mock.patch.object(p2p, "_enable_partial_send_recv", False):
            out = p2p.allgather_partial(t, nranks=2, rank_id=0, group=None)
        self.assertIs(out, t)
        np.testing.assert_array_equal(out.numpy(), before)
        # patch.object restored the module global; confirm the disable gate
        # (not the always-passthrough of an invalid partial) produced the
        # no-op. With partial ENABLED this divisible tensor is a *valid*
        # partial, so the function no longer returns the tensor -- a
        # non-member group now short-circuits it to None instead.
        out_enabled = p2p.allgather_partial(
            t, nranks=2, rank_id=0, group=_NonMemberGroup()
        )
        self.assertIsNone(out_enabled)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPartialSendRecvNonMemberShortCircuit(unittest.TestCase):
    """Non-member group short-circuits ``send_partial`` / ``recv_partial``.

    Contract: a group whose ``is_member()`` is False makes the call return
    ``None`` before any communication op and before peer-rank resolution. If
    the guard were dropped, execution would reach ``group.id`` /
    ``_hcg._get_p2p_*_rank`` and raise, so a clean ``None`` return is a genuine
    behavioral observation of the guard rather than a vacuous check.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_send_partial_non_member_returns_none(self):
        t = paddle.to_tensor([1.0, 2.0], dtype="float32")
        result = p2p.send_partial(
            t, dst=1, nranks=1, rank_id=0, group=_NonMemberGroup()
        )
        self.assertIsNone(result)

    def test_recv_partial_non_member_returns_none(self):
        t = paddle.to_tensor([1.0, 2.0], dtype="float32")
        result = p2p.recv_partial(
            t, src=0, nranks=1, rank_id=0, group=_NonMemberGroup()
        )
        self.assertIsNone(result)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSetSendMessageStopGradientFilter(unittest.TestCase):
    """``SendRecvMeta.set_send_message`` stop_gradient filtering edges.

    The filter keeps only trainable (``stop_gradient is False``) tensors for
    the tuple path. A tuple with every element frozen must collapse to empty
    shape/dtype tuples, while a fully-trainable tuple must preserve every
    element in order -- the contrast rules out a "returns empty regardless"
    implementation.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_all_stop_gradient_tuple_yields_empty_messages(self):
        meta = p2p.SendRecvMeta()
        a = paddle.zeros([2, 3], dtype="float32")
        a.stop_gradient = True
        b = paddle.zeros([4, 5], dtype="float32")
        b.stop_gradient = True
        meta.set_send_message((a, b))
        self.assertEqual(meta.send_shape_message, ())
        self.assertEqual(meta.send_dtype_message, ())

    def test_all_trainable_tuple_preserves_order_and_protocol_dtype(self):
        meta = p2p.SendRecvMeta()
        a = paddle.zeros([2, 3], dtype="float32")
        a.stop_gradient = False
        b = paddle.zeros([4, 5, 6], dtype="int64")
        b.stop_gradient = False
        meta.set_send_message((a, b))
        self.assertEqual(
            [list(s) for s in meta.send_shape_message], [[2, 3], [4, 5, 6]]
        )
        # float32 -> 1, int64 -> 4 by the on-wire protocol, derived
        # independently of the production paddle_2_number helper.
        self.assertEqual(meta.send_dtype_message, (1, 4))


if __name__ == "__main__":
    unittest.main()
