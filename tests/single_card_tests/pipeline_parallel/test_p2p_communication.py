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

"""CPU-only behavior tests for the device-independent pure logic in
``paddlefleet.pipeline_parallel.pp_utils.p2p_communication``.

Covered here are only the non-communicating helpers whose correctness is a
matter of local bookkeeping / branch selection:

* ``_is_valid_send_recv_partial`` - partial send/recv eligibility rule.
* ``SendRecvMeta.set_send_message`` / ``check_send_message`` - shape/dtype
  metadata capture, stop_gradient filtering, and the send-shape guard.
* ``P2PonCalcStream`` - op-record validation and field storage.
* ``_batch_p2p_tuple_or_tensor`` - wrapping a tensor / tuple into ordered
  ``P2PonCalcStream`` records with the right peer, group and mp fields.
* ``allgather_partial`` - the ``nranks == 1`` no-op / world-size-1 local path.
* ``_batched_p2p_ops`` - the peer, direction and *ordering* of the ops handed
  to ``batch_send_recv_on_calc_stream``. The collective itself is intercepted
  (it is a genuine collaborator, not the unit under test) so we can assert the
  exact ops the routing logic computed. Real cross-rank send/recv is NOT
  verified by this file; that requires a real pipeline process group.

Expected dtype numbers come from the on-wire protocol (float16=0, float32=1,
float64=2, int32=3, int64=4, bfloat16=5, bool=6) and are hardcoded rather than
produced by the production ``paddle_2_number`` helper, so the test and the code
under test do not share a source of truth.
"""

import unittest
from unittest import mock

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils import (
        p2p_communication as p2p,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    p2p = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


class _FakeHCG:
    """Deterministic stand-in for the hybrid-communicate group.

    It only reports peer ranks and (single-card) model-parallel sizes so that
    ``_batched_p2p_ops`` can compute its op list without a real process group.
    ``mp_degree == 1`` keeps ``allgather_partial`` a genuine no-op, so no
    collective runs during these tests.
    """

    def __init__(self, prev_rank=1, next_rank=3, mp_degree=1, mp_rank=0):
        self._prev = prev_rank
        self._next = next_rank
        self._mp_degree = mp_degree
        self._mp_rank = mp_rank
        self.pipe_group = object()
        self.mp_group = object()

    def get_pipe_parallel_group(self):
        return self.pipe_group

    def get_model_parallel_world_size(self):
        return self._mp_degree

    def get_model_parallel_rank(self):
        return self._mp_rank

    def get_model_parallel_group(self):
        return self.mp_group

    def _get_p2p_prev_rank(self):
        return self._prev

    def _get_p2p_next_rank(self):
        return self._next


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestIsValidSendRecvPartial(unittest.TestCase):
    """`_is_valid_send_recv_partial(tensor, mp_degree)` gating rule."""

    def setUp(self):
        paddle.set_device("cpu")

    def _tensor(self, shape):
        return paddle.zeros(shape, dtype="float32")

    def test_divisible_and_degree_gt_one_is_true(self):
        # numel = 8; 8 % 4 == 0 and 4 > 1 -> partial send/recv is valid.
        self.assertTrue(
            p2p._is_valid_send_recv_partial(self._tensor([2, 4]), 4)
        )

    def test_not_divisible_is_false(self):
        # numel = 8; 8 % 3 == 2 -> cannot be split evenly across ranks.
        self.assertFalse(
            p2p._is_valid_send_recv_partial(self._tensor([2, 4]), 3)
        )

    def test_degree_one_is_false_even_when_divisible(self):
        # numel = 8; 8 % 1 == 0, so only the ``mp_degree > 1`` guard forces
        # False here. Dropping that guard would wrongly return True.
        self.assertFalse(
            p2p._is_valid_send_recv_partial(self._tensor([2, 4]), 1)
        )

    def test_zero_element_raises_assertion(self):
        # numel == 0 must trip the explicit "can't send/recv zero element".
        with self.assertRaises(AssertionError):
            p2p._is_valid_send_recv_partial(self._tensor([0]), 2)

    def test_disabled_flag_forces_false_and_is_restored(self):
        t = self._tensor([2, 4])  # enabled path would be True (8 % 4 == 0)
        with mock.patch.object(p2p, "_enable_partial_send_recv", False):
            self.assertFalse(p2p._is_valid_send_recv_partial(t, 4))
        # patch.object restored the global; the gate really governed the result.
        self.assertTrue(p2p._is_valid_send_recv_partial(t, 4))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSendRecvMetaSendMessage(unittest.TestCase):
    """`SendRecvMeta` send-side metadata capture, filtering and checking."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_initial_state_is_empty(self):
        meta = p2p.SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.send_key_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_single_tensor_records_shape_dtype_and_no_key(self):
        meta = p2p.SendRecvMeta()
        meta.set_send_message(paddle.zeros([2, 3], dtype="float32"))
        self.assertEqual(list(meta.send_shape_message), [2, 3])
        self.assertEqual(meta.send_dtype_message, 1)  # float32 -> 1 (wire)
        self.assertIsNone(meta.send_key_message)

    def test_tuple_filters_stop_gradient_and_keeps_order(self):
        meta = p2p.SendRecvMeta()
        # Distinguishable shapes so an inverted / dropped filter is caught:
        # only the two trainable tensors survive, in order; the middle
        # stop_gradient tensor [9, 9] must be filtered out.
        a = paddle.zeros([2, 3], dtype="float32")
        a.stop_gradient = False
        drop = paddle.zeros([9, 9], dtype="float32")
        drop.stop_gradient = True
        c = paddle.zeros([4, 5, 6], dtype="int64")
        c.stop_gradient = False
        meta.set_send_message((a, drop, c))
        self.assertEqual(
            [list(s) for s in meta.send_shape_message], [[2, 3], [4, 5, 6]]
        )
        self.assertEqual(meta.send_dtype_message, (1, 4))  # float32, int64

    def test_key_attribute_is_captured(self):
        meta = p2p.SendRecvMeta()
        t = paddle.zeros([2, 3], dtype="float32")
        t.key = "layer.0"
        meta.set_send_message(t)
        self.assertEqual(meta.send_key_message, "layer.0")

    def test_check_send_message_noop_before_any_send_message(self):
        # send_shape_message is None -> the guard returns early, no assertion.
        meta = p2p.SendRecvMeta()
        self.assertIsNone(
            meta.check_send_message(paddle.zeros([7, 7], dtype="float32"))
        )

    def test_check_send_message_matches_same_shape(self):
        meta = p2p.SendRecvMeta()
        meta.set_send_message(paddle.zeros([2, 3], dtype="float32"))
        # Same shape and dtype -> no assertion raised.
        meta.check_send_message(paddle.zeros([2, 3], dtype="float32"))

    def test_check_send_message_rejects_shape_mismatch(self):
        meta = p2p.SendRecvMeta()
        meta.set_send_message(paddle.zeros([2, 3], dtype="float32"))
        with self.assertRaises(AssertionError):
            meta.check_send_message(paddle.zeros([4, 5], dtype="float32"))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestP2PonCalcStream(unittest.TestCase):
    """`P2PonCalcStream` validates its op and stores the call fields."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_valid_send_op_stores_fields(self):
        t = paddle.zeros([2, 3], dtype="float32")
        group = object()
        rec = p2p.P2PonCalcStream(
            p2p._send_on_calc_stream,
            t,
            peer=5,
            group=group,
            nranks=2,
            rank_id=1,
        )
        self.assertIs(rec.op, p2p._send_on_calc_stream)
        self.assertIs(rec.tensor, t)
        self.assertEqual(rec.peer, 5)
        self.assertIs(rec.group, group)
        self.assertEqual(rec.nranks, 2)
        self.assertEqual(rec.rank_id, 1)

    def test_valid_recv_op_defaults(self):
        t = paddle.zeros([2, 3], dtype="float32")
        rec = p2p.P2PonCalcStream(
            p2p._recv_on_calc_stream, t, peer=0, group=object()
        )
        self.assertIs(rec.op, p2p._recv_on_calc_stream)
        self.assertEqual(rec.nranks, 1)
        self.assertEqual(rec.rank_id, 0)

    def test_invalid_op_raises_runtime_error(self):
        # Only the two calc-stream primitives are permitted as ``op``.
        with self.assertRaises(RuntimeError):
            p2p.P2PonCalcStream(
                lambda *a, **k: None, None, peer=0, group=object()
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestBatchP2PTupleOrTensor(unittest.TestCase):
    """`_batch_p2p_tuple_or_tensor` wraps tensor(s) into ordered op records."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_single_tensor_yields_one_op_with_all_fields(self):
        t = paddle.zeros([2, 3], dtype="float32")
        group = object()
        ops = p2p._batch_p2p_tuple_or_tensor(
            t, p2p._send_on_calc_stream, 7, group, mp_degree=2, mp_rank=1
        )
        self.assertEqual(len(ops), 1)
        self.assertIs(ops[0].op, p2p._send_on_calc_stream)
        self.assertIs(ops[0].tensor, t)
        self.assertEqual(ops[0].peer, 7)
        self.assertIs(ops[0].group, group)
        self.assertEqual(ops[0].nranks, 2)
        self.assertEqual(ops[0].rank_id, 1)

    def test_tuple_preserves_element_order_and_identity(self):
        a = paddle.zeros([2, 3], dtype="float32")
        b = paddle.zeros([4, 5], dtype="float32")
        group = object()
        ops = p2p._batch_p2p_tuple_or_tensor(
            (a, b), p2p._recv_on_calc_stream, 5, group
        )
        self.assertEqual(len(ops), 2)
        self.assertIs(ops[0].tensor, a)
        self.assertIs(ops[1].tensor, b)
        for rec in ops:
            self.assertIs(rec.op, p2p._recv_on_calc_stream)
            self.assertEqual(rec.peer, 5)
            self.assertIs(rec.group, group)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestAllgatherPartialSingleRank(unittest.TestCase):
    """`allgather_partial` returns the input unchanged when ``nranks == 1``.

    This is the world-size-1 / no-partial local path: no collective is issued,
    so no process group is required. It does not prove multi-rank all-gather.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_nranks_one_returns_same_tensor_object(self):
        t = paddle.zeros([2, 3], dtype="float32")
        self.assertIs(p2p.allgather_partial(t, nranks=1), t)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestBatchedP2POpsRouting(unittest.TestCase):
    """`_batched_p2p_ops` peer / direction / ordering of the emitted ops.

    ``batch_send_recv_on_calc_stream`` (the real calc-stream collective) is a
    genuine collaborator and is intercepted so the exact op list produced by
    the routing logic can be inspected. ``mp_degree == 1`` makes the trailing
    ``allgather_partial`` a no-op, so no collective actually runs. Cross-rank
    transfer is therefore NOT verified here - only local op construction.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def _run(self, hcg, sp, rp, sn, rn):
        captured = []

        def _capture(ops):
            captured.append(ops)

        with mock.patch.object(
            p2p, "batch_send_recv_on_calc_stream", side_effect=_capture
        ):
            p2p._batched_p2p_ops(sp, rp, sn, rn, hcg)
        return captured

    def _summary(self, ops):
        # (direction, tensor identity, peer) triples in emission order.
        return [(rec.op, id(rec.tensor), rec.peer) for rec in ops]

    def test_all_none_emits_no_collective(self):
        hcg = _FakeHCG()
        captured = self._run(hcg, None, None, None, None)
        # No ops -> the ``len(ops) > 0`` guard skips the collective entirely.
        self.assertEqual(captured, [])

    def test_async_order_peer_and_direction(self):
        # Default (_sync_send False) async order:
        # send_prev, recv_prev, send_next, recv_next.
        hcg = _FakeHCG(prev_rank=1, next_rank=3)
        sp = paddle.zeros([2, 2], dtype="float32")
        rp = paddle.zeros([2, 3], dtype="float32")
        sn = paddle.zeros([2, 4], dtype="float32")
        rn = paddle.zeros([2, 5], dtype="float32")
        captured = self._run(hcg, sp, rp, sn, rn)
        self.assertEqual(len(captured), 1)
        self.assertEqual(
            self._summary(captured[0]),
            [
                (p2p._send_on_calc_stream, id(sp), 1),
                (p2p._recv_on_calc_stream, id(rp), 1),
                (p2p._send_on_calc_stream, id(sn), 3),
                (p2p._recv_on_calc_stream, id(rn), 3),
            ],
        )

    def test_sync_send_reorders_to_recvprev_sendnext_recvnext_sendprev(self):
        # PADDLE_P2P_SYNC_SEND path: recv_prev, send_next, recv_next, send_prev.
        hcg = _FakeHCG(prev_rank=1, next_rank=3)
        sp = paddle.zeros([2, 2], dtype="float32")
        rp = paddle.zeros([2, 3], dtype="float32")
        sn = paddle.zeros([2, 4], dtype="float32")
        rn = paddle.zeros([2, 5], dtype="float32")
        with mock.patch.object(p2p, "_sync_send", True):
            captured = self._run(hcg, sp, rp, sn, rn)
        self.assertEqual(len(captured), 1)
        self.assertEqual(
            self._summary(captured[0]),
            [
                (p2p._recv_on_calc_stream, id(rp), 1),
                (p2p._send_on_calc_stream, id(sn), 3),
                (p2p._recv_on_calc_stream, id(rn), 3),
                (p2p._send_on_calc_stream, id(sp), 1),
            ],
        )

    def test_send_prev_only_targets_prev_peer(self):
        hcg = _FakeHCG(prev_rank=1, next_rank=3)
        sp = paddle.zeros([2, 2], dtype="float32")
        captured = self._run(hcg, sp, None, None, None)
        self.assertEqual(len(captured), 1)
        self.assertEqual(
            self._summary(captured[0]),
            [(p2p._send_on_calc_stream, id(sp), 1)],
        )

    def test_recv_next_only_targets_next_peer(self):
        hcg = _FakeHCG(prev_rank=1, next_rank=3)
        rn = paddle.zeros([2, 5], dtype="float32")
        captured = self._run(hcg, None, None, None, rn)
        self.assertEqual(len(captured), 1)
        self.assertEqual(
            self._summary(captured[0]),
            [(p2p._recv_on_calc_stream, id(rn), 3)],
        )


if __name__ == "__main__":
    unittest.main()
