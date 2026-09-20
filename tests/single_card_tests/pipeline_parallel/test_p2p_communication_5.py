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

"""CPU-testable pure logic of pipeline-parallel p2p_communication.

Scope. Only the local, single-process logic that does not depend on a real
process group is exercised here: the partial send/recv validity predicate, the
SendRecvMeta send-side metadata derivation/check, the P2PonCalcStream op
wrapper, the batch-op builder, and the dtype<->number encoding tables that the
meta wire format relies on. Expected values are derived by hand from the source
tables (float16->0, float32->1, ..., bool->6), not read from any coverage file.

Out of scope on purpose. recv_meta/send_meta wire round-trips,
batch_send_recv_on_calc_stream, _p2p_ops and _batched_p2p_ops all perform real
collective / peer communication whose correctness (peer selection, split sizes,
direction, all-gather reassembly) is only observable across multiple ranks. They
are NOT faked here with a mock world_size + stubbed collectives, because that
would only prove call plumbing, never the cross-rank semantics. They belong in a
real multi-rank pipeline-parallel job under tests/multi_card_tests/.
"""

import unittest
from unittest import mock

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils import (
        p2p_communication as p2p_mod,
    )
    from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
        P2PonCalcStream,
        SendRecvMeta,
        _batch_p2p_tuple_or_tensor,
        _is_valid_send_recv_partial,
        _recv_on_calc_stream,
        _send_on_calc_stream,
    )
    from paddlefleet.pipeline_parallel.pp_utils.utils import (
        number_2_dtype,
        paddle_2_number,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddle/paddlefleet not importable on this CPU host (real dependency "
    f"missing, not a business-inapplicable path): {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestIsValidSendRecvPartial(unittest.TestCase):
    """_is_valid_send_recv_partial: (mp_degree > 1) AND (numel % mp_degree == 0),
    gated by the module toggle, rejecting zero-element tensors."""

    def test_divisible_and_multi_degree_is_true(self):
        # numel = 8, mp_degree = 4 -> 8 % 4 == 0 and 4 > 1 -> True (by hand)
        t = paddle.zeros([8], dtype="float32")
        self.assertTrue(_is_valid_send_recv_partial(t, 4))
        t2 = paddle.zeros([2, 3], dtype="float32")  # numel = 6
        self.assertTrue(_is_valid_send_recv_partial(t2, 3))  # 6 % 3 == 0

    def test_degree_one_is_false_even_if_divisible(self):
        # mp_degree == 1 fails the mp_degree > 1 guard regardless of divisibility.
        t = paddle.zeros([8], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(t, 1))

    def test_indivisible_is_false(self):
        # numel = 7, mp_degree = 4 -> 7 % 4 == 3 != 0 -> False (by hand)
        t = paddle.zeros([7], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(t, 4))
        t2 = paddle.zeros([2, 3], dtype="float32")  # numel = 6
        self.assertFalse(_is_valid_send_recv_partial(t2, 4))  # 6 % 4 == 2

    def test_disabled_toggle_forces_false(self):
        # When the module-level toggle is off, even a valid split returns False.
        # patch.object restores the global on exit (no cross-test pollution).
        t = paddle.zeros([8], dtype="float32")
        with mock.patch.object(p2p_mod, "_enable_partial_send_recv", False):
            self.assertFalse(_is_valid_send_recv_partial(t, 4))
        # toggle restored: the same input is valid again
        self.assertTrue(_is_valid_send_recv_partial(t, 4))

    def test_zero_element_rejected(self):
        t = paddle.zeros([0], dtype="float32")  # numel = 0
        with self.assertRaises(AssertionError):
            _is_valid_send_recv_partial(t, 4)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSendRecvMetaState(unittest.TestCase):
    """SendRecvMeta init/erase resets every field, and the send-side metadata
    derivation filters stop_gradient tensors while preserving order/dtype."""

    def test_fresh_meta_all_fields_cleared(self):
        meta = SendRecvMeta()
        for name in (
            "send_shape_message",
            "send_dtype_message",
            "send_key_message",
            "recv_shape_message",
            "recv_dtype_message",
            "recv_stop_gradient",
            "recv_key_message",
        ):
            self.assertIsNone(getattr(meta, name), name)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_init_or_erase_resets_dirtied_fields(self):
        meta = SendRecvMeta()
        meta.send_shape_message = [2, 3]
        meta.send_dtype_message = 1
        meta.recv_shape_message = [4, 5]
        meta.has_send_meta = True
        meta.has_recv_meta = True
        meta.init_or_erase_meta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_set_send_message_single_tensor(self):
        # Single tensor: (shape, paddle_2_number(dtype), key). float32 -> 1 by hand.
        meta = SendRecvMeta()
        t = paddle.zeros([2, 3], dtype="float32")
        meta.set_send_message(t)
        self.assertEqual(meta.send_shape_message, [2, 3])
        self.assertEqual(meta.send_dtype_message, 1)
        self.assertIsNone(meta.send_key_message)

    def test_set_send_message_tuple_filters_stop_gradient_keeps_order(self):
        # Tuple branch skips stop_gradient tensors and keeps the order of the
        # remaining ones. float32 -> 1, float16 -> 0 (derived by hand from the
        # PADDLE_TO_NUMBER table).
        t1 = paddle.zeros([2, 3], dtype="float32")
        t1.stop_gradient = False
        t2 = paddle.zeros([4, 5], dtype="int64")  # dropped: stop_gradient True
        t2.stop_gradient = True
        t3 = paddle.zeros([6], dtype="float16")
        t3.stop_gradient = False

        meta = SendRecvMeta()
        meta.set_send_message((t1, t2, t3))
        # t2 must be absent; t1 before t3; dtypes distinguishable (1 vs 0).
        self.assertEqual(meta.send_shape_message, ([2, 3], [6]))
        self.assertEqual(meta.send_dtype_message, (1, 0))
        self.assertEqual(meta.send_key_message, (None, None))

    def test_check_send_message_matches_and_rejects(self):
        meta = SendRecvMeta()
        base = paddle.zeros([2, 3], dtype="float32")
        meta.set_send_message(base)

        # Matching shape + dtype: no exception.
        meta.check_send_message(paddle.zeros([2, 3], dtype="float32"))

        # Shape mismatch is rejected.
        with self.assertRaises(AssertionError):
            meta.check_send_message(paddle.zeros([4, 5], dtype="float32"))

        # dtype mismatch (same shape) is rejected too -> dtype is really checked.
        with self.assertRaises(AssertionError):
            meta.check_send_message(paddle.zeros([2, 3], dtype="int64"))

    def test_check_send_message_noop_when_unset(self):
        # Fresh meta has None messages -> check short-circuits, never raises.
        meta = SendRecvMeta()
        meta.check_send_message(paddle.zeros([9, 9], dtype="float32"))


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestP2PonCalcStream(unittest.TestCase):
    """P2PonCalcStream stores every field verbatim and rejects unknown ops."""

    def test_send_op_stores_all_fields(self):
        t = paddle.zeros([2, 3], dtype="float32")
        group = object()
        op = P2PonCalcStream(
            _send_on_calc_stream, t, 7, group, nranks=4, rank_id=2
        )
        self.assertIs(op.op, _send_on_calc_stream)
        self.assertIs(op.tensor, t)
        self.assertEqual(op.peer, 7)
        self.assertIs(op.group, group)
        self.assertEqual(op.nranks, 4)
        self.assertEqual(op.rank_id, 2)

    def test_recv_op_stores_all_fields(self):
        t = paddle.zeros([2, 3], dtype="float32")
        group = object()
        op = P2PonCalcStream(
            _recv_on_calc_stream, t, 3, group, nranks=2, rank_id=1
        )
        self.assertIs(op.op, _recv_on_calc_stream)
        self.assertEqual(op.peer, 3)
        self.assertEqual(op.nranks, 2)
        self.assertEqual(op.rank_id, 1)

    def test_default_nranks_rank_id(self):
        t = paddle.zeros([2], dtype="float32")
        op = P2PonCalcStream(_send_on_calc_stream, t, 0, object())
        self.assertEqual(op.nranks, 1)
        self.assertEqual(op.rank_id, 0)

    def test_invalid_op_rejected(self):
        t = paddle.zeros([2], dtype="float32")
        with self.assertRaises(RuntimeError):
            P2PonCalcStream(lambda x: x, t, 1, object())


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestBatchP2pTupleOrTensor(unittest.TestCase):
    """_batch_p2p_tuple_or_tensor wraps inputs into one P2PonCalcStream per
    tensor, forwarding peer/group/degree/rank to each op in order."""

    def test_single_tensor_forwards_params(self):
        t = paddle.zeros([2, 3], dtype="float32")
        group = object()
        ops = _batch_p2p_tuple_or_tensor(
            t, _send_on_calc_stream, 7, group, mp_degree=4, mp_rank=2
        )
        self.assertEqual(len(ops), 1)
        self.assertIs(ops[0].op, _send_on_calc_stream)
        self.assertIs(ops[0].tensor, t)
        self.assertEqual(ops[0].peer, 7)
        self.assertIs(ops[0].group, group)
        self.assertEqual(ops[0].nranks, 4)
        self.assertEqual(ops[0].rank_id, 2)

    def test_tuple_preserves_order_and_identity(self):
        t_a = paddle.zeros([2, 3], dtype="float32")
        t_b = paddle.zeros([4, 5], dtype="float32")
        group = object()
        ops = _batch_p2p_tuple_or_tensor(
            (t_a, t_b), _recv_on_calc_stream, 5, group, mp_degree=3, mp_rank=1
        )
        self.assertEqual(len(ops), 2)
        self.assertIs(
            ops[0].tensor, t_a
        )  # order preserved, per-tensor identity
        self.assertIs(ops[1].tensor, t_b)
        for op in ops:
            self.assertIs(op.op, _recv_on_calc_stream)
            self.assertEqual(op.peer, 5)
            self.assertIs(op.group, group)
            self.assertEqual(op.nranks, 3)
            self.assertEqual(op.rank_id, 1)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDtypeNumberMapping(unittest.TestCase):
    """paddle_2_number / number_2_dtype: the exact meta wire-format tables and
    their round-trip, plus rejection of unsupported entries."""

    def test_forward_table_by_hand(self):
        expected = {
            paddle.float16: 0,
            paddle.float32: 1,
            paddle.float64: 2,
            paddle.int32: 3,
            paddle.int64: 4,
            paddle.bfloat16: 5,
            paddle.bool: 6,
        }
        for dtype, number in expected.items():
            self.assertEqual(paddle_2_number(dtype), number)

    def test_inverse_table_by_hand(self):
        expected = {
            0: "float16",
            1: "float32",
            2: "float64",
            3: "int32",
            4: "int64",
            5: "bfloat16",
            6: "bool",
        }
        for number, name in expected.items():
            self.assertEqual(number_2_dtype(number), name)

    def test_unsupported_dtype_rejected(self):
        with self.assertRaises(AssertionError):
            paddle_2_number(paddle.uint8)

    def test_unsupported_number_rejected(self):
        with self.assertRaises(AssertionError):
            number_2_dtype(99)


if __name__ == "__main__":
    unittest.main()
