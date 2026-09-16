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

"""CPU-only behavior tests for pipeline_parallel/pp_utils/p2p_communication.py.

This file exercises only the pure, single-process logic of the p2p module:
tensor-type dispatch and validation, partial send/recv eligibility arithmetic,
the P2PonCalcStream op guard, the SendRecvMeta shape/dtype/key message
bookkeeping, dtype<->number mapping, and P2pHelper construction state.

It intentionally does NOT touch real p2p communication (send/recv/broadcast,
process groups, all_gather). Those require a real multi-rank process group and
belong to multi-card tests; faking world_size + mocking collectives to assert
trivial facts would prove nothing about cross-rank behavior.
"""

import unittest
from types import SimpleNamespace

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils import (
        p2p_communication as p2p_mod,
    )
    from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
        P2pHelper,
        P2PonCalcStream,
        SendRecvMeta,
        _is_valid_send_recv_partial,
        _recv_on_calc_stream,
        _send_on_calc_stream,
    )
    from paddlefleet.pipeline_parallel.pp_utils.utils import (
        number_2_dtype,
        paddle_2_number,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    _IMPORT_ERROR = exc

PADDLE_AVAILABLE = _IMPORT_ERROR is None
SKIP_REASON = (
    f"paddle / paddlefleet p2p_communication import failed: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestDtypeNumberMapping(unittest.TestCase):
    """paddle_2_number / number_2_dtype are inverse lookup tables.

    Expected values derived by hand from the module's own PADDLE_TO_NUMBER /
    NUMBER_TO_DTYPE tables, restated here independently rather than read back
    from the production dicts.
    """

    def test_forward_mapping_exact_codes(self):
        self.assertEqual(paddle_2_number(paddle.float16), 0)
        self.assertEqual(paddle_2_number(paddle.float32), 1)
        self.assertEqual(paddle_2_number(paddle.float64), 2)
        self.assertEqual(paddle_2_number(paddle.int32), 3)
        self.assertEqual(paddle_2_number(paddle.int64), 4)
        self.assertEqual(paddle_2_number(paddle.bfloat16), 5)
        self.assertEqual(paddle_2_number(paddle.bool), 6)

    def test_reverse_mapping_exact_names(self):
        self.assertEqual(number_2_dtype(0), "float16")
        self.assertEqual(number_2_dtype(1), "float32")
        self.assertEqual(number_2_dtype(2), "float64")
        self.assertEqual(number_2_dtype(3), "int32")
        self.assertEqual(number_2_dtype(4), "int64")
        self.assertEqual(number_2_dtype(5), "bfloat16")
        self.assertEqual(number_2_dtype(6), "bool")

    def test_round_trip_dtype_to_name(self):
        cases = [
            (paddle.float16, "float16"),
            (paddle.float32, "float32"),
            (paddle.float64, "float64"),
            (paddle.int32, "int32"),
            (paddle.int64, "int64"),
            (paddle.bfloat16, "bfloat16"),
            (paddle.bool, "bool"),
        ]
        for dtype, name in cases:
            self.assertEqual(number_2_dtype(paddle_2_number(dtype)), name)

    def test_unknown_dtype_rejected(self):
        # complex64 is a real paddle dtype but is absent from the table.
        with self.assertRaises(AssertionError):
            paddle_2_number(paddle.complex64)

    def test_unknown_number_rejected(self):
        with self.assertRaises(AssertionError):
            number_2_dtype(99)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestIsValidSendRecvPartial(unittest.TestCase):
    """_is_valid_send_recv_partial(tensor, mp_degree) is pure arithmetic:

    returns True only when partial send/recv is enabled globally, mp_degree > 1,
    and the tensor element count divides evenly by mp_degree.
    """

    @staticmethod
    def _shaped(shape):
        # Only .shape is consumed (np.prod(tensor.shape)); a lightweight stub
        # avoids allocating a real device tensor for a pure numeric predicate.
        return SimpleNamespace(shape=list(shape))

    def test_degree_one_never_partial(self):
        self.assertFalse(_is_valid_send_recv_partial(self._shaped([2, 4]), 1))

    def test_divisible_numel_is_partial(self):
        # numel = 8: divisible by 2 and by 4.
        self.assertTrue(_is_valid_send_recv_partial(self._shaped([2, 4]), 2))
        self.assertTrue(_is_valid_send_recv_partial(self._shaped([2, 4]), 4))

    def test_indivisible_numel_not_partial(self):
        # numel = 8 is not divisible by 3.
        self.assertFalse(_is_valid_send_recv_partial(self._shaped([2, 4]), 3))

    def test_zero_element_tensor_rejected(self):
        with self.assertRaises(AssertionError):
            _is_valid_send_recv_partial(self._shaped([0, 4]), 2)

    def test_disabled_flag_forces_false(self):
        orig = p2p_mod._enable_partial_send_recv
        self.addCleanup(setattr, p2p_mod, "_enable_partial_send_recv", orig)
        p2p_mod._enable_partial_send_recv = False
        # Even a divisible numel with degree > 1 must be False when disabled.
        self.assertFalse(_is_valid_send_recv_partial(self._shaped([2, 4]), 2))


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestP2PonCalcStreamGuard(unittest.TestCase):
    """P2PonCalcStream stores its fields verbatim and rejects unknown ops."""

    def test_send_op_stored_verbatim(self):
        tensor = SimpleNamespace(tag="t")
        group = SimpleNamespace(tag="g")
        op = P2PonCalcStream(
            _send_on_calc_stream,
            tensor,
            peer=3,
            group=group,
            nranks=2,
            rank_id=1,
        )
        self.assertIs(op.op, _send_on_calc_stream)
        self.assertIs(op.tensor, tensor)
        self.assertEqual(op.peer, 3)
        self.assertIs(op.group, group)
        self.assertEqual(op.nranks, 2)
        self.assertEqual(op.rank_id, 1)

    def test_recv_op_accepted_with_defaults(self):
        op = P2PonCalcStream(
            _recv_on_calc_stream,
            SimpleNamespace(),
            peer=0,
            group=SimpleNamespace(),
        )
        self.assertIs(op.op, _recv_on_calc_stream)
        self.assertEqual(op.nranks, 1)
        self.assertEqual(op.rank_id, 0)

    def test_unknown_op_rejected(self):
        def not_a_stream_op(*args, **kwargs):
            return None

        with self.assertRaises(RuntimeError):
            P2PonCalcStream(
                not_a_stream_op,
                SimpleNamespace(),
                peer=0,
                group=SimpleNamespace(),
            )


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestSendRecvMetaState(unittest.TestCase):
    """SendRecvMeta bookkeeping: init state, erase, and message capture."""

    def test_fresh_meta_is_empty(self):
        meta = SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.send_key_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertIsNone(meta.recv_stop_gradient)
        self.assertIsNone(meta.recv_key_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_init_or_erase_resets_all_fields(self):
        meta = SendRecvMeta()
        meta.send_shape_message = [2, 3]
        meta.recv_shape_message = [4, 5]
        meta.has_send_meta = True
        meta.has_recv_meta = True
        meta.init_or_erase_meta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_obtain_message_single_tensor(self):
        t = paddle.zeros([2, 3], dtype="float32")
        shape, dtype_no, key = SendRecvMeta()._obtain_send_message(t)
        self.assertEqual(list(shape), [2, 3])
        self.assertEqual(dtype_no, 1)  # float32 -> 1
        self.assertIsNone(key)

    def test_obtain_message_reads_key_attribute(self):
        t = paddle.zeros([4], dtype="int64")
        t.key = "layer0"
        shape, dtype_no, key = SendRecvMeta()._obtain_send_message(t)
        self.assertEqual(list(shape), [4])
        self.assertEqual(dtype_no, 4)  # int64 -> 4
        self.assertEqual(key, "layer0")

    def test_obtain_message_list_skips_stop_gradient(self):
        # Distinguishable shapes; the middle tensor is stop_gradient and must
        # be dropped, so order and identity of the kept tensors are observable.
        t0 = paddle.zeros([2, 3], dtype="float32")
        t0.stop_gradient = False
        t1 = paddle.zeros([4, 5], dtype="float32")
        t1.stop_gradient = True
        t2 = paddle.zeros([6, 7], dtype="float32")
        t2.stop_gradient = False

        shapes, dtypes, keys = SendRecvMeta()._obtain_send_message([t0, t1, t2])
        self.assertEqual([list(s) for s in shapes], [[2, 3], [6, 7]])
        self.assertEqual(dtypes, (1, 1))
        self.assertEqual(keys, (None, None))

    def test_set_and_check_message_roundtrip(self):
        meta = SendRecvMeta()
        t = paddle.zeros([2, 3], dtype="float32")
        t.stop_gradient = False
        meta.set_send_message(t)
        self.assertEqual(list(meta.send_shape_message), [2, 3])
        self.assertEqual(meta.send_dtype_message, 1)
        self.assertIsNone(meta.send_key_message)
        # Same tensor passes the consistency check.
        meta.check_send_message(t)

    def test_check_message_rejects_shape_mismatch(self):
        meta = SendRecvMeta()
        stored = paddle.zeros([2, 3], dtype="float32")
        stored.stop_gradient = False
        meta.set_send_message(stored)

        other = paddle.zeros([2, 4], dtype="float32")
        other.stop_gradient = False
        with self.assertRaises(AssertionError):
            meta.check_send_message(other)

    def test_check_message_no_op_when_unset(self):
        # With no send message recorded, check must be a silent no-op.
        meta = SendRecvMeta()
        t = paddle.zeros([9], dtype="float32")
        t.stop_gradient = False
        meta.check_send_message(t)  # must not raise

    def test_send_meta_rejects_non_tensor_type(self):
        # send_meta first resolves a peer via the module hcg, then dispatches
        # on tensor type. Stub the hcg (a non-tested collaborator) so the pure
        # type guard is reachable; no real communication is performed.
        orig_hcg = p2p_mod._hcg
        self.addCleanup(setattr, p2p_mod, "_hcg", orig_hcg)
        p2p_mod._hcg = SimpleNamespace(
            _get_p2p_next_rank=lambda: 1,
            _get_p2p_prev_rank=lambda: 0,
        )
        meta = SendRecvMeta()
        with self.assertRaises(TypeError):
            meta.send_meta("not_a_tensor", group=SimpleNamespace())


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestP2pHelperConstruction(unittest.TestCase):
    """P2pHelper.__init__ state differs by dynamic_shape; __repr__ reports it."""

    def test_static_shape_construction(self):
        helper = P2pHelper(use_cache=True, dynamic_shape=False)
        self.assertTrue(helper._use_cache)
        self.assertFalse(helper._dynamic_shape)
        self.assertIsInstance(helper._send_recv_meta, SendRecvMeta)
        # Dynamic-only bookkeeping must be absent in static mode.
        self.assertFalse(hasattr(helper, "_dynamic_cnt"))
        self.assertFalse(hasattr(helper, "_send_recv_meta_list"))

    def test_dynamic_shape_construction(self):
        helper = P2pHelper(use_cache=False, dynamic_shape=True)
        self.assertFalse(helper._use_cache)
        self.assertTrue(helper._dynamic_shape)
        self.assertEqual(helper._dynamic_cnt, 0)
        self.assertEqual(helper._send_recv_meta_list, [])

    def test_repr_reports_cache_and_meta(self):
        helper = P2pHelper(use_cache=True, dynamic_shape=False)
        text = repr(helper)
        self.assertIn("using cache: True", text)
        # The nested SendRecvMeta repr is embedded.
        self.assertIn("send_shape_message", text)
        self.assertIn("recv_shape_message", text)


if __name__ == "__main__":
    unittest.main()
