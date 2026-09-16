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

"""Device-agnostic behavior tests for paddlefleet pipeline_parallel
p2p_communication.

These tests exercise the pure, communication-free logic of the production
module ``paddlefleet.pipeline_parallel.pp_utils.p2p_communication``. Importing
paddlefleet initializes paddlefleet_ops, which queries the CUDA device
capability, so on a GPU host the device is selected as ``gpu`` before import
(see below); the logic under test is independent of the device it runs on:

* ``_is_valid_send_recv_partial`` divisibility / enable-flag gating
* ``SendRecvMeta._obtain_send_message`` shape/dtype-number/key extraction and
  the stop_gradient skipping contract for tuples
* ``SendRecvMeta.set_send_message`` / ``check_send_message`` match + mismatch
  assertion contract
* ``_batch_p2p_tuple_or_tensor`` fan-out and per-op parameter wiring
* ``P2PonCalcStream`` op validation and attribute consumption

Real cross-rank send/recv, batched calc-stream collectives and the pipeline
schedule (``_p2p_ops``, ``_batched_p2p_ops``, ``P2pHelper`` methods) require a
real process group and are intentionally NOT exercised here; they belong to a
multi-card test.  Expected dtype numbers are hand-derived from the documented
PADDLE_TO_NUMBER mapping (independent of the production helper).
"""

import unittest
from unittest import mock

try:
    import paddle

    # Importing paddlefleet drags in paddlefleet_ops, whose module init calls
    # paddle.cuda.get_device_capability() against the *current* device. On a CPU
    # place that raises ValueError and aborts collection, so on a CUDA-capable
    # host we must select the GPU before the import. The tests below are pure,
    # communication-free logic and are device-agnostic once imported.
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

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

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    p2p_mod = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    f"paddle/paddlefleet import failed: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)

# Hand-derived reference for paddle dtype -> wire number, independent of the
# production paddle_2_number() so a change to that table cannot silently agree
# with the expected values below.
_EXPECTED_DTYPE_NUMBER = {
    "float16": 0,
    "float32": 1,
    "float64": 2,
    "int32": 3,
    "int64": 4,
    "bfloat16": 5,
    "bool": 6,
}


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestIsValidSendRecvPartial(unittest.TestCase):
    """_is_valid_send_recv_partial: enable flag + numel % mp_degree gating."""

    def test_divisibility_when_enabled(self):
        # numel = 4 * 4 = 16. Expected by hand: only mp_degree>1 that divides 16.
        with mock.patch.object(p2p_mod, "_enable_partial_send_recv", True):
            tensor = paddle.zeros([4, 4], dtype="float32")
            self.assertFalse(_is_valid_send_recv_partial(tensor, mp_degree=1))
            self.assertTrue(_is_valid_send_recv_partial(tensor, mp_degree=2))
            self.assertFalse(_is_valid_send_recv_partial(tensor, mp_degree=3))
            self.assertTrue(_is_valid_send_recv_partial(tensor, mp_degree=4))

            # numel = 3 * 3 = 9: divisible by 3, not by 2.
            odd = paddle.zeros([3, 3], dtype="float32")
            self.assertFalse(_is_valid_send_recv_partial(odd, mp_degree=2))
            self.assertTrue(_is_valid_send_recv_partial(odd, mp_degree=3))

    def test_disabled_flag_forces_false(self):
        # Even a perfectly divisible tensor is invalid once the flag is off.
        with mock.patch.object(p2p_mod, "_enable_partial_send_recv", False):
            tensor = paddle.zeros([4, 4], dtype="float32")
            self.assertFalse(_is_valid_send_recv_partial(tensor, mp_degree=2))
            self.assertFalse(_is_valid_send_recv_partial(tensor, mp_degree=4))

    def test_zero_element_rejected(self):
        with mock.patch.object(p2p_mod, "_enable_partial_send_recv", True):
            empty = paddle.zeros([0, 4], dtype="float32")
            with self.assertRaises(AssertionError):
                _is_valid_send_recv_partial(empty, mp_degree=2)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestObtainSendMessage(unittest.TestCase):
    """SendRecvMeta._obtain_send_message: shape/dtype-number/key + skipping."""

    def test_single_tensor_shape_dtype_key(self):
        meta = SendRecvMeta()
        tensor = paddle.zeros([2, 3], dtype="float32")
        shape, dtype_num, key = meta._obtain_send_message(tensor)
        self.assertEqual(list(shape), [2, 3])
        self.assertEqual(dtype_num, _EXPECTED_DTYPE_NUMBER["float32"])
        self.assertIsNone(key)

    def test_single_tensor_key_propagated(self):
        meta = SendRecvMeta()
        tensor = paddle.zeros([1, 4], dtype="int64")
        tensor.key = "layer0"
        shape, dtype_num, key = meta._obtain_send_message(tensor)
        self.assertEqual(list(shape), [1, 4])
        self.assertEqual(dtype_num, _EXPECTED_DTYPE_NUMBER["int64"])
        self.assertEqual(key, "layer0")

    def test_tuple_skips_stop_gradient_and_keeps_order(self):
        meta = SendRecvMeta()
        # t0 is stop_gradient=True and must be dropped; t1, t2 kept in order
        # with distinguishable shapes and dtypes so a swap/keep-wrong would show.
        t0 = paddle.zeros([2, 3], dtype="float32")
        t0.stop_gradient = True
        t1 = paddle.zeros([4], dtype="int64")
        t1.stop_gradient = False
        t2 = paddle.zeros([1, 5], dtype="float16")
        t2.stop_gradient = False

        shapes, dtypes, keys = meta._obtain_send_message((t0, t1, t2))
        self.assertEqual([list(s) for s in shapes], [[4], [1, 5]])
        self.assertEqual(
            list(dtypes),
            [
                _EXPECTED_DTYPE_NUMBER["int64"],
                _EXPECTED_DTYPE_NUMBER["float16"],
            ],
        )
        self.assertEqual(list(keys), [None, None])


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestCheckSendMessage(unittest.TestCase):
    """SendRecvMeta.set_send_message / check_send_message contract."""

    def test_set_records_shape_and_dtype(self):
        meta = SendRecvMeta()
        tensor = paddle.zeros([2, 3], dtype="float32")
        meta.set_send_message(tensor)
        self.assertEqual(list(meta.send_shape_message), [2, 3])
        self.assertEqual(
            meta.send_dtype_message, _EXPECTED_DTYPE_NUMBER["float32"]
        )
        self.assertIsNone(meta.send_key_message)

    def test_check_returns_early_when_unset(self):
        meta = SendRecvMeta()
        # Nothing recorded yet -> must not raise for any tensor.
        self.assertIsNone(meta.send_shape_message)
        meta.check_send_message(paddle.zeros([9, 9], dtype="float32"))

    def test_check_matching_passes(self):
        meta = SendRecvMeta()
        meta.set_send_message(paddle.zeros([2, 3], dtype="float32"))
        # Same shape and dtype -> no assertion error.
        meta.check_send_message(paddle.zeros([2, 3], dtype="float32"))

    def test_check_shape_mismatch_raises(self):
        meta = SendRecvMeta()
        meta.set_send_message(paddle.zeros([2, 3], dtype="float32"))
        with self.assertRaises(AssertionError):
            meta.check_send_message(paddle.zeros([4, 5], dtype="float32"))

    def test_check_dtype_mismatch_raises(self):
        meta = SendRecvMeta()
        meta.set_send_message(paddle.zeros([2, 3], dtype="float32"))
        with self.assertRaises(AssertionError):
            meta.check_send_message(paddle.zeros([2, 3], dtype="int64"))


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestBatchP2pTupleOrTensor(unittest.TestCase):
    """_batch_p2p_tuple_or_tensor fan-out and per-op parameter wiring."""

    def test_single_tensor_makes_one_op(self):
        group = object()  # opaque process-group stand-in; only stored, not used
        tensor = paddle.zeros([2, 2], dtype="float32")
        ops = _batch_p2p_tuple_or_tensor(
            tensor,
            _send_on_calc_stream,
            pp_rank=7,
            pp_group=group,
            mp_degree=4,
            mp_rank=2,
        )
        self.assertEqual(len(ops), 1)
        op = ops[0]
        self.assertIs(op.op, _send_on_calc_stream)
        self.assertIs(op.tensor, tensor)
        self.assertEqual(op.peer, 7)
        self.assertIs(op.group, group)
        self.assertEqual(op.nranks, 4)
        self.assertEqual(op.rank_id, 2)

    def test_tuple_preserves_count_order_and_params(self):
        group = object()
        t0 = paddle.zeros([1], dtype="float32")
        t1 = paddle.zeros([2], dtype="float32")
        t2 = paddle.zeros([3], dtype="float32")
        ops = _batch_p2p_tuple_or_tensor(
            (t0, t1, t2),
            _recv_on_calc_stream,
            pp_rank=5,
            pp_group=group,
            mp_degree=3,
            mp_rank=1,
        )
        self.assertEqual(len(ops), 3)
        # Order preserved: op i wraps tensor i (identity check catches reorder).
        for op, expected_tensor in zip(ops, (t0, t1, t2)):
            self.assertIs(op.tensor, expected_tensor)
            self.assertIs(op.op, _recv_on_calc_stream)
            self.assertEqual(op.peer, 5)
            self.assertIs(op.group, group)
            self.assertEqual(op.nranks, 3)
            self.assertEqual(op.rank_id, 1)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestP2PonCalcStream(unittest.TestCase):
    """P2PonCalcStream op validation and attribute consumption."""

    def test_invalid_op_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            P2PonCalcStream(lambda *a, **k: None, object(), 0, object())

    def test_defaults_and_stored_attributes(self):
        group = object()
        tensor = paddle.zeros([2, 2], dtype="float32")
        # Defaults: nranks=1, rank_id=0 when not supplied.
        op = P2PonCalcStream(_recv_on_calc_stream, tensor, peer=3, group=group)
        self.assertIs(op.op, _recv_on_calc_stream)
        self.assertIs(op.tensor, tensor)
        self.assertEqual(op.peer, 3)
        self.assertIs(op.group, group)
        self.assertEqual(op.nranks, 1)
        self.assertEqual(op.rank_id, 0)


if __name__ == "__main__":
    unittest.main()
