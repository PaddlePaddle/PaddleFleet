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

"""Single-card behavior tests for the p2p metadata serialization codec.

Exercises the CPU-executable pure logic of the production module
``paddlefleet.pipeline_parallel.pp_utils.p2p_communication``:

* ``SendRecvMeta.send_meta`` wire encoding: the exact int64 buffer produced
  for a single tensor, a tuple of tensors and the ``TypeError`` contract for a
  non-tensor payload, plus destination-peer selection (next rank forward,
  previous rank on ``reverse=True``).
* ``SendRecvMeta.recv_meta`` wire decoding of hand-built int64 payloads back
  into shape / dtype-number / stop_gradient / key messages, plus source-peer
  selection (previous rank forward, next rank on ``reverse=True``).
* A ``send_meta -> recv_meta`` round trip through a single in-memory buffer,
  confirming the two halves of the codec agree on shape, dtype and gradient
  flag, cross-checked through ``number_2_dtype``.
* ``paddle_2_number`` / ``number_2_dtype`` inverse-pair consistency and their
  invalid-input assertions.

Scope / substitution boundary: ``send_meta`` / ``recv_meta`` only use
``paddle.distributed.send`` / ``recv`` / ``broadcast`` as a byte transport for
an already-built buffer. Here that transport is replaced by an in-memory pipe
so the REAL encode/decode logic runs on both ends and is compared against
independently hand-derived buffers and values. No real process group runs;
this verifies the serialization protocol and peer selection ONLY, NOT
cross-rank transport, split sizes or reduction (which belong to a multi-card
test). Wire numbers are hand-derived from the documented PADDLE_TO_NUMBER
table, independent of ``paddle_2_number`` so a corrupted table cannot silently
agree with the encoder.
"""

import unittest

try:
    import numpy as np
    import paddle

    # Importing paddlefleet pulls in paddlefleet_ops, which queries the CUDA
    # device capability at import time; that requires a CUDA place to be
    # selected first. The CI runner provides a real GPU, so select it here and
    # only fall back to CPU when CUDA is unavailable.
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    from paddlefleet.pipeline_parallel.pp_utils import (
        p2p_communication as p2p_mod,
    )
    from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
        SendRecvMeta,
        _is_valid_send_recv_partial,
    )
    from paddlefleet.pipeline_parallel.pp_utils.utils import (
        number_2_dtype,
        paddle_2_number,
    )

    _IMPORT_ERROR = None
# RuntimeError/ValueError capture the import-time device-capability probe on a
# host without a usable CUDA device, so the suite skips instead of erroring at
# collection.
except (
    ImportError,
    ModuleNotFoundError,
    RuntimeError,
    ValueError,
) as exc:  # pragma: no cover
    np = None
    paddle = None
    p2p_mod = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    f"paddle/paddlefleet import failed: {_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)

# Hand-derived paddle dtype -> wire number, independent of the production
# paddle_2_number() table so a corrupted table cannot silently agree.
_FLOAT16_NUM = 0
_FLOAT32_NUM = 1
_INT64_NUM = 4
_PREV_RANK = 2
_NEXT_RANK = 3


class _RankStub:
    """Minimal stand-in for the module-global ``_hcg`` peer resolver.

    ``send_meta`` / ``recv_meta`` only read the p2p peer ranks off this object;
    the values distinguish forward (next/prev) from ``reverse`` selection.
    """

    def _get_p2p_prev_rank(self):
        return _PREV_RANK

    def _get_p2p_next_rank(self):
        return _NEXT_RANK


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSendRecvMetaCodec(unittest.TestCase):
    def setUp(self):
        self._orig_hcg = p2p_mod._hcg
        self._orig_send = paddle.distributed.send
        self._orig_recv = paddle.distributed.recv
        self.addCleanup(self._restore)
        p2p_mod._hcg = _RankStub()

    def _restore(self):
        p2p_mod._hcg = self._orig_hcg
        paddle.distributed.send = self._orig_send
        paddle.distributed.recv = self._orig_recv

    def _install_capture_send(self):
        """Replace only the byte transport; capture what send_meta emits."""
        self._sent = []

        def fake_send(tensor, dst, group):
            self._sent.append(
                (np.asarray(tensor.numpy()).reshape(-1).tolist(), dst)
            )

        paddle.distributed.send = fake_send

    def _install_replay_recv(self, buffers):
        """Feed hand-built int64 buffers into recv_meta; record src peers."""
        self._recv_srcs = []
        queue = [list(buf) for buf in buffers]

        def fake_recv(tensor, src, group):
            payload = np.asarray(queue.pop(0), dtype="int64").reshape(
                tensor.shape
            )
            tensor.set_value(paddle.to_tensor(payload, dtype=tensor.dtype))
            self._recv_srcs.append(src)

        paddle.distributed.recv = fake_recv

    def test_send_meta_single_tensor_wire_encoding(self):
        self._install_capture_send()
        meta = SendRecvMeta()
        tensor = paddle.ones([2, 3], dtype="float32")
        tensor.stop_gradient = False

        meta.send_meta(tensor, group=object())

        # Two sends: element count, then the encoded buffer.
        self.assertEqual(len(self._sent), 2)
        # Hand-derived layout:
        # [tensor_type=0, shape_len=2, 2, 3, dtype=1, stop_gradient=0, key_len=0]
        expected = [0, 2, 2, 3, _FLOAT32_NUM, 0, 0]
        self.assertEqual(self._sent[0][0], [len(expected)])
        self.assertEqual(self._sent[1][0], expected)
        # Forward send targets the NEXT peer for both messages.
        self.assertEqual(self._sent[0][1], _NEXT_RANK)
        self.assertEqual(self._sent[1][1], _NEXT_RANK)

    def test_send_meta_reverse_targets_prev_peer(self):
        self._install_capture_send()
        meta = SendRecvMeta()
        tensor = paddle.ones([2, 3], dtype="float32")
        tensor.stop_gradient = False

        meta.send_meta(tensor, group=object(), reverse=True)

        self.assertEqual(
            [dst for _, dst in self._sent], [_PREV_RANK, _PREV_RANK]
        )

    def test_send_meta_tuple_wire_encoding(self):
        self._install_capture_send()
        meta = SendRecvMeta()
        first = paddle.ones([5], dtype="int64")
        first.stop_gradient = False
        second = paddle.ones([2, 3], dtype="float16")
        second.stop_gradient = True

        meta.send_meta((first, second), group=object())

        # Hand-derived layout for a 2-tuple:
        # [type=1, num=2,
        #  shape_len=1, 5, dtype=4, stop_gradient=0, key_len=0,
        #  shape_len=2, 2, 3, dtype=0, stop_gradient=1, key_len=0]
        expected = [
            1,
            2,
            1,
            5,
            _INT64_NUM,
            0,
            0,
            2,
            2,
            3,
            _FLOAT16_NUM,
            1,
            0,
        ]
        self.assertEqual(self._sent[1][0], expected)
        self.assertEqual(self._sent[0][0], [len(expected)])

    def test_send_meta_rejects_non_tensor(self):
        self._install_capture_send()
        meta = SendRecvMeta()
        with self.assertRaises(TypeError):
            meta.send_meta("not-a-tensor", group=object())

    def test_recv_meta_single_tensor_wire_decoding(self):
        # Hand-built payload for shape [3, 4], float32, stop_gradient=True.
        payload = [0, 2, 3, 4, _FLOAT32_NUM, 1, 0]
        self._install_replay_recv([[len(payload)], payload])

        meta = SendRecvMeta()
        meta.recv_meta(object())

        self.assertEqual(meta.recv_shape_message, [3, 4])
        self.assertEqual(meta.recv_dtype_message, _FLOAT32_NUM)
        self.assertIs(meta.recv_stop_gradient, True)
        self.assertIsNone(meta.recv_key_message)
        # Forward recv reads the element count then the buffer, both from prev.
        self.assertEqual(self._recv_srcs, [_PREV_RANK, _PREV_RANK])

    def test_recv_meta_reverse_reads_from_next_peer(self):
        payload = [0, 2, 3, 4, _FLOAT32_NUM, 0, 0]
        self._install_replay_recv([[len(payload)], payload])

        meta = SendRecvMeta()
        meta.recv_meta(object(), reverse=True)

        self.assertEqual(self._recv_srcs, [_NEXT_RANK, _NEXT_RANK])
        self.assertIs(meta.recv_stop_gradient, False)

    def test_recv_meta_tuple_wire_decoding(self):
        # Hand-built payload for a 2-tuple:
        #   t0: shape [5], int64, stop_gradient=False
        #   t1: shape [2, 3], float16, stop_gradient=True
        payload = [
            1,
            2,
            1,
            5,
            _INT64_NUM,
            0,
            0,
            2,
            2,
            3,
            _FLOAT16_NUM,
            1,
            0,
        ]
        self._install_replay_recv([[len(payload)], payload])

        meta = SendRecvMeta()
        meta.recv_meta(object())

        self.assertEqual(meta.recv_shape_message, ([5], [2, 3]))
        self.assertEqual(meta.recv_dtype_message, (_INT64_NUM, _FLOAT16_NUM))
        self.assertEqual(meta.recv_stop_gradient, (False, True))
        self.assertEqual(meta.recv_key_message, (None, None))

    def test_recv_meta_rejects_trailing_bytes(self):
        # One extra trailing element must trip the "parsed zero" assertion.
        payload = [0, 2, 3, 4, _FLOAT32_NUM, 0, 0, 99]
        self._install_replay_recv([[len(payload)], payload])

        meta = SendRecvMeta()
        with self.assertRaises(AssertionError):
            meta.recv_meta(object())

    def test_send_then_recv_round_trip_agrees(self):
        self._install_capture_send()
        tensor = paddle.ones([4, 1], dtype="int64")
        tensor.stop_gradient = False
        SendRecvMeta().send_meta(tensor, group=object())

        buffers = [self._sent[0][0], self._sent[1][0]]
        self._install_replay_recv(buffers)
        decoded = SendRecvMeta()
        decoded.recv_meta(object())

        self.assertEqual(decoded.recv_shape_message, [4, 1])
        self.assertEqual(number_2_dtype(decoded.recv_dtype_message), "int64")
        self.assertIs(decoded.recv_stop_gradient, False)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDtypeNumberMapping(unittest.TestCase):
    def test_paddle_number_pairs_are_consistent_inverses(self):
        # Hand-derived from the documented PADDLE_TO_NUMBER table.
        cases = [
            (paddle.float16, 0, "float16"),
            (paddle.float32, 1, "float32"),
            (paddle.float64, 2, "float64"),
            (paddle.int32, 3, "int32"),
            (paddle.int64, 4, "int64"),
            (paddle.bfloat16, 5, "bfloat16"),
            (paddle.bool, 6, "bool"),
        ]
        for dtype, number, name in cases:
            self.assertEqual(paddle_2_number(dtype), number)
            self.assertEqual(number_2_dtype(number), name)
            self.assertEqual(number_2_dtype(paddle_2_number(dtype)), name)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(AssertionError):
            paddle_2_number("float32")  # a string is not a paddle dtype key
        with self.assertRaises(AssertionError):
            number_2_dtype(999)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSendMessageAndPartialPredicate(unittest.TestCase):
    def setUp(self):
        self._orig_enable = p2p_mod._enable_partial_send_recv
        self.addCleanup(
            lambda: setattr(
                p2p_mod, "_enable_partial_send_recv", self._orig_enable
            )
        )

    def test_check_send_message_noop_when_unset(self):
        meta = SendRecvMeta()
        probe = paddle.ones([2, 2], dtype="float32")
        # No send message recorded yet -> early return, no assertion.
        self.assertIsNone(meta.check_send_message(probe))

    def test_check_send_message_detects_each_mismatch(self):
        meta = SendRecvMeta()
        baseline = paddle.ones([2, 2], dtype="float32")
        baseline.stop_gradient = False
        baseline.key = "expected"
        meta.set_send_message(baseline)
        self.assertEqual(meta.send_shape_message, [2, 2])
        self.assertEqual(meta.send_dtype_message, _FLOAT32_NUM)
        self.assertEqual(meta.send_key_message, "expected")

        bad_shape = paddle.ones([3, 2], dtype="float32")
        bad_shape.stop_gradient = False
        bad_shape.key = "expected"
        with self.assertRaisesRegex(AssertionError, "send_shape_message"):
            meta.check_send_message(bad_shape)

        bad_dtype = paddle.ones([2, 2], dtype="float16")
        bad_dtype.stop_gradient = False
        bad_dtype.key = "expected"
        with self.assertRaisesRegex(AssertionError, "send_dtype_message"):
            meta.check_send_message(bad_dtype)

        bad_key = paddle.ones([2, 2], dtype="float32")
        bad_key.stop_gradient = False
        bad_key.key = "actual"
        with self.assertRaisesRegex(AssertionError, "send_key_message"):
            meta.check_send_message(bad_key)

    def test_set_send_message_tuple_skips_stop_gradient_tensor(self):
        meta = SendRecvMeta()
        first = paddle.ones([2, 2], dtype="float32")
        first.stop_gradient = False
        second = paddle.ones([1, 4], dtype="int64")
        second.stop_gradient = True

        meta.set_send_message((first, second))

        # Only the trainable tensor is recorded for the send-shape contract.
        self.assertEqual(meta.send_shape_message, ([2, 2],))
        self.assertEqual(meta.send_dtype_message, (_FLOAT32_NUM,))

    def test_is_valid_send_recv_partial_predicate(self):
        p2p_mod._enable_partial_send_recv = True
        numel4 = paddle.ones([2, 2], dtype="float32")  # 4 elements
        numel6 = paddle.ones([2, 3], dtype="float32")  # 6 elements
        self.assertFalse(_is_valid_send_recv_partial(numel4, 1))
        self.assertTrue(_is_valid_send_recv_partial(numel4, 2))
        self.assertFalse(_is_valid_send_recv_partial(numel4, 3))
        self.assertTrue(_is_valid_send_recv_partial(numel6, 3))

        with self.assertRaises(AssertionError):
            _is_valid_send_recv_partial(paddle.empty([0]), 2)

        # Disabled: returns False and short-circuits BEFORE the >0 assertion.
        p2p_mod._enable_partial_send_recv = False
        self.assertFalse(_is_valid_send_recv_partial(numel4, 2))
        self.assertFalse(_is_valid_send_recv_partial(paddle.empty([0]), 2))


if __name__ == "__main__":
    unittest.main()
