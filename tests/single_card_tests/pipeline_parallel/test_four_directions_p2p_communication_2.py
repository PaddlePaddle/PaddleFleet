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

"""CPU-only behavior tests for the device-independent dtype protocol used by
``paddlefleet.pipeline_parallel.pp_utils.four_directions_p2p_communication``.

The p2p meta channel encodes each tensor's dtype as a small integer on the
wire: ``SendRecvMeta.set_send_message`` stores ``paddle_2_number(dtype)`` on the
send side and ``_p2p_helper`` reconstructs the tensor with
``number_2_dtype(...)`` on the receive side. This file pins that protocol and
the send-side/receive-side reconstruction consistency, which are pure Python
lookups exercisable on CPU without any process group.

The expected dtype integers and dtype names are hand-written literals below
(``_WIRE_PROTOCOL``); they are NOT read from the production ``PADDLE_TO_NUMBER``
/ ``NUMBER_TO_DTYPE`` tables, so the test and the code under test do not share a
source of truth. A consistent swap in both production tables would still be
caught by the exact forward/backward value assertions, not only the round trip.

The actual distributed send/recv, meta exchange over process groups, and the
four-direction ``_p2p_helper`` scheduling require a real pipeline process group
and are intentionally NOT covered here: faking ``world_size`` and mocking the
collectives would only prove local orchestration, never the cross-rank
communication direction, peer selection, or partial-split semantics (see the
multi-card / anti-pattern-13 guidance). Those belong in a real multi-card test.
"""

import unittest

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils import (
        four_directions_p2p_communication as fd,
    )
    from paddlefleet.pipeline_parallel.pp_utils.utils import (
        number_2_dtype,
        paddle_2_number,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    fd = None
    number_2_dtype = None
    paddle_2_number = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDtypeWireProtocol(unittest.TestCase):
    """`paddle_2_number` / `number_2_dtype` on-wire dtype protocol.

    Contract (hand-derived, independent of the production tables):
    float16=0, float32=1, float64=2, int32=3, int64=4, bfloat16=5, bool=6.
    ``paddle_2_number`` asserts on unsupported dtypes; ``number_2_dtype``
    asserts on numbers outside the mapping.
    """

    def _wire_protocol(self):
        # (paddle dtype, wire integer, dtype name) written out by hand.
        return [
            (paddle.float16, 0, "float16"),
            (paddle.float32, 1, "float32"),
            (paddle.float64, 2, "float64"),
            (paddle.int32, 3, "int32"),
            (paddle.int64, 4, "int64"),
            (paddle.bfloat16, 5, "bfloat16"),
            (paddle.bool, 6, "bool"),
        ]

    def test_forward_mapping_exact(self):
        for dtype, number, _name in self._wire_protocol():
            self.assertEqual(paddle_2_number(dtype), number)

    def test_number_to_dtype_exact(self):
        for _dtype, number, name in self._wire_protocol():
            self.assertEqual(number_2_dtype(number), name)

    def test_bijection_round_trip_is_consistent(self):
        # dtype -> number -> name must land on the hand-written name; and the
        # set of wire integers must be exactly {0..6} with no collisions.
        seen_numbers = []
        for dtype, number, name in self._wire_protocol():
            got_number = paddle_2_number(dtype)
            self.assertEqual(got_number, number)
            self.assertEqual(number_2_dtype(got_number), name)
            seen_numbers.append(got_number)
        self.assertEqual(sorted(seen_numbers), list(range(7)))

    def test_paddle_2_number_rejects_unsupported_dtype(self):
        # uint8 is deliberately absent from the protocol table.
        with self.assertRaises(AssertionError):
            paddle_2_number(paddle.uint8)

    def test_number_2_dtype_rejects_out_of_range(self):
        for bad in (7, -1):
            with self.assertRaises(AssertionError):
                number_2_dtype(bad)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSendMessageDtypeReconstruction(unittest.TestCase):
    """`SendRecvMeta.set_send_message` capture reconstructs on the recv side.

    The send side stores ``paddle_2_number(dtype)``; the receive side rebuilds
    the tensor via ``number_2_dtype(...)``. This checks that captured metadata
    round-trips back to the original dtype name and that the tuple path filters
    out ``stop_gradient`` tensors while preserving order and per-element dtype.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_single_tensor_shape_and_reconstructed_dtype(self):
        meta = fd.SendRecvMeta()
        t = paddle.zeros([2, 3], dtype="float64")
        meta.set_send_message(t)
        self.assertEqual(list(meta.send_shape_message), [2, 3])
        # Reconstruct via the recv-side helper: float64 must come back as such.
        self.assertEqual(number_2_dtype(meta.send_dtype_message), "float64")

    def test_tuple_drops_stop_gradient_and_reconstructs_each(self):
        meta = fd.SendRecvMeta()
        # Two trainable tensors with distinguishable shapes/dtypes plus one
        # stop_gradient tensor that must be dropped. Distinct shapes make an
        # inverted filter or reordering observable.
        keep_a = paddle.zeros([2, 3], dtype="float32")
        keep_a.stop_gradient = False
        keep_b = paddle.zeros([4, 5], dtype="float64")
        keep_b.stop_gradient = False
        drop = paddle.zeros([7, 7], dtype="float32")
        drop.stop_gradient = True

        meta.set_send_message((keep_a, keep_b, drop))

        self.assertEqual(len(meta.send_shape_message), 2)
        self.assertEqual(
            [list(s) for s in meta.send_shape_message], [[2, 3], [4, 5]]
        )
        self.assertEqual(
            [number_2_dtype(n) for n in meta.send_dtype_message],
            ["float32", "float64"],
        )


if __name__ == "__main__":
    unittest.main()
