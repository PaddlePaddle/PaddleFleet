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

"""CPU-only behavior tests for the pure logic in
``paddlefleet.pipeline_parallel.pp_utils.four_directions_p2p_communication``.

Only the device-independent, non-communicating helpers are exercised here:
``_is_valid_send_recv_partial`` (partial-send eligibility rule) and
``SendRecvMeta.set_send_message`` (shape/dtype metadata capture and the
stop_gradient filtering rule), plus ``SendRecvMeta`` initial state.

The distributed send/recv, meta exchange over process groups, and the
four-direction ``_p2p_helper`` scheduling all require a real pipeline process
group and are intentionally NOT covered here: faking ``world_size`` and mocking
collectives would only prove local orchestration, not cross-rank semantics.

Expected dtype numbers are taken directly from the on-wire protocol
(float16=0, float32=1, float64=2, int32=3, int64=4, bfloat16=5, bool=6) and are
NOT produced by calling the production ``paddle_2_number`` helper, so the test
and the code under test do not share a source of truth.
"""

import unittest
from unittest import mock

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils import (
        four_directions_p2p_communication as p2p,
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


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestIsValidSendRecvPartial(unittest.TestCase):
    """`_is_valid_send_recv_partial(tensor, mp_degree)` gating rule.

    Contract: returns False if partial send/recv is disabled; asserts the
    tensor is non-empty; otherwise returns True iff ``mp_degree > 1`` AND the
    element count is divisible by ``mp_degree``.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def _tensor(self, shape):
        return paddle.zeros(shape, dtype="float32")

    def test_divisible_and_degree_gt_one_is_true(self):
        # numel = 2 * 4 = 8; 8 % 4 == 0 and 4 > 1 -> valid partial.
        t = self._tensor([2, 4])
        self.assertTrue(p2p._is_valid_send_recv_partial(t, 4))

    def test_not_divisible_is_false(self):
        # numel = 8; 8 % 3 == 2 -> not evenly splittable across ranks.
        t = self._tensor([2, 4])
        self.assertFalse(p2p._is_valid_send_recv_partial(t, 3))

    def test_degree_one_is_false_even_when_divisible(self):
        # numel = 8; 8 % 1 == 0, so only the ``mp_degree > 1`` guard makes this
        # False. Dropping that guard would wrongly return True here.
        t = self._tensor([2, 4])
        self.assertFalse(p2p._is_valid_send_recv_partial(t, 1))

    def test_zero_element_raises_assertion(self):
        # numel == 0 must trip the explicit "can't send/recv zero element".
        t = self._tensor([0])
        with self.assertRaises(AssertionError):
            p2p._is_valid_send_recv_partial(t, 2)

    def test_disabled_flag_forces_false_and_is_restored(self):
        # With the divisible + degree>1 tensor the enabled path returns True,
        # so this isolates the ``_enable_partial_send_recv`` gate.
        t = self._tensor([2, 4])
        with mock.patch.object(p2p, "_enable_partial_send_recv", False):
            self.assertFalse(p2p._is_valid_send_recv_partial(t, 4))
        # patch.object restores the module global; the gate really governed
        # the result, so the re-enabled path is True again.
        self.assertTrue(p2p._is_valid_send_recv_partial(t, 4))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSendRecvMetaSetSendMessage(unittest.TestCase):
    """`SendRecvMeta.set_send_message` shape/dtype capture and filtering."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_single_tensor_records_shape_and_protocol_dtype(self):
        meta = p2p.SendRecvMeta()
        t = paddle.zeros([2, 3], dtype="float32")
        meta.set_send_message(t)
        self.assertEqual(list(meta.send_shape_message), [2, 3])
        # float32 -> 1 by the wire protocol (independent of paddle_2_number).
        self.assertEqual(meta.send_dtype_message, 1)

    def test_tuple_filters_stop_gradient_and_keeps_survivor(self):
        meta = p2p.SendRecvMeta()
        # Distinguishable shapes so an inverted filter would be caught: only
        # the non-stop_gradient tensor's [4, 5] must survive, not [2, 3].
        t_drop = paddle.zeros([2, 3], dtype="float32")
        t_drop.stop_gradient = True
        t_keep = paddle.zeros([4, 5], dtype="float32")
        t_keep.stop_gradient = False
        meta.set_send_message((t_drop, t_keep))
        self.assertEqual(len(meta.send_shape_message), 1)
        self.assertEqual([list(s) for s in meta.send_shape_message], [[4, 5]])
        self.assertEqual(meta.send_dtype_message, (1,))

    def test_tuple_preserves_order_and_per_element_dtype(self):
        meta = p2p.SendRecvMeta()
        # Two trainable tensors with distinct shapes and dtypes: order and the
        # per-element dtype mapping must both be preserved.
        a = paddle.zeros([2, 3], dtype="float32")
        a.stop_gradient = False
        b = paddle.zeros([4, 5, 6], dtype="int64")
        b.stop_gradient = False
        meta.set_send_message((a, b))
        self.assertEqual(
            [list(s) for s in meta.send_shape_message], [[2, 3], [4, 5, 6]]
        )
        # float32 -> 1, int64 -> 4 by the wire protocol.
        self.assertEqual(meta.send_dtype_message, (1, 4))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSendRecvMetaInit(unittest.TestCase):
    """`SendRecvMeta` starts with empty, unsent/unreceived metadata."""

    def test_initial_state(self):
        meta = p2p.SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertIsNone(meta.recv_stop_gradient)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)


if __name__ == "__main__":
    unittest.main()
