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

"""CPU-only behavior tests for device-independent logic in
``paddlefleet.pipeline_parallel.pp_utils.four_directions_p2p_communication``.

This file focuses on two pieces that are genuinely CPU-testable without a real
pipeline process group and that are NOT already covered by the sibling
``test_four_directions_p2p_communication.py`` (which exercises
``_is_valid_send_recv_partial``, ``SendRecvMeta.set_send_message`` and the
``SendRecvMeta`` initial state):

1. The on-wire dtype protocol collaborators ``paddle_2_number`` /
   ``number_2_dtype`` (imported into the module under test and used to encode /
   reconstruct tensor metadata during send/recv). Expected numbers are written
   by hand from the documented protocol, NOT obtained by calling the production
   mapping, so the test and code do not share a source of truth.
2. ``initialize_p2p_groups`` configuration propagation, observed through a real
   downstream consumer (``_is_valid_send_recv_partial`` reads the module global
   ``_enable_partial_send_recv``) and through the timer assignment branch.

The distributed send/recv, ``recv_meta`` / ``send_meta`` over process groups and
the four-direction ``_p2p_helper`` scheduling all require multiple ranks with
distinguishable data; faking ``world_size`` and mocking collectives would only
prove local orchestration, so they are intentionally left to multi-card tests.
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
class TestDtypeWireProtocol(unittest.TestCase):
    """`paddle_2_number` / `number_2_dtype` encode and decode tensor dtypes.

    These are the exact collaborators the meta exchange uses to serialize a
    tensor's dtype to an integer and to rebuild an empty receive buffer. The
    encode and decode tables must stay mutually inverse or a received tensor
    would be allocated with the wrong dtype.
    """

    # Hand-written reference, independent of the production PADDLE_TO_NUMBER /
    # NUMBER_TO_DTYPE tables. (dtype attr name, wire number, decoded name).
    _PROTOCOL = [
        ("float16", 0, "float16"),
        ("float32", 1, "float32"),
        ("float64", 2, "float64"),
        ("int32", 3, "int32"),
        ("int64", 4, "int64"),
        ("bfloat16", 5, "bfloat16"),
        ("bool", 6, "bool"),
    ]

    def test_paddle_2_number_encodes_each_dtype(self):
        for attr, number, _ in self._PROTOCOL:
            dtype = getattr(paddle, attr)
            self.assertEqual(
                p2p.paddle_2_number(dtype),
                number,
                msg=f"{attr} should encode to {number}",
            )

    def test_number_2_dtype_decodes_each_number(self):
        for _, number, decoded in self._PROTOCOL:
            self.assertEqual(
                p2p.number_2_dtype(number),
                decoded,
                msg=f"{number} should decode to {decoded}",
            )

    def test_encode_decode_roundtrip_is_identity(self):
        # Sending a tensor's dtype then rebuilding it must recover the same
        # dtype name; an off-by-one or swapped table entry would break here.
        for attr, _, decoded in self._PROTOCOL:
            dtype = getattr(paddle, attr)
            self.assertEqual(
                p2p.number_2_dtype(p2p.paddle_2_number(dtype)), decoded
            )

    def test_paddle_2_number_rejects_unsupported_dtype(self):
        # int8 is deliberately absent from the protocol table.
        with self.assertRaises(AssertionError):
            p2p.paddle_2_number(paddle.int8)

    def test_number_2_dtype_rejects_unknown_number(self):
        # 7 is one past the last valid wire number (6 -> bool).
        with self.assertRaises(AssertionError):
            p2p.number_2_dtype(7)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestInitializeP2PGroupsConfigPropagation(unittest.TestCase):
    """`initialize_p2p_groups` stores the hcg and partial-send configuration.

    It mutates module globals, so every test snapshots and restores them via
    ``addCleanup`` to avoid leaking a fake hcg / flag into other tests.
    """

    def setUp(self):
        paddle.set_device("cpu")
        orig_hcg = p2p._hcg
        orig_flag = p2p._enable_partial_send_recv
        orig_timers = p2p._timers

        def _restore():
            p2p._hcg = orig_hcg
            p2p._enable_partial_send_recv = orig_flag
            p2p._timers = orig_timers

        self.addCleanup(_restore)

    def _fake_hcg(self):
        hcg = mock.MagicMock(name="hcg")
        # get_p2p_groups is unpacked into exactly four groups by the function.
        hcg.get_p2p_groups.return_value = (
            "send_next",
            "send_prev",
            "recv_next",
            "recv_prev",
        )
        return hcg

    def test_stores_hcg_and_queries_p2p_groups(self):
        hcg = self._fake_hcg()
        p2p.initialize_p2p_groups(hcg, enable_partial_send_recv=True)
        self.assertIs(p2p._hcg, hcg)
        hcg.get_p2p_groups.assert_called_once_with()

    def test_partial_flag_propagates_to_validity_gate(self):
        # Divisible tensor with degree > 1: the gate returns True only when the
        # partial flag configured here is enabled. This observes the flag being
        # consumed downstream rather than merely stored.
        tensor = paddle.zeros([2, 4], dtype="float32")

        p2p.initialize_p2p_groups(
            self._fake_hcg(), enable_partial_send_recv=False
        )
        self.assertFalse(p2p._enable_partial_send_recv)
        self.assertFalse(p2p._is_valid_send_recv_partial(tensor, 4))

        p2p.initialize_p2p_groups(
            self._fake_hcg(), enable_partial_send_recv=True
        )
        self.assertTrue(p2p._enable_partial_send_recv)
        self.assertTrue(p2p._is_valid_send_recv_partial(tensor, 4))

    def test_enable_timer_false_leaves_timers_untouched(self):
        p2p._timers = None
        p2p.initialize_p2p_groups(self._fake_hcg(), enable_timer=False)
        self.assertIsNone(p2p._timers)

    def test_enable_timer_true_assigns_timers_from_helper(self):
        # The timer_helper is not the logic under test; give it a distinguishable
        # return and confirm initialize_p2p_groups assigns exactly that object.
        p2p._timers = None
        sentinel = object()
        with mock.patch.object(
            p2p.timer, "get_timers", return_value=sentinel
        ) as get_timers:
            p2p.initialize_p2p_groups(self._fake_hcg(), enable_timer=True)
        get_timers.assert_called_once_with()
        self.assertIs(p2p._timers, sentinel)


if __name__ == "__main__":
    unittest.main()
