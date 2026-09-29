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

"""CPU-only behavior tests for the pure logic of PaddleFleet's pipeline
parallel p2p_communication module.

Scope note (distributed module rules): the real send/recv, batched p2p and
_p2p_helper paths depend on a real pipeline process group and cross-rank
exchange. Those are NOT exercised here and MUST be validated with a real
multi-card process group. This file only pins the CPU-executable pure logic:
dtype<->number mapping, partial send/recv eligibility, SendRecvMeta message
derivation / caching / reset, P2PonCalcStream op validation, and P2pHelper
construction. Expected values are derived independently by hand.
"""

import unittest

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
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_IMPORT_OK = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle / paddlefleet.pipeline_parallel.pp_utils.p2p_communication is not "
    f"importable on this CPU host (honest skip, not a pass): {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestDtypeNumberMapping(unittest.TestCase):
    """number_2_dtype / paddle_2_number are collaborators of the meta
    serialization path. Reference table is written by hand here, independent
    of the module's own PADDLE_TO_NUMBER / NUMBER_TO_DTYPE dicts."""

    def test_paddle_2_number_exact_and_inverse(self):
        # Independent hand-written reference (not imported from production).
        ref_num = {
            paddle.float16: 0,
            paddle.float32: 1,
            paddle.float64: 2,
            paddle.int32: 3,
            paddle.int64: 4,
            paddle.bfloat16: 5,
            paddle.bool: 6,
        }
        ref_name = {
            0: "float16",
            1: "float32",
            2: "float64",
            3: "int32",
            4: "int64",
            5: "bfloat16",
            6: "bool",
        }
        for dt, num in ref_num.items():
            self.assertEqual(paddle_2_number(dt), num)
            # round-trip: the number must map back to the matching dtype name.
            self.assertEqual(number_2_dtype(num), ref_name[num])

    def test_unknown_dtype_and_number_rejected(self):
        # complex64 is a real paddle dtype deliberately absent from the table.
        with self.assertRaises(AssertionError):
            paddle_2_number(paddle.complex64)
        with self.assertRaises(AssertionError):
            number_2_dtype(99)


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestIsValidSendRecvPartial(unittest.TestCase):
    """_is_valid_send_recv_partial gates partial (model-parallel) send/recv.
    Pure arithmetic on numel and mp_degree plus the module-global toggle."""

    def test_divisibility_and_degree_gate(self):
        t12 = paddle.ones([3, 4], dtype="float32")  # numel == 12
        # degree > 1 and numel % degree == 0 -> True
        self.assertTrue(_is_valid_send_recv_partial(t12, 4))
        self.assertTrue(_is_valid_send_recv_partial(t12, 3))
        # not divisible -> False
        self.assertFalse(_is_valid_send_recv_partial(t12, 5))
        # degree == 1 is not "partial" even though 12 % 1 == 0
        self.assertFalse(_is_valid_send_recv_partial(t12, 1))

    def test_zero_element_rejected(self):
        empty = paddle.zeros([0], dtype="float32")  # numel == 0
        with self.assertRaises(AssertionError):
            _is_valid_send_recv_partial(empty, 2)

    def test_disabled_flag_short_circuits(self):
        # Global toggle: save + restore so we never pollute other tests.
        orig = p2p_mod._enable_partial_send_recv
        self.addCleanup(setattr, p2p_mod, "_enable_partial_send_recv", orig)
        p2p_mod._enable_partial_send_recv = False
        t12 = paddle.ones([3, 4], dtype="float32")
        # Disabled -> always False regardless of divisibility ...
        self.assertFalse(_is_valid_send_recv_partial(t12, 4))
        # ... and it short-circuits before the zero-element assertion.
        self.assertFalse(
            _is_valid_send_recv_partial(paddle.zeros([0], dtype="float32"), 2)
        )


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestSendRecvMetaMessage(unittest.TestCase):
    """SendRecvMeta derives send-side shape/dtype/key metadata, caches it,
    validates subsequent tensors against it, and can reset."""

    def test_obtain_single_tensor_message(self):
        meta = SendRecvMeta()
        t = paddle.ones([2, 3], dtype="float32")
        shape, dtype_num, key = meta._obtain_send_message(t)
        self.assertEqual(shape, [2, 3])
        self.assertEqual(dtype_num, 1)  # float32 -> 1 (hand-derived)
        self.assertIsNone(key)

    def test_obtain_single_tensor_message_with_key(self):
        meta = SendRecvMeta()
        t = paddle.ones([5], dtype="int64")
        t.key = "hidden_states"
        shape, dtype_num, key = meta._obtain_send_message(t)
        self.assertEqual(shape, [5])
        self.assertEqual(dtype_num, 4)  # int64 -> 4
        self.assertEqual(key, "hidden_states")

    def test_obtain_tuple_skips_stop_gradient_members(self):
        # Distinguishable shapes/dtypes so a failure to skip the
        # stop_gradient member would change the observed tuples.
        meta = SendRecvMeta()
        a = paddle.ones([2, 3], dtype="float32")
        a.stop_gradient = False
        b = paddle.ones([4, 5], dtype="int64")
        b.stop_gradient = True  # must be dropped from the message
        c = paddle.ones([6], dtype="float16")
        c.stop_gradient = False

        shapes, dtypes, keys = meta._obtain_send_message((a, b, c))
        self.assertEqual(shapes, ([2, 3], [6]))  # b's [4, 5] excluded
        self.assertEqual(dtypes, (1, 0))  # float32 -> 1, float16 -> 0
        self.assertEqual(keys, (None, None))

    def test_set_and_check_send_message_roundtrip(self):
        meta = SendRecvMeta()
        meta.set_send_message(paddle.ones([2, 3], dtype="float32"))
        self.assertEqual(meta.send_shape_message, [2, 3])
        self.assertEqual(meta.send_dtype_message, 1)
        self.assertIsNone(meta.send_key_message)

        # Matching tensor: check passes silently.
        meta.check_send_message(paddle.ones([2, 3], dtype="float32"))

        # Shape mismatch is a hard contract violation.
        with self.assertRaises(AssertionError):
            meta.check_send_message(paddle.ones([2, 4], dtype="float32"))
        # Dtype mismatch too.
        with self.assertRaises(AssertionError):
            meta.check_send_message(paddle.ones([2, 3], dtype="float16"))

    def test_check_send_message_noop_when_unset(self):
        # Before any set_send_message, send_shape_message is None and
        # check_send_message returns early without asserting anything.
        meta = SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        meta.check_send_message(paddle.ones([9, 9], dtype="float32"))

    def test_init_or_erase_meta_resets_all_fields(self):
        meta = SendRecvMeta()
        meta.set_send_message(paddle.ones([2, 3], dtype="float32"))
        meta.has_send_meta = True
        meta.has_recv_meta = True
        meta.recv_shape_message = [7]
        meta.recv_dtype_message = 1
        meta.recv_stop_gradient = True
        meta.recv_key_message = "k"

        meta.init_or_erase_meta()

        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.send_key_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertIsNone(meta.recv_stop_gradient)
        self.assertIsNone(meta.recv_key_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestP2PonCalcStream(unittest.TestCase):
    """P2PonCalcStream stores its p2p descriptor and rejects invalid ops."""

    def test_valid_send_op_stores_fields(self):
        t = paddle.ones([2, 2], dtype="float32")
        op = P2PonCalcStream(
            _send_on_calc_stream, t, peer=3, group="grp", nranks=2, rank_id=1
        )
        self.assertIs(op.op, _send_on_calc_stream)
        self.assertIs(op.tensor, t)
        self.assertEqual(op.peer, 3)
        self.assertEqual(op.group, "grp")
        self.assertEqual(op.nranks, 2)
        self.assertEqual(op.rank_id, 1)

    def test_valid_recv_op_defaults(self):
        t = paddle.ones([2, 2], dtype="float32")
        op = P2PonCalcStream(_recv_on_calc_stream, t, 0, "grp")
        self.assertIs(op.op, _recv_on_calc_stream)
        self.assertEqual(op.nranks, 1)  # default
        self.assertEqual(op.rank_id, 0)  # default

    def test_invalid_op_rejected(self):
        t = paddle.ones([2, 2], dtype="float32")
        with self.assertRaises(RuntimeError):
            P2PonCalcStream(lambda *a, **k: None, t, 0, "grp")


@unittest.skipUnless(_IMPORT_OK, _SKIP_REASON)
class TestP2pHelperConstruction(unittest.TestCase):
    """P2pHelper construction flags and clear_meta_cache delegation."""

    def test_default_construction(self):
        helper = P2pHelper()
        self.assertTrue(helper._use_cache)
        self.assertFalse(helper._dynamic_shape)
        self.assertIsInstance(helper._send_recv_meta, SendRecvMeta)
        # Non-dynamic helpers must not allocate the dynamic meta list.
        self.assertFalse(hasattr(helper, "_send_recv_meta_list"))

    def test_dynamic_shape_construction(self):
        helper = P2pHelper(use_cache=False, dynamic_shape=True)
        self.assertFalse(helper._use_cache)
        self.assertTrue(helper._dynamic_shape)
        self.assertEqual(helper._dynamic_cnt, 0)
        self.assertEqual(helper._send_recv_meta_list, [])

    def test_clear_meta_cache_resets_underlying_meta(self):
        helper = P2pHelper()
        helper._send_recv_meta.set_send_message(
            paddle.ones([2, 3], dtype="float32")
        )
        helper._send_recv_meta.has_send_meta = True
        helper._send_recv_meta.has_recv_meta = True

        helper.clear_meta_cache()

        self.assertIsNone(helper._send_recv_meta.send_shape_message)
        self.assertIsNone(helper._send_recv_meta.send_dtype_message)
        self.assertFalse(helper._send_recv_meta.has_send_meta)
        self.assertFalse(helper._send_recv_meta.has_recv_meta)


if __name__ == "__main__":
    unittest.main()
