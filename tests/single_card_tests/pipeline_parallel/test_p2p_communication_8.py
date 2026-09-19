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

"""CPU-only behavior tests for the pure logic in paddlefleet's own
pipeline_parallel p2p_communication module.

Scope note (why these particular targets):
    Real point-to-point send/recv, meta broadcast, partial all-gather and the
    P2pHelper stage orchestration all require a real pipeline process group and
    a live ``_hcg``. Faking ``world_size`` + mocking the collectives to assert
    "was called" would only exercise local plumbing and could not reject a
    wrong peer / wrong split / dropped reduction (see antipattern #13). Those
    behaviors belong in a multi-card test. This file therefore verifies only the
    genuinely CPU-executable, communication-free pure logic that lives in the
    same production files and independently derives every expected value:
      * dtype <-> wire-number mapping (utils.paddle_2_number / number_2_dtype)
      * SendRecvMeta metadata extraction / check (shape/dtype/key, stop_gradient
        filtering of tuples)
      * _is_valid_send_recv_partial predicate
      * tensor dict <-> keyed tuple round-trip helpers
"""

import unittest

try:
    import paddle

    paddle.set_device("cpu")

    from paddlefleet.pipeline_parallel.pp_utils import (
        p2p_communication as p2p,
    )
    from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
        SendRecvMeta,
    )
    from paddlefleet.pipeline_parallel.pp_utils.utils import (
        NUMBER_TO_DTYPE,
        PADDLE_TO_NUMBER,
        convert_tensor_dict_to_tuple,
        convert_tensor_tuple_to_dict,
        dict_to_tuple_helper,
        number_2_dtype,
        paddle_2_number,
        tuple_to_dict_helper,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest, narrow guard
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)

# Independently authored reference mapping (NOT imported from production). If
# the production table drifts, the round-trip test below must fail.
_REFERENCE_DTYPE_TABLE = [
    ("float16", 0),
    ("float32", 1),
    ("float64", 2),
    ("int32", 3),
    ("int64", 4),
    ("bfloat16", 5),
    ("bool", 6),
]


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDtypeNumberMapping(unittest.TestCase):
    """paddle_2_number / number_2_dtype are exact, mutually-inverse encoders."""

    def test_round_trip_matches_independent_reference(self):
        for name, num in _REFERENCE_DTYPE_TABLE:
            dtype = getattr(paddle, name)
            # forward: dtype -> number equals hand-derived number
            self.assertEqual(paddle_2_number(dtype), num)
            # reverse: number -> dtype name equals hand-derived name
            self.assertEqual(number_2_dtype(num), name)
        # the two production tables must have identical, non-overlapping domains
        self.assertEqual(
            sorted(PADDLE_TO_NUMBER.values()),
            sorted(NUMBER_TO_DTYPE.keys()),
        )
        self.assertEqual(
            len(set(PADDLE_TO_NUMBER.values())), len(PADDLE_TO_NUMBER)
        )

    def test_paddle_2_number_rejects_unregistered_dtype(self):
        # int8 is deliberately absent from PADDLE_TO_NUMBER
        self.assertNotIn(paddle.int8, PADDLE_TO_NUMBER)
        with self.assertRaises(AssertionError):
            paddle_2_number(paddle.int8)

    def test_number_2_dtype_rejects_unknown_number(self):
        self.assertNotIn(999, NUMBER_TO_DTYPE)
        with self.assertRaises(AssertionError):
            number_2_dtype(999)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSendRecvMetaExtraction(unittest.TestCase):
    """SendRecvMeta._obtain_send_message / set / check pure metadata logic."""

    def test_single_tensor_reports_shape_dtype_and_no_key(self):
        meta = SendRecvMeta()
        t = paddle.ones([2, 3], dtype="float32")
        shape, dtype, key = meta._obtain_send_message(t)
        self.assertEqual(shape, [2, 3])
        self.assertEqual(dtype, paddle_2_number(paddle.float32))  # == 1
        self.assertEqual(dtype, 1)
        self.assertIsNone(key)

    def test_tuple_skips_stop_gradient_tensors(self):
        # a requires grad and must be reported; b is stop_gradient and must be
        # filtered out. Distinguishable shapes/dtypes so a dropped filter, or a
        # swapped element, would change the exact result below.
        a = paddle.ones([2, 3], dtype="float32")
        a.stop_gradient = False
        b = paddle.ones([5], dtype="int64")
        b.stop_gradient = True

        meta = SendRecvMeta()
        shapes, dtypes, keys = meta._obtain_send_message((a, b))
        self.assertEqual(shapes, ([2, 3],))
        self.assertEqual(dtypes, (paddle_2_number(paddle.float32),))
        self.assertEqual(dtypes, (1,))
        self.assertEqual(keys, (None,))

    def test_tuple_preserves_per_tensor_key(self):
        a = paddle.ones([4], dtype="float32")
        a.stop_gradient = False
        a.key = "hidden_states"
        c = paddle.ones([4], dtype="float32")
        c.stop_gradient = False
        c.key = "router_logits"

        meta = SendRecvMeta()
        shapes, dtypes, keys = meta._obtain_send_message((a, c))
        self.assertEqual(shapes, ([4], [4]))
        self.assertEqual(keys, ("hidden_states", "router_logits"))

    def test_set_then_check_accepts_matching_tensor(self):
        meta = SendRecvMeta()
        meta.set_send_message(paddle.ones([2, 3], dtype="float32"))
        self.assertEqual(meta.send_shape_message, [2, 3])
        self.assertEqual(meta.send_dtype_message, 1)
        self.assertIsNone(meta.send_key_message)
        # a matching tensor must pass check without raising
        meta.check_send_message(paddle.zeros([2, 3], dtype="float32"))

    def test_check_detects_shape_mismatch(self):
        meta = SendRecvMeta()
        meta.set_send_message(paddle.ones([2, 3], dtype="float32"))
        with self.assertRaises(AssertionError):
            meta.check_send_message(paddle.ones([2, 4], dtype="float32"))

    def test_check_detects_dtype_mismatch(self):
        meta = SendRecvMeta()
        meta.set_send_message(paddle.ones([2, 3], dtype="float32"))
        with self.assertRaises(AssertionError):
            meta.check_send_message(paddle.ones([2, 3], dtype="int64"))

    def test_check_is_noop_before_set(self):
        # Before set_send_message, both messages are None and check must be a
        # silent no-op (not a false rejection). Returns None without raising.
        meta = SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(
            meta.check_send_message(paddle.ones([9, 9], dtype="float32"))
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestIsValidSendRecvPartial(unittest.TestCase):
    """_is_valid_send_recv_partial: numel-divisibility predicate over mp_degree."""

    def setUp(self):
        # This module global gates the predicate; save and restore it so a test
        # that flips it cannot pollute sibling tests (antipattern #11).
        self._orig_flag = p2p._enable_partial_send_recv
        self.addCleanup(
            setattr, p2p, "_enable_partial_send_recv", self._orig_flag
        )
        p2p._enable_partial_send_recv = True

    def test_mp_degree_one_is_never_partial(self):
        t = paddle.ones([2, 4], dtype="float32")  # numel 8
        self.assertFalse(p2p._is_valid_send_recv_partial(t, 1))

    def test_divisible_numel_is_partial(self):
        t = paddle.ones([2, 4], dtype="float32")  # numel 8
        self.assertTrue(p2p._is_valid_send_recv_partial(t, 2))
        self.assertTrue(p2p._is_valid_send_recv_partial(t, 4))
        self.assertTrue(p2p._is_valid_send_recv_partial(t, 8))

    def test_indivisible_numel_is_not_partial(self):
        t = paddle.ones([2, 4], dtype="float32")  # numel 8
        self.assertFalse(p2p._is_valid_send_recv_partial(t, 3))
        self.assertFalse(p2p._is_valid_send_recv_partial(t, 5))

    def test_zero_element_tensor_asserts(self):
        t = paddle.zeros([0], dtype="float32")  # numel 0
        with self.assertRaises(AssertionError):
            p2p._is_valid_send_recv_partial(t, 2)

    def test_disabled_flag_short_circuits_to_false(self):
        p2p._enable_partial_send_recv = False
        t = paddle.ones([2, 4], dtype="float32")  # divisible, but disabled
        self.assertFalse(p2p._is_valid_send_recv_partial(t, 2))


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestTensorDictTupleRoundTrip(unittest.TestCase):
    """convert_tensor_dict_to_tuple / convert_tensor_tuple_to_dict and helpers."""

    def _make_dict(self):
        ta = paddle.to_tensor([1.0, 2.0])
        t0 = paddle.to_tensor([3.0, 4.0])
        t1 = paddle.to_tensor([5.0, 6.0])
        return ta, t0, t1, {"a": ta, "b": [t0, t1]}

    def test_dict_to_tuple_flattens_and_assigns_keys(self):
        ta, t0, t1, d = self._make_dict()
        tup = convert_tensor_dict_to_tuple(d)
        # order: single "a" first, then list "b" elements in order
        self.assertEqual(len(tup), 3)
        self.assertIs(tup[0], ta)
        self.assertIs(tup[1], t0)
        self.assertIs(tup[2], t1)
        # keys encode the dict key; list members get " <idx>" suffixes
        self.assertEqual(tup[0].key, "a")
        self.assertEqual(tup[1].key, "b 0")
        self.assertEqual(tup[2].key, "b 1")
        # content untouched
        self.assertEqual(tup[0].tolist(), [1.0, 2.0])
        self.assertEqual(tup[2].tolist(), [5.0, 6.0])

    def test_round_trip_restores_structure_and_strips_keys(self):
        ta, t0, t1, d = self._make_dict()
        tup = convert_tensor_dict_to_tuple(d)
        restored = convert_tensor_tuple_to_dict(tup)

        self.assertEqual(set(restored.keys()), {"a", "b"})
        self.assertIs(restored["a"], ta)
        self.assertEqual(restored["b"], [t0, t1])  # order + identity preserved
        self.assertEqual(restored["a"].tolist(), [1.0, 2.0])
        self.assertEqual(restored["b"][1].tolist(), [5.0, 6.0])
        # keys must be consumed (deleted) on the way back to a dict
        for t in (ta, t0, t1):
            self.assertFalse(hasattr(t, "key"))

    def test_tuple_to_dict_helper_detects_keyed_vs_plain(self):
        _, _, _, d = self._make_dict()
        keyed_tuple = convert_tensor_dict_to_tuple(d)
        out, use_dict = tuple_to_dict_helper(keyed_tuple)
        self.assertTrue(use_dict)
        self.assertEqual(set(out.keys()), {"a", "b"})

        # a plain tuple with no ".key" attribute must be returned unchanged
        plain = (paddle.to_tensor([7.0]), paddle.to_tensor([8.0]))
        out2, use_dict2 = tuple_to_dict_helper(plain)
        self.assertFalse(use_dict2)
        self.assertIs(out2, plain)

    def test_dict_to_tuple_helper_passthrough_for_non_dict(self):
        _, _, _, d = self._make_dict()
        self.assertIsInstance(dict_to_tuple_helper(d), tuple)

        lone = paddle.to_tensor([1.0])
        self.assertIs(dict_to_tuple_helper(lone), lone)


if __name__ == "__main__":
    unittest.main()
