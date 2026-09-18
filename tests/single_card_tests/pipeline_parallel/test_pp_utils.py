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
``paddlefleet.pipeline_parallel.pp_utils.utils``.

Covered (device-independent, no collectives):
  * ``paddle_2_number`` / ``number_2_dtype`` -- the on-wire dtype<->int codec.
  * ``convert_tensor_dict_to_tuple`` / ``convert_tensor_tuple_to_dict`` -- the
    dict<->tuple carrier used to smuggle dict outputs through the tuple-only
    pipeline send/recv path, including the ``.key`` tagging scheme.
  * ``tuple_to_dict_helper`` / ``dict_to_tuple_helper`` -- the thin dispatch
    wrappers that decide whether to invoke the converters.
  * ``profile_pipeline_details`` -- the CPU (no-CUDA) logging branch.

Expected dtype numbers are taken directly from the on-wire protocol
(float16=0, float32=1, float64=2, int32=3, int64=4, bfloat16=5, bool=6) and are
hand-written here, NOT read back from the production ``PADDLE_TO_NUMBER`` /
``NUMBER_TO_DTYPE`` tables, so the test and the code under test do not share a
source of truth.

Out of scope: ``pp_comm_utils.broadcast_data_obj`` and
``init_magic_send_comm_group`` require a real multi-rank pipeline process group
(cross-rank broadcast, subgroup formation); faking ``world_size`` + mocking the
collectives would only prove local orchestration, not the actual cross-rank
transfer, so they are intentionally left to the multi-card suite.
"""

import unittest
from unittest import mock

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils import utils

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    utils = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)

# Independent, hand-written copy of the wire protocol. Do NOT import the
# production tables here -- this is the reference the codec is checked against.
_WIRE_DTYPE_TO_NUMBER = [
    ("float16", 0),
    ("float32", 1),
    ("float64", 2),
    ("int32", 3),
    ("int64", 4),
    ("bfloat16", 5),
    ("bool", 6),
]


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestWireProtocolCodec(unittest.TestCase):
    """``paddle_2_number`` / ``number_2_dtype`` encode the send/recv dtype tag."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_paddle_2_number_maps_each_supported_dtype(self):
        # Each real paddle dtype must encode to its hand-known protocol int.
        for name, number in _WIRE_DTYPE_TO_NUMBER:
            dtype = getattr(paddle, name)
            self.assertEqual(
                utils.paddle_2_number(dtype),
                number,
                msg=f"{name} should encode to {number}",
            )

    def test_number_2_dtype_maps_each_number(self):
        # Each protocol int must decode to the hand-known dtype *name string*.
        for name, number in _WIRE_DTYPE_TO_NUMBER:
            self.assertEqual(utils.number_2_dtype(number), name)

    def test_number_2_dtype_returns_str_not_paddle_dtype(self):
        # Contract: the decoder yields a plain name string (used to build a
        # paddle.empty later), never a paddle.dtype object.
        decoded = utils.number_2_dtype(1)
        self.assertIsInstance(decoded, str)
        self.assertEqual(decoded, "float32")

    def test_roundtrip_dtype_to_number_to_name(self):
        # paddle_2_number then number_2_dtype recovers the dtype's name for
        # every supported dtype -- a swapped/duplicated table row would break
        # at least one of these pairs.
        for name, _ in _WIRE_DTYPE_TO_NUMBER:
            dtype = getattr(paddle, name)
            self.assertEqual(
                utils.number_2_dtype(utils.paddle_2_number(dtype)), name
            )

    def test_paddle_2_number_rejects_unsupported_real_dtype(self):
        # uint8 is a genuine paddle dtype that is intentionally absent from the
        # protocol table; encoding it must trip the assert, not return garbage.
        with self.assertRaises(AssertionError):
            utils.paddle_2_number(paddle.uint8)

    def test_number_2_dtype_rejects_out_of_range_number(self):
        with self.assertRaises(AssertionError):
            utils.number_2_dtype(99)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestConvertTensorDictToTuple(unittest.TestCase):
    """``convert_tensor_dict_to_tuple`` flattens a dict into a keyed tuple."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_single_tensors_keep_order_identity_and_get_named(self):
        # Distinguishable contents + distinct dict keys: the flattened tuple
        # must preserve insertion order, hand back the *same* tensor objects,
        # and tag each with its dict key.
        a = paddle.to_tensor([1.0, 2.0])
        b = paddle.to_tensor([3.0, 4.0, 5.0])
        out = utils.convert_tensor_dict_to_tuple({"a": a, "b": b})
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIs(out[0], a)
        self.assertIs(out[1], b)
        self.assertEqual(out[0].key, "a")
        self.assertEqual(out[1].key, "b")

    def test_list_value_expands_with_indexed_keys_in_order(self):
        # A list value must be spread into consecutive entries whose keys carry
        # the "<key> <idx>" index suffix, in list order.
        t0 = paddle.to_tensor([10.0])
        t1 = paddle.to_tensor([11.0])
        t2 = paddle.to_tensor([12.0])
        out = utils.convert_tensor_dict_to_tuple({"x": [t0, t1, t2]})
        self.assertEqual([t.key for t in out], ["x 0", "x 1", "x 2"])
        self.assertEqual(
            [t.numpy().tolist() for t in out], [[10.0], [11.0], [12.0]]
        )

    def test_mixed_single_and_list_preserve_full_layout(self):
        s = paddle.to_tensor([7.0])
        l0 = paddle.to_tensor([8.0])
        l1 = paddle.to_tensor([9.0])
        out = utils.convert_tensor_dict_to_tuple({"solo": s, "pair": [l0, l1]})
        self.assertEqual([t.key for t in out], ["solo", "pair 0", "pair 1"])
        self.assertIs(out[0], s)
        self.assertIs(out[1], l0)
        self.assertIs(out[2], l1)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestConvertTensorTupleToDict(unittest.TestCase):
    """``convert_tensor_tuple_to_dict`` rebuilds the dict from ``.key`` tags."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_plain_keys_rebuild_dict_and_strip_key_attr(self):
        # Keys with no space map straight to single-tensor dict entries; the
        # transient ``.key`` tag must be removed afterwards.
        a = paddle.to_tensor([1.0])
        b = paddle.to_tensor([2.0])
        a.key = "a"
        b.key = "b"
        out = utils.convert_tensor_tuple_to_dict((a, b))
        self.assertEqual(set(out), {"a", "b"})
        self.assertIs(out["a"], a)
        self.assertIs(out["b"], b)
        self.assertFalse(hasattr(a, "key"))
        self.assertFalse(hasattr(b, "key"))

    def test_indexed_keys_group_into_ordered_list(self):
        # "<key> <idx>" tagged tensors must collapse back into a single list
        # value, preserving the encounter order.
        t0 = paddle.to_tensor([20.0])
        t1 = paddle.to_tensor([21.0])
        t0.key = "g 0"
        t1.key = "g 1"
        out = utils.convert_tensor_tuple_to_dict((t0, t1))
        self.assertEqual(list(out), ["g"])
        self.assertEqual(len(out["g"]), 2)
        self.assertIs(out["g"][0], t0)
        self.assertIs(out["g"][1], t1)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDictTupleRoundTrip(unittest.TestCase):
    """The two converters are meant to be mutual inverses across send/recv."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_roundtrip_preserves_structure_and_values(self):
        # dict -> tuple -> dict must return the original grouping, keys, and
        # tensor identities for ordinary (space-free) keys.
        solo = paddle.to_tensor([1.0, 2.0])
        p0 = paddle.to_tensor([3.0])
        p1 = paddle.to_tensor([4.0])
        original = {"solo": solo, "pair": [p0, p1]}
        tup = utils.convert_tensor_dict_to_tuple(original)
        back = utils.convert_tensor_tuple_to_dict(tup)
        self.assertEqual(list(back), ["solo", "pair"])
        self.assertIs(back["solo"], solo)
        self.assertEqual(
            [t.numpy().tolist() for t in back["pair"]], [[3.0], [4.0]]
        )
        self.assertIs(back["pair"][0], p0)
        self.assertIs(back["pair"][1], p1)

    @unittest.expectedFailure
    def test_key_with_space_should_roundtrip_losslessly(self):
        # BUG: convert_tensor_tuple_to_dict (utils.py:97-110, line 102
        # ``real_key, _ = key.split(" ")``) reuses the space character both as
        # the list-index separator and as an ordinary character in the key.
        # A single-tensor value under a key that itself contains a space is
        # therefore mis-parsed: {"weird key": t} round-trips to {"weird": [t]}
        # -- the key is truncated AND the value is wrongly wrapped in a list.
        # The correct behavior is a lossless round-trip; asserting it here
        # fails today, so this is marked expectedFailure to flag the defect
        # without editing production code.
        t = paddle.to_tensor([1.0, 2.0])
        tup = utils.convert_tensor_dict_to_tuple({"weird key": t})
        back = utils.convert_tensor_tuple_to_dict(tup)
        self.assertEqual(list(back), ["weird key"])
        self.assertIs(back["weird key"], t)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestTupleToDictHelper(unittest.TestCase):
    """``tuple_to_dict_helper`` only converts when the payload is key-tagged."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_single_tensor_without_key_is_passed_through(self):
        # No ``.key`` -> plain tensor recv path: returned unchanged, use_dict
        # False, and no dict conversion attempted.
        t = paddle.to_tensor([1.0, 2.0])
        out, use_dict = utils.tuple_to_dict_helper(t)
        self.assertFalse(use_dict)
        self.assertIs(out, t)

    def test_tuple_without_keys_is_passed_through(self):
        a = paddle.to_tensor([1.0])
        b = paddle.to_tensor([2.0])
        payload = (a, b)
        out, use_dict = utils.tuple_to_dict_helper(payload)
        self.assertFalse(use_dict)
        self.assertIs(out, payload)

    def test_tuple_with_keys_is_converted_to_dict(self):
        # First element carries a ``.key`` -> the helper must route through the
        # converter and yield the reconstructed dict.
        a = paddle.to_tensor([5.0])
        b = paddle.to_tensor([6.0])
        a.key = "a"
        b.key = "b"
        out, use_dict = utils.tuple_to_dict_helper((a, b))
        self.assertTrue(use_dict)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"a", "b"})
        self.assertIs(out["a"], a)
        self.assertIs(out["b"], b)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDictToTupleHelper(unittest.TestCase):
    """``dict_to_tuple_helper`` converts only dict outputs, else passes through."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_dict_output_is_converted_to_keyed_tuple(self):
        a = paddle.to_tensor([1.0])
        b = paddle.to_tensor([2.0])
        out = utils.dict_to_tuple_helper({"a": a, "b": b})
        self.assertIsInstance(out, tuple)
        self.assertEqual([t.key for t in out], ["a", "b"])
        self.assertIs(out[0], a)
        self.assertIs(out[1], b)

    def test_single_tensor_output_is_passed_through(self):
        t = paddle.to_tensor([1.0, 2.0])
        out = utils.dict_to_tuple_helper(t)
        self.assertIs(out, t)

    def test_tuple_output_is_passed_through(self):
        payload = (paddle.to_tensor([1.0]), paddle.to_tensor([2.0]))
        out = utils.dict_to_tuple_helper(payload)
        self.assertIs(out, payload)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestProfilePipelineDetails(unittest.TestCase):
    """``profile_pipeline_details`` CPU (no-CUDA) logging branch."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_cpu_branch_logs_message_with_zero_memory(self):
        # Force the non-CUDA branch so the numbers are deterministic, then
        # assert the *exact* string handed to the logger: the caller message
        # must be embedded and both memory figures pinned to 0.00.
        logger = mock.MagicMock()
        with (
            mock.patch.object(
                paddle.base.core, "is_compiled_with_cuda", return_value=False
            ),
            mock.patch.object(utils, "get_sync_logger", return_value=logger),
        ):
            utils.profile_pipeline_details("stage-marker")
        logger.info.assert_called_once()
        (logged,) = logger.info.call_args[0]
        self.assertEqual(
            logged,
            "stage-marker: memory_allocated_size=0.00, "
            "memory_reserved_size=0.00",
        )


if __name__ == "__main__":
    unittest.main()
