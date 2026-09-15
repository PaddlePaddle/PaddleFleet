# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for paddlefleet/utils/distributed.py.

Environment notes:
- The pure helpers (convert_file_size_to_int, dtype_byte_size, reduce_tensor
  slicing) run 无卡 on CPU; they still require the ``paddle`` dependency because
  the production module imports paddle at load time.
- distributed_gather / distributed_allgather perform real collective
  communication. The rank/world-size/group *mapping* around those calls is
  verified 无卡 with a communication stub (allowed by the 分布式训练 rules), and
  the stub tests explicitly declare that no real process group runs. The true
  cross-rank numeric behavior is 多卡-only and is marked @unittest.skip below;
  faking world_size + a mock collective is NOT multi-card numeric evidence.
"""

import unittest
from unittest import mock

import numpy as np
import paddle

from paddlefleet.utils.distributed import (
    convert_file_size_to_int,
    distributed_allgather,
    distributed_gather,
    dtype_byte_size,
    reduce_tensor,
)


class TestConvertFileSizeToInt(unittest.TestCase):
    """convert_file_size_to_int is a pure integer helper (no paddle math).

    Expected values are hand-derived from the unit definitions, never by
    calling the function under test.
    """

    def test_int_passthrough(self):
        # An int input must be returned unchanged.
        self.assertEqual(convert_file_size_to_int(1024), 1024)
        self.assertEqual(convert_file_size_to_int(0), 0)

    def test_binary_units_use_powers_of_two(self):
        self.assertEqual(convert_file_size_to_int("1GiB"), 1073741824)  # 2**30
        self.assertEqual(convert_file_size_to_int("2MiB"), 2097152)  # 2 * 2**20
        self.assertEqual(convert_file_size_to_int("1KiB"), 1024)  # 2**10
        self.assertEqual(convert_file_size_to_int("10GiB"), 10 * 1073741824)

    def test_decimal_byte_units_use_powers_of_ten(self):
        # Upper-case trailing "B" means bytes: no division by 8.
        self.assertEqual(convert_file_size_to_int("1GB"), 1000000000)  # 10**9
        self.assertEqual(convert_file_size_to_int("1MB"), 1000000)  # 10**6
        self.assertEqual(convert_file_size_to_int("1KB"), 1000)  # 10**3

    def test_decimal_bit_units_divide_by_eight(self):
        # Lower-case trailing "b" means bits: divide the byte count by 8.
        self.assertEqual(convert_file_size_to_int("1Gb"), 1000000000 // 8)
        self.assertEqual(convert_file_size_to_int("1Mb"), 1000000 // 8)
        self.assertEqual(convert_file_size_to_int("1Kb"), 1000 // 8)

    def test_case_insensitive_prefix_but_bit_flag_is_case_sensitive(self):
        # The unit prefix is matched case-insensitively via .upper()...
        self.assertEqual(convert_file_size_to_int("1gb"), 1000000000 // 8)
        # ...but only a lower-case final "b" triggers the bit division, so a
        # value ending in upper-case "B" stays in bytes.
        self.assertEqual(convert_file_size_to_int("1GB"), 1000000000)

    def test_gib_takes_priority_over_gb_branch(self):
        # "GiB" upper-cases to "GIB" and must match the binary branch, not the
        # decimal "GB" branch; a wrong branch order would give 5 * 10**9.
        self.assertEqual(convert_file_size_to_int("5GiB"), 5 * (2**30))

    def test_unknown_unit_raises_value_error(self):
        with self.assertRaises(ValueError):
            convert_file_size_to_int("invalid")

    def test_bare_bytes_suffix_is_not_supported(self):
        # A plain "B" suffix matches none of the endswith checks -> ValueError.
        with self.assertRaises(ValueError):
            convert_file_size_to_int("16B")


class TestDtypeByteSize(unittest.TestCase):
    """dtype_byte_size maps a paddle dtype to bytes-per-element."""

    def test_bool_is_one_eighth_byte(self):
        # bool is special-cased to 1/8 byte (one bit).
        self.assertEqual(dtype_byte_size(paddle.bool), 1 / 8)

    def test_float8_dtypes_are_one_byte(self):
        self.assertEqual(dtype_byte_size(paddle.float8_e4m3fn), 1)
        self.assertEqual(dtype_byte_size(paddle.float8_e5m2), 1)

    def test_standard_widths_from_trailing_bit_count(self):
        # bytes = trailing bit width // 8, derived independently here.
        self.assertEqual(dtype_byte_size(paddle.float32), 32 // 8)
        self.assertEqual(dtype_byte_size(paddle.float16), 16 // 8)
        self.assertEqual(dtype_byte_size(paddle.bfloat16), 16 // 8)
        self.assertEqual(dtype_byte_size(paddle.int64), 64 // 8)
        self.assertEqual(dtype_byte_size(paddle.int32), 32 // 8)
        self.assertEqual(dtype_byte_size(paddle.int8), 8 // 8)

    def test_dtype_without_trailing_digits_raises(self):
        # A dtype-like whose str has no trailing digits cannot be parsed. Use a
        # dedicated stand-in with a safe __eq__ so the branch is reached without
        # relying on paddle dtype equality semantics against a string.
        class _FakeDtype:
            def __eq__(self, other):
                return False

            def __str__(self):
                return "fakedtype"

        with self.assertRaises(ValueError):
            dtype_byte_size(_FakeDtype())


class TestReduceTensor(unittest.TestCase):
    """reduce_tensor flattens a tensor and yields buffer-sized slices.

    Slice boundaries are hand-derived: send_size = buffer_size //
    dtype_byte_size(dtype). buffer_size is passed as an int so it is used
    verbatim (int input is returned unchanged by convert_file_size_to_int).
    """

    def test_float32_slice_boundaries_and_contents(self):
        x = paddle.arange(10, dtype="float32")
        # dtype_byte_size(float32) == 4, so send_size = 16 // 4 == 4.
        parts = list(reduce_tensor(x, buffer_size=16))

        indices = [idx for _, idx in parts]
        self.assertEqual(indices, [(0, 4), (4, 8), (8, 10)])

        contents = [p.numpy().tolist() for p, _ in parts]
        self.assertEqual(contents, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
        # Concatenated slices reconstruct the full flattened tensor.
        np.testing.assert_array_equal(
            np.concatenate([p.numpy() for p, _ in parts]),
            np.arange(10, dtype="float32"),
        )

    def test_buffer_size_changes_slice_width(self):
        # A smaller buffer must produce narrower slices, proving the parameter
        # is actually consumed. send_size = 8 // 4 == 2.
        x = paddle.arange(6, dtype="float32")
        parts = list(reduce_tensor(x, buffer_size=8))
        indices = [idx for _, idx in parts]
        self.assertEqual(indices, [(0, 2), (2, 4), (4, 6)])
        contents = [p.numpy().tolist() for p, _ in parts]
        self.assertEqual(contents, [[0, 1], [2, 3], [4, 5]])

    def test_int8_uses_shape_product_and_one_byte_width(self):
        # int8 takes the np.prod(shape) numel branch; dtype_byte_size == 1 so
        # send_size == buffer_size == 4.
        x = paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype="int8")
        parts = list(reduce_tensor(x, buffer_size=4))
        indices = [idx for _, idx in parts]
        self.assertEqual(indices, [(0, 4), (4, 6)])
        contents = [p.numpy().tolist() for p, _ in parts]
        self.assertEqual(contents, [[0, 1, 2, 3], [4, 5]])


class _FakeGroup:
    """Minimal stand-in for a Fleet communication group.

    Only ``ranks`` (group-local index -> global rank) is exercised by the
    distributed_gather dst mapping.
    """

    def __init__(self, ranks):
        self.ranks = list(ranks)


class TestDistributedGatherMapping(unittest.TestCase):
    """无卡 orchestration checks for distributed_gather.

    A communication stub replaces the collective so the rank/group mapping and
    argument forwarding can be observed on CPU. This does NOT run a real
    process group and is NOT multi-card numeric evidence.
    """

    def test_group_local_dst_maps_to_global_rank(self):
        tensor = paddle.zeros([2, 2], dtype="float32")
        group = _FakeGroup([3, 7])
        captured = {}

        def fake_gather(t, out, dst=None, group=None, **kwargs):
            captured["tensor"] = t
            captured["dst"] = dst
            captured["group"] = group
            captured["kwargs"] = kwargs

        # get_rank != dst so is_dst is False and no output buffers are built;
        # the code path still forwards the mapped dst to the collective.
        with (
            mock.patch("paddle.distributed.get_rank", return_value=999),
            mock.patch(
                "paddle.distributed.communication.stream.gather",
                side_effect=fake_gather,
            ),
        ):
            distributed_gather(tensor, dst=1, group=group)

        # dst=1 is a group-local index; it must be mapped through group.ranks.
        self.assertEqual(captured["dst"], 7)
        self.assertIs(captured["group"], group)
        self.assertIs(captured["tensor"], tensor)
        self.assertTrue(captured["kwargs"]["sync_op"])
        self.assertFalse(captured["kwargs"]["use_calc_stream"])

    def test_index_based_mapping_selects_correct_global_rank(self):
        tensor = paddle.zeros([1], dtype="float32")
        group = _FakeGroup([5, 6, 9])
        captured = {}

        with (
            mock.patch("paddle.distributed.get_rank", return_value=999),
            mock.patch(
                "paddle.distributed.communication.stream.gather",
                side_effect=lambda t, out, dst=None, **kw: captured.update(
                    dst=dst
                ),
            ),
        ):
            distributed_gather(tensor, dst=2, group=group)
        # group.ranks[2] == 9, not the raw dst 2.
        self.assertEqual(captured["dst"], 9)

    def test_no_group_forwards_dst_unchanged(self):
        tensor = paddle.zeros([1], dtype="float32")
        captured = {}

        with (
            mock.patch("paddle.distributed.get_rank", return_value=999),
            mock.patch(
                "paddle.distributed.communication.stream.gather",
                side_effect=lambda t, out, dst=None, **kw: captured.update(
                    dst=dst
                ),
            ),
        ):
            distributed_gather(tensor, dst=0, group=None)
        # With no group there is no remapping table, so dst passes through.
        self.assertEqual(captured["dst"], 0)

    def test_container_types_and_keys_are_preserved(self):
        # Recursion dispatch is pure orchestration: tuple stays tuple, dict
        # keeps its keys. Non-dst ranks receive None from the collective.
        t1 = paddle.zeros([1], dtype="float32")
        t2 = paddle.zeros([1], dtype="float32")

        with (
            mock.patch("paddle.distributed.get_rank", return_value=999),
            mock.patch(
                "paddle.distributed.communication.stream.gather",
                return_value=None,
            ),
        ):
            tuple_out = distributed_gather((t1, t2), dst=0, group=None)
            dict_out = distributed_gather({"a": t1, "b": t2}, dst=0, group=None)

        self.assertIsInstance(tuple_out, tuple)
        self.assertEqual(len(tuple_out), 2)
        self.assertEqual([x for x in tuple_out], [None, None])

        self.assertIsInstance(dict_out, dict)
        self.assertEqual(set(dict_out.keys()), {"a", "b"})
        self.assertEqual(dict_out["a"], None)
        self.assertEqual(dict_out["b"], None)


class TestDistributedAllgatherMapping(unittest.TestCase):
    """无卡 orchestration checks for distributed_allgather.

    Verifies that exactly world_size receive buffers are allocated and that the
    input tensor and group are forwarded to the collective. The collective is
    stubbed, so no real cross-rank data movement occurs; this is not multi-card
    numeric evidence.
    """

    def test_allocates_world_size_buffers_and_forwards_args(self):
        tensor = paddle.zeros([2], dtype="float32")
        group = _FakeGroup([0, 1, 2])
        captured = {}

        def fake_all_gather(out_list, t, group=None):
            captured["count"] = len(out_list)
            captured["input"] = t
            captured["group"] = group
            # Fill each buffer with a rank-identifiable value so we can confirm
            # the returned list is the same set of allocated buffers.
            for i, buf in enumerate(out_list):
                buf.set_value(paddle.full_like(buf, float(i + 1)))

        with (
            mock.patch("paddle.distributed.get_world_size", return_value=3),
            mock.patch(
                "paddle.distributed.all_gather", side_effect=fake_all_gather
            ),
        ):
            result = distributed_allgather(tensor, group=group)

        # world_size == 3 -> exactly 3 receive buffers, input and group passed.
        self.assertEqual(captured["count"], 3)
        self.assertIs(captured["input"], tensor)
        self.assertIs(captured["group"], group)
        self.assertEqual(len(result), 3)
        for i, buf in enumerate(result):
            np.testing.assert_array_equal(
                buf.numpy(), np.full((2,), float(i + 1), dtype="float32")
            )

    def test_container_recursion_preserves_list_type(self):
        t1 = paddle.zeros([1], dtype="float32")
        t2 = paddle.zeros([1], dtype="float32")

        with (
            mock.patch("paddle.distributed.get_world_size", return_value=1),
            mock.patch("paddle.distributed.all_gather", return_value=None),
        ):
            out = distributed_allgather([t1, t2], group=None)

        # A list input yields a list of per-element gather results.
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        for entry in out:
            self.assertEqual(len(entry), 1)  # world_size == 1


class TestDistributedCollectivesMultiCard(unittest.TestCase):
    """多卡-only numeric contracts. Cannot be proven 无卡 / single process."""

    @unittest.skip(
        "多卡-only: real cross-rank gather requires a >=2-rank process group. "
        "A proper test launches N ranks, each holding distinguishable content, "
        "and asserts the destination rank receives every rank's tensor in rank "
        "order (and non-dst ranks receive None). Faking world_size + a mock "
        "collective in one process is NOT multi-card numeric evidence."
    )
    def test_gather_collects_all_ranks_in_order(self):
        raise AssertionError("must run under paddle.distributed.launch")

    @unittest.skip(
        "多卡-only: real all_gather requires a >=2-rank process group. A proper "
        "test launches N ranks with distinguishable per-rank content and asserts "
        "every rank receives all buffers in rank order. Single-process buffer "
        "allocation only proves the local world_size mapping, not the transfer."
    )
    def test_allgather_replicates_all_ranks_in_order(self):
        raise AssertionError("must run under paddle.distributed.launch")


if __name__ == "__main__":
    unittest.main()
