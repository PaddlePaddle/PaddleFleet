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

"""Behavior tests for ``paddlefleet.tensor_parallel.data``.

Repository module map: "分布式训练" (tensor parallel data broadcast helpers).
This file (``_3``) targets branches of ``data.py`` that are NOT the rank-0
happy path of the broadcast pipeline, focusing on:

  * ``_check_data_types`` -- the per-key dtype guard: it must iterate over
    *every* listed key (not just the first), name the offending key in its
    message, accept non-float dtypes, and touch only the keys passed in
    ``keys`` (never the rest of ``data``).
  * ``_build_key_size_numel_dictionaries`` -- the size pack/unpack round trip
    that turns a data dict into ``(key_size, key_numel, total_numel)``.

Environment: these helpers require the real ``paddle`` tensor API. ``paddle``
is not installed in the authoring environment, so the suite is honestly
skipped rather than faked. Distributed collaborators
(``get_tensor_model_parallel_group_if_none`` and ``distributed.broadcast``)
are replaced by a single-process, rank-0, world-size-1 stand-in: with rank 0
as the broadcast source a real broadcast is a no-op, so the local pack/unpack
logic is exercised for real. No real process group is run; multi-rank
broadcast semantics are out of scope here.
"""

import unittest

try:
    import paddle

    from paddlefleet.tensor_parallel import data as data_mod
    from paddlefleet.tensor_parallel.data import (
        _build_key_size_numel_dictionaries,
        _check_data_types,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False


def _single_rank_group():
    """A rank-0, world-size-1 tensor-parallel group stand-in.

    ``ranks == [0]`` and ``rank == 0`` mean the production code takes the
    rank-0 packing path and treats rank 0 as the broadcast source, so a
    no-op broadcast faithfully models the single-process case.
    """

    class _Group:
        rank = 0
        world_size = 1
        ranks = [0]

    return _Group()


@unittest.skipUnless(HAS_PADDLE, "paddle is not installed in this environment")
class TestCheckDataTypes(unittest.TestCase):
    """Real behavior of ``_check_data_types`` (no distributed, no mocks)."""

    def test_mismatch_on_second_key_is_detected_and_named(self):
        # The offending key is deliberately the SECOND one: a guard that only
        # inspected keys[0] would wrongly pass this input.
        data = {
            "a": paddle.zeros([2, 3], dtype="float32"),
            "b": paddle.zeros([2, 3], dtype="float64"),
        }
        with self.assertRaises(AssertionError) as ctx:
            _check_data_types(["a", "b"], data, paddle.float32)
        # The message must identify the actual offender ("b"), not "a".
        message = str(ctx.exception)
        self.assertIn("b", message)
        self.assertNotIn("a has data type", message)

    def test_accepts_all_matching_including_non_float_dtype(self):
        # A dtype other than float32 exercises the equality check against an
        # arbitrary target and confirms every listed key is walked without
        # raising. The helper returns None on success.
        data = {
            "x": paddle.zeros([2, 2], dtype="int64"),
            "y": paddle.zeros([5], dtype="int64"),
        }
        self.assertIsNone(_check_data_types(["x", "y"], data, paddle.int64))

    def test_only_keys_argument_is_inspected(self):
        # A key present in ``data`` but absent from ``keys`` must be ignored,
        # even when its dtype disagrees with the target. This pins the loop to
        # ``keys`` rather than ``data.keys()``.
        data = {
            "a": paddle.zeros([2], dtype="float32"),
            "ignored": paddle.zeros([2], dtype="float64"),
        }
        self.assertIsNone(_check_data_types(["a"], data, paddle.float32))


@unittest.skipUnless(HAS_PADDLE, "paddle is not installed in this environment")
class TestBuildKeySizeNumelDictionaries(unittest.TestCase):
    """Pack/unpack round trip of ``_build_key_size_numel_dictionaries``."""

    def test_round_trip_recovers_sizes_and_numels(self):
        # Rank-0 packs the per-key shapes into the flat size buffer, the mocked
        # no-op broadcast leaves them intact for the single process, and the
        # unpack loop reconstructs exact per-key sizes / numels / running total.
        from unittest import mock

        group = _single_rank_group()
        # Two keys with DISTINCT shapes so the per-key size/numel mapping is
        # discriminating (a swap or off-by-one in the offset stride would move
        # the numbers). "a": 4*2=8 elements, "b": 3 elements -> total 11.
        data = {
            "a": paddle.zeros([4, 2], dtype="float32"),
            "b": paddle.zeros([3], dtype="float32"),
        }

        with (
            mock.patch.object(
                data_mod,
                "get_tensor_model_parallel_group_if_none",
                return_value=group,
            ),
            mock.patch.object(
                paddle.distributed,
                "broadcast",
                side_effect=lambda *a, **k: None,
            ),
        ):
            key_size, key_numel, total_numel = (
                _build_key_size_numel_dictionaries(["a", "b"], data)
            )

        self.assertEqual([int(s) for s in key_size["a"]], [4, 2])
        self.assertEqual([int(s) for s in key_size["b"]], [3])
        self.assertEqual(int(key_numel["a"]), 8)
        self.assertEqual(int(key_numel["b"]), 3)
        self.assertEqual(int(total_numel), 11)


if __name__ == "__main__":
    unittest.main()
