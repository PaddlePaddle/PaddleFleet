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

"""Behavior tests for tensor-parallel data broadcasting helpers (part 2).

Module under test: ``paddlefleet.tensor_parallel.data``. In the repository
module map this belongs to "分布式训练" (TP data plumbing): rank 0 packs the
per-key tensor sizes, broadcasts them to the TP group, then every rank
reconstructs ``key_size`` / ``key_numel`` / ``total_numel`` and finally the
flattened payload is scattered back into per-key tensors.

Scope of this file (distinct from the base data test, which focuses on
``broadcast_data`` end-to-end and simple dtype checks):

* ``_build_key_size_numel_dictionaries`` -- the rank-0 size *packing* and the
  *unpacking* loop: exact per-key sizes, the ``_MAX_DATA_DIM`` stride between
  keys, and the derived element counts for MULTIPLE keys with DISTINGUISHABLE
  shapes (so an off-by-stride or key-swap is caught, not merely a length).
* ``_check_data_types`` -- which key/dtype the raised ``AssertionError``
  actually names when a *later* key in the list is the offender (iteration and
  error-reporting branch), rather than the generic "a mismatch raises".

Environment: 无卡 (CPU / no accelerator). The real collective
``paddle.distributed.broadcast`` is a genuine not-under-test collaborator here
and is stubbed with an identity no-op so the locally packed sizes survive; the
packing/unpacking arithmetic under test runs for real. This exercises only the
single-rank local path -- it does NOT verify real cross-rank communication,
which requires an actual TP process group (see the multi-card suite).

Paddle is imported guardedly: when it is unavailable the whole module is
skipped with an honest reason instead of fake-passing.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import paddle

    from paddlefleet.tensor_parallel.data import (
        _build_key_size_numel_dictionaries,
        _check_data_types,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle (and paddlefleet.tensor_parallel.data) is not importable"


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestBuildKeySizeNumelDictionaries(unittest.TestCase):
    """Rank-0 size packing and the reconstruction of size/numel dictionaries."""

    @staticmethod
    def _identity_broadcast(tensor, src, group=None):
        # Single-rank stand-in for the collective: rank 0 already holds the
        # authoritative sizes, so a no-op leaves the packed tensor intact.
        # Real cross-rank broadcast is NOT exercised here.
        return None

    # This asserts the CORRECT reconstruction: rank 0 packs the per-key shapes
    # into the flat size buffer, the mocked no-op broadcast leaves them intact
    # for the single process, and the unpack loop recovers exact sizes / numels
    # / running total from distinguishable shapes.
    def test_multi_key_distinguishable_shapes(self):
        tp_group = SimpleNamespace(rank=0, ranks=[0])
        # Distinguishable shapes: a stride/offset error or key swap changes the
        # reconstructed sizes, not just a count.
        data = {
            "x": paddle.zeros([2, 3], dtype=paddle.float32),
            "y": paddle.zeros([4, 5, 6], dtype=paddle.float32),
        }

        with (
            mock.patch("paddle.distributed.is_initialized", return_value=True),
            mock.patch(
                "paddle.distributed.broadcast",
                side_effect=self._identity_broadcast,
            ),
        ):
            key_size, key_numel, total_numel = (
                _build_key_size_numel_dictionaries(
                    ["x", "y"], data, tp_group=tp_group
                )
            )

        self.assertEqual([int(s) for s in key_size["x"]], [2, 3])
        self.assertEqual([int(s) for s in key_size["y"]], [4, 5, 6])
        self.assertEqual(int(key_numel["x"]), 6)
        self.assertEqual(int(key_numel["y"]), 120)
        self.assertEqual(int(total_numel), 126)

    def test_key_order_independent_of_alpha_order(self):
        # Sizes must follow the requested key order, not dict/alphabetical order.
        # "b" (2-D) precedes "a" (3-D): a swap would exchange the sizes.
        tp_group = SimpleNamespace(rank=0, ranks=[0])
        data = {
            "a": paddle.zeros([7, 8, 9], dtype=paddle.float32),
            "b": paddle.zeros([2, 3], dtype=paddle.float32),
        }

        with (
            mock.patch("paddle.distributed.is_initialized", return_value=True),
            mock.patch(
                "paddle.distributed.broadcast",
                side_effect=self._identity_broadcast,
            ),
        ):
            key_size, key_numel, total_numel = (
                _build_key_size_numel_dictionaries(
                    ["b", "a"], data, tp_group=tp_group
                )
            )

        self.assertEqual([int(s) for s in key_size["b"]], [2, 3])
        self.assertEqual([int(s) for s in key_size["a"]], [7, 8, 9])
        self.assertEqual(int(key_numel["b"]), 6)
        self.assertEqual(int(key_numel["a"]), 504)
        self.assertEqual(int(total_numel), 510)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestCheckDataTypesReportsOffendingKey(unittest.TestCase):
    """The mismatch branch must name the actual offending key and dtypes."""

    def test_later_key_mismatch_names_that_key(self):
        # First key matches target; the SECOND key is the offender. Verifies the
        # function keeps iterating and reports the real offender, not the first
        # key or a stale value.
        data = {
            "good": paddle.zeros([2, 2], dtype=paddle.float32),
            "bad": paddle.zeros([2, 2], dtype=paddle.float16),
        }
        with self.assertRaises(AssertionError) as ctx:
            _check_data_types(["good", "bad"], data, paddle.float32)

        message = str(ctx.exception)
        self.assertIn("bad", message)
        self.assertNotIn("good has data type", message)
        self.assertIn("float16", message)
        self.assertIn("float32", message)

    def test_all_matching_keys_pass_silently(self):
        # Every key matches the target -> no exception, returns None. Uses two
        # distinct keys so a short-circuit that only checks the first is exposed
        # by pairing with the mismatch test above.
        data = {
            "p": paddle.zeros([3], dtype=paddle.int64),
            "q": paddle.ones([1, 4], dtype=paddle.int64),
        }
        self.assertIsNone(_check_data_types(["p", "q"], data, paddle.int64))


if __name__ == "__main__":
    unittest.main()
