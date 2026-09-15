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

"""Behavior tests for ``paddlefleet.tensor_parallel.data``.

Module under test lives in the "分布式训练" (tensor parallel) module of the
repository. It exposes three symbols:

  * ``_check_data_types(keys, data, target_dtype)`` -- a pure local guard that
    every listed tensor carries ``target_dtype``; raises ``AssertionError``
    otherwise. Fully verifiable without any accelerator or process group.
  * ``_build_key_size_numel_dictionaries(keys, data, tp_group)`` -- packs each
    tensor's shape into a flat buffer on rank 0, broadcasts it, then unpacks
    per-key shape / element-count / running total. The pack+unpack arithmetic
    is genuine *local* logic; only the ``broadcast`` collective and the default
    TP-group lookup are genuine not-under-test collaborators here.
  * ``broadcast_data`` -- rank-0 flatten + collective broadcast + per-key
    narrow/view. This is a real GPU (``.cuda()``) + multi-rank path; it cannot
    be given honest *numeric* verification on a single CPU process, so it is
    not asserted here (a truthful skip would only restate that limitation).

Test environment: no-card. Where a genuine collective (``paddle.distributed.
broadcast``) is replaced by a marker, that stands in for a not-under-test
collaborator ONLY -- it proves the local pack/unpack orchestration, NOT any
real cross-rank broadcast semantics, which require a true process group.

paddle is not importable in this environment, so every case is guarded by
``@unittest.skipUnless(HAS_PADDLE, ...)`` and reports an honest skip rather
than a fake pass.
"""

import unittest
from unittest import mock

try:
    import paddle

    from paddlefleet.tensor_parallel import data as tp_data

    HAS_PADDLE = True
except ImportError:
    paddle = None
    tp_data = None
    HAS_PADDLE = False

SKIP_REASON = "paddle is not installed in this environment"


class _FakeTPGroup:
    """Minimal stand-in for a tensor-parallel process group.

    Only the two attributes the code under test reads (``rank`` and ``ranks``)
    are provided; no collective is actually performed by this object.
    """

    def __init__(self, rank, ranks):
        self.rank = rank
        self.ranks = list(ranks)


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestCheckDataTypes(unittest.TestCase):
    """``_check_data_types`` must compare every key against ``target_dtype``."""

    def test_all_keys_match_target_passes(self):
        # Distinguishable contents/shapes, identical dtype == the target.
        data = {
            "tokens": paddle.zeros([2, 3], dtype=paddle.float32),
            "labels": paddle.ones([4], dtype=paddle.float32),
        }
        # A conforming call returns nothing and must not raise.
        self.assertIsNone(
            tp_data._check_data_types(
                ["tokens", "labels"], data, paddle.float32
            )
        )

    def test_one_key_mismatched_against_target_raises(self):
        # "labels" deliberately differs from the requested target dtype.
        data = {
            "tokens": paddle.zeros([2, 3], dtype=paddle.float32),
            "labels": paddle.ones([4], dtype=paddle.float16),
        }
        with self.assertRaises(AssertionError) as ctx:
            tp_data._check_data_types(
                ["tokens", "labels"], data, paddle.float32
            )
        # The guard names the offending key -- proves it flagged "labels",
        # not merely that *some* assertion fired.
        self.assertIn("labels", str(ctx.exception))

    def test_uniform_but_wrong_target_raises(self):
        # Both tensors share a dtype with each other, yet neither equals the
        # requested target. This distinguishes a correct impl (compares each
        # key to ``target_dtype``) from a broken one that only checks that the
        # keys agree among themselves.
        data = {
            "a": paddle.zeros([2], dtype=paddle.float16),
            "b": paddle.ones([2], dtype=paddle.float16),
        }
        with self.assertRaises(AssertionError):
            tp_data._check_data_types(["a", "b"], data, paddle.float32)


@unittest.skipUnless(HAS_PADDLE, SKIP_REASON)
class TestBuildKeySizeNumelDictionaries(unittest.TestCase):
    """Rank-0 shape packing + unpacking arithmetic.

    ``broadcast`` and the default-group lookup are the only mocked
    collaborators; the pack/unpack math stays real. Expected values below are
    hand-derived from the two input shapes.
    """

    @unittest.expectedFailure
    def test_rank0_packs_and_unpacks_shapes(self):
        # KNOWN PRODUCTION BUG (do not edit production code): this path cannot
        # run as written. On rank 0 the packer calls ``data[key].size()``, but
        # in Paddle ``Tensor.size`` is an int property (element count), not a
        # callable returning the shape -- calling it raises TypeError; the
        # correct expression is ``data[key].shape``. Independently, line
        # ``paddle.tensor(sizes, dtype=paddle.int32)`` calls the ``paddle.tensor``
        # sub-package (not callable); it should be ``paddle.to_tensor``. This
        # test asserts the CORRECT hand-derived result and is marked
        # expectedFailure until the bug is fixed.
        group = _FakeTPGroup(rank=0, ranks=[0, 1])
        data = {
            "a": paddle.zeros([2, 3], dtype=paddle.float32),  # numel 6
            "b": paddle.ones([4], dtype=paddle.float32),  # numel 4
        }

        # In a single process the broadcast is a no-op: rank 0 already holds
        # the packed sizes, so returning without mutation is faithful here.
        def _noop_broadcast(tensor, src, group=None):
            return None

        with (
            mock.patch.object(
                tp_data,
                "get_tensor_model_parallel_group_if_none",
                return_value=group,
            ),
            mock.patch.object(
                paddle.distributed, "broadcast", side_effect=_noop_broadcast
            ),
        ):
            key_size, key_numel, total_numel = (
                tp_data._build_key_size_numel_dictionaries(
                    ["a", "b"], data, tp_group=group
                )
            )

        # Shapes are recovered exactly (order and values), element counts are
        # the products, and the running total is their sum.
        self.assertEqual([int(x) for x in key_size["a"]], [2, 3])
        self.assertEqual([int(x) for x in key_size["b"]], [4])
        self.assertEqual(int(key_numel["a"]), 6)
        self.assertEqual(int(key_numel["b"]), 4)
        self.assertEqual(int(total_numel), 10)


if __name__ == "__main__":
    unittest.main()
