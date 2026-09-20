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

"""Behavior tests for grouped-GEMM m_indices generation.

Target production entry:
    paddlefleet.transformer.moe.fp8_utils
        ExpertsGroupGemmContiguousNode.gen_m_indices

``gen_m_indices`` turns a per-expert token-count vector ``tokens_per_expert``
into the flat ``grouped_layout`` tensor consumed by the contiguous grouped
GEMM kernels: a length ``sum(tokens_per_expert)`` int32 tensor whose i-th
entry is the id of the expert that owns row i. Every downstream deep_gemm /
fp8 grouped call in this module indexes rows through this tensor, so an
off-by-one, a wrong ordering, or a mishandled zero-count expert would send
tokens to the wrong expert's weights.

The function is pure integer bookkeeping (arange + repeat_interleave) and runs
on CPU, so these tests exercise the real production method against expected
layouts derived by hand from the token-ownership definition -- no kernels are
mocked and no expected value is produced by calling the function under test.

CPU-only: this file needs a working Paddle install. When Paddle is not
importable the whole case is skipped with an honest reason rather than faked
as passing.
"""

import types
import unittest

try:
    import paddle

    from paddlefleet.transformer.moe.fp8_utils import (
        ExpertsGroupGemmContiguousNode,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet unavailable in this env
    paddle = None
    ExpertsGroupGemmContiguousNode = None
    _IMPORT_ERROR = repr(exc)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}",
)
class TestGenMIndices(unittest.TestCase):
    """gen_m_indices maps per-expert counts to a flat token->expert layout."""

    @classmethod
    def setUpClass(cls):
        # Keep the pure integer op off any visible accelerator; this logic is
        # device-independent and we only claim to have verified the CPU path.
        paddle.set_device("cpu")

    def _make_node(self):
        """Construct a real node via the real constructor.

        Only ``custom_map.experts`` is consulted on the default
        (use_fp8_mlp=True, moe_deep_gemm=False, moe_expert_fusion=False,
        expert_id=None) construction branch, so a lightweight namespace of
        placeholder experts is a genuine collaborator here -- gen_m_indices
        itself is not mocked or replaced.
        """
        custom_map = types.SimpleNamespace(
            experts=[object(), object(), object()]
        )
        return ExpertsGroupGemmContiguousNode(custom_map)

    def test_maps_each_row_to_owning_expert(self):
        """counts [2, 1, 3] -> hand-derived layout [0,0,1,2,2,2]."""
        node = self._make_node()
        out = node.gen_m_indices([2, 1, 3])
        # Row 0,1 belong to expert 0; row 2 to expert 1; rows 3,4,5 to expert 2.
        self.assertEqual(out.tolist(), [0, 0, 1, 2, 2, 2])
        self.assertEqual(out.dtype, paddle.int32)
        self.assertEqual(out.shape, [6])  # length == sum(counts)

    def test_zero_count_expert_contributes_no_rows(self):
        """An expert with 0 tokens is skipped; neighbours keep their ids.

        counts [2, 0, 3] -> [0,0,2,2,2]: expert 1 owns nothing, and its id
        never appears. A naive arange-per-row implementation would wrongly
        emit a 1 here, so this pins the skip behaviour.
        """
        node = self._make_node()
        out = node.gen_m_indices([2, 0, 3])
        self.assertEqual(out.tolist(), [0, 0, 2, 2, 2])
        self.assertNotIn(1, out.tolist())
        self.assertEqual(out.dtype, paddle.int32)

    def test_leading_and_trailing_zero_counts(self):
        """counts [0, 3, 0, 1] -> [1,1,1,3]; boundary zeros drop out."""
        node = self._make_node()
        out = node.gen_m_indices([0, 3, 0, 1])
        self.assertEqual(out.tolist(), [1, 1, 1, 3])
        self.assertEqual(out.shape, [4])

    def test_single_expert_owns_all_rows(self):
        """counts [4] -> [0,0,0,0]: every row maps to the only expert."""
        node = self._make_node()
        out = node.gen_m_indices([4])
        self.assertEqual(out.tolist(), [0, 0, 0, 0])
        self.assertEqual(out.dtype, paddle.int32)

    def test_empty_counts_returns_empty_int32(self):
        """No experts -> empty int32 tensor of length 0 (documented guard)."""
        node = self._make_node()
        out = node.gen_m_indices([])
        self.assertEqual(out.shape, [0])
        self.assertEqual(out.tolist(), [])
        self.assertEqual(out.dtype, paddle.int32)

    def test_accepts_tensor_counts_and_casts_to_int32(self):
        """Tensor input path yields the same layout and int32 output.

        counts tensor [1, 2, 1] -> [0,1,1,2]. Passing int64 also checks the
        documented cast("int32") on the tensor branch.
        """
        node = self._make_node()
        counts = paddle.to_tensor([1, 2, 1], dtype="int64")
        out = node.gen_m_indices(counts)
        self.assertEqual(out.tolist(), [0, 1, 1, 2])
        self.assertEqual(out.dtype, paddle.int32)

    def test_length_matches_total_tokens(self):
        """Independent invariant: layout length == total dispatched tokens."""
        node = self._make_node()
        counts = [3, 0, 2, 5, 1]
        out = node.gen_m_indices(counts)
        self.assertEqual(out.shape[0], sum(counts))
        # Non-decreasing: contiguous grouped layout never revisits an expert.
        ids = out.tolist()
        self.assertEqual(ids, sorted(ids))
        # Exactly the experts with a positive count appear.
        self.assertEqual(
            sorted(set(ids)),
            [i for i, c in enumerate(counts) if c > 0],
        )


if __name__ == "__main__":
    unittest.main()
