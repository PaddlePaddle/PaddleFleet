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

"""Behavior tests for the CPU-observable pure logic in
paddlefleet.context_parallel_utils.

Scope. The flashmask-mode dispatch (FlashMaskContextParallel / the
cp_flashmask_* forward+backward) is covered concurrently by the sibling
test_context_parallel_flashmask_modes.py. This file deliberately restricts
itself to logic that is fully decidable on a single process without a real
process group:

  * mark_context_parallel_parameter_disable_scale_grad / the getter
  * preprocess_index / preprocess_index_dual_chunks (pure index math)
  * the world-size == 1 local fallback of the scatter/gather/reduce-scatter
    helpers (they ``return input_tensor.clone()`` before touching any
    collective -- this only proves the local passthrough, NOT cross-rank
    communication, which belongs to a multi-card test)
  * scatter_contiguous's divisibility guard, which raises before any
    collective is issued
  * scatter_with_padding, a per-rank LOCAL split helper that performs no
    collective at all; we hand-derive each rank's shard. This validates only
    the local shard computation, not cross-rank reassembly.

No collective is ever faked or mocked here (see antipattern #13): every
assertion is on a genuinely local code path. Every expected value below is
hand-derived from the source arithmetic, not read back from the function
under test.
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Honest capability probe. context_parallel_utils imports paddle and
# paddle.nn.functional.flash_attention at module load. Only a genuine missing
# dependency (ImportError) may skip; any other error must surface as a real
# failure instead of a fake pass.
try:
    import paddle

    import paddlefleet.context_parallel_utils as cpu

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    paddle = None
    cpu = None
    _IMPORT_ERROR = exc

_SKIP_MSG = (
    "paddlefleet.context_parallel_utils could not be imported "
    f"(missing dependency): {_IMPORT_ERROR}"
)

# all_gather_balance / reduce_scatter_any_axis_balance do ``import triton`` at
# the very top of the call, before the world-size==1 short circuit, so even
# the local passthrough is unreachable without triton installed.
try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

_NO_TRITON_MSG = "triton is required to reach all_gather_balance/reduce_scatter_any_axis_balance"


class _FakeGroup:
    """A plain stand-in for a distributed Group.

    It carries only ``nranks`` and ``rank``. It is used exclusively on code
    paths that read those two integers and then either return locally
    (nranks == 1) or raise on input validation -- no collective is invoked, so
    this is not a mocked collective (cf. antipattern #13).
    """

    def __init__(self, nranks, rank=0):
        self.nranks = nranks
        self.rank = rank


@unittest.skipUnless(cpu is not None, _SKIP_MSG)
class TestMarkDisableScaleGrad(unittest.TestCase):
    """mark_context_parallel_parameter_disable_scale_grad sets the flag."""

    def test_layer_marks_both_weight_and_bias(self):
        layer = paddle.nn.Linear(4, 4)
        self.assertIsNotNone(layer.bias)  # default Linear has a bias
        cpu.mark_context_parallel_parameter_disable_scale_grad(layer)
        self.assertIs(layer.weight.context_parallel_disable_scale_grad, True)
        self.assertIs(layer.bias.context_parallel_disable_scale_grad, True)

    def test_layer_without_bias_marks_only_weight(self):
        layer = paddle.nn.Linear(4, 4, bias_attr=False)
        self.assertIsNone(layer.bias)
        cpu.mark_context_parallel_parameter_disable_scale_grad(layer)
        self.assertIs(layer.weight.context_parallel_disable_scale_grad, True)

    def test_tensor_marks_flag_true(self):
        t = paddle.zeros([2, 2])
        self.assertFalse(
            getattr(t, "context_parallel_disable_scale_grad", False)
        )
        cpu.mark_context_parallel_parameter_disable_scale_grad(t)
        self.assertIs(t.context_parallel_disable_scale_grad, True)

    def test_invalid_type_raises_typeerror(self):
        with self.assertRaises(TypeError):
            cpu.mark_context_parallel_parameter_disable_scale_grad(
                "not a param"
            )


@unittest.skipUnless(cpu is not None, _SKIP_MSG)
class TestQueryDisableScaleGrad(unittest.TestCase):
    """context_parallel_parameter_disable_scale_grad reads the flag."""

    def test_unmarked_returns_false(self):
        t = paddle.zeros([2, 2])
        self.assertIs(
            cpu.context_parallel_parameter_disable_scale_grad(t), False
        )

    def test_marked_returns_true(self):
        # genuine collaborator: mark via the production setter, then read back
        t = paddle.zeros([2, 2])
        cpu.mark_context_parallel_parameter_disable_scale_grad(t)
        self.assertIs(
            cpu.context_parallel_parameter_disable_scale_grad(t), True
        )


@unittest.skipUnless(cpu is not None, _SKIP_MSG)
class TestPreprocessIndex(unittest.TestCase):
    """preprocess_index: (indices - chunk_id*blocksize) clipped to [0, max]."""

    def test_subtracts_offset_and_clips_both_ends(self):
        # chunk_id=1, blocksize=16 -> subtract 16 -> [-6, 4, 14, 24]
        # clip to [0, 16] -> [0, 4, 14, 16]
        idx = paddle.to_tensor([10, 20, 30, 40], dtype=paddle.int32)
        out = cpu.preprocess_index(
            idx, chunk_id=1, seq_blocksize=16, max_seqlen_q=16
        )
        self.assertEqual(out.tolist(), [0, 4, 14, 16])

    def test_clips_low_values_to_zero(self):
        # chunk_id=2 -> subtract 32 -> [-22, -12, -2, 8] -> clip -> [0, 0, 0, 8]
        idx = paddle.to_tensor([10, 20, 30, 40], dtype=paddle.int32)
        out = cpu.preprocess_index(
            idx, chunk_id=2, seq_blocksize=16, max_seqlen_q=16
        )
        self.assertEqual(out.tolist(), [0, 0, 0, 8])

    def test_clips_high_values_to_max(self):
        # chunk_id=0 -> subtract 0 -> [50, 60, 70, 80] -> clip to 16 -> all 16
        idx = paddle.to_tensor([50, 60, 70, 80], dtype=paddle.int32)
        out = cpu.preprocess_index(
            idx, chunk_id=0, seq_blocksize=16, max_seqlen_q=16
        )
        self.assertEqual(out.tolist(), [16, 16, 16, 16])


@unittest.skipUnless(cpu is not None, _SKIP_MSG)
class TestPreprocessIndexDualChunks(unittest.TestCase):
    """preprocess_index_dual_chunks: per-chunk clip, +max offset on the
    non-zero second chunk, then elementwise maximum of the two."""

    def test_combines_first_and_offset_second(self):
        # idx = [10, 20, 30, 40], first_id=0, second_id=1, blocksize=max=16.
        # first  = clip([10,20,30,40], 0, 16)            = [10, 16, 16, 16]
        # second = clip([10,20,30,40]-16, 0, 16)         = [0, 4, 14, 16]
        # where(second!=0, second+16, second)           = [0, 20, 30, 32]
        # maximum(first, second)                         = [10, 20, 30, 32]
        idx = paddle.to_tensor([10, 20, 30, 40], dtype=paddle.int32)
        out = cpu.preprocess_index_dual_chunks(
            idx,
            chunk_id_first=0,
            chunk_id_second=1,
            seq_blocksize=16,
            max_seqlen_q=16,
        )
        self.assertEqual(out.tolist(), [10, 20, 30, 32])

    def test_second_chunk_fully_clipped_leaves_first(self):
        # second_id=3 -> subtract 48 -> all negative -> clip 0 -> all zero,
        # so the offset branch is skipped and maximum() returns the first chunk.
        # first = clip([10,20,30,40], 0, 16) = [10, 16, 16, 16]
        idx = paddle.to_tensor([10, 20, 30, 40], dtype=paddle.int32)
        out = cpu.preprocess_index_dual_chunks(
            idx,
            chunk_id_first=0,
            chunk_id_second=3,
            seq_blocksize=16,
            max_seqlen_q=16,
        )
        self.assertEqual(out.tolist(), [10, 16, 16, 16])


@unittest.skipUnless(cpu is not None, _SKIP_MSG)
class TestSingleRankLocalFallback(unittest.TestCase):
    """world-size == 1 fallback: each helper returns a fresh clone of the
    input (a distinct object with identical content) before any collective is
    reached. This proves only the local passthrough, not cross-rank behavior.
    """

    def _assert_clone(self, out, src):
        self.assertIsNot(out, src)  # a clone, not the same object handed back
        self.assertEqual(out.tolist(), src.tolist())

    def test_scatter_balance_clones(self):
        x = paddle.arange(8, dtype="float32")
        out = cpu.scatter_balance(x, group=_FakeGroup(nranks=1), axis=0)
        self._assert_clone(out, x)

    def test_reduce_scatter_any_axis_clones(self):
        x = paddle.arange(8, dtype="float32")
        out = cpu.reduce_scatter_any_axis(x, 0, group=_FakeGroup(nranks=1))
        self._assert_clone(out, x)

    def test_scatter_contiguous_clones(self):
        x = paddle.arange(8, dtype="float32")
        out = cpu.scatter_contiguous(x, group=_FakeGroup(nranks=1), axis=0)
        self._assert_clone(out, x)

    def test_all_gather_contiguous_clones(self):
        x = paddle.arange(8, dtype="float32")
        out = cpu.all_gather_contiguous(x, group=_FakeGroup(nranks=1), axis=0)
        self._assert_clone(out, x)

    def test_reduce_scatter_contiguous_clones(self):
        x = paddle.arange(8, dtype="float32")
        out = cpu.reduce_scatter_contiguous(x, 0, group=_FakeGroup(nranks=1))
        self._assert_clone(out, x)


@unittest.skipUnless(cpu is not None, _SKIP_MSG)
@unittest.skipUnless(_HAS_TRITON, _NO_TRITON_MSG)
class TestSingleRankLocalFallbackTriton(unittest.TestCase):
    """Same world-size == 1 clone fallback, for the two helpers that ``import
    triton`` unconditionally at call entry (so triton must be importable even
    to reach the local short circuit)."""

    def test_all_gather_balance_clones(self):
        x = paddle.arange(8, dtype="float32")
        out = cpu.all_gather_balance(x, group=_FakeGroup(nranks=1), axis=0)
        self.assertIsNot(out, x)
        self.assertEqual(out.tolist(), x.tolist())

    def test_reduce_scatter_any_axis_balance_clones(self):
        x = paddle.arange(8, dtype="float32")
        out = cpu.reduce_scatter_any_axis_balance(
            x, 0, group=_FakeGroup(nranks=1)
        )
        self.assertIsNot(out, x)
        self.assertEqual(out.tolist(), x.tolist())


@unittest.skipUnless(cpu is not None, _SKIP_MSG)
class TestScatterContiguousValidation(unittest.TestCase):
    """scatter_contiguous rejects a length that is not divisible by the group
    size. With nranks=2 this guard raises BEFORE any slice/collective, so it
    is a purely local input-validation check (no communication happens)."""

    def test_non_divisible_length_raises_valueerror(self):
        x = paddle.arange(5, dtype="float32")  # 5 % 2 != 0
        with self.assertRaises(ValueError):
            cpu.scatter_contiguous(
                x, group=_FakeGroup(nranks=2, rank=0), axis=0
            )


@unittest.skipUnless(cpu is not None, _SKIP_MSG)
class TestScatterWithPadding(unittest.TestCase):
    """scatter_with_padding is a per-rank LOCAL split+pad helper: it issues no
    collective, so its per-rank shard is fully decidable in one process. We
    hand-derive each rank's output. This validates the local shard math only,
    NOT the cross-rank reassembly done by the matching all_gather.
    """

    def test_axis0_no_pad_rank0(self):
        # total=8, avg=(8+0)//2=4, sections=[4,4]; rank0 -> first 4 elements
        x = paddle.arange(8, dtype="float32")
        out = cpu.scatter_with_padding(x, 0, 0, _FakeGroup(nranks=2, rank=0))
        self.assertEqual(out.tolist(), [0, 1, 2, 3])

    def test_axis0_no_pad_rank1(self):
        x = paddle.arange(8, dtype="float32")
        out = cpu.scatter_with_padding(x, 0, 0, _FakeGroup(nranks=2, rank=1))
        self.assertEqual(out.tolist(), [4, 5, 6, 7])

    def test_axis0_last_rank_gets_padded(self):
        # total=3, num_pad=1, avg=(3+1)//2=2, sections=[2,1], rank_pad=1.
        # rank1 receives [2] then is right-padded by 1 zero -> [2, 0].
        x = paddle.arange(3, dtype="float32")
        out = cpu.scatter_with_padding(x, 1, 0, _FakeGroup(nranks=2, rank=1))
        self.assertEqual(out.tolist(), [2.0, 0.0])

    def test_axis0_first_rank_not_padded(self):
        # same config as above; rank0 gets [0, 1] with no padding applied.
        x = paddle.arange(3, dtype="float32")
        out = cpu.scatter_with_padding(x, 1, 0, _FakeGroup(nranks=2, rank=0))
        self.assertEqual(out.tolist(), [0.0, 1.0])

    def test_rank_beyond_data_gets_zeros(self):
        # total=2, num_pad=2, avg=2; only rank_idx=1 consumes data, so rank1
        # falls into the else branch and gets a zeros(avg_num=2) tensor with
        # stop_gradient cleared.
        x = paddle.arange(2, dtype="float32")
        out = cpu.scatter_with_padding(x, 2, 0, _FakeGroup(nranks=2, rank=1))
        self.assertEqual(out.tolist(), [0.0, 0.0])
        self.assertFalse(out.stop_gradient)

    @unittest.expectedFailure
    def test_axis1_split_is_broken(self):
        """Real bug: scatter_with_padding is broken for axis != 0.

        context_parallel_utils.py:1008 calls
            paddle.split(input_tensor, num_or_sections=split_sections)
        WITHOUT ``axis=axis``, so it always splits along axis 0 regardless of
        the requested ``axis``. context_parallel_utils.py:1012 then builds the
        pad index as ``axis * input_tensor.ndim * 2 + 1`` instead of
        ``axis * 2 + 1``, which indexes out of the length-(2*ndim) pad list for
        any axis >= 1.

        Correct axis=1 behavior on a [2, 8] tensor with nranks=2 would give
        rank0 the first 4 columns, i.e. [[0,1,2,3],[8,9,10,11]]. Production
        instead raises while trying to split the (size-2) axis 0 into sections
        summing to 8. Marked expectedFailure; production is NOT modified.
        """
        x = paddle.arange(16, dtype="float32").reshape([2, 8])
        out = cpu.scatter_with_padding(x, 0, 1, _FakeGroup(nranks=2, rank=0))
        self.assertEqual(out.tolist(), [[0, 1, 2, 3], [8, 9, 10, 11]])


if __name__ == "__main__":
    unittest.main()
