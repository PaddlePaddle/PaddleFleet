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

"""``reduce_scatter_contiguous`` semantics under a simulated CP group.

Both branches must satisfy ``out_r == split(sum_j x_j, nranks, axis)[r]``. A
whole CP world lives in one process: the mocked collectives hold every rank's
input, so they return what a rank would really receive.
"""

import os
import sys

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_project_root, "src"))

import unittest
from unittest import mock

import paddle
import paddle.distributed as dist

NRANKS = 4


def _mock_group(rank):
    group = mock.MagicMock()
    group.nranks = NRANKS
    group.rank = rank
    return group


def _run_rank(xs, rank, axis):
    """Call reduce_scatter_contiguous as ``rank`` would see it in a CP group."""
    from paddlefleet.context_parallel_utils import reduce_scatter_contiguous

    def fake_reduce_scatter(output, input_tensor, op, group, use_calc_stream):
        # NCCL semantics: rank r keeps chunk r of the sum over the whole world.
        flat = [
            x.reshape([-1, *x.shape[axis + 1 :]]).cast("float32") for x in xs
        ]
        total = flat[0]
        for f in flat[1:]:
            total = total + f
        output[:] = paddle.split(total, NRANKS, axis=0)[rank].cast(output.dtype)

    def fake_alltoall(output_list, input_list, group, use_calc_stream):
        # rank r receives chunk r of every rank's split.
        for j, buf in enumerate(output_list):
            buf[:] = paddle.split(xs[j], NRANKS, axis=axis)[rank].cast(
                buf.dtype
            )

    with (
        mock.patch.object(
            dist.stream, "reduce_scatter", side_effect=fake_reduce_scatter
        ),
        mock.patch.object(dist.stream, "alltoall", side_effect=fake_alltoall),
    ):
        return reduce_scatter_contiguous(
            xs[rank], axis=axis, group=_mock_group(rank)
        )


def _expected(xs, rank, axis):
    """Ground truth in fp64, independent of the implementation."""
    total = xs[0].cast("float64")
    for x in xs[1:]:
        total = total + x.cast("float64")
    return paddle.split(total, NRANKS, axis=axis)[rank]


class TestReduceScatterContiguousSemantics(unittest.TestCase):
    def _check(self, shape, axis, dtype="float32", tol=1e-5):
        paddle.seed(0)
        xs = [paddle.randn(shape).cast(dtype) for _ in range(NRANKS)]
        for rank in range(NRANKS):
            out = _run_rank(xs, rank, axis)
            want = _expected(xs, rank, axis)
            self.assertEqual(list(out.shape), list(want.shape))
            self.assertEqual(out.dtype, xs[rank].dtype)
            err = float((out.cast("float64") - want).abs().max()) / max(
                float(want.abs().max()), 1e-30
            )
            self.assertLess(err, tol, f"shape={shape} axis={axis} rank={rank}")

    # ---- the flat path (leading dims all 1, axis == 0 included) ----

    def test_axis0_fp32(self):
        self._check([8, 6], axis=0)

    def test_axis0_bf16(self):
        self._check([8, 6], axis=0, dtype="bfloat16", tol=1e-2)

    def test_axis1_leading_one_bf16(self):
        # The production shape family: [b=1, s, n, d] key/value gradients.
        self._check([1, 8, 2, 3], axis=1, dtype="bfloat16", tol=1e-2)

    def test_axis2_leading_ones(self):
        self._check([1, 1, 8, 3], axis=2)

    # ---- the alltoall path (a leading dim != 1 forces it) ----

    def test_axis1_batch_two_takes_slow_path(self):
        self._check([2, 8, 3], axis=1, dtype="bfloat16", tol=1e-2)

    def test_both_paths_agree(self):
        """[1, s, ...] (flat path) vs [2, s, ...] (alltoall) on the same data."""
        paddle.seed(0)
        xs = [paddle.randn([2, 8, 3]).cast("bfloat16") for _ in range(NRANKS)]
        for rank in range(NRANKS):
            slow = _run_rank(xs, rank, axis=1)
            for b in range(2):
                sliced = [x[b : b + 1] for x in xs]
                fast = _run_rank(sliced, rank, axis=1)
                self.assertTrue(
                    paddle.equal_all(
                        fast.cast("float32"),
                        slow[b : b + 1].cast("float32"),
                    ),
                    f"rank={rank} b={b}",
                )

    # ---- fp32 accumulation is what keeps bf16 inputs accurate ----

    def test_fp32_accumulation_beats_bf16(self):
        """The unified fp32 reduce is at least as accurate as a bf16 one."""
        paddle.seed(0)
        xs = [paddle.randn([8, 6]).cast("bfloat16") for _ in range(NRANKS)]
        want = _expected(xs, 0, 0)
        got = _run_rank(xs, 0, axis=0)

        in_bf16 = xs[0].cast("float32")
        for x in xs[1:]:
            in_bf16 = (
                (in_bf16 + x.cast("float32")).cast("bfloat16").cast("float32")
            )
        in_bf16 = paddle.split(in_bf16, NRANKS, axis=0)[0]

        err_fp32 = float((got.cast("float64") - want).abs().max())
        err_bf16 = float((in_bf16.cast("float64") - want).abs().max())
        self.assertLessEqual(err_fp32, err_bf16)

    # ---- guards ----

    def test_indivisible_axis_rejected(self):
        paddle.seed(0)
        xs = [paddle.randn([1, 7, 3]) for _ in range(NRANKS)]
        with self.assertRaises(AssertionError):
            _run_rank(xs, 0, axis=1)

    def test_nranks_one_is_a_clone(self):
        from paddlefleet.context_parallel_utils import (
            reduce_scatter_contiguous,
        )

        group = mock.MagicMock()
        group.nranks = 1
        x = paddle.randn([4, 6])
        out = reduce_scatter_contiguous(x, axis=1, group=group)
        self.assertTrue(paddle.equal_all(out, x))


if __name__ == "__main__":
    unittest.main()
