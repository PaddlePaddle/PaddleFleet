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

"""Comparison-aid helpers in ``inspect_util`` that the probe-off tests skip.

Two independent groups, both exercised only when a dump is actually compared:

* ``to_interleave_rope_layout_`` + its ``interleave_rope_segment`` /
  ``interleave_rope_segment_inplace`` wrappers -- rewrite a post-RoPE tensor
  from the training kernel's **half-split** channel layout to sglang's
  **interleaved** layout so the two sides line up. The permutation is
  ``interleaved[2j] == half[j]`` and ``interleaved[2j+1] == half[j+half]``;
  applied to q_pe and k_pe together it leaves ``q . k`` unchanged, so it is
  purely a comparison view.
* ``_print_detail_diff`` -- the per-element breakdown printed when two dumps
  fail to match. Its shape-reconciliation head has three early exits (element
  counts differ -> skip; same count, different shape -> reshape; already
  identical -> report and stop) that the "diverging tensor" tests never reach.

These are pure, kernel-free reshapes/prints, so every check is exact.
"""

from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
import paddle

from paddlefleet.train_infer_consistent_ops.inspect_util import (
    _print_detail_diff,
    interleave_rope_segment,
    interleave_rope_segment_inplace,
    to_interleave_rope_layout_,
)


def _f32(rows):
    return paddle.to_tensor(rows, dtype="float32")


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda(),
    "paddlefleet import chain needs a CUDA build",
)
class TestInterleaveRopeLayout(unittest.TestCase):
    def setUp(self) -> None:
        paddle.set_device("gpu")

    def test_half_split_becomes_interleaved_over_the_whole_dim(self) -> None:
        """``[h0 h1 | h2 h3]`` -> ``[h0 h2 h1 h3]`` (default spans all).

        With half = 2 the permutation pairs channel ``j`` with ``j + half``:
        output slot ``2j`` takes ``half[j]`` and slot ``2j+1`` takes
        ``half[j+half]``.
        """
        out = to_interleave_rope_layout_(_f32([[10, 11, 12, 13]]))
        self.assertEqual(out.numpy().tolist(), [[10, 12, 11, 13]])

    def test_a_leading_nope_head_is_passed_through(self) -> None:
        """``start>0`` leaves ``[..., :start]`` verbatim, permutes the rest.

        This is the ``[nope | rope]`` case: RoPE never touches the nope head,
        so only the ``[start:]`` rope block is rewritten.
        """
        out = to_interleave_rope_layout_(
            _f32([[0, 1, 10, 11, 12, 13]]), start=2
        )
        self.assertEqual(out.numpy().tolist(), [[0, 1, 10, 12, 11, 13]])

    def test_a_trailing_tail_past_end_is_passed_through(self) -> None:
        """``end<width`` leaves ``[..., end:]`` alone, permutes the front."""
        out = to_interleave_rope_layout_(
            _f32([[10, 11, 12, 13, 99, 98]]), start=0, end=4
        )
        self.assertEqual(out.numpy().tolist(), [[10, 12, 11, 13, 99, 98]])

    def test_an_odd_length_segment_is_rejected(self) -> None:
        """Pairing ``(2j, 2j+1)`` needs an even segment, else ``ValueError``.

        An odd-length rope segment has no partner for its last channel.
        """
        with self.assertRaises(ValueError):
            to_interleave_rope_layout_(_f32([[1, 2, 3]]))

    def test_segment_view_returns_only_the_interleaved_rope(self) -> None:
        """``interleave_rope_segment`` drops the nope head, keeps only rope.

        It is a ``pre_save_func`` view: the dump holds just the rope segment
        ``t[..., nope_dim:]`` rewritten to the interleaved layout.
        """
        t = _f32([[100, 101, 10, 11, 12, 13]])
        out = interleave_rope_segment(t, 2)
        self.assertEqual(out.numpy().tolist(), [[10, 12, 11, 13]])

    def test_inplace_wrapper_rebuilds_the_full_width_buffer(self) -> None:
        """``interleave_rope_segment_inplace`` keeps the nope head in place.

        It scatters the interleaved rope block back so the full-width tensor
        is returned with ``[..., :nope_dim]`` untouched -- and via ``concat``,
        never an in-place write into the live buffer.
        """
        t = _f32([[100, 101, 10, 11, 12, 13]])
        out = interleave_rope_segment_inplace(t, 2)
        self.assertEqual(out.numpy().tolist(), [[100, 101, 10, 12, 11, 13]])

    def test_applying_it_to_a_pair_preserves_their_dot_product(self) -> None:
        """The permutation is orthogonal: ``q . k`` is invariant under it.

        Interleaving q_pe and k_pe with the same channel permutation cannot
        change their inner product, which is the guarantee that lets the probe
        use it as a pure comparison aid.
        """
        paddle.seed(0)
        q = paddle.randn([2, 8], dtype="float32")
        k = paddle.randn([2, 8], dtype="float32")
        before = paddle.sum(q * k, axis=-1)
        qi = to_interleave_rope_layout_(q)
        ki = to_interleave_rope_layout_(k)
        after = paddle.sum(qi * ki, axis=-1)
        np.testing.assert_allclose(
            after.numpy(), before.numpy(), rtol=0, atol=1e-5
        )


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda(),
    "paddlefleet import chain needs a CUDA build",
)
class TestPrintDetailDiffShapeHead(unittest.TestCase):
    """The three shape-reconciliation exits of ``_print_detail_diff``."""

    def _emit(self, infer, train):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_detail_diff("tag", 0, 0, infer, train)
        return buf.getvalue()

    def test_a_different_element_count_skips_the_comparison(self) -> None:
        """Unequal ``size`` can't be reshaped into agreement, so it bails."""
        out = self._emit(
            np.zeros((2, 3), np.float32), np.zeros((2, 2), np.float32)
        )
        self.assertIn("skip: element count differs", out)

    def test_same_count_different_shape_is_reshaped(self) -> None:
        """A ``(2,3)`` infer vs ``(6,)`` train reshapes, then matches exactly.

        Same element count but different shape takes the reshape branch; the
        values then agree, so it reports the identical case rather than a diff.
        """
        out = self._emit(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(6, dtype=np.float32),
        )
        self.assertIn("identical element-for-element", out)

    def test_an_exact_match_reports_zero_diffs_and_stops(self) -> None:
        """Bit-identical inputs short-circuit to the identical report."""
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        out = self._emit(arr, arr.copy())
        self.assertIn("identical element-for-element", out)
        self.assertNotIn("skip: element count differs", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
