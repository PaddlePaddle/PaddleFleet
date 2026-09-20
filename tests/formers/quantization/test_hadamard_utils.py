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

"""Behavior tests for paddlefleet.quantization.hadamard_utils.

These exercise the real Fast Walsh-Hadamard Transform entry points and compare
their outputs against an INDEPENDENT reference: an unnormalized Sylvester
Hadamard matrix built with numpy via the recursive block construction
H_{2n} = [[H_n, H_n], [H_n, -H_n]].  That construction shares no code with the
butterfly algorithm in the module under test, so a scale error, a transposed /
mis-ordered row, a dropped block boundary or a broken batch loop is rejected
rather than mirrored.  Small vectors additionally carry expected values derived
purely by hand (e.g. H4 @ [1,2,3,4] = [10, -2, -4, 0]).

Environment: all ops here (reshape, elementwise add/sub, matmul) run on CPU, so
this is valid 无卡/CPU coverage.  It does not exercise any GPU-only kernel.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.quantization.hadamard_utils import (
    apply_hadamard_matmul,
    create_hadamard_matrix,
    hadamard_matmul,
    matmul_hadU,
)
from paddlefleet.utils import infohub


def sylvester_hadamard(n):
    """Independent unnormalized Hadamard matrix (Sylvester ordering).

    Built by recursive Kronecker/block doubling with numpy only; it does NOT
    call any production code, so it is a genuine reference.
    """
    assert n >= 1 and (n & (n - 1)) == 0, "n must be a power of 2"
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


class TestMatmulHadU(unittest.TestCase):
    """matmul_hadU applies an unnormalized FWHT along the last dimension."""

    def test_known_vector_matches_hand_derived(self):
        # H4 @ [1,2,3,4]:
        #   1+2+3+4 = 10 ; 1-2+3-4 = -2 ; 1+2-3-4 = -4 ; 1-2-3+4 = 0
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        out = matmul_hadU(x)
        np.testing.assert_allclose(
            out.numpy(), np.array([10.0, -2.0, -4.0, 0.0]), atol=1e-5
        )

    def test_vector_matches_independent_matrix(self):
        for n in [2, 4, 8, 16]:
            vec = np.arange(1, n + 1, dtype=np.float64)
            expected = sylvester_hadamard(n) @ vec
            x = paddle.to_tensor(vec.astype("float32"))
            out = matmul_hadU(x)
            np.testing.assert_allclose(
                out.numpy(), expected.astype("float32"), atol=1e-4, rtol=1e-5
            )

    def test_batched_rows_transformed_independently(self):
        # Two distinguishable rows; each row must be transformed on its own,
        # with no cross-row leakage.
        rows = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        H = sylvester_hadamard(4)
        expected = rows @ H.T  # H @ row for each row (H symmetric)
        x = paddle.to_tensor(rows.astype("float32"))
        out = matmul_hadU(x)
        self.assertEqual(out.shape, [2, 4])
        np.testing.assert_allclose(
            out.numpy(), expected.astype("float32"), atol=1e-4
        )

    def test_transform_is_involution_up_to_scale(self):
        # H @ H = n * I  =>  applying the transform twice scales by n.
        vec = np.array([2.0, -1.0, 5.0, 3.0, 0.0, -4.0, 1.0, 7.0])
        x = paddle.to_tensor(vec.astype("float32"))
        twice = matmul_hadU(matmul_hadU(x))
        np.testing.assert_allclose(
            twice.numpy(), (len(vec) * vec).astype("float32"), atol=1e-3
        )


class TestCreateHadamardMatrix(unittest.TestCase):
    """create_hadamard_matrix returns the unnormalized Sylvester Hadamard."""

    def test_size_2_exact(self):
        H = create_hadamard_matrix(2, paddle.float32)
        self.assertEqual(H.shape, [2, 2])
        np.testing.assert_array_equal(
            H.numpy(), np.array([[1.0, 1.0], [1.0, -1.0]], dtype="float32")
        )

    def test_size_4_exact(self):
        H = create_hadamard_matrix(4, paddle.float32)
        np.testing.assert_array_equal(
            H.numpy(), sylvester_hadamard(4).astype("float32")
        )

    def test_size_8_matches_independent(self):
        H = create_hadamard_matrix(8, paddle.float32)
        np.testing.assert_array_equal(
            H.numpy(), sylvester_hadamard(8).astype("float32")
        )

    def test_orthogonality(self):
        for n in [2, 4, 8, 16]:
            H = create_hadamard_matrix(n, paddle.float32)
            HHT = paddle.matmul(H, H.T)
            np.testing.assert_allclose(
                HHT.numpy(), (n * np.eye(n)).astype("float32"), atol=1e-3
            )

    def test_float16_dtype_and_content(self):
        # +/-1 entries are exact in float16, so both dtype and values hold.
        H = create_hadamard_matrix(4, paddle.float16)
        self.assertEqual(H.dtype, paddle.float16)
        np.testing.assert_array_equal(
            H.astype("float32").numpy(), sylvester_hadamard(4).astype("float32")
        )


class TestHadamardMatmul(unittest.TestCase):
    """hadamard_matmul does block-wise right/left Hadamard multiplication."""

    def setUp(self):
        self.H4_np = sylvester_hadamard(4)
        self.H4 = paddle.to_tensor(self.H4_np.astype("float32"))

    def test_right_side_matches_x_times_H(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        expected = x @ self.H4_np  # right side: input @ H
        out = hadamard_matmul(
            paddle.to_tensor(x.astype("float32")), "right", self.H4, 4
        )
        self.assertEqual(out.shape, [2, 4])
        np.testing.assert_allclose(
            out.numpy(), expected.astype("float32"), atol=1e-4
        )

    def test_right_side_respects_block_boundaries(self):
        # Width 8 == two independent size-4 blocks. e0 -> row0 of H, e1 in the
        # second block -> row1 of H. A leak across the block boundary changes
        # the result.
        x = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        b0 = x[:, :4] @ self.H4_np
        b1 = x[:, 4:] @ self.H4_np
        expected = np.concatenate([b0, b1], axis=1)
        out = hadamard_matmul(
            paddle.to_tensor(x.astype("float32")), "right", self.H4, 4
        )
        np.testing.assert_allclose(
            out.numpy(), expected.astype("float32"), atol=1e-4
        )
        # Sanity: the two blocks are genuinely different rows of H.
        np.testing.assert_array_equal(
            out.numpy()[0, :4], np.array([1, 1, 1, 1], dtype="float32")
        )
        np.testing.assert_array_equal(
            out.numpy()[0, 4:], np.array([1, -1, 1, -1], dtype="float32")
        )

    def test_left_side_matches_H_times_x(self):
        # left -> H.T @ input; H symmetric so this equals H @ input, applied
        # along the leading (block) dimension.
        x = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        )
        expected = self.H4_np @ x
        out = hadamard_matmul(
            paddle.to_tensor(x.astype("float32")), "left", self.H4, 4
        )
        self.assertEqual(out.shape, [4, 3])
        np.testing.assert_allclose(
            out.numpy(), expected.astype("float32"), atol=1e-4
        )


class TestApplyHadamardMatmul(unittest.TestCase):
    """apply_hadamard_matmul builds+caches the matrix then delegates."""

    def setUp(self):
        # apply_hadamard_matmul mutates the process-global infohub.hadamard
        # cache; save and restore it so tests do not pollute each other.
        self._had_present = "hadamard" in infohub
        self._had_orig = infohub.get("hadamard")
        self.addCleanup(self._restore_infohub)
        infohub.hadamard = None

    def _restore_infohub(self):
        if self._had_present:
            infohub["hadamard"] = self._had_orig
        else:
            infohub.pop("hadamard", None)

    def test_result_matches_independent_reference(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        expected = x @ sylvester_hadamard(4)  # right side
        out = apply_hadamard_matmul(
            paddle.to_tensor(x.astype("float32")), "right", 4
        )
        np.testing.assert_allclose(
            out.numpy(), expected.astype("float32"), atol=1e-4
        )

    def test_cached_matrix_is_correct_and_reused(self):
        x = paddle.to_tensor(np.array([[1.0, 2.0, 3.0, 4.0]]).astype("float32"))
        out1 = apply_hadamard_matmul(x, "right", 4)

        # The cached entry must be the correct independent Hadamard matrix,
        # not merely present.
        self.assertIn(4, infohub.hadamard)
        cached = infohub.hadamard[4]
        np.testing.assert_array_equal(
            cached.numpy(), sylvester_hadamard(4).astype("float32")
        )

        # A second call reuses the same cached object and yields the same
        # result.
        out2 = apply_hadamard_matmul(x, "right", 4)
        self.assertIs(infohub.hadamard[4], cached)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)

    def test_distinct_block_sizes_cached_separately(self):
        x4 = paddle.to_tensor(np.arange(1, 5, dtype="float32").reshape([1, 4]))
        x8 = paddle.to_tensor(np.arange(1, 9, dtype="float32").reshape([1, 8]))
        out4 = apply_hadamard_matmul(x4, "right", 4)
        out8 = apply_hadamard_matmul(x8, "right", 8)

        self.assertIn(4, infohub.hadamard)
        self.assertIn(8, infohub.hadamard)
        np.testing.assert_allclose(
            out4.numpy(),
            (np.arange(1, 5) @ sylvester_hadamard(4)).astype("float32"),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            out8.numpy(),
            (np.arange(1, 9) @ sylvester_hadamard(8)).astype("float32"),
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
