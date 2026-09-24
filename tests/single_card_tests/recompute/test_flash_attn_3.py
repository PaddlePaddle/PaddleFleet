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

"""Behaviour tests for ``flashattn_auto_cast``.

``flashattn_auto_cast`` is the dtype-normalisation helper that the refined
recompute FlashAttention path (``RefinedRcomputeFlashAttention._first_fwd`` /
``_second_fwd``) runs on the Q/K/V tensors before every kernel call. The
contract that matters for correctness -- and that a rewrite could break without
changing any tensor shape -- is:

  * each of q, k, v is inspected and cast *independently*: a tensor already at
    the target dtype is returned *as-is* (same object, no copy), a tensor at a
    different dtype is replaced by a freshly cast tensor;
  * the numeric content is preserved by the cast (exactly, for values that are
    representable in the target dtype) and never swapped between q/k/v.

These are CPU-observable: ``astype`` runs on CPU, no accelerator is required.
Expected values below are hand-written, never produced by calling the function
under test.
"""

import unittest

# paddlefleet imports paddle at import time, and paddle is not installed in
# every environment (e.g. the no-card lint box). Guard the import honestly and
# skip -- never fake a pass -- when the dependency is genuinely absent.
try:
    import numpy as np
    import paddle

    from paddlefleet.refined_recompute.flash_attn import flashattn_auto_cast

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without paddle
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)
class TestFlashattnAutoCast(unittest.TestCase):
    """Genuine content/identity behaviour of ``flashattn_auto_cast``."""

    def setUp(self):
        # Keep everything on CPU: this helper is pure dtype conversion.
        paddle.set_device("cpu")

    def test_casts_each_tensor_independently_by_dtype(self):
        """Each of q/k/v is cast on its own dtype, not q's dtype alone.

        q differs from target -> new bfloat16 tensor.
        k already IS the target -> returned as the same object (no copy).
        v differs from target -> new bfloat16 tensor.

        An implementation that casts all three unconditionally would fail the
        ``k_out is k`` identity check; one that decides from q's dtype only
        would fail to cast v (or would wrongly copy k).
        """
        q = paddle.ones([2, 3], dtype=paddle.float32)
        k = paddle.ones([2, 3], dtype=paddle.bfloat16)
        v = paddle.ones([2, 3], dtype=paddle.float16)

        q_out, k_out, v_out = flashattn_auto_cast(
            q, k, v, dtype=paddle.bfloat16
        )

        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertIsNot(q_out, q)  # differing dtype -> a new tensor

        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertIs(k_out, k)  # already target dtype -> same object, no copy

        self.assertEqual(v_out.dtype, paddle.bfloat16)
        self.assertIsNot(v_out, v)  # differing dtype -> a new tensor

    def test_preserves_distinct_values_without_swapping(self):
        """Values survive the cast exactly and stay with their own tensor.

        The chosen values (1.0, -2.0, 0.5, 4.0 and scalings thereof) are all
        exactly representable in bfloat16, so a lossless round-trip must return
        the identical numbers. Distinct per-tensor content catches any q/k/v
        mix-up in the return order.
        """
        q = paddle.to_tensor([[1.0, -2.0, 0.5, 4.0]], dtype=paddle.float32)
        k = paddle.to_tensor([[2.0, -4.0, 1.0, 8.0]], dtype=paddle.float32)
        v = paddle.to_tensor([[0.5, -1.0, 0.25, 2.0]], dtype=paddle.float32)

        q_out, k_out, v_out = flashattn_auto_cast(
            q, k, v, dtype=paddle.bfloat16
        )

        # Hand-written expectations (independent of the function under test).
        np.testing.assert_array_equal(
            q_out.astype(paddle.float32).numpy(),
            np.array([[1.0, -2.0, 0.5, 4.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            k_out.astype(paddle.float32).numpy(),
            np.array([[2.0, -4.0, 1.0, 8.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            v_out.astype(paddle.float32).numpy(),
            np.array([[0.5, -1.0, 0.25, 2.0]], dtype=np.float32),
        )

    def test_default_dtype_is_bfloat16(self):
        """Called with no dtype argument, all three are normalised to bfloat16.

        Also confirms the value carries through, so the assertion is not
        satisfied by a dtype change alone.
        """
        q = paddle.to_tensor([[1.0, 4.0]], dtype=paddle.float32)
        k = paddle.to_tensor([[-2.0, 0.5]], dtype=paddle.float32)
        v = paddle.to_tensor([[8.0, -1.0]], dtype=paddle.float32)

        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)

        for out in (q_out, k_out, v_out):
            self.assertEqual(out.dtype, paddle.bfloat16)
        np.testing.assert_array_equal(
            q_out.astype(paddle.float32).numpy(),
            np.array([[1.0, 4.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            k_out.astype(paddle.float32).numpy(),
            np.array([[-2.0, 0.5]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            v_out.astype(paddle.float32).numpy(),
            np.array([[8.0, -1.0]], dtype=np.float32),
        )

    def test_float16_input_is_recast_not_passed_through(self):
        """float16 differs from bfloat16, so it must be recast, not returned.

        Guards against an implementation that only special-cases float32. The
        values are exactly representable in both float16 and bfloat16, so the
        recast round-trip is lossless.
        """
        q = paddle.to_tensor([[1.0, -1.0, 0.5, 2.0]], dtype=paddle.float16)
        k = paddle.to_tensor([[2.0, -2.0, 0.25, 4.0]], dtype=paddle.float16)
        v = paddle.to_tensor([[0.5, -0.5, 1.0, 8.0]], dtype=paddle.float16)

        q_out, k_out, v_out = flashattn_auto_cast(
            q, k, v, dtype=paddle.bfloat16
        )

        for out, src in ((q_out, q), (k_out, k), (v_out, v)):
            self.assertEqual(out.dtype, paddle.bfloat16)
            self.assertIsNot(out, src)  # dtype differs -> new tensor
        np.testing.assert_array_equal(
            q_out.astype(paddle.float32).numpy(),
            np.array([[1.0, -1.0, 0.5, 2.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            k_out.astype(paddle.float32).numpy(),
            np.array([[2.0, -2.0, 0.25, 4.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            v_out.astype(paddle.float32).numpy(),
            np.array([[0.5, -0.5, 1.0, 8.0]], dtype=np.float32),
        )

    def test_same_dtype_returns_all_inputs_unchanged(self):
        """When every tensor already matches, all three are the same objects.

        No spurious copies are made -- a distinct behaviour from the mixed-dtype
        case above, and one a blanket ``astype`` would violate.
        """
        q = paddle.to_tensor([[1.0, 2.0]], dtype=paddle.bfloat16)
        k = paddle.to_tensor([[3.0, 4.0]], dtype=paddle.bfloat16)
        v = paddle.to_tensor([[5.0, 6.0]], dtype=paddle.bfloat16)

        q_out, k_out, v_out = flashattn_auto_cast(
            q, k, v, dtype=paddle.bfloat16
        )

        self.assertIs(q_out, q)
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)


if __name__ == "__main__":
    unittest.main()
