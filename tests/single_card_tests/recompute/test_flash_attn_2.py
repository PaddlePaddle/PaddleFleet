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

"""Behavior tests for ``flashattn_auto_cast`` in the refined-recompute
FlashAttention module.

Facet under test: the precision-normalisation utility that FlashAttention
kernels rely on to receive Q/K/V in a single supported dtype. The genuine,
CPU-observable contract of ``flashattn_auto_cast`` is:

  * each of q, k, v is examined independently against the target dtype;
  * a tensor already at the target dtype is passed through *by identity*
    (no copy is made -- the source guards every ``astype`` with
    ``if x.dtype != dtype``);
  * a tensor of any other dtype is converted to the target dtype while
    preserving values that are exactly representable there;
  * the default target dtype is ``bfloat16``.

These are checked with hand-picked, exactly-bfloat16/float16-representable
values (powers of two and small integers) so that a correct cast reproduces
the input bit-for-bit and any wrong-dtype / dropped-cast / copy-instead-of-
passthrough regression is observable without a GPU.
"""

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.refined_recompute.flash_attn import flashattn_auto_cast

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet unavailable in this env
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None


def _to_f32_numpy(tensor):
    """Return tensor values as float32 numpy (numpy has no bfloat16)."""
    return tensor.astype("float32").numpy()


@unittest.skipUnless(
    _HAS_DEPS,
    f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR}",
)
class TestFlashattnAutoCast(unittest.TestCase):
    """Independent-per-tensor cast + identity-passthrough contract."""

    # Values chosen to be exact in both bfloat16 and float16 (small integers
    # and negative powers of two), so a correct cast is loss-free and value
    # comparisons are exact rather than tolerance-based.
    EXACT = [[1.0, -2.0, 0.5, 4.0]]

    def test_default_target_is_bfloat16(self):
        """No explicit dtype -> everything ends up bfloat16."""
        q = paddle.to_tensor(self.EXACT, dtype=paddle.float32)
        k = paddle.to_tensor(self.EXACT, dtype=paddle.float32)
        v = paddle.to_tensor(self.EXACT, dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)

    def test_already_target_dtype_is_passed_through_by_identity(self):
        """A tensor already at the target dtype is returned unchanged.

        The source only calls ``astype`` when ``x.dtype != dtype``; when the
        dtype already matches the very same object must come back, and its
        values must be untouched. ``assertIs`` would fail a naive
        ``x = x.astype(dtype)`` unconditional implementation.
        """
        q = paddle.to_tensor(self.EXACT, dtype=paddle.bfloat16)
        k = paddle.to_tensor(self.EXACT, dtype=paddle.bfloat16)
        v = paddle.to_tensor(self.EXACT, dtype=paddle.bfloat16)
        q_out, k_out, v_out = flashattn_auto_cast(
            q, k, v, dtype=paddle.bfloat16
        )
        self.assertIs(q_out, q)
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)
        np.testing.assert_array_equal(
            _to_f32_numpy(q_out), np.array(self.EXACT)
        )

    def test_cast_from_float32_preserves_exact_values(self):
        """float32 inputs are converted to bfloat16 with exact values kept."""
        q = paddle.to_tensor(self.EXACT, dtype=paddle.float32)
        k = paddle.to_tensor(self.EXACT, dtype=paddle.float32)
        v = paddle.to_tensor(self.EXACT, dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        # A new object is produced (conversion happened), not the float32 input.
        self.assertIsNot(q_out, q)
        expected = np.array(self.EXACT, dtype=np.float32)
        for out in (q_out, k_out, v_out):
            self.assertEqual(out.dtype, paddle.bfloat16)
            np.testing.assert_array_equal(_to_f32_numpy(out), expected)

    def test_custom_target_dtype_is_respected(self):
        """An explicit dtype overrides the bfloat16 default for every tensor."""
        q = paddle.to_tensor(self.EXACT, dtype=paddle.float32)
        k = paddle.to_tensor(self.EXACT, dtype=paddle.float32)
        v = paddle.to_tensor(self.EXACT, dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v, dtype=paddle.float16)
        expected = np.array(self.EXACT, dtype=np.float32)
        for out in (q_out, k_out, v_out):
            self.assertEqual(out.dtype, paddle.float16)
            np.testing.assert_array_equal(_to_f32_numpy(out), expected)

    def test_each_tensor_cast_decision_is_independent(self):
        """Mixed input dtypes -> per-tensor decision, target one passed through.

        q is float32 (must convert to a new bfloat16 tensor), k is already
        bfloat16 (must be returned by identity, no copy), v is float16 (must
        convert to a new bfloat16 tensor). A shared/whole-batch cast or a swap
        of the per-tensor branches would break exactly one of these.
        """
        q = paddle.to_tensor(self.EXACT, dtype=paddle.float32)
        k = paddle.to_tensor(self.EXACT, dtype=paddle.bfloat16)
        v = paddle.to_tensor(self.EXACT, dtype=paddle.float16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)

        # k already matched the target: identity passthrough.
        self.assertIs(k_out, k)
        # q and v were converted: new objects, target dtype.
        self.assertIsNot(q_out, q)
        self.assertIsNot(v_out, v)

        expected = np.array(self.EXACT, dtype=np.float32)
        for out in (q_out, k_out, v_out):
            self.assertEqual(out.dtype, paddle.bfloat16)
            np.testing.assert_array_equal(_to_f32_numpy(out), expected)


if __name__ == "__main__":
    unittest.main()
