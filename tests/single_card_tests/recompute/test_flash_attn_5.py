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

import os
import sys
import unittest

import numpy as np

# Make the in-repo `src/` importable when running the file directly, matching
# how the Fleet single-card runner exposes the paddlefleet package.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# paddlefleet imports paddle at import time. The no-card environment used here
# has no paddle installed, so guard the import and skip honestly rather than
# faking a pass. Only a precise ImportError is treated as "dependency missing".
try:
    import paddle

    from paddlefleet.refined_recompute.flash_attn import flashattn_auto_cast

    _PADDLE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without paddle
    paddle = None
    flashattn_auto_cast = None
    _PADDLE_IMPORT_ERROR = exc

_PADDLE_AVAILABLE = _PADDLE_IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle is not importable in this environment: {_PADDLE_IMPORT_ERROR}"
    if not _PADDLE_AVAILABLE
    else ""
)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestFlashattnAutoCastIdentityAndPairing(unittest.TestCase):
    """Behavioral tests for ``flashattn_auto_cast``.

    Distinct facet from the shared casting-dtype checks: this suite pins the
    *selective* nature of the cast (tensors already at the target dtype are
    returned by identity, not rebuilt), the exact q/k/v argument pairing, and
    lossless value preservation. Independent expectations are hand-written
    literals; the function under test is never used to compute its own expected.

    Uses float32/float64 which are CPU-executable and exercise the identical
    dtype-agnostic branch logic (``if x.dtype != dtype: x = x.astype(dtype)``).
    """

    def test_matching_dtype_returns_same_objects(self):
        """When all inputs already match the target dtype, the exact same
        tensor objects are returned. The production code guards each cast with
        ``if x.dtype != dtype``; an unconditional ``astype`` would allocate new
        objects and break this identity contract."""
        q = paddle.to_tensor([[1.0, 2.0]], dtype=paddle.float32)
        k = paddle.to_tensor([[3.0, 4.0]], dtype=paddle.float32)
        v = paddle.to_tensor([[5.0, 6.0]], dtype=paddle.float32)

        oq, ok, ov = flashattn_auto_cast(q, k, v, dtype=paddle.float32)

        self.assertIs(oq, q)
        self.assertIs(ok, k)
        self.assertIs(ov, v)

    def test_selective_cast_only_mismatched_tensor_is_rebuilt(self):
        """Mixed dtypes: only the mismatched tensor is cast (new object); the
        already-matching tensors keep their identity. Catches a bug that casts
        all three unconditionally, or one that skips the needed cast."""
        q = paddle.to_tensor([[1.5, -2.0]], dtype=paddle.float32)
        k = paddle.to_tensor([[0.25, 4.0]], dtype=paddle.float64)
        v = paddle.to_tensor([[-8.0, 0.5]], dtype=paddle.float32)

        oq, ok, ov = flashattn_auto_cast(q, k, v, dtype=paddle.float32)

        # Matching tensors preserved by identity.
        self.assertIs(oq, q)
        self.assertIs(ov, v)
        # Mismatched tensor is a distinct, newly-cast object.
        self.assertIsNot(ok, k)
        self.assertEqual(ok.dtype, paddle.float32)
        # Values chosen to be exactly representable, so the cast is lossless.
        np.testing.assert_array_equal(
            ok.numpy(), np.array([[0.25, 4.0]], dtype=np.float32)
        )
        # Original mismatched tensor is left untouched.
        self.assertEqual(k.dtype, paddle.float64)

    def test_cast_preserves_argument_order_and_values(self):
        """All three inputs need casting and carry distinct content. Each output
        must correspond to its own input (not a swapped one) and preserve the
        exact values. Independent expectations are literal float32 arrays."""
        q = paddle.to_tensor([[10.0, 11.0], [12.0, 13.0]], dtype=paddle.float64)
        k = paddle.to_tensor([[20.0, 21.0], [22.0, 23.0]], dtype=paddle.float64)
        v = paddle.to_tensor([[30.0, 31.0], [32.0, 33.0]], dtype=paddle.float64)

        oq, ok, ov = flashattn_auto_cast(q, k, v, dtype=paddle.float32)

        expected_q = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
        expected_k = np.array([[20.0, 21.0], [22.0, 23.0]], dtype=np.float32)
        expected_v = np.array([[30.0, 31.0], [32.0, 33.0]], dtype=np.float32)

        self.assertEqual(oq.dtype, paddle.float32)
        self.assertEqual(ok.dtype, paddle.float32)
        self.assertEqual(ov.dtype, paddle.float32)
        np.testing.assert_array_equal(oq.numpy(), expected_q)
        np.testing.assert_array_equal(ok.numpy(), expected_k)
        np.testing.assert_array_equal(ov.numpy(), expected_v)

    def test_downcast_rounds_to_target_precision(self):
        """A value not exactly representable in float32 must round to the
        float32 nearest value on cast. Hand-derived expectation via Python's
        struct round-trip, independent of the function under test."""
        import struct

        raw = 1.0 + 2.0**-40  # differs from 1.0 only below float32 resolution
        # Independent reference: round-trip through IEEE-754 single precision.
        expected = struct.unpack("f", struct.pack("f", raw))[0]
        self.assertEqual(expected, 1.0)  # sanity: rounds down to 1.0 in float32

        q = paddle.to_tensor([[raw]], dtype=paddle.float64)
        k = paddle.to_tensor([[raw]], dtype=paddle.float64)
        v = paddle.to_tensor([[raw]], dtype=paddle.float64)

        oq, _, _ = flashattn_auto_cast(q, k, v, dtype=paddle.float32)

        self.assertEqual(oq.dtype, paddle.float32)
        np.testing.assert_array_equal(
            oq.numpy(), np.array([[expected]], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
