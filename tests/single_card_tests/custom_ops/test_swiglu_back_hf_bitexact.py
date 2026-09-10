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
"""Tests for the ``"hf"`` SwiGLU backward in ``fusions/fused_bias_swiglu``.

``swiglu_back_hf_bitexact`` reproduces what ``silu(y_1) * y_2`` autograd does in
torch under ``autocast(bfloat16)``. Three properties are load-bearing and each
gets its own test:

* the intermediate ``gs = g * y_2`` is rounded to the activation dtype before it
  enters ``silu_backward`` (the Megatron branch stays in one dtype throughout);
* ``silu_backward`` promotes to FP32 and evaluates ``1 + x * (1 - s)`` as written;
* the multiplication **left-associates** -- ``(go * s) * inner``, not
  ``go * (s * inner)``. Only the left-associated form matches ATen, and the
  difference is a single ULP on near-tie inputs.

The target is threaded in as a parameter, so the three ``PyLayer``s are also
checked to forward it rather than consulting a module-level flag.
"""

import os
import sys
import unittest

import numpy as np
import paddle

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddlefleet.fusions.fused_bias_swiglu import (
    swiglu_back_eager,
    swiglu_back_hf_bitexact,
)


def _reference_hf_grads(g, y_1, y_2):
    """Independent transcription of ATen's silu-backward, left-associated."""
    dtype = y_1.dtype
    with paddle.amp.auto_cast(False):
        gs = (g * y_2).astype("float32")
        x = y_1.astype("float32")
        s = paddle.nn.functional.sigmoid(x)
        gy1 = ((gs * s) * (1.0 + x * (1.0 - s))).astype(dtype)
        gy2 = g * paddle.nn.functional.silu(y_1)
    return gy1, gy2


class TestSwigluBackHFBitexact(unittest.TestCase):
    """Arithmetic of ``swiglu_back_hf_bitexact`` itself."""

    def setUp(self):
        paddle.seed(20260908)
        self.y_1 = paddle.randn([64, 32], dtype=paddle.float32)
        self.y_2 = paddle.randn([64, 32], dtype=paddle.float32)
        self.g = paddle.randn([64, 32], dtype=paddle.float32)

    def test_matches_independent_reference(self):
        """Both returned grads equal a separately written ATen transcription."""
        gy1, gy2 = swiglu_back_hf_bitexact(self.g, self.y_1, self.y_2)
        ref1, ref2 = _reference_hf_grads(self.g, self.y_1, self.y_2)
        np.testing.assert_array_equal(gy1.numpy(), ref1.numpy())
        np.testing.assert_array_equal(gy2.numpy(), ref2.numpy())

    def test_gy2_is_g_times_silu(self):
        """The second output is exactly ``g * silu(y_1)``, no FP32 detour."""
        _, gy2 = swiglu_back_hf_bitexact(self.g, self.y_1, self.y_2)
        expected = self.g * paddle.nn.functional.silu(self.y_1)
        np.testing.assert_array_equal(gy2.numpy(), expected.numpy())

    def test_left_association_is_not_right_association(self):
        """``(go*s)*inner`` must not be silently replaced by ``go*(s*inner)``.

        In BF16 the two differ on near-tie elements; the docstring records one
        real element out of 27136 that broke a run. Using BF16 here makes the
        distinction observable at a practical tensor size.
        """
        y_1 = self.y_1.astype(paddle.bfloat16)
        y_2 = self.y_2.astype(paddle.bfloat16)
        g = self.g.astype(paddle.bfloat16)
        gy1, _ = swiglu_back_hf_bitexact(g, y_1, y_2)
        with paddle.amp.auto_cast(False):
            gs = (g * y_2).astype("float32")
            x = y_1.astype("float32")
            s = paddle.nn.functional.sigmoid(x)
            right = (gs * (s * (1.0 + x * (1.0 - s)))).astype(paddle.bfloat16)
        # Same value to within a ULP, but the implementation must be the
        # left-associated one, i.e. equal to the left reference bit-for-bit.
        left = _reference_hf_grads(g, y_1, y_2)[0]
        np.testing.assert_array_equal(gy1.numpy(), left.numpy())
        np.testing.assert_allclose(
            gy1.astype("float32").numpy(),
            right.astype("float32").numpy(),
            rtol=8e-3,
            atol=0,
        )

    def test_preserves_activation_dtype(self):
        for dtype in (paddle.float32, paddle.bfloat16, paddle.float16):
            with self.subTest(dtype=dtype):
                gy1, gy2 = swiglu_back_hf_bitexact(
                    self.g.astype(dtype),
                    self.y_1.astype(dtype),
                    self.y_2.astype(dtype),
                )
                self.assertEqual(gy1.dtype, dtype)
                self.assertEqual(gy2.dtype, dtype)

    def test_finite_on_large_magnitudes(self):
        """``sigmoid`` saturation must not produce NaN/Inf in the product."""
        y_1 = paddle.to_tensor([[-80.0, 80.0, 0.0, -1e-8]], dtype="float32")
        y_2 = paddle.ones_like(y_1)
        g = paddle.ones_like(y_1)
        gy1, gy2 = swiglu_back_hf_bitexact(g, y_1, y_2)
        self.assertTrue(bool(paddle.all(paddle.isfinite(gy1))))
        self.assertTrue(bool(paddle.all(paddle.isfinite(gy2))))


class TestSwigluBackEagerTargetDispatch(unittest.TestCase):
    """``swiglu_back_eager`` picks its branch from ``accuracy_target``."""

    def setUp(self):
        paddle.seed(7)
        self.y = paddle.randn([32, 16], dtype=paddle.float32)
        self.g = paddle.randn([32, 8], dtype=paddle.float32)

    def _megatron_reference(self, g, y):
        y_1, y_2 = paddle.chunk(y, 2, axis=-1)
        sig = paddle.nn.functional.sigmoid
        return paddle.concat(
            (
                g * sig(y_1) * (1 + y_1 * (1 - sig(y_1))) * y_2,
                g * paddle.nn.functional.silu(y_1),
            ),
            axis=-1,
        )

    def test_hf_target_takes_the_bitexact_branch(self):
        out = swiglu_back_eager(self.g, self.y, "hf")
        y_1, y_2 = paddle.chunk(self.y, 2, axis=-1)
        expected = paddle.concat(
            swiglu_back_hf_bitexact(self.g, y_1, y_2), axis=-1
        )
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_default_and_true_and_megatron_take_the_megatron_branch(self):
        """The historical two-arg call and both Megatron spellings agree."""
        ref = self._megatron_reference(self.g, self.y)
        outs = [
            swiglu_back_eager(self.g, self.y),
            swiglu_back_eager(self.g, self.y, True),
            swiglu_back_eager(self.g, self.y, "megatron"),
        ]
        for i, out in enumerate(outs):
            with self.subTest(variant=i):
                np.testing.assert_array_equal(out.numpy(), ref.numpy())

    def test_false_target_also_takes_the_megatron_branch(self):
        """``swiglu_back_eager`` only runs under an alignment mode anyway."""
        np.testing.assert_array_equal(
            swiglu_back_eager(self.g, self.y, False).numpy(),
            self._megatron_reference(self.g, self.y).numpy(),
        )

    def test_hf_differs_from_megatron(self):
        """The two branches are genuinely different arithmetic, not aliases."""
        y = self.y.astype(paddle.bfloat16)
        g = self.g.astype(paddle.bfloat16)
        hf = swiglu_back_eager(g, y, "hf").astype("float32").numpy()
        mg = swiglu_back_eager(g, y, "megatron").astype("float32").numpy()
        self.assertFalse(np.array_equal(hf, mg))

    def test_output_shape_matches_input(self):
        out = swiglu_back_eager(self.g, self.y, "hf")
        self.assertEqual(out.shape, self.y.shape)


if __name__ == "__main__":
    unittest.main()
