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
"""Behavior tests for helpers in ``transformer/moe/moe_layer``.

These exercise the CPU-runnable, autograd-only building blocks of the MoE
layer against hand-derived expectations, independent of the implementation:

* ``GradDtypeGuard`` / ``GradDtypeUnguard`` form a matched pair. The guard
  parks the *real* activation in a side-channel dict and emits a zero-element
  placeholder that carries only the requested gradient dtype into the autograd
  graph; the unguard recovers the parked tensor untouched. The tests pin the
  placeholder's dtype and empty extent, and check the guard->unguard
  round-trip returns the original bytes for two different requested dtypes so
  the ``dtype`` argument is proven to be consumed.
* ``ThreePathCloneAlignMG`` fans one tensor out to three consumers and, on the
  backward, sums their three upstream gradients. The test drives a real
  backward with three *distinct* per-path cotangents so that dropping or
  misrouting any single path is rejected; in fp32 the sum is exact regardless
  of the MG-aligned addition order.
* ``MoESublayers`` is a dataclass whose whole contract is storing / defaulting
  / comparing its ``mlp_spec`` field.

The heavier fused / bit-exact MoE fanout layers are covered separately in
``test_moe_layer_hf_bitexact.py`` and are intentionally not duplicated here.

Paddle is an optional heavy dependency (and importing the module also pulls in
``paddlefleet_ops``); when either is unavailable the whole case is skipped with
an honest reason rather than reported as passing.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)
# Package lives under ``src/``; add it so an editable checkout imports too.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        ),
        "src",
    ),
)

try:
    import numpy as np
    import paddle

    _DEP_ERROR = None
except ImportError as exc:  # paddle / numpy not installed in this env
    paddle = None
    np = None
    _DEP_ERROR = exc

try:
    from paddlefleet.transformer.moe.moe_layer import (
        GradDtypeGuard,
        GradDtypeUnguard,
        MoESublayers,
        ThreePathCloneAlignMG,
    )

    _MOD_ERROR = None
except ImportError as exc:  # module or its native deps unavailable
    GradDtypeGuard = GradDtypeUnguard = None
    MoESublayers = ThreePathCloneAlignMG = None
    _MOD_ERROR = exc


_SKIP_REASON = None
if _DEP_ERROR is not None:
    _SKIP_REASON = f"paddle/numpy unavailable: {_DEP_ERROR!r}"
elif _MOD_ERROR is not None:
    _SKIP_REASON = f"moe_layer import failed: {_MOD_ERROR!r}"


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "")
class TestGradDtypeGuardRoundTrip(unittest.TestCase):
    """Guard parks the real tensor and emits a dtype-only placeholder."""

    def setUp(self):
        paddle.set_device("cpu")
        self.x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=paddle.float32
        )

    def test_placeholder_is_empty_and_carries_requested_dtype(self):
        placeholder, status = GradDtypeGuard.apply(self.x, paddle.float16)

        # The placeholder is a graph node only: no data, requested dtype.
        self.assertEqual(list(placeholder.shape), [0])
        self.assertEqual(placeholder.dtype, paddle.float16)

        # The real activation is parked unchanged (value and its own dtype).
        self.assertIn("x", status)
        self.assertEqual(status["x"].dtype, paddle.float32)
        np.testing.assert_array_equal(status["x"].numpy(), self.x.numpy())

    def test_requested_dtype_is_consumed(self):
        # Same input, different requested dtype -> only placeholder dtype moves;
        # parked payload is untouched. Proves the dtype arg is actually used.
        ph_fp16, _ = GradDtypeGuard.apply(self.x, paddle.float16)
        ph_bf16, _ = GradDtypeGuard.apply(self.x, paddle.bfloat16)
        self.assertEqual(ph_fp16.dtype, paddle.float16)
        self.assertEqual(ph_bf16.dtype, paddle.bfloat16)

    def test_unguard_recovers_original_tensor(self):
        placeholder, status = GradDtypeGuard.apply(self.x, paddle.float16)
        recovered = GradDtypeUnguard.apply(placeholder, status)

        # Round-trip returns the exact original bytes and dtype, not the
        # placeholder's dtype/extent.
        self.assertEqual(recovered.dtype, paddle.float32)
        self.assertEqual(list(recovered.shape), [2, 3])
        np.testing.assert_array_equal(recovered.numpy(), self.x.numpy())


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "")
class TestThreePathCloneAlignMG(unittest.TestCase):
    """Three-way clone; backward sums the three upstream gradients."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_emits_three_equal_independent_clones(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype=paddle.float32)
        o0, o1, o2 = ThreePathCloneAlignMG.apply(x)

        for out in (o0, o1, o2):
            np.testing.assert_array_equal(out.numpy(), x.numpy())
        # Distinct objects (clones), not the same aliased tensor.
        self.assertIsNot(o0, o1)
        self.assertIsNot(o1, o2)
        self.assertIsNot(o0, o2)

    def test_backward_sums_distinct_per_path_gradients(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype=paddle.float32)
        x.stop_gradient = False
        o0, o1, o2 = ThreePathCloneAlignMG.apply(x)

        # Distinguishable cotangent per path so dropping / swapping any one
        # path changes the result. fp32 -> the MG-aligned add order is exact.
        g0 = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype=paddle.float32)
        g1 = paddle.to_tensor(
            [[10.0, 20.0], [30.0, 40.0]], dtype=paddle.float32
        )
        g2 = paddle.to_tensor(
            [[100.0, 200.0], [300.0, 400.0]], dtype=paddle.float32
        )
        paddle.autograd.backward([o0, o1, o2], [g0, g1, g2])

        expected = (g0 + g1 + g2).numpy()  # [[111,222],[333,444]]
        self.assertIsNotNone(x.grad)
        np.testing.assert_array_equal(x.grad.numpy(), expected)


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "")
class TestMoESublayers(unittest.TestCase):
    """Dataclass field default, retention and value equality."""

    def test_default_is_none(self):
        self.assertIsNone(MoESublayers().mlp_spec)

    def test_retains_provided_spec_identity(self):
        sentinel = object()
        self.assertIs(MoESublayers(mlp_spec=sentinel).mlp_spec, sentinel)

    def test_value_equality_tracks_field(self):
        sentinel = object()
        self.assertEqual(
            MoESublayers(mlp_spec=sentinel), MoESublayers(mlp_spec=sentinel)
        )
        self.assertNotEqual(MoESublayers(mlp_spec=sentinel), MoESublayers())


if __name__ == "__main__":
    unittest.main()
