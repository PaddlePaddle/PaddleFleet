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

"""Behaviour tests for the CPU-testable surface of
``paddlefleet.triton_ops.fused_linear_cross_entropy``.

The fused loss/grad numerics live in Triton kernels
(``liger_cross_entropy_kernel`` / ``liger_cross_entropy_multimax_kernel`` and
``element_mul_kernel``) that only run on a real CUDA device, so they are NOT
exercised here -- faking a GPU or mocking those kernels would only test the
test. What IS pure-python and therefore CPU-testable:

* ``_select_ce_launch_config`` -- derives ``(BLOCK_SIZE, num_warps)`` from the
  documented per-thread element budget; compared against an independent numpy
  reimplementation plus hand-computed anchors.
* the control flow of ``fused_linear_cross_entropy_backward`` that runs BEFORE
  any kernel launch: the ``grad_output == 1.0`` identity short-circuit and the
  all-``None`` gradient handling (both return without touching the kernel).

On a dependency-less / CPU-only collection the heavy imports fail and every
class skips with the real import-error repr (never a fabricated pass).
"""

import unittest

import numpy as np

_IMPORT_ERROR = None
try:
    import paddle

    from paddlefleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
        CE_BLOCK_SIZE_CAP,
        CE_ELEMENTS_PER_THREAD,
        MAX_FUSED_SIZE,
        _select_ce_launch_config,
        fused_linear_cross_entropy_backward,
    )
except (ImportError, ModuleNotFoundError) as exc:  # dependency may be absent
    _IMPORT_ERROR = exc
    paddle = None
    _select_ce_launch_config = None
    fused_linear_cross_entropy_backward = None
    CE_BLOCK_SIZE_CAP = CE_ELEMENTS_PER_THREAD = MAX_FUSED_SIZE = None

_IMPORTS_OK = _IMPORT_ERROR is None
_SKIP_IMPORT = (
    f"fused_linear_cross_entropy import unavailable: {_IMPORT_ERROR!r}"
)


def _independent_launch_config(n_cols, max_fused, cap, elements_per_thread):
    """Reimplement the documented launch-config budget independently.

    next-power-of-2 is computed here via ``bit_length`` (not the production
    helper) so this reference cannot silently share a bug with the code under
    test. block_size is the smallest of {max_fused cap, next_pow2, tile cap};
    num_warps floors the tile over the per-thread element budget, clamped to
    [1, 32].
    """
    npow2 = 1 if n_cols <= 1 else 1 << (n_cols - 1).bit_length()
    block_size = min(max_fused, npow2, cap)
    num_warps = block_size // (32 * elements_per_thread)
    num_warps = max(1, min(32, num_warps))
    return block_size, num_warps


@unittest.skipUnless(_IMPORTS_OK, _SKIP_IMPORT)
class TestSelectCeLaunchConfig(unittest.TestCase):
    """``_select_ce_launch_config`` is pure python (only ``triton.next_power_of_2``
    plus min/floor-div/clamp); no kernel launch, so fully CPU-testable."""

    # (n_cols, expected_block_size, expected_num_warps), hand-derived from the
    # documented budget: block = min(32768, next_pow2(V), 2048);
    # warps = clamp(block // (32*16), 1, 32).
    CASES = [
        (1, 1, 1),
        (16, 16, 1),
        (100, 128, 1),  # next_pow2(100) == 128
        (512, 512, 1),  # 512 // 512 == 1
        (1024, 1024, 2),
        (1536, 2048, 4),  # next_pow2(1536) == 2048
        (2048, 2048, 4),
        (201216, 2048, 4),  # large vocab: capped at CE_BLOCK_SIZE_CAP
    ]

    def test_matches_hand_derived_anchors(self):
        for n_cols, exp_block, exp_warps in self.CASES:
            block_size, num_warps = _select_ce_launch_config(n_cols)
            self.assertEqual(
                (block_size, num_warps),
                (exp_block, exp_warps),
                msg=f"n_cols={n_cols}",
            )

    def test_matches_independent_reference(self):
        for n_cols in [1, 3, 16, 17, 100, 512, 1024, 1536, 2048, 4096, 201216]:
            expected = _independent_launch_config(
                n_cols,
                MAX_FUSED_SIZE,
                CE_BLOCK_SIZE_CAP,
                CE_ELEMENTS_PER_THREAD,
            )
            self.assertEqual(
                _select_ce_launch_config(n_cols),
                expected,
                msg=f"n_cols={n_cols}",
            )

    def test_cap_binds_for_large_vocab(self):
        # A vocab whose next_pow2 (262144) far exceeds the tile cap must be
        # clamped to CE_BLOCK_SIZE_CAP; if the cap were dropped this would be
        # MAX_FUSED_SIZE (32768) with num_warps == 32.
        block_size, num_warps = _select_ce_launch_config(201216)
        self.assertEqual(block_size, CE_BLOCK_SIZE_CAP)
        self.assertLess(block_size, MAX_FUSED_SIZE)
        self.assertEqual(
            num_warps, CE_BLOCK_SIZE_CAP // (32 * CE_ELEMENTS_PER_THREAD)
        )

    def test_invariants_hold(self):
        for n_cols in [1, 7, 33, 129, 777, 4096, 131072]:
            block_size, num_warps = _select_ce_launch_config(n_cols)
            self.assertLessEqual(block_size, CE_BLOCK_SIZE_CAP)
            self.assertLessEqual(block_size, MAX_FUSED_SIZE)
            # block_size is a power of two
            self.assertEqual(block_size & (block_size - 1), 0)
            self.assertGreaterEqual(num_warps, 1)
            self.assertLessEqual(num_warps, 32)


@unittest.skipUnless(_IMPORTS_OK, _SKIP_IMPORT)
class TestBackwardControlFlow(unittest.TestCase):
    """Pure-python branches of ``fused_linear_cross_entropy_backward`` that run
    before any ``element_mul_kernel`` launch. The actual per-element scaling is
    a Triton kernel (GPU-only) and is intentionally NOT asserted here."""

    def test_unit_scalar_grad_output_is_identity(self):
        # grad_output == 1.0 must short-circuit and return the SAME grad
        # objects untouched -- no kernel, no scaling.
        grad_input = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        grad_weight = paddle.to_tensor([[5.0, 6.0, 7.0]], dtype="float32")
        grad_bias = paddle.to_tensor([8.0, 9.0, 10.0], dtype="float32")
        gi_before = grad_input.numpy().copy()
        gw_before = grad_weight.numpy().copy()
        gb_before = grad_bias.numpy().copy()

        grad_output = paddle.to_tensor(1.0, dtype="float32")
        self.assertEqual(grad_output.shape, [])

        out_gi, out_gw, out_gb = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )

        self.assertIs(out_gi, grad_input)
        self.assertIs(out_gw, grad_weight)
        self.assertIs(out_gb, grad_bias)
        np.testing.assert_array_equal(out_gi.numpy(), gi_before)
        np.testing.assert_array_equal(out_gw.numpy(), gw_before)
        np.testing.assert_array_equal(out_gb.numpy(), gb_before)

    def test_unit_scalar_preserves_none_slots(self):
        grad_input = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        grad_output = paddle.to_tensor(1.0, dtype="float32")
        out_gi, out_gw, out_gb = fused_linear_cross_entropy_backward(
            grad_output, grad_input, None, None
        )
        self.assertIs(out_gi, grad_input)
        self.assertIsNone(out_gw)
        self.assertIsNone(out_gb)

    def test_nonunit_scalar_all_none_is_noop(self):
        # grad_output != 1.0 skips the short-circuit; with every grad None no
        # kernel is launched and all slots stay None.
        grad_output = paddle.to_tensor(2.0, dtype="float32")
        result = fused_linear_cross_entropy_backward(
            grad_output, None, None, None
        )
        self.assertEqual(result, (None, None, None))

    def test_vector_grad_output_all_none_is_noop(self):
        # A [BT] vector grad_output exercises the ``ndim >= 1`` collapse branch
        # (max().reshape([])) without a kernel launch when grads are absent.
        grad_output = paddle.to_tensor([0.25, 0.25, 0.0, 0.0], dtype="float32")
        self.assertEqual(grad_output.ndim, 1)
        out_gi, out_gw, out_gb = fused_linear_cross_entropy_backward(
            grad_output, None, None, None
        )
        self.assertIsNone(out_gi)
        self.assertIsNone(out_gw)
        self.assertIsNone(out_gb)


if __name__ == "__main__":
    unittest.main()
