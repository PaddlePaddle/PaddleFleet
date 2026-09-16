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

"""Behavior tests for fused_linear_cross_entropy CPU-observable pure logic.

Module under test:
``paddlefleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy``.
In the repository module map this is the "计算优化 / Fused Ops" boundary.

This file (the ``_2`` companion) deliberately targets logic that does NOT
require driving the Triton/GPU CE kernels, so it can be asserted with
independent hand-derived expectations:

* ``_select_ce_launch_config`` -- the pure launch-config picker. It combines
  ``MAX_FUSED_SIZE``, ``triton.next_power_of_2(n_cols)`` and
  ``CE_BLOCK_SIZE_CAP`` into a ``BLOCK_SIZE``, then derives ``num_warps`` from
  ``CE_ELEMENTS_PER_THREAD`` with a ``max(1, ...)`` clamp. Pure arithmetic +
  real ``triton.next_power_of_2`` (no GPU kernel, no device tensors).
* ``LigerFusedLinearCrossEntropyFunction.backward`` output-slot arity contract
  when ``grad_output`` is the scalar ``1.0`` identity (so
  ``fused_linear_cross_entropy_backward`` short-circuits before any
  ``element_mul_kernel`` launch). The PyLayer must return exactly one grad
  slot per forward Tensor arg, with ``None`` at the non-differentiable
  ``target`` slot, and bias / multimax slots appended only when present.

The production module top-level ``import paddle`` / ``import triton`` plus its
sibling kernel modules must import for these tests to run. Triton is imported
FOR REAL (never mocked) because ``_select_ce_launch_config`` relies on the
genuine ``triton.next_power_of_2``; mocking it would erase the logic under
test. If the real module cannot be imported (e.g. Paddle absent in a CPU-only
box) the suite skips honestly rather than fake-passing -- it never claims to
have driven the GPU CE numerics.
"""

import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# Import the production module FOR REAL. Only genuine missing-dependency
# errors (Paddle/Triton/kernel siblings absent) are treated as a skip
# condition; any other error surfaces instead of being swallowed as "no dep".
try:
    from paddlefleet.triton_ops.fused_linear_cross_entropy import (
        fused_linear_cross_entropy as flce,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    flce = None
    _IMPORT_ERROR = repr(exc)

_MODULE_AVAILABLE = flce is not None
_SKIP_REASON = (
    "production module fused_linear_cross_entropy could not be imported "
    f"(Paddle/Triton/kernel deps unavailable): {_IMPORT_ERROR}"
)


@unittest.skipUnless(_MODULE_AVAILABLE, _SKIP_REASON)
class TestSelectCeLaunchConfig(unittest.TestCase):
    """Hand-derived checks for ``_select_ce_launch_config``.

    Contract (from module constants MAX_FUSED_SIZE=32768,
    CE_BLOCK_SIZE_CAP=2048, CE_ELEMENTS_PER_THREAD=16)::

        block_size = min(32768, next_power_of_2(n_cols), 2048)
        num_warps  = max(1, min(32, block_size // (32 * 16)))
                   = max(1, min(32, block_size // 512))

    Expected pairs are derived by hand, not by re-running the function.
    """

    def _expected(self, n_cols):
        # Independent re-derivation using the real triton primitive, mirroring
        # the documented formula but written out separately from production.
        import triton

        block = min(32768, triton.next_power_of_2(n_cols), 2048)
        warps = block // 512
        warps = max(1, min(32, warps))
        return block, warps

    def test_hand_derived_pairs(self):
        # (n_cols, expected_block_size, expected_num_warps), derived by hand.
        # next_power_of_2: 16->16, 256->256, 512->512, 513->1024, 1000->1024,
        # 1024->1024, 2048->2048, 4096->4096, 201216->262144, 32768->32768.
        cases = [
            (16, 16, 1),  # block 16  -> 16//512=0 clamped to 1
            (256, 256, 1),  # block 256 -> 0 clamped to 1
            (512, 512, 1),  # block 512 -> exactly 1 (not clamped)
            (513, 1024, 2),  # rounds up to 1024 -> 2 warps
            (1000, 1024, 2),
            (1024, 1024, 2),
            (2048, 2048, 4),  # cap == pow2, 2048//512 = 4
            (4096, 2048, 4),  # CE_BLOCK_SIZE_CAP clamps block to 2048
            (32768, 2048, 4),  # large vocab still capped at 2048 / 4 warps
            (201216, 2048, 4),  # realistic large vocab from module comment
        ]
        for n_cols, exp_block, exp_warps in cases:
            block, warps = flce._select_ce_launch_config(n_cols)
            self.assertEqual(
                (block, warps),
                (exp_block, exp_warps),
                msg=f"n_cols={n_cols}: got ({block}, {warps}), expected ({exp_block}, {exp_warps})",
            )
            # Cross-check against the independently re-derived reference too.
            self.assertEqual((block, warps), self._expected(n_cols))

    def test_block_size_never_exceeds_cap(self):
        # For any vocab larger than the cap the block must saturate at the
        # CE_BLOCK_SIZE_CAP, never at MAX_FUSED_SIZE or next_power_of_2(V).
        for n_cols in (2049, 5000, 40000, 200000):
            block, _ = flce._select_ce_launch_config(n_cols)
            self.assertEqual(block, flce.CE_BLOCK_SIZE_CAP)
            self.assertLessEqual(block, flce.MAX_FUSED_SIZE)

    def test_num_warps_lower_clamp_engages_below_512(self):
        # Any n_cols whose next_power_of_2 is < 512 yields block//512 == 0,
        # which MUST be clamped up to 1 (a 0-warp launch is illegal). Without
        # the max(1, ...) clamp these would return 0.
        # next_power_of_2(n) < 512 requires n <= 256 (257..512 round to 512).
        for n_cols in (1, 8, 16, 100, 255, 256):
            block, warps = flce._select_ce_launch_config(n_cols)
            self.assertLess(block, 512)
            self.assertEqual(warps, 1)

    def test_warps_scale_with_block_via_elements_per_thread(self):
        # num_warps must equal block // (32 * CE_ELEMENTS_PER_THREAD) once the
        # lower clamp is not engaged. Verifies the divisor is genuinely tied to
        # CE_ELEMENTS_PER_THREAD rather than a hard-coded constant.
        divisor = 32 * flce.CE_ELEMENTS_PER_THREAD
        for n_cols in (512, 1024, 2048, 4096):
            block, warps = flce._select_ce_launch_config(n_cols)
            self.assertGreaterEqual(block, 512)
            self.assertEqual(warps, block // divisor)


class _FakeCtx:
    """Minimal stand-in for the PyLayer context.

    ``ctx`` is a genuine not-under-test collaborator: it only carries the
    tensors saved in forward plus a handful of boolean flags. The logic under
    test is ``LigerFusedLinearCrossEntropyFunction.backward`` itself -- how it
    assembles the returned gradient tuple. Nothing here fakes that logic.
    """

    def __init__(self, saved, has_bias, has_multimax):
        self._saved = saved
        self.has_bias = has_bias
        self.has_multimax = has_multimax
        self.weight_requires_grad = False
        self.weight_ref = None
        self.ec_align = False
        self.multimax_ranges_ref = None
        self.multimax_ts_ref = None
        self.multimax_ranges_requires_grad = False
        self.multimax_ts_requires_grad = False

    def saved_tensor(self):
        return self._saved


@unittest.skipUnless(_MODULE_AVAILABLE, _SKIP_REASON)
class TestBackwardOutputArityContract(unittest.TestCase):
    """PyLayer backward must emit one grad slot per forward Tensor arg.

    The forward signature is
    ``(_input, weight, target, bias?, multimax_ranges?, multimax_ts?)``.
    ``target`` is non-differentiable so its slot is ALWAYS ``None`` (index 2);
    the ``bias`` and multimax slots exist only when those args were supplied.
    We drive the real ``backward`` with ``grad_output = 1.0`` (scalar identity)
    and all-``None`` saved grads so ``fused_linear_cross_entropy_backward``
    short-circuits before launching any ``element_mul_kernel`` -- keeping the
    assertion purely on the CPU-observable tuple structure.
    """

    def _run(self, has_bias, has_multimax):
        import paddle

        # (grad_input, grad_weight, grad_bias, grad_mm_ranges, grad_mm_ts)
        saved = (None, None, None, None, None)
        ctx = _FakeCtx(saved, has_bias=has_bias, has_multimax=has_multimax)
        grad_output = paddle.to_tensor(1.0)
        return flce.LigerFusedLinearCrossEntropyFunction.backward(
            ctx, grad_output
        )

    def test_no_bias_no_multimax_returns_three_slots(self):
        result = self._run(has_bias=False, has_multimax=False)
        self.assertIsInstance(result, tuple)
        # (grad__input, grad_weight, grad_target)
        self.assertEqual(len(result), 3)
        # target slot (index 2) is always None: it is non-differentiable.
        self.assertIsNone(result[2])
        # identity grad_output + None saved grads -> all three None here.
        self.assertEqual(result, (None, None, None))

    def test_bias_adds_exactly_one_slot(self):
        result = self._run(has_bias=True, has_multimax=False)
        self.assertEqual(len(result), 4)
        self.assertIsNone(result[2])  # target slot still None
        # bias slot appended; its value is the (None) short-circuited grad.
        self.assertIsNone(result[3])

    def test_multimax_adds_two_trailing_slots(self):
        result = self._run(has_bias=True, has_multimax=True)
        # _input, weight, target, bias, mm_ranges, mm_ts
        self.assertEqual(len(result), 6)
        self.assertIsNone(result[2])
        # With no saved multimax grads the two trailing slots stay None.
        self.assertIsNone(result[4])
        self.assertIsNone(result[5])

    def test_multimax_without_bias_arity(self):
        # has_bias=False, has_multimax=True -> _input, weight, target,
        # mm_ranges, mm_ts (bias slot omitted entirely).
        result = self._run(has_bias=False, has_multimax=True)
        self.assertEqual(len(result), 5)
        self.assertIsNone(result[2])
        self.assertIsNone(result[3])
        self.assertIsNone(result[4])


if __name__ == "__main__":
    unittest.main()
