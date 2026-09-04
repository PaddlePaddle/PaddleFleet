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

"""`q_rms_norm_fusion` schedules rows with a compile-time `UNROLL` so that
several row loads are in flight before the first reduction. The claim that this
does not move a single output bit rests on every row still being reduced by the
same lane set, which is a property of `(BLOCK_N2, num_warps)` and not of the
unroll factor or the grid. That is an argument, not a proof, so this file checks
it: each case runs the kernels at `UNROLL == 0` -- the pre-optimization
grid-stride loop, still in the kernel bodies for exactly this reason -- and
compares y, invvar and dx bit-for-bit against the shipped schedule.

Note what is *not* claimed. The fused kernel is not bitwise equal to Paddle's
eager path at every shape (see the module docstring); that gap predates the
unroll and is not what these tests measure.
"""

import unittest

import paddle
import triton

from paddlefleet.triton_ops.q_rms_norm_fusion import (
    _num_programs,
    _num_warps,
    _pick_unroll,
    fused_q_rms_norm,
    q_rms_norm_bwd_kernel,
    q_rms_norm_fwd_kernel,
)

EPS = 1e-5


def _run_fwd(x, unroll):
    """Forward at an explicit ``unroll``, mirroring the shipped launcher."""
    n2 = x.shape[-1]
    block_n2 = triton.next_power_of_2(n2)
    n1 = 1
    for s in x.shape[:-1]:
        n1 *= s
    stride_x_row = x.stride()[x.ndim - 2]

    y = paddle.empty(x.shape, dtype=x.dtype)
    invvar = paddle.empty([n1], dtype=paddle.float32)
    q_rms_norm_fwd_kernel[(_num_programs(n1, unroll),)](
        x,
        y,
        invvar,
        stride_x_row,
        n2,
        n1,
        n2,
        BLOCK_N2=block_n2,
        eps=EPS,
        UNROLL=unroll,
        num_warps=_num_warps(block_n2),
    )
    return y, invvar


def _run_bwd(dy, x, invvar, unroll):
    """Backward at an explicit ``unroll``, mirroring the shipped launcher."""
    n2 = x.shape[-1]
    block_n2 = triton.next_power_of_2(n2)
    n1 = 1
    for s in x.shape[:-1]:
        n1 *= s

    dx = paddle.empty(dy.shape, dtype=dy.dtype)
    q_rms_norm_bwd_kernel[(_num_programs(n1, unroll),)](
        dy,
        x,
        invvar,
        dx,
        dy.stride()[dy.ndim - 2],
        x.stride()[x.ndim - 2],
        n2,
        n1,
        n2,
        BLOCK_N2=block_n2,
        UNROLL=unroll,
        num_warps=_num_warps(block_n2),
    )
    return dx


def _bit_equal(a, b):
    """``equal_all`` has no bfloat16 kernel; widening to fp32 is exact."""
    return bool(paddle.equal_all(a.astype("float32"), b.astype("float32")))


# (shape, expected forward unroll). n1 = prod(shape[:-1]).
CASES = [
    ([4, 512, 24, 512], 8),  # n1 = 49152, the production geometry
    ([1, 64, 2, 128], 8),  # block_n2=128 -> num_warps=1
    ([3, 7, 512], 1),  # n1 = 21, odd: _pick_unroll must decline
    ([2, 34, 256], 4),  # n1 = 68 = 4*17, divisible by 4 but not 8
]


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "fused_q_rms_norm requires GPU with SM80+ (bf16)",
)
class TestQRMSNormUnroll(unittest.TestCase):
    def _inputs(self, shape, seed):
        paddle.seed(seed)
        x = paddle.randn(shape, dtype="bfloat16")
        dy = paddle.randn(shape, dtype="bfloat16")
        return x, dy

    def test_pick_unroll_matches_expected(self):
        for shape, expected in CASES:
            with self.subTest(shape=shape):
                n1 = 1
                for s in shape[:-1]:
                    n1 *= s
                block_n2 = triton.next_power_of_2(shape[-1])
                self.assertEqual(
                    _pick_unroll(n1, block_n2, cap=8, elem_budget=4096),
                    expected,
                )

    def test_pick_unroll_never_exceeds_the_cap(self):
        # A factor above the cap would shrink the grid while the kernel body
        # still handled `cap` rows, leaving the output tail in uninitialized
        # memory. Nothing about that failure is loud, so the bound is asserted
        # rather than trusted.
        for cap in (4, 8):
            for n1 in (16, 64, 4096, 49152, 786432):
                for block_n2 in (128, 256, 512, 1024):
                    u = _pick_unroll(
                        n1, block_n2, cap=cap, elem_budget=cap * 512
                    )
                    with self.subTest(cap=cap, n1=n1, block_n2=block_n2):
                        self.assertGreaterEqual(u, 1)
                        self.assertLessEqual(u, cap)
                        self.assertEqual(n1 % u, 0)
                        self.assertEqual(u & (u - 1), 0)  # power of two

    def test_unrolled_schedule_is_bit_exact_against_grid_stride_loop(self):
        for shape, _ in CASES:
            with self.subTest(shape=shape):
                x, dy = self._inputs(shape, seed=20260903)
                n1 = 1
                for s in shape[:-1]:
                    n1 *= s
                block_n2 = triton.next_power_of_2(shape[-1])

                y_ref, iv_ref = _run_fwd(x, 0)
                y_new, iv_new = _run_fwd(
                    x, _pick_unroll(n1, block_n2, cap=8, elem_budget=4096)
                )
                self.assertTrue(_bit_equal(y_new, y_ref), "y differs")
                self.assertTrue(_bit_equal(iv_new, iv_ref), "invvar differs")

                dx_ref = _run_bwd(dy, x, iv_ref, 0)
                dx_new = _run_bwd(
                    dy,
                    x,
                    iv_ref,
                    _pick_unroll(n1, block_n2, cap=4, elem_budget=2048),
                )
                self.assertTrue(_bit_equal(dx_new, dx_ref), "dx differs")

    def test_public_entry_matches_the_grid_stride_loop(self):
        # Same check one level up, through fused_q_rms_norm + autograd, so a
        # launcher that picked a factor the kernel does not implement would show
        # up here even if the raw-kernel comparison above were passed a matching
        # pair of wrong values.
        shape = [4, 512, 24, 512]
        x, dy = self._inputs(shape, seed=7)
        y_ref, iv_ref = _run_fwd(x, 0)
        dx_ref = _run_bwd(dy, x, iv_ref, 0)

        xv = x.detach()
        xv.stop_gradient = False
        y = fused_q_rms_norm(xv, eps=EPS)
        y.backward(dy)

        self.assertTrue(_bit_equal(y, y_ref), "forward differs")
        self.assertTrue(_bit_equal(xv.grad, dx_ref), "backward differs")


if __name__ == "__main__":
    unittest.main()
