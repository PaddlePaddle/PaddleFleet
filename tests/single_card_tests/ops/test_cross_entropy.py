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

"""Behavior tests for the fused cross-entropy Triton kernels.

Module under test:
``paddlefleet.triton_ops.fused_linear_cross_entropy.cross_entropy``. In the
repository module map this belongs to "计算优化 / Fused Ops": two ``@triton.jit``
kernels that compute per-row cross-entropy loss with an online-softmax lse pass
and write the softmax gradient back in place:

  * ``liger_cross_entropy_kernel``          -- plain CE.
  * ``liger_cross_entropy_multimax_kernel`` -- CE fused with the learnable
    SegLU modulation ``SegLU(x) = x + t0*max(r0-x,0) + t1*max(x-r1,0)
    + t2*max(r2-x,0)^2 + t3*max(x-r3,0)^2`` plus its closed-form backward for
    grad_x, grad_ranges and grad_ts.

Both kernel *bodies* compile to GPU PTX and are not Python-instrumentable, and
the module exposes no CPU-observable pure-Python helper (no arg validation,
stride math, or block-size selection lives here). The only honest way to verify
real behavior is to LAUNCH the kernels on a CUDA device and compare against an
independently hand-derived NumPy reference. These tests therefore skip honestly
on a CPU-only / no-paddle host (this environment has no paddle) rather than
faking a pass; the numeric contract below runs unchanged on a GPU box.
"""

import unittest

import numpy as np

# --- Honest import guards ---------------------------------------------------
# Only genuine missing-dependency errors (ImportError / ModuleNotFoundError)
# flip a capability flag to False. Compile/API errors are NOT swallowed as a
# "missing dependency" -- they would surface as real failures on a real host.
try:
    import paddle

    _HAS_PADDLE = True
except ImportError:
    _HAS_PADDLE = False

try:
    import triton

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

_HAS_MODULE = False
if _HAS_PADDLE and _HAS_TRITON:
    try:
        from paddlefleet.triton_ops.fused_linear_cross_entropy.cross_entropy import (
            liger_cross_entropy_kernel,
            liger_cross_entropy_multimax_kernel,
        )

        _HAS_MODULE = True
    except ImportError:
        _HAS_MODULE = False


def _has_cuda():
    if not _HAS_PADDLE:
        return False
    try:
        return (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except Exception:
        return False


_HAS_GPU_ENV = _HAS_PADDLE and _HAS_TRITON and _HAS_MODULE and _has_cuda()
_SKIP_REASON = (
    "requires paddle + triton + a CUDA GPU: the cross_entropy kernels are "
    "@triton.jit bodies that only execute as GPU PTX and cannot be verified "
    "on a CPU-only / no-paddle host"
)


# --- Independent NumPy references (hand-derived, no production code) ---------
def _ref_cross_entropy(x, y, ignore_index, reduction):
    """Reference for ``liger_cross_entropy_kernel``.

    Per row r with label y[r] != ignore_index:
        lse = logsumexp(x[r]); loss = lse - x[r, y]
        grad = softmax(x[r]);  grad[y] -= 1
    ``mean`` divides both loss and grad by the count of non-ignored rows.
    Ignored rows produce loss 0 and an all-zero gradient row (the kernel
    writes 0 into X and never writes loss for those rows).
    """
    x = x.astype(np.float64)
    n_rows, _ = x.shape
    n_non_ignore = int(np.sum(y != ignore_index))
    loss = np.zeros(n_rows, dtype=np.float64)
    grad = np.zeros_like(x)
    for r in range(n_rows):
        label = int(y[r])
        if label == ignore_index:
            continue
        row = x[r]
        m = row.max()
        lse = m + np.log(np.exp(row - m).sum())
        loss[r] = lse - row[label]
        p = np.exp(row - lse)
        p[label] -= 1.0
        grad[r] = p
        if reduction == "mean":
            loss[r] /= n_non_ignore
            grad[r] /= n_non_ignore
    return loss, grad


def _seglu(x, r, t):
    m0 = np.maximum(r[0] - x, 0.0)
    m1 = np.maximum(x - r[1], 0.0)
    m2 = np.maximum(r[2] - x, 0.0)
    m3 = np.maximum(x - r[3], 0.0)
    val = x + t[0] * m0 + t[1] * m1 + t[2] * m2 * m2 + t[3] * m3 * m3
    return val, m0, m1, m2, m3


def _ref_multimax(x, y, ignore_index, reduction, r, t):
    """Reference for ``liger_cross_entropy_multimax_kernel``.

    Forward:  L = lse(SegLU(x)) - SegLU(x)[y]
    grad_x  = grad_out * dSegLU/dx, where grad_out = softmax(SegLU(x)) with the
              one-hot at y subtracted (mean-divided when reduction=="mean") and
                dSegLU/dx = 1 - t0*1{r0>x} + t1*1{x>r1}
                            - 2*t2*max(r2-x,0) + 2*t3*max(x-r3,0)
    Param grads accumulate over all non-ignored rows/cols:
        gt0=sum(grad_out*m0) gt1=sum(grad_out*m1)
        gt2=sum(grad_out*m2^2) gt3=sum(grad_out*m3^2)
        gr0= t0*sum(grad_out*1{r0>x})  gr1=-t1*sum(grad_out*1{x>r1})
        gr2= 2*t2*sum(grad_out*m2)     gr3=-2*t3*sum(grad_out*m3)
    """
    x = x.astype(np.float64)
    r = [float(v) for v in r]
    t = [float(v) for v in t]
    n_rows, _ = x.shape
    n_non_ignore = int(np.sum(y != ignore_index))
    loss = np.zeros(n_rows, dtype=np.float64)
    grad_x = np.zeros_like(x)
    grad_r = np.zeros(4, dtype=np.float64)
    grad_t = np.zeros(4, dtype=np.float64)
    for row_idx in range(n_rows):
        label = int(y[row_idx])
        if label == ignore_index:
            continue
        row = x[row_idx]
        seglu, m0, m1, m2, m3 = _seglu(row, r, t)
        m = seglu.max()
        lse = m + np.log(np.exp(seglu - m).sum())
        loss[row_idx] = lse - seglu[label]
        grad_out = np.exp(seglu - lse)
        grad_out[label] -= 1.0
        if reduction == "mean":
            loss[row_idx] /= n_non_ignore
            grad_out = grad_out / n_non_ignore
        mask0 = (m0 > 0.0).astype(np.float64)
        mask1 = (m1 > 0.0).astype(np.float64)
        dseglu = (
            1.0
            - t[0] * mask0
            + t[1] * mask1
            - 2.0 * t[2] * m2
            + 2.0 * t[3] * m3
        )
        grad_x[row_idx] = grad_out * dseglu
        grad_t[0] += np.sum(grad_out * m0)
        grad_t[1] += np.sum(grad_out * m1)
        grad_t[2] += np.sum(grad_out * m2 * m2)
        grad_t[3] += np.sum(grad_out * m3 * m3)
        grad_r[0] += t[0] * np.sum(grad_out * mask0)
        grad_r[1] += -t[1] * np.sum(grad_out * mask1)
        grad_r[2] += 2.0 * t[2] * np.sum(grad_out * m2)
        grad_r[3] += -2.0 * t[3] * np.sum(grad_out * m3)
    return loss, grad_x, grad_r, grad_t


# --- Launch helpers (mirror the production kernel call signature) -----------
def _launch_plain(x_np, y_np, ignore_index, reduction):
    logits = paddle.to_tensor(x_np, dtype="float32").contiguous()
    target = paddle.to_tensor(y_np, dtype="int64").contiguous()
    n_rows, v = logits.shape
    loss = paddle.zeros([n_rows], dtype="float32")
    n_non_ignore = (
        int(np.sum(y_np != ignore_index)) if reduction == "mean" else 0
    )
    liger_cross_entropy_kernel[(n_rows,)](
        X_ptr=logits,
        X_stride=logits.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        loss_ptr=loss,
        loss_stride=loss.stride(-1),
        n_cols=v,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        reduction=reduction,
        HAS_GRADIENTS=True,
        BLOCK_SIZE=triton.next_power_of_2(v),
        num_warps=4,
    )
    # kernel wrote grad in place into `logits`.
    return loss.numpy(), logits.numpy()


def _launch_multimax(x_np, y_np, ignore_index, reduction, r, t):
    logits = paddle.to_tensor(x_np, dtype="float32").contiguous()
    target = paddle.to_tensor(y_np, dtype="int64").contiguous()
    n_rows, v = logits.shape
    loss = paddle.zeros([n_rows], dtype="float32")
    grad_r = paddle.zeros([4], dtype="float32")
    grad_t = paddle.zeros([4], dtype="float32")
    n_non_ignore = (
        int(np.sum(y_np != ignore_index)) if reduction == "mean" else 0
    )
    liger_cross_entropy_multimax_kernel[(n_rows,)](
        X_ptr=logits,
        X_stride=logits.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        loss_ptr=loss,
        loss_stride=loss.stride(-1),
        n_cols=v,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        r0=float(r[0]),
        r1=float(r[1]),
        r2=float(r[2]),
        r3=float(r[3]),
        t0=float(t[0]),
        t1=float(t[1]),
        t2=float(t[2]),
        t3=float(t[3]),
        grad_r_ptr=grad_r,
        grad_t_ptr=grad_t,
        reduction=reduction,
        HAS_GRADIENTS=True,
        HAS_MULTIMAX_GRADIENTS=True,
        BLOCK_SIZE=triton.next_power_of_2(v),
        num_warps=4,
    )
    return loss.numpy(), logits.numpy(), grad_r.numpy(), grad_t.numpy()


# Fixed, non-degenerate fixture: distinct per-row logits, distinct labels, one
# ignored row (label == ignore_index), vocab spanning below/above every SegLU
# range so each ReLU branch is exercised.
_IGNORE = -100
_X = np.array(
    [
        [0.2, 2.7, -1.3, 3.1, 0.5, -0.4],
        [-2.1, 1.4, 0.9, -0.7, 2.2, 1.1],
        [3.3, -1.8, 0.6, 1.9, -0.2, 2.5],
        [0.1, 0.4, -0.9, 1.2, -1.5, 0.8],
    ],
    dtype=np.float32,
)
_Y = np.array([3, 4, 0, _IGNORE], dtype=np.int64)
_R = [1.0, 2.0, 0.5, 1.5]
_T = [0.1, 0.2, 0.05, 0.15]


@unittest.skipUnless(_HAS_GPU_ENV, _SKIP_REASON)
class TestLigerCrossEntropyKernel(unittest.TestCase):
    """Numeric contract of the plain fused cross-entropy Triton kernel."""

    def test_forward_and_grad_none_reduction(self):
        ref_loss, ref_grad = _ref_cross_entropy(_X, _Y, _IGNORE, "none")
        loss, grad = _launch_plain(_X, _Y, _IGNORE, "none")
        np.testing.assert_allclose(loss, ref_loss, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(grad, ref_grad, rtol=1e-5, atol=1e-5)
        # Ignored row: zero gradient, zero (unwritten) loss.
        np.testing.assert_array_equal(grad[3], np.zeros(_X.shape[1]))
        self.assertEqual(loss[3], 0.0)

    def test_forward_and_grad_mean_reduction(self):
        ref_loss, ref_grad = _ref_cross_entropy(_X, _Y, _IGNORE, "mean")
        loss, grad = _launch_plain(_X, _Y, _IGNORE, "mean")
        np.testing.assert_allclose(loss, ref_loss, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(grad, ref_grad, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(_HAS_GPU_ENV, _SKIP_REASON)
class TestLigerCrossEntropyMultimaxKernel(unittest.TestCase):
    """Numeric contract of the SegLU-fused multimax cross-entropy kernel."""

    def test_forward_and_grad_x(self):
        ref_loss, ref_gx, _, _ = _ref_multimax(_X, _Y, _IGNORE, "mean", _R, _T)
        loss, grad_x, _, _ = _launch_multimax(_X, _Y, _IGNORE, "mean", _R, _T)
        np.testing.assert_allclose(loss, ref_loss, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(grad_x, ref_gx, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(grad_x[3], np.zeros(_X.shape[1]))

    def test_param_grads_ranges_and_ts(self):
        _, _, ref_gr, ref_gt = _ref_multimax(_X, _Y, _IGNORE, "mean", _R, _T)
        _, _, grad_r, grad_t = _launch_multimax(_X, _Y, _IGNORE, "mean", _R, _T)
        np.testing.assert_allclose(grad_r, ref_gr, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(grad_t, ref_gt, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
