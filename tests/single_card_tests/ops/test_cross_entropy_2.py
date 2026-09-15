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

"""Behavior tests for the fused multimax cross-entropy Triton kernel.

Module under test:
``paddlefleet.triton_ops.fused_linear_cross_entropy.cross_entropy``. In the
repository module map this is the "计算优化 / Fused Ops" boundary. This file
targets ``liger_cross_entropy_multimax_kernel`` (the SegLU variant) and its
branches -- SegLU forward, in-place ``grad_x`` write (HAS_GRADIENTS), the
per-batch ``grad_ranges`` / ``grad_ts`` reductions (HAS_MULTIMAX_GRADIENTS),
the ``ignore_index`` early-return, and mean-vs-sum reduction. The sibling
base file covers the plain ``liger_cross_entropy_kernel``; this ``_2`` file
deliberately covers a different function and its distinct branches.

Environment: this is a Triton GPU kernel that compiles to PTX; its numerics
are only observable by launching it on a real accelerator. The production
import chain (``triton_compat`` -> ``paddle``) also hard-requires Paddle built
with CUDA. Where Paddle+CUDA are absent (no-card env) the tests skip honestly
rather than fake-pass -- they do not mock the kernel or simulate GPU numerics
on CPU. Where a single H20 GPU is available the actual kernel is launched and
compared against an independent, hand-derived NumPy reference of the SegLU
cross-entropy math.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"
    ),
)

# Honest capability probe: only real ImportError/ModuleNotFoundError means the
# dependency is absent. We do NOT catch broad Exception (that would hide real
# compile/API regressions as "missing dep"). Launching the kernel additionally
# needs Paddle compiled with CUDA and a visible device.
try:
    import paddle
    import triton  # noqa: F401

    from paddlefleet.triton_ops.fused_linear_cross_entropy.cross_entropy import (
        liger_cross_entropy_multimax_kernel,
    )

    _IMPORT_OK = True
except (ImportError, ModuleNotFoundError):
    _IMPORT_OK = False


def _gpu_ready():
    if not _IMPORT_OK:
        return False
    try:
        return (
            paddle.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except Exception:
        return False


_RUN = _gpu_ready()
_SKIP_REASON = (
    "requires Triton + Paddle-CUDA + a visible GPU; this kernel compiles to "
    "PTX and has no CPU-observable numerics, so it is skipped honestly here"
)

# Sentinel written into loss slots before launch; the kernel must leave the
# slot untouched for an ignore_index row (it early-returns before writing).
_LOSS_SENTINEL = -123.5


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _seglu(x, r, t):
    """SegLU(x) = x + t0*relu(r0-x) + t1*relu(x-r1)
                    + t2*relu(r2-x)^2 + t3*relu(x-r3)^2.

    Independent NumPy expansion derived from the documented SegLU definition,
    not copied from the Triton kernel body. Returns the SegLU output plus the
    four ReLU intermediates, which the closed-form backward reuses.
    """
    m0 = np.maximum(r[0] - x, 0.0)
    m1 = np.maximum(x - r[1], 0.0)
    m2 = np.maximum(r[2] - x, 0.0)
    m3 = np.maximum(x - r[3], 0.0)
    out = x + t[0] * m0 + t[1] * m1 + t[2] * m2 * m2 + t[3] * m3 * m3
    return out, (m0, m1, m2, m3)


def _reference(logits, targets, r, t, ignore_index, reduction):
    """Hand-derived reference for the multimax CE kernel.

    Forward per valid row i (target y):
        s = SegLU(x); L = logsumexp(s) - s[y]        (then /n_non_ignore if mean)
    Backward writes grad_x = dL/dx in place (SegLU chain rule) and accumulates
    grad_ranges/grad_ts summed over every valid row and column.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    n_rows, _ = logits.shape
    valid = targets != ignore_index
    n_non_ignore = int(valid.sum())

    loss = np.full(n_rows, _LOSS_SENTINEL, dtype=np.float64)
    grad_x = np.zeros_like(logits)
    grad_r = np.zeros(4, dtype=np.float64)
    grad_t = np.zeros(4, dtype=np.float64)

    for i in range(n_rows):
        if not valid[i]:
            grad_x[i] = 0.0  # ignore_index row is explicitly zeroed
            continue
        x = logits[i]
        y = int(targets[i])
        s, (m0, m1, m2, m3) = _seglu(x, r, t)
        m = s.max()
        d = np.exp(s - m).sum()
        lse = m + np.log(d)
        row_loss = lse - s[y]

        prob = np.exp(s - m) / d
        grad_out = prob.copy()
        grad_out[y] -= 1.0
        if reduction == "mean":
            row_loss = row_loss / n_non_ignore
            grad_out = grad_out / n_non_ignore
        loss[i] = row_loss

        mask0 = (m0 > 0.0).astype(np.float64)
        mask1 = (m1 > 0.0).astype(np.float64)
        dseglu_dx = (
            1.0
            - t[0] * mask0
            + t[1] * mask1
            - 2.0 * t[2] * m2
            + 2.0 * t[3] * m3
        )
        grad_x[i] = grad_out * dseglu_dx

        grad_t[0] += (grad_out * m0).sum()
        grad_t[1] += (grad_out * m1).sum()
        grad_t[2] += (grad_out * m2 * m2).sum()
        grad_t[3] += (grad_out * m3 * m3).sum()
        grad_r[0] += t[0] * (grad_out * mask0).sum()
        grad_r[1] += -t[1] * (grad_out * mask1).sum()
        grad_r[2] += 2.0 * t[2] * (grad_out * m2).sum()
        grad_r[3] += -2.0 * t[3] * (grad_out * m3).sum()

    return {
        "loss": loss,
        "grad_x": grad_x,
        "grad_r": grad_r,
        "grad_t": grad_t,
        "n_non_ignore": n_non_ignore,
    }


def _launch_multimax(logits, targets, r, t, ignore_index, reduction):
    """Launch the real production kernel on GPU and return NumPy outputs.

    Enables both HAS_GRADIENTS and HAS_MULTIMAX_GRADIENTS so grad_x and the
    param-grad reductions are all produced. grad_x is written in place, so the
    input logits are cloned first. loss is pre-filled with a sentinel to expose
    the ignore_index early-return contract.
    """
    n_rows, n_cols = logits.shape
    x = paddle.to_tensor(logits, dtype="float32", place=paddle.CUDAPlace(0))
    y = paddle.to_tensor(targets, dtype="int64", place=paddle.CUDAPlace(0))
    loss = paddle.full([n_rows], _LOSS_SENTINEL, dtype="float32").cuda()
    grad_r = paddle.zeros([4], dtype="float32").cuda()
    grad_t = paddle.zeros([4], dtype="float32").cuda()

    block = _next_pow2(n_cols)
    valid = np.asarray(targets) != ignore_index
    n_non_ignore = int(valid.sum())

    liger_cross_entropy_multimax_kernel[(n_rows,)](
        x,
        n_cols,  # X_stride: row stride for a contiguous [rows, cols] tensor
        y,
        1,  # Y_stride
        loss,
        1,  # loss_stride
        n_cols,
        n_non_ignore,
        ignore_index,
        float(r[0]),
        float(r[1]),
        float(r[2]),
        float(r[3]),
        float(t[0]),
        float(t[1]),
        float(t[2]),
        float(t[3]),
        grad_r,
        grad_t,
        reduction,
        block,
        True,  # HAS_GRADIENTS
        True,  # HAS_MULTIMAX_GRADIENTS
    )
    paddle.device.cuda.synchronize()
    return {
        "loss": np.asarray(loss.numpy(), dtype=np.float64),
        "grad_x": np.asarray(x.numpy(), dtype=np.float64),
        "grad_r": np.asarray(grad_r.numpy(), dtype=np.float64),
        "grad_t": np.asarray(grad_t.numpy(), dtype=np.float64),
    }


# Fixed, non-uniform, uniquely valued logits so that transposes / mis-indexing
# would change the result. Targets are distinct per row. SegLU params are
# chosen so every ReLU branch is active on some element (some x < r0, some
# x > r1, some x < r2, some x > r3). No element sits exactly on a linear-ReLU
# kink (x == r0 or x == r1) so the (m > 0) subgradient tie-break is unambiguous
# between the fp32 kernel and the fp64 reference.
_LOGITS = np.array(
    [
        [-2.0, -0.6, 1.5, 0.25, 2.0, -1.25],
        [0.75, -1.5, 0.0, 1.0, -0.25, 0.6],
        [1.25, 2.5, -2.0, -0.75, 0.6, -1.0],
    ],
    dtype=np.float64,
)
_TARGETS = np.array([2, 5, 0], dtype=np.int64)
_R = (0.5, -0.5, 1.0, -1.0)
_T = (0.3, 0.2, 0.1, 0.15)
_IGNORE = -100


@unittest.skipUnless(_RUN, _SKIP_REASON)
class TestLigerCrossEntropyMultimaxKernel(unittest.TestCase):
    """Behavior tests for liger_cross_entropy_multimax_kernel (SegLU CE)."""

    def test_forward_loss_matches_reference_mean(self):
        ref = _reference(_LOGITS, _TARGETS, _R, _T, _IGNORE, "mean")
        got = _launch_multimax(_LOGITS, _TARGETS, _R, _T, _IGNORE, "mean")
        # Guard the fixture is non-degenerate: losses must be finite & varied.
        self.assertTrue(np.isfinite(ref["loss"]).all())
        self.assertGreater(ref["loss"].std(), 1e-3)
        np.testing.assert_allclose(
            got["loss"], ref["loss"], rtol=1e-4, atol=1e-5
        )

    def test_grad_x_matches_reference_mean(self):
        ref = _reference(_LOGITS, _TARGETS, _R, _T, _IGNORE, "mean")
        got = _launch_multimax(_LOGITS, _TARGETS, _R, _T, _IGNORE, "mean")
        # Non-degenerate, scale-sensitive check: reference grad has real signal.
        self.assertGreater(np.abs(ref["grad_x"]).max(), 1e-3)
        np.testing.assert_allclose(
            got["grad_x"], ref["grad_x"], rtol=1e-4, atol=1e-5
        )

    def test_param_grad_reductions_match_reference_mean(self):
        ref = _reference(_LOGITS, _TARGETS, _R, _T, _IGNORE, "mean")
        got = _launch_multimax(_LOGITS, _TARGETS, _R, _T, _IGNORE, "mean")
        # grad_ranges / grad_ts are summed over all valid rows and columns.
        self.assertGreater(np.abs(ref["grad_t"]).max(), 1e-3)
        self.assertGreater(np.abs(ref["grad_r"]).max(), 1e-3)
        np.testing.assert_allclose(
            got["grad_t"], ref["grad_t"], rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            got["grad_r"], ref["grad_r"], rtol=1e-4, atol=1e-4
        )

    def test_sum_reduction_has_no_division(self):
        ref_sum = _reference(_LOGITS, _TARGETS, _R, _T, _IGNORE, "sum")
        ref_mean = _reference(_LOGITS, _TARGETS, _R, _T, _IGNORE, "mean")
        got = _launch_multimax(_LOGITS, _TARGETS, _R, _T, _IGNORE, "sum")
        n = ref_sum["n_non_ignore"]
        # sum-reduction result must equal mean-reduction * n_non_ignore, i.e.
        # the /n division genuinely happens only for "mean".
        np.testing.assert_allclose(
            ref_sum["loss"], ref_mean["loss"] * n, rtol=1e-6, atol=1e-6
        )
        self.assertGreater(n, 1)  # otherwise the *n check is vacuous
        np.testing.assert_allclose(
            got["loss"], ref_sum["loss"], rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            got["grad_x"], ref_sum["grad_x"], rtol=1e-4, atol=1e-5
        )

    def test_ignore_index_row_is_zeroed_and_loss_untouched(self):
        targets = np.array([2, _IGNORE, 0], dtype=np.int64)
        ref = _reference(_LOGITS, targets, _R, _T, _IGNORE, "mean")
        got = _launch_multimax(_LOGITS, targets, _R, _T, _IGNORE, "mean")

        # Ignored row: grad_x row fully zeroed, loss slot left at the sentinel
        # (kernel returns before writing it), and it is excluded from the
        # n_non_ignore normalizer (so only 2 rows contribute).
        self.assertEqual(ref["n_non_ignore"], 2)
        np.testing.assert_array_equal(
            got["grad_x"][1], np.zeros(_LOGITS.shape[1])
        )
        self.assertAlmostEqual(float(got["loss"][1]), _LOSS_SENTINEL, places=4)

        # Valid rows: the sentinel must be overwritten with real losses that
        # match the /2 normalized reference.
        for i in (0, 2):
            self.assertNotAlmostEqual(
                float(got["loss"][i]), _LOSS_SENTINEL, places=4
            )
        np.testing.assert_allclose(
            got["loss"][[0, 2]], ref["loss"][[0, 2]], rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            got["grad_x"][[0, 2]], ref["grad_x"][[0, 2]], rtol=1e-4, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
