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

"""Behavior tests for the CPU-observable core of
``paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op``.

The heavy attention math lives in the Triton kernels ``gemm_fuse_softmax_*``
and requires a GPU, so it is NOT asserted here. Instead we pin the pure
CPU-observable logic of the primary entry ``rr_attn_estimate_triton_func`` and
its helpers:

* ``_extract_raw_ptrs`` -- the mode/causal -> column selection (index math).
* input validation (``_require``) that must fire before any kernel launch.
* the kernel-launch orchestration: which kernel is chosen (causal vs
  non-causal), the launch grid, the softmax ``scale``, the constexpr meta and
  that the extracted raw / stride-maxmin pointers are forwarded in order.

All expected values are hand-derived from the input construction rule; we
never call the function under test to produce its own expected values, and the
GPU kernels / ``prepare_maxmin`` / ``find_blocks_topp`` collaborators are
replaced by spies that return distinguishable markers so their routing can be
observed without a device.
"""

import math
import unittest
from unittest import mock

# ``paddlefleet_ops`` imports ``paddle`` and Triton at import time; the local
# CI image for this file may lack a working ``paddle``/``triton`` install.
# Guard the import so the suite skips honestly instead of erroring at
# collection. Only genuine missing-dependency errors are swallowed -- any other
# error (e.g. an API change or a decorator failure) must surface, not skip.
try:
    import numpy as np
    import paddle
    from paddlefleet_ops._extensions.flashmask import (
        rr_attn_estimate_triton_op as rr_op,
    )
    from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
        RawPtrs,
        StrideMaxMinPtrs,
        _extract_raw_ptrs,
        rr_attn_estimate_triton_func,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:
    np = None
    paddle = None
    rr_op = None
    RawPtrs = StrideMaxMinPtrs = None
    _extract_raw_ptrs = rr_attn_estimate_triton_func = None
    _IMPORT_ERROR = repr(exc)

_MODULE_AVAILABLE = _IMPORT_ERROR is None

# 1 / ln(2), the module's documented LOG2E constant (independent literal).
_LOG2E_REF = 1.4426950408889634
# BLOCK_SIZE is a module-private launch constant baked into the entry point.
_EXPECTED_BLOCK_SIZE = 128


def _startend_columns(mode, seqlen=3):
    """[1, 1, seqlen, mode] int32 with value[0, 0, i, j] == i * 10 + j.

    Every column ``j`` is distinguishable (column ``j`` == [j, 10 + j, ...]),
    so we can tell which input column ``_extract_raw_ptrs`` routed to each
    RawPtrs field.
    """
    data = [[[[i * 10 + j for j in range(mode)] for i in range(seqlen)]]]
    return paddle.to_tensor(data, dtype="int32")


def _startend_const_columns(bsz, hids, seqlen_k, mode):
    """[bsz, hids, seqlen_k, mode] int32 with every entry of column j == j."""
    col = paddle.arange(mode, dtype="int32").reshape([1, 1, 1, mode])
    return paddle.tile(col, [bsz, hids, seqlen_k, 1])


class _KernelLaunchSpy:
    """Stand-in for a GPU-only ``gemm_fuse_softmax_*`` Triton kernel.

    Triton kernels are launched as ``kernel[grid](*args, **kwargs)``; this
    records the grid captured by ``__getitem__`` and the positional/keyword
    arguments captured by the returned launcher, so the test can assert what
    ``rr_attn_estimate_triton_func`` actually forwarded. No device work runs.
    """

    def __init__(self):
        self.grid = None
        self.args = None
        self.kwargs = None
        self.launched = False

    def __getitem__(self, grid):
        self.grid = grid

        def _launch(*args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.launched = True

        return _launch


def _make_prepare_maxmin_spy(calls):
    """Spy for ``prepare_maxmin``: returns per-call distinguishable markers.

    Call ``i`` (1-based) returns (max, min) int32 tensors filled with
    ``1000 + i`` and ``2000 + i`` respectively, shaped like the real output
    ``[bsz, num_heads, ceil(seq_len / chunk_size)]`` so ``n_strides`` and all
    downstream shapes stay consistent.
    """

    def _spy(inp, chunk_size):
        calls.append((inp, chunk_size))
        bsz, num_heads, seq_len = inp.shape
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        idx = len(calls)
        mx = paddle.full(
            [bsz, num_heads, num_chunks], 1000 + idx, dtype="int32"
        )
        mn = paddle.full(
            [bsz, num_heads, num_chunks], 2000 + idx, dtype="int32"
        )
        return mx, mn

    return _spy


@unittest.skipUnless(
    _MODULE_AVAILABLE, f"module import failed: {_IMPORT_ERROR}"
)
class TestLog2EConstant(unittest.TestCase):
    def test_log2e_value(self):
        # LOG2E == 1 / ln(2); used to convert exp() into exp2() in the kernel.
        self.assertEqual(rr_op.LOG2E, _LOG2E_REF)
        self.assertAlmostEqual(rr_op.LOG2E, 1.0 / math.log(2.0), places=15)


@unittest.skipUnless(
    _MODULE_AVAILABLE, f"module import failed: {_IMPORT_ERROR}"
)
class TestExtractRawPtrs(unittest.TestCase):
    """``_extract_raw_ptrs`` selects input columns by (mode, causal).

    Contract (from the source docstring):
      * mode=1: only lt_start; lt_end/ut_start/ut_end alias lt_start.
      * mode=2 causal:  (lt_start, lt_end); ut_* alias lt_start.
      * mode=2 !causal: (lt_start, ut_end); lt_end/ut_start alias lt_start.
      * mode=4: (lt_start, lt_end, ut_start, ut_end) from columns 0..3.
    """

    def _col(self, seqlen, j):
        return np.array([[[i * 10 + j for i in range(seqlen)]]], dtype="int32")

    def test_mode1_all_alias_lt_start(self):
        x = _startend_columns(mode=1)
        mode, raw = _extract_raw_ptrs(x, causal=True)
        self.assertEqual(mode, 1)
        self.assertIsInstance(raw, RawPtrs)
        np.testing.assert_array_equal(raw.lt_start.numpy(), self._col(3, 0))
        # Unused fields must alias lt_start (identity), not fresh columns.
        self.assertIs(raw.lt_end, raw.lt_start)
        self.assertIs(raw.ut_start, raw.lt_start)
        self.assertIs(raw.ut_end, raw.lt_start)

    def test_mode2_causal_uses_lt_end(self):
        x = _startend_columns(mode=2)
        mode, raw = _extract_raw_ptrs(x, causal=True)
        self.assertEqual(mode, 2)
        np.testing.assert_array_equal(raw.lt_start.numpy(), self._col(3, 0))
        np.testing.assert_array_equal(raw.lt_end.numpy(), self._col(3, 1))
        # causal=True: upper-triangle bounds are unused and alias lt_start.
        self.assertIs(raw.ut_start, raw.lt_start)
        self.assertIs(raw.ut_end, raw.lt_start)

    def test_mode2_non_causal_uses_ut_end(self):
        x = _startend_columns(mode=2)
        mode, raw = _extract_raw_ptrs(x, causal=False)
        self.assertEqual(mode, 2)
        np.testing.assert_array_equal(raw.lt_start.numpy(), self._col(3, 0))
        # Column 1 must land in ut_end (NOT lt_end) when non-causal.
        np.testing.assert_array_equal(raw.ut_end.numpy(), self._col(3, 1))
        self.assertIs(raw.lt_end, raw.lt_start)
        self.assertIs(raw.ut_start, raw.lt_start)

    def test_mode4_all_distinct_columns(self):
        x = _startend_columns(mode=4)
        for causal in (True, False):
            mode, raw = _extract_raw_ptrs(x, causal=causal)
            self.assertEqual(mode, 4)
            np.testing.assert_array_equal(raw.lt_start.numpy(), self._col(3, 0))
            np.testing.assert_array_equal(raw.lt_end.numpy(), self._col(3, 1))
            np.testing.assert_array_equal(raw.ut_start.numpy(), self._col(3, 2))
            np.testing.assert_array_equal(raw.ut_end.numpy(), self._col(3, 3))

    def test_unsupported_mode_raises(self):
        x = _startend_columns(mode=3)
        with self.assertRaisesRegex(ValueError, "Unsupported mode=3"):
            _extract_raw_ptrs(x, causal=True)


@unittest.skipUnless(
    _MODULE_AVAILABLE, f"module import failed: {_IMPORT_ERROR}"
)
class TestValidationGuards(unittest.TestCase):
    """``_require`` guards in the entry point must fire before any kernel."""

    def _q(self, bsz=1, q_len=8, hq=4, hd=16):
        return paddle.zeros([bsz, q_len, hq, hd], dtype="float32")

    def _k(self, bsz=1, k_len=8, hkv=2, hd=16):
        return paddle.zeros([bsz, k_len, hkv, hd], dtype="float32")

    def test_startend_must_be_4d(self):
        se = paddle.zeros([1, 8, 4], dtype="int32")  # 3D
        with self.assertRaisesRegex(ValueError, "startend_row_indices must be"):
            rr_attn_estimate_triton_func(self._q(), self._k(), se)

    def test_batch_mismatch(self):
        se = paddle.zeros([2, 1, 8, 1], dtype="int32")  # bsz=2 != q bsz=1
        with self.assertRaisesRegex(ValueError, "batch mismatch"):
            rr_attn_estimate_triton_func(self._q(), self._k(), se)

    def test_seqlen_k_mismatch(self):
        se = paddle.zeros([1, 1, 7, 1], dtype="int32")  # 7 != kv_len 8
        with self.assertRaisesRegex(ValueError, "seqlen_k mismatch"):
            rr_attn_estimate_triton_func(self._q(), self._k(), se)

    def test_head_not_divisible_by_kv_heads(self):
        se = paddle.zeros([1, 1, 8, 1], dtype="int32")
        with self.assertRaisesRegex(ValueError, "num_q_heads % num_kv_heads"):
            rr_attn_estimate_triton_func(self._q(hq=4), self._k(hkv=3), se)

    def test_head_not_divisible_by_index_heads(self):
        se = paddle.zeros([1, 3, 8, 1], dtype="int32")  # HIDS=3, 4 % 3 != 0
        with self.assertRaisesRegex(
            ValueError, "num_q_heads % num_indices_heads"
        ):
            rr_attn_estimate_triton_func(self._q(hq=4), self._k(hkv=2), se)

    def test_non_positive_stride(self):
        se = paddle.zeros([1, 1, 8, 1], dtype="int32")
        with self.assertRaisesRegex(ValueError, "stride must be positive"):
            rr_attn_estimate_triton_func(self._q(), self._k(), se, stride=0)


@unittest.skipUnless(
    _MODULE_AVAILABLE, f"module import failed: {_IMPORT_ERROR}"
)
class TestKernelLaunchOrchestration(unittest.TestCase):
    """Observe the real ``rr_attn_estimate_triton_func`` orchestration.

    The GPU kernels, ``prepare_maxmin`` and ``find_blocks_topp`` are replaced
    with spies so the whole call runs on CPU tensors (no device required). We
    assert kernel selection, launch grid, softmax scale, constexpr meta, and
    that the extracted raw columns and per-field stride-maxmin markers are
    forwarded to the kernel in the documented positional order.
    """

    # Positional argument indices in kernel[grid](...):
    #   0:q 1:k 2:out 3:boundary
    #   4:lt_start 5:lt_end 6:ut_start 7:ut_end
    #   8:lt_start_max 9:lt_start_min 10:lt_end_max 11:lt_end_min
    #   12:ut_start_max 13:ut_start_min 14:ut_end_max 15:ut_end_min

    def _patch(self, calls):
        causal_spy = _KernelLaunchSpy()
        non_causal_spy = _KernelLaunchSpy()
        fbt_marker = object()
        captured = {}

        def _fbt(x, p):
            captured["fbt_x"] = x
            captured["fbt_p"] = p
            return fbt_marker

        patches = [
            mock.patch.object(rr_op, "gemm_fuse_softmax_causal", causal_spy),
            mock.patch.object(
                rr_op, "gemm_fuse_softmax_non_causal", non_causal_spy
            ),
            mock.patch.object(
                rr_op, "prepare_maxmin", _make_prepare_maxmin_spy(calls)
            ),
            mock.patch.object(rr_op, "find_blocks_topp", _fbt),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return causal_spy, non_causal_spy, fbt_marker, captured

    def _assert_all_equal(self, tensor, value):
        arr = tensor.numpy()
        self.assertTrue(
            (arr == value).all(), f"expected all == {value}, got {arr!r}"
        )

    def test_non_causal_mode4_launch(self):
        calls = []
        causal_spy, non_causal_spy, fbt_marker, captured = self._patch(calls)

        bsz, q_len, hq, hd = 2, 200, 4, 16
        kv_len, hkv, hids, mode, stride = 256, 2, 1, 4, 8
        q = paddle.zeros([bsz, q_len, hq, hd], dtype="float32")
        k = paddle.zeros([bsz, kv_len, hkv, hd], dtype="float32")
        se = _startend_const_columns(bsz, hids, kv_len, mode)

        result = rr_attn_estimate_triton_func(
            q, k, se, stride=stride, causal=False, threshold=0.5
        )

        # Kernel selection: non-causal chosen, causal untouched.
        self.assertTrue(non_causal_spy.launched)
        self.assertFalse(causal_spy.launched)

        # Launch grid = (num_q_blocks, num_q_heads, bsz) = (ceil(200/128), 4, 2)
        self.assertEqual(non_causal_spy.grid, (2, 4, 2))

        args = non_causal_spy.args
        self.assertIs(args[0], q)
        self.assertIs(args[1], k)
        # out / boundary shapes and dtypes.
        self.assertEqual(list(args[2].shape), [bsz, hq, 2, 2])
        self.assertEqual(args[2].dtype, q.dtype)
        self.assertEqual(list(args[3].shape), [bsz, hq, 2, 2])
        self.assertEqual(args[3].dtype, paddle.bool)

        # Raw column routing: column j landed in field order lt_start..ut_end.
        self._assert_all_equal(args[4], 0)  # lt_start = col 0
        self._assert_all_equal(args[5], 1)  # lt_end   = col 1
        self._assert_all_equal(args[6], 2)  # ut_start = col 2
        self._assert_all_equal(args[7], 3)  # ut_end   = col 3

        # Stride maxmin routing: mode=4 computes all four -> calls 1..4.
        self._assert_all_equal(args[8], 1001)  # lt_start_max
        self._assert_all_equal(args[9], 2001)  # lt_start_min
        self._assert_all_equal(args[10], 1002)  # lt_end_max
        self._assert_all_equal(args[11], 2002)  # lt_end_min
        self._assert_all_equal(args[12], 1003)  # ut_start_max
        self._assert_all_equal(args[13], 2003)  # ut_start_min
        self._assert_all_equal(args[14], 1004)  # ut_end_max
        self._assert_all_equal(args[15], 2004)  # ut_end_min
        self.assertEqual(len(calls), 4)

        kw = non_causal_spy.kwargs
        expected_scale = _LOG2E_REF / math.sqrt(hd) / stride
        self.assertAlmostEqual(kw["scale"], expected_scale, places=12)
        self.assertEqual(kw["seqlen_q"], q_len)
        self.assertEqual(kw["seqlen_k"], kv_len)
        self.assertEqual(kw["num_q_blocks"], 2)
        self.assertEqual(kw["num_k_blocks"], 2)
        self.assertEqual(kw["N_STRIDES"], (kv_len + stride - 1) // stride)  # 32
        self.assertEqual(kw["STRIDE"], stride)
        self.assertEqual(kw["HQ"], hq)
        self.assertEqual(kw["H"], hkv)
        self.assertEqual(kw["HIDS"], hids)
        self.assertEqual(kw["K"], hd)
        self.assertEqual(kw["BLOCK_SIZE"], _EXPECTED_BLOCK_SIZE)
        self.assertEqual(kw["mode"], mode)

        # Return tuple: (attn_sums, boundary_mask, find_blocks_topp(attn_sums)).
        self.assertIs(result[0], args[2])
        self.assertIs(result[1], args[3])
        self.assertIs(result[2], fbt_marker)
        self.assertIs(captured["fbt_x"], args[2])
        self.assertEqual(captured["fbt_p"], 0.5)

    def test_causal_mode2_launch(self):
        calls = []
        causal_spy, non_causal_spy, _fbt_marker, _cap = self._patch(calls)

        bsz, q_len, hq, hd = 1, 128, 4, 16
        kv_len, hkv, hids, mode, stride = 128, 2, 1, 2, 8
        q = paddle.zeros([bsz, q_len, hq, hd], dtype="float32")
        k = paddle.zeros([bsz, kv_len, hkv, hd], dtype="float32")
        se = _startend_const_columns(bsz, hids, kv_len, mode)

        rr_attn_estimate_triton_func(
            q, k, se, stride=stride, causal=True, threshold=0.3
        )

        # Kernel selection: causal chosen, non-causal untouched.
        self.assertTrue(causal_spy.launched)
        self.assertFalse(non_causal_spy.launched)
        self.assertEqual(causal_spy.grid, (1, 4, 1))

        args = causal_spy.args
        # Raw routing for mode=2 causal: lt_end = col1, ut_* alias lt_start.
        self._assert_all_equal(args[4], 0)  # lt_start = col 0
        self._assert_all_equal(args[5], 1)  # lt_end   = col 1
        self._assert_all_equal(args[6], 0)  # ut_start aliases lt_start
        self._assert_all_equal(args[7], 0)  # ut_end   aliases lt_start

        # Stride maxmin: only lt_start (call 1) and lt_end (call 2) computed;
        # ut_* max fields alias lt_start_max (never dereferenced for mode 2).
        self._assert_all_equal(args[8], 1001)  # lt_start_max
        self._assert_all_equal(args[10], 1002)  # lt_end_max
        self._assert_all_equal(args[12], 1001)  # ut_start_max aliases lt_start
        self._assert_all_equal(args[14], 1001)  # ut_end_max   aliases lt_start
        self.assertEqual(len(calls), 2)

        kw = causal_spy.kwargs
        self.assertEqual(kw["mode"], 2)
        self.assertEqual(kw["N_STRIDES"], (kv_len + stride - 1) // stride)  # 16
        self.assertEqual(kw["num_q_blocks"], 1)
        self.assertEqual(kw["num_k_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
