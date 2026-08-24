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

"""
Unit tests for ``paddlefleet.triton_ops.local_to_global_idxs_fusion``.

The eager ``_local_to_global_flat`` is the ground truth; the fused kernel must
match it *bit-for-bit* (these are index tables -- there is no tolerance).
"""

import unittest

import numpy as np
import paddle

from paddlefleet.fusions.csa_sparse_attn_utils import (
    _local_to_global_flat,
    local_to_global_flat,
)
from paddlefleet.triton_ops.local_to_global_idxs_fusion import (
    local_to_global_flat_triton,
)


def _require_sm100_sparse_kernels(testcase):
    """Skip unless the FlashMLA / DSA sparse-attention kernels can actually run.

    Importing ``paddlefleet_ops.flash_mla`` is not a capability check: the
    library loads on Hopper (the CI runner is SM 9.0 and the log shows
    "Successfully loaded ecosystem library: flash_mla") while the kernels these
    tests reach are SM100-only -- the backward the docstrings quote lives in
    ``sparse_attention_backward/dsa_bwd_sm100.py``. Calling them on SM 9.x is
    not a meaningful test, so gate on the capability the way
    ``test_hysparse_online_tilelang_train_step`` does for the TileLang kernels.
    """
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")
    major = paddle.device.cuda.get_device_capability()[0]
    if major < 10:
        testcase.skipTest(
            f"FlashMLA / DSA sparse kernels require SM 10.x; got SM {major}.x"
        )
    try:
        from paddlefleet_ops.flash_mla import (  # noqa: F401
            flash_mla_sparse_fwd,
        )
    except (ImportError, RuntimeError):
        testcase.skipTest("flash_mla is not available")


def _assert_bitwise(case, ref, got):
    assert ref.dtype == got.dtype, f"{case}: dtype {ref.dtype} vs {got.dtype}"
    assert ref.shape == got.shape, f"{case}: shape {ref.shape} vs {got.shape}"
    r, g = ref.numpy(), got.numpy()
    if not np.array_equal(r, g):
        bad = np.argwhere(r != g)
        p = tuple(bad[0])
        raise AssertionError(
            f"{case}: {len(bad)}/{r.size} mismatches, "
            f"first at {p}: ref={r[p]} got={g[p]}"
        )


def _make_idxs(b, sq, topk, seqlen_kv, dtype, rng, pad_ratio=0.3):
    """Random indices in [0, seqlen_kv), ``pad_ratio`` of them set to -1.

    The pad mask is drawn as int16 rather than float64: these tables reach tens
    of millions of entries and the mask would otherwise be the single largest
    host allocation in the file.
    """
    shape = (b, sq, topk)
    idxs = rng.integers(0, max(seqlen_kv, 1), size=shape).astype(np.int64)
    pad = rng.integers(0, 1000, size=shape, dtype=np.int16)
    idxs[pad < int(pad_ratio * 1000)] = -1
    return paddle.to_tensor(idxs, dtype=dtype)


def _low32(x):
    """Two's-complement truncation to int32, not relying on numpy cast rules."""
    y = np.asarray(x, dtype=np.int64)
    return ((y + 2**31) % 2**32 - 2**31).astype(np.int32)


def _numpy_oracle(idxs, seqlen_kv):
    """Independent reference derived from the spec, not from the Paddle code.

    ``out[b * sq + s, k] = idx + b * seqlen_kv`` when ``idx >= 0``, else ``idx``
    unchanged, truncated to int32.

    One oracle covers both input dtypes: the eager int32 path (wrapping ``add``
    then a no-op ``cast``) and the eager int64 path (exact ``add`` then a
    truncating ``cast``) both reduce to the low 32 bits of the exact sum,
    because low32(a + low32(p)) == low32(a + p).
    """
    np_idxs = idxs.numpy().astype(np.int64)
    b, sq, topk = np_idxs.shape
    flat = np_idxs.reshape(b * sq, topk)
    off = np.repeat(np.arange(b, dtype=np.int64) * int(seqlen_kv), sq)[:, None]
    return _low32(np.where(flat >= 0, flat + off, flat))


class TestLocalToGlobalFlatFusion(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def _check(self, case, idxs, seqlen_kv):
        ref = _local_to_global_flat(idxs, seqlen_kv)
        got = local_to_global_flat_triton(idxs, seqlen_kv)
        _assert_bitwise(case, ref, got)
        return ref

    def test_smallest_possible_launch(self):
        """One row, one column: the cheapest possible kernel launch.

        Deliberately first in the file so the CI log distinguishes "the very
        first Triton compile stalls" (no progress at all) from "the volume of a
        later test is the problem" (this one's dot appears, then the stall).
        """
        idxs = paddle.to_tensor([[[3]]], dtype="int32")
        got = local_to_global_flat_triton(idxs, 8)
        self.assertEqual(list(got.shape), [1, 1])
        self.assertEqual(int(got[0, 0]), 3)

    def test_ernielite_hca_shape(self):
        """Real ernielite HCA widths: window+compress=640, seqlen_kv=65536+512.

        ``sq`` is only a grid multiplier -- the kernel is one program per row and
        ``cdiv(topk, BLOCK_K)`` along topk -- so it is kept small here. The real
        sq_local=16384 costs 10.5M entries per dtype and, with the eager
        reference plus the host round-trip in the comparison, dominated this
        file's runtime.
        """
        for dtype in ("int32", "int64"):
            idxs = _make_idxs(1, 1024, 640, 65536 + 512, dtype, self.rng)
            self._check(f"hca[{dtype}]", idxs, 65536 + 512)

    def test_shapes_and_dtypes(self):
        cases = [
            (1, 1, 1),
            (1, 7, 5),
            (1, 16, 640),
            (2, 4, 6),
            (3, 5, 1),
            (4, 33, 64),
            (2, 8, 1025),  # topk > BLOCK_K -> multi-block along topk
            (5, 3, 2047),  # non power-of-2, multi-block
        ]
        for dtype in ("int32", "int64"):
            for b, sq, topk in cases:
                seqlen_kv = max(topk, 8) * 2 + 3
                idxs = _make_idxs(b, sq, topk, seqlen_kv, dtype, self.rng)
                case = f"[{dtype}] b={b},sq={sq},topk={topk}"
                self._check(case, idxs, seqlen_kv)

    def test_edge_values(self):
        """All-padding rows, all-valid rows, index 0, and non -1 negatives."""
        for dtype in ("int32", "int64"):
            all_pad = paddle.full([3, 4, 5], -1, dtype=dtype)
            self._check(f"all_pad[{dtype}]", all_pad, 32)

            all_zero = paddle.zeros([3, 4, 5], dtype=dtype)
            self._check(f"all_zero[{dtype}]", all_zero, 32)

            # the reference passes negatives through unchanged, it does not
            # normalise them to -1; make sure the fusion does the same.
            other_neg = paddle.to_tensor(
                [[[-5, 0, -1, 7], [-2, 3, -9, 0]]], dtype=dtype
            )
            ref = self._check(f"other_neg[{dtype}]", other_neg, 16)
            np.testing.assert_array_equal(
                ref.numpy(), np.array([[-5, 0, -1, 7], [-2, 3, -9, 0]])
            )

    def test_non_contiguous_input(self):
        """CSA slices the index table before this call; keep that path exact."""
        for dtype in ("int32", "int64"):
            full = _make_idxs(2, 16, 128, 256, dtype, self.rng)
            sliced = full[:, 4:12, 8:72]
            self.assertEqual(sliced.shape, [2, 8, 64])
            self._check(f"sliced[{dtype}]", sliced, 256)

    def test_large_batch_offset_int32_wraparound(self):
        """b * seqlen_kv beyond int32 must wrap the same way in both paths."""
        seqlen_kv = 2**30
        rows = [[[0, 5, -1]], [[1, -1, 9]], [[3, 4, 0]]]
        idxs = paddle.to_tensor(rows, dtype="int32")
        self._check("int32_wrap", idxs, seqlen_kv)

        idxs64 = paddle.to_tensor(rows, dtype="int64")
        self._check("int64_no_wrap", idxs64, seqlen_kv)

    def test_allow_alias_matches_when_batch_is_one(self):
        for dtype in ("int32", "int64"):
            idxs = _make_idxs(1, 64, 96, 512, dtype, self.rng)
            ref = _local_to_global_flat(idxs, 512)
            got = local_to_global_flat_triton(idxs, 512, allow_alias=True)
            _assert_bitwise(f"alias[{dtype}]", ref, got)

    def test_not_differentiable(self):
        """Integer index tables carry no gradient; fusion must not add one."""
        idxs = _make_idxs(2, 4, 8, 32, "int32", self.rng)
        self.assertTrue(idxs.stop_gradient)
        out = local_to_global_flat_triton(idxs, 32)
        self.assertTrue(out.stop_gradient)

    def test_dispatcher_matches_eager(self):
        """``local_to_global_flat(fused=...)`` must agree with the reference."""
        for dtype in ("int32", "int64"):
            idxs = _make_idxs(3, 32, 48, 256, dtype, self.rng)
            ref = _local_to_global_flat(idxs, 256)
            for fused in (False, True):
                got = local_to_global_flat(idxs, 256, fused=fused)
                _assert_bitwise(f"dispatch[{dtype},fused={fused}]", ref, got)


class TestConfigSwitch(unittest.TestCase):
    """``sparse_attn_global_kv_idx_remap_fusion`` must be inert numerically."""

    def test_config_field_defaults_off(self):
        from paddlefleet.transformer.transformer_config import TransformerConfig

        field = TransformerConfig.__dataclass_fields__[
            "sparse_attn_global_kv_idx_remap_fusion"
        ]
        self.assertIs(field.default, False)

    def test_cudnn_sparse_attn_end_to_end(self):
        """Flipping the switch must not move out / dq / d_sink at all.

        ``dkv`` is deliberately excluded: ``reduce_dKV_from_reg`` in
        ``paddlefleet_ops/.../sparse_attention_backward/dsa_bwd_sm100.py``
        accumulates it with ``cute.arch.atomic_add`` scattered by the top-k
        index, so it is not bitwise reproducible even eager-vs-eager whenever a
        KV column has more than one writer (measured: flakes on 19/19 repeats,
        up to ~100 bf16 elements, 1 ULP). The switch adds no deviation beyond
        that pre-existing nondeterminism.

        ``d_sink`` IS checked, but only because ``sq <= dSink_block_q``: the
        separate ``sum_dSink`` kernel launches ``ceil_div(seqlen_q, 256)``
        q-blocks and each one does its own ``atomic_add``, so a single block
        has no contention and is exactly reproducible. Raising ``sq`` above 256
        makes ``d_sink`` flake too (measured: 19/19 repeats at sq=2048,
        max|diff| ~1e-6) -- keep the assert honest by keeping sq <= 256.

        ``dq`` needs no such caveat: ``store_dQ`` writes it with a TMA store
        from the single CTA that owns each query row.
        """
        _require_sm100_sparse_kernels(self)

        from paddlefleet.fusions.csa_sparse_attn import csa_sparse_attn

        b, sq, h, d = 1, 256, 64, 512  # FlashMLA sparse fixes h_q=64, d_v=512
        assert sq <= 256, "sq must stay within one sum_dSink block (see above)"
        s_kv, topk = 384, 128
        rng = np.random.default_rng(0)
        idx = rng.integers(0, s_kv, (b, sq, topk)).astype(np.int32)
        idx[rng.random((b, sq, topk)) < 0.3] = -1
        q_np = rng.standard_normal((b, sq, h, d)).astype(np.float32)
        kv_np = rng.standard_normal((b, s_kv, d)).astype(np.float32)
        sink_np = rng.standard_normal((h,)).astype(np.float32)
        go_np = rng.standard_normal((b, sq, h * d)).astype(np.float32)

        def run(fused):
            q = paddle.to_tensor(q_np, dtype="bfloat16")
            kv = paddle.to_tensor(kv_np, dtype="bfloat16")
            sink = paddle.to_tensor(sink_np, dtype="float32")
            for t in (q, kv, sink):
                t.stop_gradient = False
            out = csa_sparse_attn(
                q,
                kv,
                sink,
                paddle.to_tensor(idx, dtype="int32"),
                d**-0.5,
                backend="cudnn",
                global_kv_idx_remap_fusion=fused,
            )
            out.backward(paddle.to_tensor(go_np, dtype=out.dtype))
            return out, q.grad, sink.grad

        ref, got = run(False), run(True)
        for name, a, c in zip(("out", "dq", "d_sink"), ref, got):
            _assert_bitwise(f"switch[{name}]", a, c)

    def test_cudnn_sparse_attn_unaligned_topk(self):
        """topk not a multiple of the arch alignment exercises the F.pad path.

        ``flash_mla_sparse_attn`` pads ``global_idxs`` up to a multiple of 64
        (SM100) with ``-1`` right after the remap, so the fused output has to
        survive that padding untouched.
        """
        _require_sm100_sparse_kernels(self)

        from paddlefleet.fusions.csa_sparse_attn import csa_sparse_attn

        b, sq, h, d = 1, 256, 64, 512
        s_kv, topk = 320, 100  # 100 -> padded to 128
        rng = np.random.default_rng(3)
        idx = rng.integers(0, s_kv, (b, sq, topk)).astype(np.int32)
        idx[rng.random((b, sq, topk)) < 0.3] = -1
        q_np = rng.standard_normal((b, sq, h, d)).astype(np.float32)
        kv_np = rng.standard_normal((b, s_kv, d)).astype(np.float32)
        sink_np = rng.standard_normal((h,)).astype(np.float32)
        go_np = rng.standard_normal((b, sq, h * d)).astype(np.float32)

        def run(fused):
            q = paddle.to_tensor(q_np, dtype="bfloat16")
            kv = paddle.to_tensor(kv_np, dtype="bfloat16")
            sink = paddle.to_tensor(sink_np, dtype="float32")
            for t in (q, kv, sink):
                t.stop_gradient = False
            out = csa_sparse_attn(
                q,
                kv,
                sink,
                paddle.to_tensor(idx, dtype="int32"),
                d**-0.5,
                backend="cudnn",
                global_kv_idx_remap_fusion=fused,
            )
            out.backward(paddle.to_tensor(go_np, dtype=out.dtype))
            return out, q.grad, sink.grad

        ref, got = run(False), run(True)
        for name, a, c in zip(("out", "dq", "d_sink"), ref, got):
            _assert_bitwise(f"unaligned[{name}]", a, c)

    def test_mqa_sparse_attn_end_to_end(self):
        """The switch is also wired into the absorbed-MQA path; pin that too.

        Swept over ``mqa_sparse_attn_backward_backend``, because the remap has a
        different reach on each branch: the FlashMLA forward always builds the
        flat-global table, while only the ``"cudnn"`` backward consumes one --
        the ``"tilelang"`` backward indexes ``token_indices`` per batch itself.

        ``dkv`` is only checked on ``"tilelang"``. On ``"cudnn"`` it comes from
        the atomic epilogue and is not reproducible even against itself, which
        is the same exclusion the CSA case makes; the deterministic kernel is
        what lets this test pin the one output that branch has to leave out.
        ``s`` stays within one ``sum_dSink`` block so ``d_sink`` is exact.
        """
        _require_sm100_sparse_kernels(self)

        from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

        b, s, h, d_qk, d_v = 1, 128, 64, 576, 512  # absorbed MQA layout
        s_kv, width = 256, 64
        assert s <= 256, "keep s within one sum_dSink block"
        rng = np.random.default_rng(11)
        tok = rng.integers(0, s_kv, (b, s, width)).astype(np.int32)
        tok[rng.random((b, s, width)) < 0.3] = -1
        q_np = rng.standard_normal((b, s, h, d_qk)).astype(np.float32)
        kv_np = rng.standard_normal((b, s_kv, d_qk)).astype(np.float32)
        sink_np = rng.standard_normal((h,)).astype(np.float32)
        go_np = rng.standard_normal((b, s, h * d_v)).astype(np.float32)

        def run(fused, backend):
            q = paddle.to_tensor(q_np, dtype="bfloat16")
            kv = paddle.to_tensor(kv_np, dtype="bfloat16")
            sink = paddle.to_tensor(sink_np, dtype="float32")
            for t in (q, kv, sink):
                t.stop_gradient = False
            out = mqa_sparse_attn(
                q,
                kv,
                paddle.to_tensor(tok, dtype="int32"),
                d_qk**-0.5,
                d_v,
                attn_sink=sink,
                global_kv_idx_remap_fusion=fused,
                backward_backend=backend,
            )
            out.backward(paddle.to_tensor(go_np, dtype=out.dtype))
            return {
                "out": out,
                "dq": q.grad,
                "dkv": kv.grad,
                "d_sink": sink.grad,
            }

        for backend in ("cudnn", "tilelang"):
            names = ("out", "dq", "d_sink")
            if backend == "tilelang":
                names += ("dkv",)
            with self.subTest(backward_backend=backend):
                ref, got = run(False, backend), run(True, backend)
                for name in names:
                    _assert_bitwise(
                        f"mqa[{backend}][{name}]", ref[name], got[name]
                    )


class TestBitwiseAgainstOracle(unittest.TestCase):
    """eager == independent numpy oracle == fused, in every scenario.

    Comparing the fusion only against the eager code would pass even if both
    were wrong, so every case is pinned to ``_numpy_oracle`` as well.
    """

    def setUp(self):
        self.rng = np.random.default_rng(1234)

    def _check3(self, case, idxs, seqlen_kv):
        oracle = _numpy_oracle(idxs, seqlen_kv)
        ref = _local_to_global_flat(idxs, seqlen_kv)
        got = local_to_global_flat_triton(idxs, seqlen_kv)
        self.assertEqual(ref.dtype, paddle.int32, case)
        self.assertEqual(got.dtype, paddle.int32, case)
        np.testing.assert_array_equal(
            ref.numpy(), oracle, err_msg=f"{case}: eager vs oracle"
        )
        np.testing.assert_array_equal(
            got.numpy(), oracle, err_msg=f"{case}: fused vs oracle"
        )

    def test_topk_block_boundaries(self):
        """BLOCK_K = min(next_pow2(topk), 1024); probe both sides of edges."""
        for topk in (
            1,
            2,
            3,
            7,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            511,
            512,
            513,
            640,
            1023,
            1024,
            1025,
            1536,
            2047,
            2048,
            3072,
            4097,
        ):
            for dtype in ("int32", "int64"):
                skv = topk * 2 + 5
                idxs = _make_idxs(2, 3, topk, skv, dtype, self.rng)
                self._check3(f"topk={topk} [{dtype}]", idxs, skv)

    def test_batch_and_seq_shapes(self):
        for b in (1, 2, 3, 7, 8, 16, 33):
            for sq in (1, 2, 7, 64):
                for dtype in ("int32", "int64"):
                    idxs = _make_idxs(b, sq, 96, 512, dtype, self.rng)
                    self._check3(f"b={b} sq={sq} [{dtype}]", idxs, 512)

    def test_pad_ratios(self):
        """No padding, partial padding, all padding."""
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            for dtype in ("int32", "int64"):
                idxs = _make_idxs(
                    4, 16, 130, 1024, dtype, self.rng, pad_ratio=ratio
                )
                self._check3(f"pad={ratio} [{dtype}]", idxs, 1024)

    def test_value_boundaries(self):
        """0, seqlen_kv-1, -1 and other negatives, hand-written per row."""
        skv = 97
        rows = [
            [0, skv - 1, -1, 1, skv - 2],
            [-1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0],
            [-5, -2, -9, -128, -1],
            [skv - 1, skv - 1, 0, -1, 3],
        ]
        for dtype in ("int32", "int64"):
            idxs = paddle.to_tensor([rows, rows, rows], dtype=dtype)
            self._check3(f"boundaries [{dtype}]", idxs, skv)

    def test_int32_overflow_matrix(self):
        """b * seqlen_kv near and beyond 2**31 must wrap identically."""
        rows = [[[0, 5, -1, 17]], [[1, -1, 9, 0]], [[3, 4, 0, -7]]]
        for skv in (2**20, 2**28, 2**30, 2**31 - 1, 2**31, 3 * 10**8):
            for dtype in ("int32", "int64"):
                idxs = paddle.to_tensor(rows, dtype=dtype)
                self._check3(f"skv={skv} [{dtype}]", idxs, skv)

    def test_ernielite_real_shapes(self):
        """The two index-table widths this actually runs on in ernielite."""
        # sq trimmed for the same reason as test_ernielite_hca_shape; the two
        # topk widths and the seqlen_kv that sets the int32 offset range -- the
        # things this test is about -- are the real ones.
        cases = [
            (1, 1024, 128 + 512, 65536 + 512),  # HCA: window + compressed
            (1, 1024, 128 + 2048, 65536 + 512),  # DSA: window + index_topk
        ]
        for b, sq, topk, skv in cases:
            for dtype in ("int32", "int64"):
                idxs = _make_idxs(b, sq, topk, skv, dtype, self.rng)
                self._check3(f"real b={b} topk={topk} [{dtype}]", idxs, skv)

    def test_non_contiguous_and_strided(self):
        """Sliced / stepped views must give the same answer as a dense copy."""
        for dtype in ("int32", "int64"):
            full = _make_idxs(4, 32, 256, 1024, dtype, self.rng)
            for name, view in (
                ("slice", full[:, 4:20, 8:200]),
                ("step2", full[:, ::2, ::2]),
                ("tail", full[1:, -8:, -65:]),
            ):
                self._check3(f"{name} [{dtype}]", view, 1024)
                dense = paddle.to_tensor(view.numpy(), dtype=dtype)
                _assert_bitwise(
                    f"{name} view vs dense [{dtype}]",
                    local_to_global_flat_triton(dense, 1024),
                    local_to_global_flat_triton(view, 1024),
                )


class TestFusedKernelProperties(unittest.TestCase):
    """Properties of the fused kernel itself, independent of the reference."""

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_repeatable_across_runs(self):
        """No atomics in the kernel, so repeated launches must be identical."""
        idxs = _make_idxs(3, 512, 640, 4096, "int32", self.rng)
        base = local_to_global_flat_triton(idxs, 4096).numpy()
        for i in range(8):
            again = local_to_global_flat_triton(idxs, 4096).numpy()
            np.testing.assert_array_equal(base, again, err_msg=f"run {i}")

    def test_output_is_fresh_storage_by_default(self):
        """The reference always allocates; the default path must match that."""
        idxs = _make_idxs(1, 32, 64, 256, "int32", self.rng)
        out = local_to_global_flat_triton(idxs, 256)
        self.assertNotEqual(out.data_ptr(), idxs.data_ptr())
        out[0, 0] = 12345  # writing the result must not touch the input
        self.assertNotEqual(int(idxs[0, 0, 0]), 12345)

    def test_alias_shares_storage_only_when_safe(self):
        """``allow_alias`` aliases exactly for b == 1 and int32, never else."""
        i32 = _make_idxs(1, 32, 64, 256, "int32", self.rng)
        self.assertEqual(
            local_to_global_flat_triton(i32, 256, allow_alias=True).data_ptr(),
            i32.data_ptr(),
        )
        i64 = _make_idxs(1, 32, 64, 256, "int64", self.rng)
        self.assertNotEqual(
            local_to_global_flat_triton(i64, 256, allow_alias=True).data_ptr(),
            i64.data_ptr(),
        )
        many = _make_idxs(3, 32, 64, 256, "int32", self.rng)
        self.assertNotEqual(
            local_to_global_flat_triton(many, 256, allow_alias=True).data_ptr(),
            many.data_ptr(),
        )

    def test_rejects_wrong_rank(self):
        for shape in ([8, 16], [2, 4, 8, 16]):
            with self.assertRaises(AssertionError):
                local_to_global_flat_triton(
                    paddle.zeros(shape, dtype="int32"), 64
                )

    def test_degenerate_shapes_skip_the_kernel(self):
        """An empty row or topk axis must return early, not launch a 0-size grid.

        ``triton.cdiv(0, BLOCK_K)`` is 0, so without the guard the launch is a
        no-op on some Triton versions and an error on others.
        """
        for b, sq, topk in ((0, 8, 16), (2, 0, 16), (2, 8, 0)):
            out = local_to_global_flat_triton(
                paddle.zeros([b, sq, topk], dtype="int32"), 64
            )
            self.assertEqual(list(out.shape), [b * sq, topk])
            self.assertEqual(out.dtype, paddle.int32)


if __name__ == "__main__":
    unittest.main()
