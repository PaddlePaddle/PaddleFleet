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

"""Reproducibility of the tilelang CSA/DSA indexer forward and backward.

The forward has no atomics and no RNG, so it is bit-reproducible as written; the
tests below pin that as a contract rather than an accident. The backward scatters
into ``dIndexKComp``, which is where run-to-run drift comes from: the default path
uses ``T.atomic_add``, so the fp32 accumulation order follows block scheduling.
``FLAGS_cudnn_deterministic=1`` switches to a staging buffer plus a CSR-ordered
reduction, which these tests exercise -- including the row-chunked regime, where
the staging buffer is capped and chunks are accumulated in ascending row order.
"""

import unittest
from importlib import import_module

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)


def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _inputs(b, sq, sk, h, d, topk, seed=2026, dtype="bfloat16"):
    """Indexer backward inputs with a deliberately collision-heavy index table.

    Every row selects the same few compressed positions, so many rows scatter
    into one output slot. That is what makes the atomic path drift and the CSR
    path worth testing.
    """
    paddle.seed(seed)
    q = paddle.randn([b, sq, h, d]).astype(dtype)
    k = paddle.randn([b, sk, d]).astype(dtype)
    w = paddle.randn([b, sq, h]).astype("float32")
    idx = paddle.randint(0, sk, [b, sq, topk]).astype("int32")
    # Punch some invalid slots; they must contribute nothing on either path.
    idx = paddle.where(
        paddle.randint(0, 8, [b, sq, topk]) == 0,
        paddle.full_like(idx, -1),
        idx,
    )
    grad = paddle.randn([b, sq, topk]).astype("float32")
    return q, k, w, idx, grad


def _bit_equal(a, b):
    """Exact equality that works for bf16/fp16.

    ``paddle.equal_all`` has no bf16 kernel, and the widening cast to fp32 is
    lossless, so comparing the casts is still an exact bit comparison of the
    originals.
    """
    if a.dtype in (paddle.bfloat16, paddle.float16):
        a, b = a.cast("float32"), b.cast("float32")
    return paddle.equal_all(a, b).item()


class _DeterministicFlag:
    """Scoped ``FLAGS_cudnn_deterministic`` so a failure cannot leak the flag."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.saved = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
            "FLAGS_cudnn_deterministic"
        ]
        paddle.set_flags({"FLAGS_cudnn_deterministic": self.value})
        return self

    def __exit__(self, *exc):
        paddle.set_flags({"FLAGS_cudnn_deterministic": self.saved})
        return False


class TestTilelangIndexerForwardReproducible(unittest.TestCase):
    def setUp(self):
        _cuda_or_skip(self)

    def test_forward_is_bit_identical_across_runs(self):
        from paddlefleet.tilelang_ops import csa_indexer_topk_fwd

        b, sq, sk, h, d = 1, 64, 64, 64, 128
        q, k, w, _, _ = _inputs(b, sq, sk, h, d, topk=32)
        valid_range = paddle.stack(
            [
                paddle.zeros([b, sq], dtype="int32"),
                paddle.full([b, sq], sk, dtype="int32"),
            ],
            axis=-1,
        )
        first = csa_indexer_topk_fwd(
            q, k, w, ratio=1, topk_effective=sk, valid_range=valid_range
        )
        for _ in range(2):
            again = csa_indexer_topk_fwd(
                q, k, w, ratio=1, topk_effective=sk, valid_range=valid_range
            )
            self.assertTrue(
                _bit_equal(first[0], again[0]),
                "indexer forward columns are not reproducible",
            )
            self.assertTrue(
                _bit_equal(first[1], again[1]),
                "indexer forward scores are not reproducible",
            )


class TestTilelangIndexerBackwardDeterministic(unittest.TestCase):
    def setUp(self):
        _cuda_or_skip(self)
        self.b, self.sq, self.sk, self.h, self.d = 1, 64, 32, 64, 128
        self.topk = 32

    def _bwd(self, deterministic, chunk_elems=None, b=None, sq=None):
        # ``import_module`` on purpose: ``indexer/__init__.py`` re-exports a
        # *function* named ``csa_indexer_bwd``, so ``from ... import
        # csa_indexer_bwd`` hands back the function, not the module.
        mod = import_module("paddlefleet.tilelang_ops.indexer.csa_indexer_bwd")

        q, k, w, idx, grad = _inputs(
            self.b if b is None else b,
            self.sq if sq is None else sq,
            self.sk,
            self.h,
            self.d,
            self.topk,
        )
        saved = mod._DET_ROW_CHUNK_ELEMS
        if chunk_elems is not None:
            mod._DET_ROW_CHUNK_ELEMS = chunk_elems
        try:
            with _DeterministicFlag(deterministic):
                return mod.csa_indexer_bwd_interface(q, w, k, idx, grad)
        finally:
            mod._DET_ROW_CHUNK_ELEMS = saved

    def test_deterministic_backward_is_bit_identical(self):
        first = self._bwd(True)
        for _ in range(3):
            again = self._bwd(True)
            for name, a, bb in zip(
                ("grad_q", "grad_weights", "grad_k_comp"), first, again
            ):
                self.assertTrue(
                    _bit_equal(a, bb),
                    f"{name} is not reproducible under "
                    "FLAGS_cudnn_deterministic=1",
                )

    def test_row_chunked_backward_is_bit_identical(self):
        # One row per chunk: exercises the tail path and the cross-chunk
        # accumulation order.
        chunk = self.b * self.topk * self.d
        first = self._bwd(True, chunk_elems=chunk)
        again = self._bwd(True, chunk_elems=chunk)
        for name, a, bb in zip(
            ("grad_q", "grad_weights", "grad_k_comp"), first, again
        ):
            self.assertTrue(
                _bit_equal(a, bb),
                f"{name} is not reproducible with a one-row chunk",
            )

    def test_chunked_matches_unchunked_within_fp32_regrouping(self):
        # grad_q / grad_weights are per-row and must match exactly; grad_k_comp
        # is the reduced one, so only its value is compared, not its bits.
        whole = self._bwd(True)
        chunked = self._bwd(True, chunk_elems=self.b * self.topk * self.d)
        self.assertTrue(_bit_equal(whole[0], chunked[0]))
        self.assertTrue(_bit_equal(whole[1], chunked[1]))
        diff = (whole[2] - chunked[2]).abs().max().item()
        scale = whole[2].abs().max().item()
        self.assertLess(diff, 1e-4 * max(scale, 1.0))

    def test_deterministic_matches_atomic_path(self):
        det = self._bwd(True)
        atomic = self._bwd(False)
        for name, a, bb in zip(
            ("grad_q", "grad_weights", "grad_k_comp"), det, atomic
        ):
            a32, b32 = a.cast("float32"), bb.cast("float32")
            diff = (a32 - b32).abs().max().item()
            scale = max(a32.abs().max().item(), 1.0)
            self.assertLess(
                diff,
                1e-3 * scale,
                f"{name} disagrees between the deterministic and atomic paths",
            )

    def test_batch_gt_one_with_tail_matches_unchunked(self):
        # Review P1: the tail chunk used to be ``dindexk_buf[:, :rows]``, whose
        # batch stride is that of the *full* buffer while the kernel addresses
        # its argument as dense -- so every batch past the first read and wrote
        # at the wrong offset. Only reproducible with batch > 1 *and* a tail.
        b, sq = 2, 20
        chunk = b * 7 * self.topk * self.d  # 7 rows -> 20 = 2*7 + tail 6
        whole = self._bwd(
            True, chunk_elems=b * sq * self.topk * self.d, b=b, sq=sq
        )
        chunked = self._bwd(True, chunk_elems=chunk, b=b, sq=sq)
        self.assertTrue(_bit_equal(whole[0], chunked[0]), "grad_q")
        self.assertTrue(_bit_equal(whole[1], chunked[1]), "grad_weights")
        diff = (whole[2] - chunked[2]).abs().max().item()
        scale = max(whole[2].abs().max().item(), 1.0)
        self.assertLess(diff, 1e-4 * scale, "grad_k_comp")

    def test_empty_sequence_returns_empty_grads(self):
        # Review P1: ``rows_per_chunk = min(..., seq_len)`` was 0 for an empty
        # sequence, and ``range(start, stop, 0)`` raises. The atomic path returns
        # empty gradients for it, so the deterministic one must too.
        out = self._bwd(True, sq=0)
        self.assertEqual(list(out[0].shape), [self.b, 0, self.h, self.d])
        self.assertEqual(list(out[1].shape), [self.b, 0, self.h])
        self.assertEqual(list(out[2].shape), [self.b, self.sk, self.d])
        self.assertEqual(float(out[2].abs().sum().item()), 0.0)

    def test_only_one_staging_buffer_is_alive_at_a_time(self):
        # Review P1 (second round): giving the tail its own allocation must not
        # leave the full-size buffer alive alongside it, or the peak becomes
        # ~2x the advertised budget. Track liveness with weakrefs rather than
        # counting allocations, since reuse across full chunks is also expected.
        import weakref

        live, peak = set(), 0
        real_empty = paddle.empty

        def spy(*args, **kwargs):
            nonlocal peak
            t = real_empty(*args, **kwargs)
            if (
                args
                and isinstance(args[0], list | tuple)
                and len(args[0]) == 4
                and str(kwargs.get("dtype", ""))
                in ("float32", "paddle.float32")
            ):
                key = id(t)
                live.add(key)
                weakref.finalize(t, live.discard, key)
                peak = max(peak, len(live))
            return t

        paddle.empty = spy
        try:
            # 7 rows per chunk over 20 rows -> 2 full chunks + a 6-row tail.
            self._bwd(True, chunk_elems=7 * self.topk * self.d, sq=20)
        finally:
            paddle.empty = real_empty
        self.assertEqual(
            peak, 1, f"{peak} staging buffers were alive at once, expected 1"
        )

    def test_staging_buffer_respects_the_row_chunk_budget(self):
        # The point of the chunking is the bound, so assert it directly instead
        # of trusting that a large shape merely happened not to OOM.
        seen = []
        real_empty = paddle.empty

        def spy(*args, **kwargs):
            # ``paddle.empty`` accepts both ``empty([d0, d1])`` and
            # ``empty(d0, d1)``. The staging buffer is the 4-D fp32 one; the
            # per-chunk ``grad_q`` is also 4-D but carries the input dtype, so
            # filter on both.
            if (
                args
                and isinstance(args[0], list | tuple)
                and len(args[0]) == 4
                and str(kwargs.get("dtype", ""))
                in ("float32", "paddle.float32")
            ):
                seen.append(list(args[0]))
            return real_empty(*args, **kwargs)

        paddle.empty = spy
        try:
            self._bwd(True, chunk_elems=8 * self.topk * self.d)
        finally:
            paddle.empty = real_empty
        self.assertTrue(seen, "no staging buffer was allocated")
        for shape in seen:
            self.assertLessEqual(
                shape[0] * shape[1] * shape[2] * shape[3],
                8 * self.topk * self.d,
                f"staging buffer {shape} exceeds the row-chunk budget",
            )
            self.assertEqual(shape[1], 8)


if __name__ == "__main__":
    unittest.main()
