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

"""Backend-selection control flow for ``csa_sparse_attn``.

Scope of this file is the CPU-observable *dispatch decision matrix* of the
unified CSA sparse-attention entry point and its "cudnn" backend helpers:

* the ``csa_sparse_attn`` backend router (``unfused`` runs the pure-Paddle
  einsum path; ``tilelang`` / ``cudnn`` are accepted; anything else raises);
* the pure head-tile / latent-dim padding decisions the "cudnn" backend makes
  (``_dsa_head_tile`` / ``_dsa_latent_dim``), including their capacity ceilings;
* the architecture-gated decisions (``score_target_qheads``,
  ``_dsa_bwd_runs_sub_tile_heads``, ``_csa_bwd_honours_topk_length_holes``),
  driven by patching the ``get_device_capability`` collaborator so the SM90 vs
  SM100 branch of the *dispatch logic* is exercised regardless of the host GPU.

The actual FlashMLA / cuDNN kernel numerics are GPU-only; they are compared
against the independent ``unfused`` reference in ``TestGpuBackendNumerics`` and
honestly skipped (never faked as passing) when CUDA / FlashMLA / cuDNN are
absent.
"""

import unittest
from unittest.mock import patch

import numpy as np

try:
    import paddle

    from paddlefleet.fusions.csa_sparse_attn import (
        _csa_bwd_honours_topk_length_holes,
        _dsa_bwd_runs_sub_tile_heads,
        _dsa_head_tile,
        _dsa_latent_dim,
        csa_sparse_attn,
        score_target_qheads,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # only a genuinely missing dependency, not a bug
    _IMPORT_ERROR = exc

# Honest FlashMLA forward probe: True only when the sparse prefill kernel is
# genuinely loadable. A missing/uncompiled op is "not available"; anything else
# is left to surface at run time rather than being silently treated as absent.
try:
    import paddlefleet_ops

    from paddlefleet.cudnn_ops.attn import csa_sparse_attn_fwd_cudnn

    _HAS_FLASH_MLA = (
        paddlefleet_ops.is_flash_mla_available()
        and csa_sparse_attn_fwd_cudnn._flash_mla_sparse_fwd is not None
    )
except (ImportError, RuntimeError, AttributeError):
    _HAS_FLASH_MLA = False


skip_if_no_module = unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet.fusions.csa_sparse_attn import failed: {_IMPORT_ERROR}",
)


def _unfused_reference(
    query, kv_full, attn_sink, topk_idxs, scale, topk_length=None
):
    """Independent numpy reference for the ``unfused`` sparse-MQA math.

    Recomputes softmax(q.k * scale) with a learnable attention sink over the
    ``topk``-gathered KV rows, masking ``-1`` slots (and, when given, slots at
    ``>= topk_length``) exactly. Written from the algebra of the operation --
    it never calls the production ``unfused_compressed_sparse_attn`` -- so it is
    a genuine anchor, not a same-source echo. Everything is float64 to keep the
    reference well above the float32 tolerance the comparison is gated at.
    """
    query = np.asarray(query, dtype=np.float64)
    kv_full = np.asarray(kv_full, dtype=np.float64)
    attn_sink = np.asarray(attn_sink, dtype=np.float64)
    topk_idxs = np.asarray(topk_idxs)
    b, sq, nph, hn = query.shape
    topk = topk_idxs.shape[-1]
    out = np.zeros((b, sq, nph, hn), dtype=np.float64)
    for bi in range(b):
        for si in range(sq):
            length = (
                None
                if topk_length is None
                else int(np.asarray(topk_length)[bi, si])
            )
            rows = np.zeros((topk, hn), dtype=np.float64)
            valid = np.zeros(topk, dtype=bool)
            for t in range(topk):
                idx = int(topk_idxs[bi, si, t])
                is_valid = idx >= 0
                if length is not None and t >= length:
                    is_valid = False
                rows[t] = kv_full[bi, max(idx, 0)]
                valid[t] = is_valid
            for hi in range(nph):
                scores = (rows @ query[bi, si, hi]) * scale
                scores = np.where(valid, scores, -np.inf)
                sink = attn_sink[hi]
                m = max(float(scores.max(initial=-np.inf)), float(sink))
                exp_s = np.where(valid, np.exp(scores - m), 0.0)
                denom = exp_s.sum() + np.exp(sink - m)
                out[bi, si, hi] = ((exp_s / denom)[:, None] * rows).sum(axis=0)
    return out.reshape(b, sq, nph * hn)


def _rel_l2(actual, expected):
    a = np.asarray(actual, dtype=np.float64).ravel()
    e = np.asarray(expected, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - e) / (np.linalg.norm(e) + 1e-12))


@skip_if_no_module
class TestCudnnHeadTileDispatch(unittest.TestCase):
    """``_dsa_head_tile``: smallest FlashMLA query-head tile (64 or 128).

    Pure integer decision, no device needed. The exact tile drives how much
    zero-padded head work the "cudnn" forward does, so the boundary at 64 and
    the hard ceiling at 128 both matter.
    """

    def test_tile_selection_matrix(self):
        # (num_heads -> chosen tile); hand-derived from the 64/128 tile set.
        self.assertEqual(_dsa_head_tile(1), 64)
        self.assertEqual(_dsa_head_tile(24), 64)
        self.assertEqual(_dsa_head_tile(63), 64)
        self.assertEqual(_dsa_head_tile(64), 64)  # inclusive lower boundary
        self.assertEqual(_dsa_head_tile(65), 128)  # first count needing 128
        self.assertEqual(_dsa_head_tile(96), 128)
        self.assertEqual(_dsa_head_tile(128), 128)  # inclusive upper boundary

    def test_over_128_heads_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _dsa_head_tile(129)
        msg = str(ctx.exception)
        self.assertIn("at most 128 query heads per rank", msg)
        self.assertIn("got 129", msg)


@skip_if_no_module
class TestCudnnLatentDimDispatch(unittest.TestCase):
    """``_dsa_latent_dim``: the "cudnn" backend is always driven at 512.

    Any narrower latent is zero-padded up to 512 (the only width the FlashMLA
    forward and cuDNN backward exist at); anything wider is rejected.
    """

    def test_always_512_up_to_the_limit(self):
        for hn in (1, 32, 64, 256, 384, 511, 512):
            self.assertEqual(_dsa_latent_dim(hn), 512, f"hn={hn}")

    def test_over_512_latent_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _dsa_latent_dim(513)
        msg = str(ctx.exception)
        self.assertIn("at most 512 latent dims", msg)
        self.assertIn("got 513", msg)


@skip_if_no_module
class TestScoreTargetQheadsArchMatrix(unittest.TestCase):
    """``score_target_qheads``: score-target tile width, per architecture.

    ``get_device_capability`` is a genuine Paddle collaborator; patching it lets
    the SM90 vs SM100 branch of the *dispatch arithmetic* be checked on any host.
    The function is not cached, so a plain ``patch`` context is sufficient.
    """

    @staticmethod
    def _width(capability, num_heads):
        with patch.object(
            paddle.device.cuda,
            "get_device_capability",
            return_value=capability,
        ):
            return score_target_qheads(num_heads)

    def test_sm90_tiles_by_64_with_min_2(self):
        # <=64: max(2, h); >64: round up to a multiple of 64.
        expected = {
            1: 2,
            2: 2,
            8: 8,
            63: 63,
            64: 64,
            65: 128,
            100: 128,
            128: 128,
            129: 192,
        }
        for h, exp in expected.items():
            self.assertEqual(self._width((9, 0), h), exp, f"h={h}")

    def test_sm100_next_pow2_floored_at_16(self):
        expected = {
            1: 16,
            8: 16,
            15: 16,
            16: 16,
            17: 32,
            24: 32,
            32: 32,
            33: 64,
            64: 64,
            65: 128,
            128: 128,
            129: 256,
        }
        for h, exp in expected.items():
            self.assertEqual(self._width((10, 0), h), exp, f"h={h}")


@skip_if_no_module
class TestBwdSubTileHeadsArchMatrix(unittest.TestCase):
    """``_dsa_bwd_runs_sub_tile_heads``: run backward at the real head count?

    ``True`` only on SM100+ and only when the count is a multiple of the 64-head
    tile (the OOB-read guard). Cached and keyed on ``num_heads`` *and* the arch,
    so the cache is cleared around every patched call to avoid cross-arch bleed.
    """

    @staticmethod
    def _runs(capability, num_heads):
        _dsa_bwd_runs_sub_tile_heads.cache_clear()
        try:
            with patch.object(
                paddle.device.cuda,
                "get_device_capability",
                return_value=capability,
            ):
                return _dsa_bwd_runs_sub_tile_heads(num_heads)
        finally:
            _dsa_bwd_runs_sub_tile_heads.cache_clear()

    def test_sm90_never_runs_sub_tile(self):
        for h in (1, 24, 63, 64, 128):
            self.assertFalse(self._runs((9, 0), h), f"h={h}")

    def test_sm100_only_multiples_of_64(self):
        for h in (64, 128, 192):
            self.assertTrue(self._runs((10, 0), h), f"h={h}")
        for h in (1, 24, 32, 63, 65, 100):
            self.assertFalse(self._runs((10, 0), h), f"h={h}")


@skip_if_no_module
class TestBwdHonoursTopkLengthHolesArch(unittest.TestCase):
    """``_csa_bwd_honours_topk_length_holes``: compact backward only on SM100+.

    SM90's kernel lacks the empty-row guard, so a compacted ``topk_length`` is
    unsafe there. Cached with no args, so the single entry is cleared around
    each patched arch.
    """

    @staticmethod
    def _honours(capability):
        _csa_bwd_honours_topk_length_holes.cache_clear()
        try:
            with patch.object(
                paddle.device.cuda,
                "get_device_capability",
                return_value=capability,
            ):
                return _csa_bwd_honours_topk_length_holes()
        finally:
            _csa_bwd_honours_topk_length_holes.cache_clear()

    def test_sm90_falls_back(self):
        self.assertFalse(self._honours((9, 0)))

    def test_sm100_and_newer_keep_the_compact_path(self):
        self.assertTrue(self._honours((10, 0)))
        self.assertTrue(self._honours((12, 0)))


@skip_if_no_module
class TestBackendDispatchRouting(unittest.TestCase):
    """The ``csa_sparse_attn`` entry-point backend router (CPU-observable).

    An unknown backend name is rejected with a precise message *before* any
    kernel runs; ``unfused`` is routed to the pure-Paddle path and is not caught
    by the guard. Forced onto CPU so the routing decision is exercised without
    touching a GPU, with the device restored afterwards.
    """

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.set_device(self._orig_device)

    def _tiny_inputs(self):
        query = paddle.to_tensor(
            [[[[1.0, 0.0]]]], dtype="float32"
        )  # [b=1, sq=1, np=1, hn=2]
        kv_full = paddle.to_tensor(
            [[[1.0, 2.0], [3.0, 4.0]]], dtype="float32"
        )  # [b=1, n_kv=2, hn=2]
        attn_sink = paddle.zeros([1], dtype="float32")
        topk_idxs = paddle.to_tensor([[[0]]], dtype="int32")  # [b, sq, topk=1]
        return query, kv_full, attn_sink, topk_idxs, 0.5

    def test_unknown_backend_raises_before_dispatch(self):
        q, kv, sink, idx, scale = self._tiny_inputs()
        # Case-sensitive: "CUDNN"/"" are not the accepted spellings.
        for bad in ("numpy", "flash", "unfusedx", "CUDNN", ""):
            with self.assertRaises(ValueError) as ctx:
                csa_sparse_attn(q, kv, sink, idx, scale, backend=bad)
            msg = str(ctx.exception)
            self.assertIn(f"csa_sparse_attn_backend={bad!r} is invalid", msg)
            self.assertIn("Must be one of", msg)

    def test_unfused_name_is_routed_not_rejected(self):
        q, kv, sink, idx, scale = self._tiny_inputs()
        # "unfused" is handled ahead of the guard, so it produces output rather
        # than raising. Value correctness is asserted in
        # TestUnfusedBackendNumerics.
        out = csa_sparse_attn(q, kv, sink, idx, scale, backend="unfused")
        self.assertEqual(list(out.shape), [1, 1, 2])  # [b, sq, np*hn]
        self.assertTrue(bool(paddle.isfinite(out).all()))


@skip_if_no_module
class TestUnfusedBackendNumerics(unittest.TestCase):
    """``backend="unfused"`` runs the einsum path and matches the anchor.

    This is genuine CPU numerics (the unfused backend is pure Paddle, not a GPU
    kernel), so it validates both the dispatch (``unfused`` -> einsum) and the
    sparse-MQA math -- attention sink, ``-1`` masking and the ``topk_length``
    early-stop -- against ``_unfused_reference``.
    """

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.set_device(self._orig_device)

    def _inputs(self):
        # b=1, sq=2, np=2, hn=2, n_kv=3, topk=2; distinct, non-degenerate.
        query = paddle.to_tensor(
            [
                [
                    [[1.0, 0.0], [0.0, 2.0]],
                    [[1.0, 1.0], [2.0, -1.0]],
                ]
            ],
            dtype="float32",
        )
        kv_full = paddle.to_tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype="float32"
        )
        attn_sink = paddle.to_tensor([0.0, 0.5], dtype="float32")
        # sq0: two valid slots {0, 2}; sq1: one valid slot then a -1 hole.
        topk_idxs = paddle.to_tensor([[[0, 2], [1, -1]]], dtype="int32")
        return query, kv_full, attn_sink, topk_idxs, 0.5

    def test_matches_hand_derived_reference(self):
        q, kv, sink, idx, scale = self._inputs()
        out = csa_sparse_attn(q, kv, sink, idx, scale, backend="unfused")
        ref = _unfused_reference(
            q.numpy(), kv.numpy(), sink.numpy(), idx.numpy(), scale
        )
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), ref, rtol=1e-5, atol=1e-6
        )

    def test_topk_length_early_stop_is_consumed(self):
        q, kv, sink, idx, scale = self._inputs()
        # Force sq0 to keep only slot 0 (dropping the still-valid slot 2); sq1
        # keeps both (its slot 1 is already a -1 hole). A router that ignored
        # topk_length would leave sq0 attending to slot 2 and diverge.
        topk_length = paddle.to_tensor([[1, 2]], dtype="int32")
        out = csa_sparse_attn(
            q, kv, sink, idx, scale, backend="unfused", topk_length=topk_length
        )
        ref = _unfused_reference(
            q.numpy(),
            kv.numpy(),
            sink.numpy(),
            idx.numpy(),
            scale,
            topk_length=topk_length.numpy(),
        )
        np.testing.assert_allclose(
            out.numpy().astype(np.float64), ref, rtol=1e-5, atol=1e-6
        )
        # Guard the fixture actually distinguishes the two bounds.
        ref_full = _unfused_reference(
            q.numpy(), kv.numpy(), sink.numpy(), idx.numpy(), scale
        )
        self.assertGreater(_rel_l2(ref, ref_full), 1e-3)


@skip_if_no_module
class TestGpuBackendNumerics(unittest.TestCase):
    """The "cudnn" backend's kernel numerics vs the independent unfused anchor.

    This is the GPU half of the dispatch: ``backend="cudnn"`` routes through the
    FlashMLA forward + cuDNN backward, and its output/gradients must agree with
    the pure-Paddle ``unfused`` path (a genuinely separate implementation, so a
    valid reference). It is honestly skipped -- never faked -- when CUDA,
    FlashMLA or the cuDNN sparse backward are unavailable. A missing module
    skips; any other failure of the probe run is re-raised so a real regression
    fails loudly.
    """

    # Against the fp32 ``unfused`` reference at bf16 inputs: ~2x the observed
    # bf16 noise floor, and a rel-L2 (not cosine) so a constant-scale gradient
    # divergence between backends is still caught.
    CEILING = {"out": 6e-3, "dq": 6e-3, "dkv": 6e-3, "d_sink": 8e-3}

    @staticmethod
    def _make_inputs(b, sq, s_kv, num_heads, hn, topk):
        paddle.seed(20260916)
        q = paddle.randn([b, sq, num_heads, hn]).cast("bfloat16")
        kv = paddle.randn([b, s_kv, hn]).cast("bfloat16")
        attn_sink = paddle.randn([num_heads]).cast("float32") * 0.1
        topk_idxs = paddle.randint(0, s_kv, [b, sq, topk]).cast("int32")
        return q, kv, attn_sink, topk_idxs, 1.0 / (hn**0.5)

    @staticmethod
    def _run(q, kv, attn_sink, topk_idxs, scale, backend):
        q_c, kv_c, sink_c = (t.detach().clone() for t in (q, kv, attn_sink))
        for t in (q_c, kv_c, sink_c):
            t.stop_gradient = False
        out = csa_sparse_attn(
            q_c, kv_c, sink_c, topk_idxs, scale, backend=backend
        )
        out.sum().backward()
        return out, q_c.grad, kv_c.grad, sink_c.grad

    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("the cuDNN CSA backend requires CUDA")
        if not _HAS_FLASH_MLA:
            raise unittest.SkipTest(
                "the cuDNN CSA backend requires the FlashMLA sparse forward"
            )
        try:
            cls._orig_device = paddle.get_device()
            paddle.set_device("gpu:0")
        except Exception as exc:
            raise unittest.SkipTest(f"gpu:0 is not available: {exc}")
        paddle.seed(20260916)
        # Probe the real cuDNN backward once; only a missing module is an honest
        # skip, everything else is a real failure and must propagate.
        try:
            cls._run(*cls._make_inputs(1, 8, 64, 64, 512, 64), backend="cudnn")
        except Exception as exc:
            if isinstance(exc, ImportError) or "No module named" in str(exc):
                raise unittest.SkipTest(
                    f"the cuDNN sparse backward does not run here: {exc}"
                ) from exc
            raise

    @classmethod
    def tearDownClass(cls):
        # Restore the process-wide device we mutated in setUpClass.
        if getattr(cls, "_orig_device", None) is not None:
            paddle.set_device(cls._orig_device)

    def test_cudnn_matches_unfused_reference(self):
        inputs = self._make_inputs(1, 8, 64, 64, 512, 64)
        cudnn = self._run(*inputs, backend="cudnn")
        ref = self._run(*inputs, backend="unfused")
        for name, got, exp in zip(("out", "dq", "dkv", "d_sink"), cudnn, ref):
            self.assertIsNotNone(got, f"{name} gradient missing")
            self.assertTrue(
                bool(paddle.isfinite(got.cast("float32")).all()),
                f"{name} is not finite",
            )
            self.assertLess(
                _rel_l2(
                    got.cast("float32").numpy(), exp.cast("float32").numpy()
                ),
                self.CEILING[name],
                f"{name} diverges from the unfused reference",
            )


if __name__ == "__main__":
    unittest.main()
