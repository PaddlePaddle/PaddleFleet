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

"""Query-head counts below a kernel tile on the CSA "cudnn" backend.

FlashMLA's sparse prefill only instantiates 64- and 128-head tiles, so an HCA /
CSA layer with 32 or 24 heads zero-pads the forward up to the tile and (on
SM100) drives the backward at the real head count. Three properties are pinned:

1. agreement with the fp32 ``unfused`` reference at the same error level as the
   tile-native 64-head case -- the padding must not cost accuracy;
2. non-interference -- whatever occupies the padded rows cannot change the real
   heads' ``out`` / ``dq``, bit-for-bit;
3. the backward really runs at the real head count on SM100, which is what stops
   the padded forward from making a sub-tile layer cost more than a 64-head one.
"""

import functools
import unittest

import paddle

try:
    import paddlefleet_ops

    from paddlefleet.cudnn_ops.attn import csa_sparse_attn_fwd_cudnn

    _HAS_FLASH_MLA = (
        paddlefleet_ops.is_flash_mla_available()
        and csa_sparse_attn_fwd_cudnn._flash_mla_sparse_fwd is not None
    )
except (ImportError, RuntimeError, AttributeError):
    _HAS_FLASH_MLA = False

try:
    import paddlefleet_ops

    from paddlefleet.cudnn_ops import csa_sparse_attn_bwd_cudnn

    _HAS_CUDNN_FRONTEND = paddlefleet_ops.is_cudnn_frontend_available() and (
        callable(csa_sparse_attn_bwd_cudnn)
    )
except (ImportError, RuntimeError, AttributeError):
    _HAS_CUDNN_FRONTEND = False

_HEAD_TILE = 64
_D = 512

# b, sq, skv, topk, invalid_frac, name. ``invalid_frac`` seeds ``-1`` columns
# anywhere in the row (not just a trailing pad), which is what document masking
# produces and what the backward's trailing ``topk_length`` bound must survive.
_SHAPES = [
    (1, 128, 256, 64, 0.0, "dense"),
    (1, 128, 256, 64, 0.3, "holes"),
    (2, 256, 512, 192, 0.2, "batch2-topk192"),
    (1, 64, 8192, 192, 0.1, "hca-like"),
    (1, 1, 64, 64, 0.0, "single-token"),
]


def _rel_l2(a, b):
    a_f = a.flatten().cast("float32")
    b_f = b.flatten().cast("float32")
    return float(
        paddle.linalg.norm(a_f - b_f) / (paddle.linalg.norm(b_f) + 1e-12)
    )


def _max_abs_diff(a, b):
    return float((a.cast("float32") - b.cast("float32")).abs().max())


def _make_inputs(b, sq, skv, num_heads, topk, invalid_frac, seed=0, sink=None):
    paddle.seed(seed)
    q = paddle.randn([b, sq, num_heads, _D]).cast("bfloat16")
    kv = paddle.randn([b, skv, _D]).cast("bfloat16")
    attn_sink = (paddle.randn([num_heads]) * 0.5).cast("float32")
    if sink is not None:
        attn_sink = _make_sink(num_heads, sink)
    topk_idxs = paddle.randint(0, skv, [b, sq, topk]).cast("int32")
    if invalid_frac > 0:
        holes = paddle.rand([b, sq, topk]) < invalid_frac
        topk_idxs = paddle.where(
            holes, paddle.full_like(topk_idxs, -1), topk_idxs
        )
        # Keep one valid column per row; an all-invalid row has no reference.
        topk_idxs[:, :, 0] = paddle.randint(0, skv, [b, sq]).cast("int32")
    return q, kv, attn_sink, topk_idxs, 1.0 / _D**0.5


def _make_sink(num_heads, spec):
    """``spec`` is a constant logit, or "split" for half dominant/half ignored."""
    if spec == "split":
        half = num_heads // 2
        return paddle.concat(
            [
                paddle.full([half], 8.0, dtype="float32"),
                paddle.full([num_heads - half], -8.0, dtype="float32"),
            ]
        )
    return paddle.full([num_heads], float(spec), dtype="float32")


def _run(q, kv, attn_sink, topk_idxs, scale, backend, junk_heads=0):
    """One fwd+bwd. ``junk_heads`` appends random extra heads.

    Returns the first ``num_heads`` heads of ``out`` and ``dq`` plus ``dkv`` and
    ``d_sink``, so a junk-head run stays comparable to a plain one.
    """
    from paddlefleet.fusions.csa_sparse_attn import csa_sparse_attn

    b, sq, num_heads, _ = q.shape
    if junk_heads:
        q = paddle.concat(
            [q, paddle.randn([b, sq, junk_heads, _D]).cast(q.dtype)], axis=2
        )
        attn_sink = paddle.concat(
            [attn_sink, (paddle.randn([junk_heads]) * 0.5).cast("float32")],
            axis=0,
        )
    q = q.detach().clone()
    q.stop_gradient = False
    kv = kv.detach().clone()
    kv.stop_gradient = False
    attn_sink = attn_sink.detach().clone()
    attn_sink.stop_gradient = False

    out = csa_sparse_attn(q, kv, attn_sink, topk_idxs, scale, backend=backend)
    out.sum().backward()
    out = out.reshape([b, sq, num_heads + junk_heads, _D])[:, :, :num_heads, :]
    return {
        "out": out,
        "dq": q.grad[:, :, :num_heads, :],
        "dkv": kv.grad,
        "d_sink": attn_sink.grad[:num_heads],
    }


# Ceilings are ~2x the worst observed bf16 error of the tile-native 64-head
# case; ``test_no_accuracy_loss_vs_tile_native`` additionally ties the sub-tile
# error to the 64-head one, which is the part that catches a real regression.
_REL_L2_CEILING = {"out": 6e-3, "dq": 6e-3, "dkv": 6e-3, "d_sink": 8e-3}


@functools.lru_cache(maxsize=1)
def _cudnn_sparse_bwd_runs():
    """Whether the cuDNN sparse backward can actually execute here.

    ``is_cudnn_frontend_available()`` is not a usable proxy. Some builds of the
    vendored frontend import a *top-level* ``cudnn`` module while executing,
    which the loader has already renamed to ``paddlefleet_ops.cudnn``, so the
    call dies with ``ModuleNotFoundError`` while the probe says "available".
    One tile-native backward settles it; anything other than a missing module
    is re-raised, so a real kernel regression still fails loudly.
    """
    if not (_HAS_FLASH_MLA and _HAS_CUDNN_FRONTEND):
        return False
    try:
        _run(*_make_inputs(1, 8, 64, _HEAD_TILE, 64, 0.0), backend="cudnn")
    except Exception as exc:
        if isinstance(exc, ImportError) or "No module named" in str(exc):
            return False
        raise
    return True


def _skip_without_cudnn_sparse_bwd():
    if not _cudnn_sparse_bwd_runs():
        raise unittest.SkipTest(
            "the cuDNN sparse backward does not run in this environment"
        )


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "sub-tile head tests require CUDA"
)
class TestSubTileHeads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _skip_without_cudnn_sparse_bwd()

    def test_matches_unfused_reference(self):
        for num_heads in (32, 24):
            for b, sq, skv, topk, inv, name in _SHAPES:
                with self.subTest(heads=num_heads, shape=name):
                    args = _make_inputs(b, sq, skv, num_heads, topk, inv)
                    ref = _run(*args, backend="unfused")
                    got = _run(*args, backend="cudnn")
                    for key, ceiling in _REL_L2_CEILING.items():
                        self.assertLess(
                            _rel_l2(got[key], ref[key]),
                            ceiling,
                            f"{key} diverges from the fp32 reference",
                        )

    def test_no_accuracy_loss_vs_tile_native(self):
        """Padding must not make a sub-tile layer less accurate than h=64.

        ``d_sink`` is excluded: it is one value per head, so its relative L2 is
        a 24- to 64-element statistic that swings an order of magnitude with the
        head count alone (measured 2.1e-3 at h=64, 2.4e-4 at h=32, 1.2e-3 at
        h=24 on SM100, and the same numbers whether the backward runs at the
        real head count or fully padded). A ratio against the 64-head draw
        therefore says nothing here; the sink is bounded absolutely by
        ``test_matches_unfused_reference`` and ``test_sink_magnitudes``, and
        pinned bit-exactly by ``TestBackwardHeadWidth``.
        """
        ratio_keys = [key for key in _REL_L2_CEILING if key != "d_sink"]
        b, sq, skv, topk, inv = 2, 256, 512, 192, 0.2
        baseline = _make_inputs(b, sq, skv, _HEAD_TILE, topk, inv)
        base_ref = _run(*baseline, backend="unfused")
        base_got = _run(*baseline, backend="cudnn")
        base_err = {
            key: _rel_l2(base_got[key], base_ref[key]) for key in ratio_keys
        }
        for num_heads in (32, 24):
            args = _make_inputs(b, sq, skv, num_heads, topk, inv)
            ref = _run(*args, backend="unfused")
            got = _run(*args, backend="cudnn")
            for key in ratio_keys:
                with self.subTest(heads=num_heads, tensor=key):
                    self.assertLess(
                        _rel_l2(got[key], ref[key]),
                        2.0 * base_err[key],
                        f"{key} is materially worse than at {_HEAD_TILE} heads",
                    )

    def test_padded_rows_cannot_leak(self):
        """Real heads are bit-identical whatever occupies the padded rows.

        The library pads with zeros; here the same call is repeated with
        *random* q and *random* finite sinks in the trailing rows. ``out`` and
        ``dq`` are per-head, so they must not move at all. ``dkv`` / ``d_sink``
        are sums over heads and legitimately change, so they are not compared.
        """
        for num_heads in (32, 24):
            for b, sq, skv, topk, inv, name in _SHAPES:
                with self.subTest(heads=num_heads, shape=name):
                    args = _make_inputs(b, sq, skv, num_heads, topk, inv)
                    padded = _run(*args, backend="cudnn")
                    junked = _run(
                        *args,
                        backend="cudnn",
                        junk_heads=_HEAD_TILE - num_heads,
                    )
                    for key in ("out", "dq"):
                        self.assertEqual(
                            _max_abs_diff(padded[key], junked[key]),
                            0.0,
                            f"{key} depends on the padded rows",
                        )

    def test_sink_magnitudes(self):
        """The learnable sink must survive the padding at any magnitude.

        The padded rows carry a ``-1e30`` sink while the real ones keep the
        learnable logit, and ``d_sink`` comes back from a differently shaped
        kernel call, so the sink is the part of this path most likely to break
        silently. A dominant sink (``+8``: the sink wins the denominator and the
        output nearly vanishes) and a per-head split are the cases where a
        leaked pad row or a mis-aligned ``d_sink`` slice would show up.
        """
        for num_heads in (32, 24):
            for spec in (-8.0, 0.0, 8.0, "split"):
                with self.subTest(heads=num_heads, sink=spec):
                    args = _make_inputs(
                        2, 256, 512, num_heads, 192, 0.2, sink=spec
                    )
                    ref = _run(*args, backend="unfused")
                    got = _run(*args, backend="cudnn")
                    for key, ceiling in _REL_L2_CEILING.items():
                        self.assertLess(
                            _rel_l2(got[key], ref[key]),
                            ceiling,
                            f"{key} diverges at sink={spec}",
                        )

    def test_odd_head_count_uses_the_padded_backward(self):
        """An odd head count must still be correct, via the padded backward.

        The cuDNN backward reads the ``[N, H]`` fp32 LSE with a 2-float vector
        access and dies with CUDA 716 on an odd ``H``, so those fall back to the
        (even) padded tile. Without the fallback this test crashes the process
        rather than failing, which is exactly why it is here.
        """
        from paddlefleet.fusions.csa_sparse_attn import (
            _dsa_bwd_runs_sub_tile_heads,
        )

        self.assertFalse(_dsa_bwd_runs_sub_tile_heads(31))
        args = _make_inputs(1, 128, 256, 31, 64, 0.2)
        ref = _run(*args, backend="unfused")
        got = _run(*args, backend="cudnn")
        for key, ceiling in _REL_L2_CEILING.items():
            self.assertLess(_rel_l2(got[key], ref[key]), ceiling, key)


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "sub-tile head tests require CUDA"
)
@unittest.skipUnless(
    _HAS_FLASH_MLA, "sub-tile head forward tests require FlashMLA"
)
class TestSubTileHeadsForward(unittest.TestCase):
    """Forward-only properties, so they also run where the backward cannot."""

    def test_rejects_more_heads_than_the_widest_tile(self):
        from paddlefleet.fusions.csa_sparse_attn import csa_sparse_attn

        args = _make_inputs(1, 8, 64, 160, 64, 0.0)
        with self.assertRaisesRegex(ValueError, "at most 128 query heads"):
            csa_sparse_attn(*args, backend="cudnn")

    def test_indexer_lse_is_returned_at_the_real_head_count(self):
        """``lse_indexer`` feeds the indexer loss and must lose the pad too."""
        from paddlefleet.fusions.csa_sparse_attn import (
            _pad_query_heads,
            csa_sparse_attn,
        )

        indexer_topk = 512
        for num_heads in (32, 24):
            with self.subTest(heads=num_heads):
                q, kv, sink, idxs, scale = _make_inputs(
                    1, 64, 1024, num_heads, indexer_topk + 128, 0.0
                )
                _, lse_indexer = csa_sparse_attn(
                    q,
                    kv,
                    sink,
                    idxs,
                    scale,
                    backend="cudnn",
                    indexer_topk=indexer_topk,
                )
                self.assertEqual(list(lse_indexer.shape), [1, 64, num_heads])
                q_pad, sink_pad = _pad_query_heads(q, sink, _HEAD_TILE)
                _, lse_ref = csa_sparse_attn(
                    q_pad,
                    kv,
                    sink_pad,
                    idxs,
                    scale,
                    backend="cudnn",
                    indexer_topk=indexer_topk,
                )
                self.assertEqual(
                    _max_abs_diff(lse_indexer, lse_ref[:, :, :num_heads]),
                    0.0,
                )


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "sub-tile head tests require CUDA"
)
class TestBackwardHeadWidth(unittest.TestCase):
    """The backward is driven at the real head count, not the forward's tile.

    That is the whole reason a 32-head layer costs about what a 64-head one does
    instead of ~1.6x: only the forward pays for the padded tile. If the cuDNN
    kernel ever stops predicating its partial head tile this test fails, and
    ``_dsa_bwd_runs_sub_tile_heads`` has to be switched off for the arch.
    """

    @classmethod
    def setUpClass(cls):
        _skip_without_cudnn_sparse_bwd()

    def _bwd(self, q, kv, out, dout, lse, sink, gidx, scale, num_heads):
        from paddlefleet.fusions.csa_sparse_attn import _csa_compact_topk_idxs

        b, sq = q.shape[0], q.shape[1]
        s_kv = kv.shape[1]
        # The backward's compact KV-load path is unguarded against interior -1,
        # so ``topk_length`` must come from compacted indices (production does
        # the same in ``csa_attention.py``). Passing the trailing bound of
        # ``_csa_compute_topk_length`` together with holey indices corrupts
        # dq/dkv by ~50%, or raises CUDA 700 / yields nan. Compacting is
        # order-preserving, so the bit-exact assertions below still hold.
        gidx, topk_length = _csa_compact_topk_idxs(gidx)
        return csa_sparse_attn_bwd_cudnn(
            q.reshape([b * sq, num_heads, _D]),
            kv.reshape([b * s_kv, _D]),
            out.reshape([b * sq, num_heads, _D]),
            dout.reshape([b * sq, num_heads, _D]),
            lse.reshape([b * sq, num_heads]),
            sink,
            gidx,
            softmax_scale=scale,
            topk_length=topk_length,
        )

    def test_real_head_count_matches_the_padded_backward(self):
        from paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
            flash_mla_sparse_attn,
        )
        from paddlefleet.fusions.csa_sparse_attn import (
            _dsa_bwd_runs_sub_tile_heads,
            _pad_query_heads,
        )
        from paddlefleet.fusions.csa_sparse_attn_utils import (
            _local_to_global_flat,
        )

        for num_heads in (32, 24):
            if not _dsa_bwd_runs_sub_tile_heads(num_heads):
                self.skipTest("this arch keeps the fully padded backward")
            for b, sq, skv, topk, inv, name in _SHAPES:
                with self.subTest(heads=num_heads, shape=name):
                    q, kv, sink, idxs, scale = _make_inputs(
                        b, sq, skv, num_heads, topk, inv
                    )
                    dout = paddle.randn([b, sq, num_heads, _D]).cast("bfloat16")
                    q_pad, sink_pad = _pad_query_heads(q, sink, _HEAD_TILE)
                    dout_pad, _ = _pad_query_heads(dout, sink, _HEAD_TILE)
                    out_pad, lse_pad, _ = flash_mla_sparse_attn(
                        q_pad, kv, sink_pad, idxs, sm_scale=scale
                    )
                    gidx = _local_to_global_flat(idxs, skv)

                    dq_t, dkv_t, dsink_t = self._bwd(
                        q_pad,
                        kv,
                        out_pad,
                        dout_pad,
                        lse_pad,
                        sink_pad,
                        gidx,
                        scale,
                        _HEAD_TILE,
                    )
                    dq_h, dkv_h, dsink_h = self._bwd(
                        q,
                        kv,
                        out_pad.reshape([b, sq, _HEAD_TILE, _D])[
                            :, :, :num_heads, :
                        ].contiguous(),
                        dout,
                        lse_pad[:, :, :num_heads].contiguous(),
                        sink,
                        gidx,
                        scale,
                        num_heads,
                    )
                    dq_t = dq_t.reshape([b * sq, _HEAD_TILE, _D])[
                        :, :num_heads, :
                    ]
                    dq_h = dq_h.reshape([b * sq, num_heads, _D])
                    self.assertEqual(_max_abs_diff(dq_h, dq_t), 0.0, "dq moved")
                    self.assertEqual(
                        _max_abs_diff(dsink_h, dsink_t[:num_heads]),
                        0.0,
                        "d_sink moved",
                    )
                    # dKV is accumulated with atomics, so only near-equal.
                    self.assertLess(_rel_l2(dkv_h, dkv_t), 1e-3)


if __name__ == "__main__":
    unittest.main()
