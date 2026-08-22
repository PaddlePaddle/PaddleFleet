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

"""CSA / HCA layers whose latent width is below the sparse-attn kernel's 512.

The FlashMLA sparse prefill accepts ``d_qk in {512, 576}`` and requires
``d_v == 512`` (``csrc/api/sparse_fwd.h``), and the cuDNN DSA backward unrolls
``dQ``/``dKV`` into exactly four 128-column sub-tiles, so neither runs at
``v_head_dim = 256``. ``_pad_latent_dim`` zero-pads ``q`` and ``kv`` up to 512
instead. Since ``kv`` is one latent head serving as both key and value, zeros on
both sides leave the scores bit-identical and make the padded output columns --
and the ``dq``/``dkv`` columns over them -- exactly zero.

This pins that claim: agreement with the fp32 ``unfused`` reference in fwd and
bwd, the padded columns being exactly (not approximately) zero straight out of
both kernels, the ``hn == 512`` path staying copy-free and therefore
bit-for-bit unchanged, and the >512 rejection.
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

_KERNEL_LATENT = 512
# b, sq, skv, topk, invalid_frac, name. ``invalid_frac`` seeds ``-1`` columns
# anywhere in the row, which is what document masking produces.
_SHAPES = [
    (1, 128, 256, 64, 0.0, "dense"),
    (1, 128, 256, 64, 0.3, "holes"),
    (2, 256, 512, 192, 0.2, "batch2-topk192"),
    (1, 64, 8192, 192, 0.1, "hca-like"),
    (1, 1, 64, 64, 0.0, "single-token"),
]
# The exp2 ernielite layer shape this was added for, plus two other widths to
# keep the padding generic rather than 256-specific.
_LATENTS = (256, 384, 128)


def _rel_l2(a, b):
    a_f = a.flatten().cast("float32")
    b_f = b.flatten().cast("float32")
    return float(
        paddle.linalg.norm(a_f - b_f) / (paddle.linalg.norm(b_f) + 1e-12)
    )


def _make_inputs(
    b, sq, skv, num_heads, hn, topk, invalid_frac, seed=0, sink=None
):
    paddle.seed(seed)
    q = paddle.randn([b, sq, num_heads, hn]).cast("bfloat16")
    kv = paddle.randn([b, skv, hn]).cast("bfloat16")
    attn_sink = (paddle.randn([num_heads]) * 0.5).cast("float32")
    if sink is not None:
        attn_sink = _make_sink(num_heads, sink)
    topk_idxs = paddle.randint(0, skv, [b, sq, topk]).cast("int32")
    if invalid_frac > 0:
        holes = paddle.rand([b, sq, topk]) < invalid_frac
        topk_idxs = paddle.where(
            holes, paddle.full_like(topk_idxs, -1), topk_idxs
        )
    return q, kv, attn_sink, topk_idxs, hn**-0.5


def _make_sink(num_heads, spec):
    """Attention-sink bias presets.

    ``"split"`` puts half the heads at +8 and half at -8 in one call, so a bug
    that mixes head rows up cannot hide behind a uniform sink.
    """
    if spec == "split":
        half = num_heads // 2
        return paddle.concat(
            [
                paddle.full([half], 8.0, dtype="float32"),
                paddle.full([num_heads - half], -8.0, dtype="float32"),
            ]
        )
    return paddle.full([num_heads], float(spec), dtype="float32")


def _run(
    q,
    kv,
    attn_sink,
    topk_idxs,
    scale,
    backend,
    grad_seed=None,
    topk_length=None,
):
    """One fwd+bwd through the public entry point.

    ``grad_seed`` seeds a random ``dO`` instead of the all-ones one an
    ``out.sum()`` loss gives. That matters here: with ``dO == 1`` every latent
    column carries the same gradient, so mis-padding ``dO`` at the wrong end --
    or slicing ``dq``/``dkv`` from the wrong side -- would not show up. The
    reference and the kernel run re-seed identically, so both see the same
    ``dO``.
    """
    from paddlefleet.fusions.csa_sparse_attn import csa_sparse_attn

    b, sq, num_heads, hn = q.shape
    q = q.detach().clone()
    q.stop_gradient = False
    kv = kv.detach().clone()
    kv.stop_gradient = False
    attn_sink = attn_sink.detach().clone()
    attn_sink.stop_gradient = False

    kwargs = {} if topk_length is None else {"topk_length": topk_length}
    out = csa_sparse_attn(
        q, kv, attn_sink, topk_idxs, scale, backend=backend, **kwargs
    )
    if grad_seed is None:
        out.sum().backward()
    else:
        paddle.seed(grad_seed)
        out.backward(paddle.randn(out.shape).cast(out.dtype))
    return {
        "out": out.reshape([b, sq, num_heads, hn]),
        "dq": q.grad,
        "dkv": kv.grad,
        "d_sink": attn_sink.grad,
    }


# Worst rel-L2 measured against the fp32 reference with a random dO, over every
# shape in ``_SHAPES`` and every latent width in 64..512: out 2.4e-3, dq 3.1e-3,
# dkv 3.3e-3, d_sink 3.9e-3 -- and the hn=512 native case sits at 2.4/2.8/3.2/
# 3.5e-3, i.e. the padding contributes nothing above bf16 noise. Ceilings are
# ~2x that; ``test_no_accuracy_loss_vs_native_512`` is the part that ties the
# padded error to the native one and so catches a real regression.
_REL_L2_CEILING = {"out": 6e-3, "dq": 6e-3, "dkv": 6e-3, "d_sink": 8e-3}


@functools.lru_cache(maxsize=1)
def _cudnn_sparse_bwd_runs():
    """Whether the cuDNN sparse backward can actually execute here.

    ``is_cudnn_frontend_available()`` is not a usable proxy: some builds import
    a top-level ``cudnn`` module while executing, which the loader has already
    renamed to ``paddlefleet_ops.cudnn``, so the call dies with
    ``ModuleNotFoundError`` while the probe says "available". One native-width
    backward settles it; anything other than a missing module is re-raised so a
    real kernel regression still fails loudly.
    """
    if not (_HAS_FLASH_MLA and _HAS_CUDNN_FRONTEND):
        return False
    try:
        _run(
            *_make_inputs(1, 8, 64, 64, _KERNEL_LATENT, 64, 0.0),
            backend="cudnn",
        )
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
    paddle.is_compiled_with_cuda(), "sub-512 latent tests require CUDA"
)
class TestSubLatentDim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _skip_without_cudnn_sparse_bwd()

    def test_matches_unfused_reference(self):
        for hn in _LATENTS:
            for b, sq, skv, topk, inv, name in _SHAPES:
                with self.subTest(hn=hn, shape=name):
                    args = _make_inputs(b, sq, skv, 64, hn, topk, inv)
                    ref = _run(*args, backend="unfused", grad_seed=7)
                    got = _run(*args, backend="cudnn", grad_seed=7)
                    for key, ceiling in _REL_L2_CEILING.items():
                        self.assertLess(
                            _rel_l2(got[key], ref[key]),
                            ceiling,
                            f"{key} at hn={hn} shape={name}",
                        )

    def test_arbitrary_latent_widths(self):
        """Any width up to 512, including ones that are not multiples of 64.

        The pad target is always 512 and ``paddle.concat`` returns a fresh
        contiguous tensor, so the kernel's alignment requirements are met by the
        padded copy regardless of how odd the real width is. Verified for 100 /
        200 / 260 as well as the tidy multiples.
        """
        for hn in (64, 100, 192, 200, 260, 320, 448):
            with self.subTest(hn=hn):
                args = _make_inputs(1, 64, 256, 64, hn, 64, 0.1)
                ref = _run(*args, backend="unfused", grad_seed=7)
                got = _run(*args, backend="cudnn", grad_seed=7)
                for key, ceiling in _REL_L2_CEILING.items():
                    self.assertLess(
                        _rel_l2(got[key], ref[key]), ceiling, f"{key} hn={hn}"
                    )

    def test_sink_magnitudes(self):
        """The learnable sink bias must survive the latent padding.

        The sink enters the softmax denominator only, so a padding bug that
        perturbed the scores would show up here first: at sink=+20 the sink
        dominates every row, at -1e4 it drops out entirely (``d_sink`` is then
        exactly 0 on both sides), and "split" makes neighbouring heads disagree.
        """
        for sink in (-8.0, 0.0, 8.0, 20.0, -1e4, "split"):
            for hn in (256, 128):
                with self.subTest(sink=sink, hn=hn):
                    args = _make_inputs(
                        1, 128, 512, 64, hn, 192, 0.2, sink=sink
                    )
                    ref = _run(*args, backend="unfused", grad_seed=7)
                    got = _run(*args, backend="cudnn", grad_seed=7)
                    for key, ceiling in _REL_L2_CEILING.items():
                        self.assertLess(
                            _rel_l2(got[key], ref[key]),
                            ceiling,
                            f"{key} at sink={sink} hn={hn}",
                        )

    def test_no_accuracy_loss_vs_native_512(self):
        """Padding must not make a narrow latent less accurate than hn=512.

        Both runs are compared to their own fp32 reference, so the ratio is
        scale-free; a real regression (wrong columns, stale padding) blows it up
        far past the slack allowed here.
        """
        b, sq, skv, topk = 1, 128, 512, 192
        base_args = _make_inputs(b, sq, skv, 64, _KERNEL_LATENT, topk, 0.2)
        base = {
            k: _rel_l2(v, _run(*base_args, backend="unfused", grad_seed=7)[k])
            for k, v in _run(*base_args, backend="cudnn", grad_seed=7).items()
        }
        for hn in _LATENTS:
            with self.subTest(hn=hn):
                args = _make_inputs(b, sq, skv, 64, hn, topk, 0.2)
                ref = _run(*args, backend="unfused", grad_seed=7)
                got = _run(*args, backend="cudnn", grad_seed=7)
                for key in ("out", "dq", "dkv"):
                    self.assertLess(
                        _rel_l2(got[key], ref[key]),
                        max(3.0 * base[key], 1e-4),
                        f"{key} at hn={hn} is worse than the hn=512 baseline "
                        f"({base[key]:.3e})",
                    )

    def test_native_512_is_copy_free(self):
        """``hn == 512`` must stay bit-for-bit unchanged, i.e. no pad at all."""
        from paddlefleet.fusions.csa_sparse_attn import _pad_latent_dim

        x = paddle.randn([2, 4, _KERNEL_LATENT]).cast("bfloat16")
        self.assertIs(_pad_latent_dim(x, _KERNEL_LATENT), x)

    def test_kernel_sees_zero_padded_512_and_output_is_sliced_back(self):
        """Pin the plumbing bit-exactly by spying on the kernel call itself.

        Checked at the boundary rather than against a hand-rolled second
        pipeline, because the forward also pre-compacts ``topk_idxs`` and derives
        its own ``topk_length``; reproducing that outside would either duplicate
        the wrapper or change the summation order and stop being bit-exact.

        Asserts the kernel is handed exactly 512 columns with zeros above ``hn``
        (so the scores cannot move), and that what the entry point returns is
        bit-for-bit the kernel's output sliced to ``hn`` -- which a wrong axis, an
        off-by-one, or a stale non-contiguous view would all break.
        """
        import paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn as fwd_mod
        from paddlefleet.fusions.csa_sparse_attn import csa_sparse_attn

        for hn in _LATENTS:
            with self.subTest(hn=hn):
                q, kv, sink, idxs, scale = _make_inputs(
                    1, 128, 512, 64, hn, 192, 0.2
                )
                seen = {}
                original = fwd_mod.flash_mla_sparse_attn

                def spy(q_in, kv_in, *args, **kwargs):
                    res = original(q_in, kv_in, *args, **kwargs)
                    seen["q"], seen["kv"], seen["out"] = q_in, kv_in, res[0]
                    return res

                fwd_mod.flash_mla_sparse_attn = spy
                try:
                    wrapped = csa_sparse_attn(
                        q, kv, sink, idxs, scale, backend="cudnn"
                    )
                finally:
                    fwd_mod.flash_mla_sparse_attn = original

                self.assertEqual(seen["q"].shape[-1], _KERNEL_LATENT)
                self.assertEqual(seen["kv"].shape[-1], _KERNEL_LATENT)
                for name in ("q", "kv"):
                    self.assertEqual(
                        float(seen[name][..., hn:].abs().max()),
                        0.0,
                        f"{name} padding is not zero",
                    )
                sliced = seen["out"][..., :hn].reshape(wrapped.shape)
                self.assertEqual(
                    float(
                        (wrapped.cast("float32") - sliced.cast("float32"))
                        .abs()
                        .max()
                    ),
                    0.0,
                )

    def test_forward_is_deterministic(self):
        """Two identical forwards must agree bit-for-bit.

        Training runs this layer under ``recompute_modules=["full_attn", ...]``,
        so the forward executes twice and the backward differentiates the second
        run. Zero padding is deterministic, but this pins it.
        """
        from paddlefleet.fusions.csa_sparse_attn import csa_sparse_attn

        args = _make_inputs(1, 128, 512, 24, 256, 192, 0.2)
        first = csa_sparse_attn(*args, backend="cudnn")
        second = csa_sparse_attn(*args, backend="cudnn")
        self.assertEqual(
            float((first.cast("float32") - second.cast("float32")).abs().max()),
            0.0,
        )

    def test_topk_length_early_stop(self):
        """The forward early-stop hint must still be honoured under padding.

        ``topk_length`` is a forward-only hint: the backward ignores it and
        rebuilds its own bound from the ``-1`` entries in ``topk_idxs``
        (``_csa_compute_topk_length``). So the slots beyond the prefix have to be
        marked ``-1`` as well, or the two halves differentiate different supports
        -- measured ``dq`` rel-L2 1.1e+01 without the ``-1`` padding, identically
        at hn=512 and hn=256, i.e. that is the API's contract and not something
        the latent padding introduced.
        """
        b, sq, skv, hn, topk = 1, 128, 512, 256, 192
        q, kv, sink, idxs, scale = _make_inputs(b, sq, skv, 64, hn, topk, 0.0)
        paddle.seed(11)
        topk_length = paddle.randint(1, topk + 1, [b, sq]).cast("int32")
        slots = paddle.arange(topk, dtype="int32").reshape([1, 1, topk])
        idxs = paddle.where(
            slots >= topk_length.reshape([b, sq, 1]),
            paddle.full_like(idxs, -1),
            idxs,
        )
        args = (q, kv, sink, idxs, scale)
        ref = _run(
            *args, backend="unfused", grad_seed=7, topk_length=topk_length
        )
        got = _run(*args, backend="cudnn", grad_seed=7, topk_length=topk_length)
        for key, ceiling in _REL_L2_CEILING.items():
            self.assertLess(_rel_l2(got[key], ref[key]), ceiling, key)

    def test_rejects_latent_wider_than_512(self):
        from paddlefleet.fusions.csa_sparse_attn import (
            _dsa_latent_dim,
            csa_sparse_attn,
        )

        self.assertEqual(_dsa_latent_dim(256), _KERNEL_LATENT)
        self.assertEqual(_dsa_latent_dim(_KERNEL_LATENT), _KERNEL_LATENT)
        with self.assertRaisesRegex(ValueError, "at most 512 latent dims"):
            _dsa_latent_dim(576)
        args = _make_inputs(1, 8, 64, 64, 576, 64, 0.0)
        with self.assertRaisesRegex(ValueError, "at most 512 latent dims"):
            csa_sparse_attn(*args, backend="cudnn")


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "sub-512 latent tests require CUDA"
)
class TestPaddedLatentColumnsAreZero(unittest.TestCase):
    """The claim the whole approach rests on, checked at the kernel boundary.

    Exactly zero, not small: if either kernel wrote anything into the padded
    columns the slice-back in ``CSASparseAttention`` would silently discard real
    contributions, so this is asserted as an exact 0 rather than a tolerance.
    """

    @classmethod
    def setUpClass(cls):
        _skip_without_cudnn_sparse_bwd()

    def test_forward_and_backward_leave_padding_at_zero(self):
        from paddlefleet.cudnn_ops import csa_sparse_attn_bwd_cudnn
        from paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
            flash_mla_sparse_attn,
        )
        from paddlefleet.fusions.csa_sparse_attn import (
            _csa_compute_topk_length,
            _pad_latent_dim,
        )
        from paddlefleet.fusions.csa_sparse_attn_utils import (
            _local_to_global_flat,
        )

        b, sq, skv, heads, topk = 1, 128, 512, 64, 192
        for hn in _LATENTS:
            with self.subTest(hn=hn):
                q, kv, sink, idxs, scale = _make_inputs(
                    b, sq, skv, heads, hn, topk, 0.2
                )
                q_pad = _pad_latent_dim(q, _KERNEL_LATENT)
                kv_pad = _pad_latent_dim(kv, _KERNEL_LATENT)
                out, lse, _ = flash_mla_sparse_attn(
                    q_pad, kv_pad, sink, idxs, sm_scale=scale
                )
                self.assertEqual(
                    float(out[:, :, :, hn:].abs().max()), 0.0, "fwd out"
                )

                idxs_flat = _local_to_global_flat(idxs, skv)
                dout = _pad_latent_dim(
                    paddle.randn([b, sq, heads, hn]).cast(out.dtype),
                    _KERNEL_LATENT,
                )
                dq, dkv, _ = csa_sparse_attn_bwd_cudnn(
                    q_pad.reshape([b * sq, heads, _KERNEL_LATENT]),
                    kv_pad.reshape([b * skv, _KERNEL_LATENT]),
                    out.reshape([b * sq, heads, _KERNEL_LATENT]),
                    dout.reshape([b * sq, heads, _KERNEL_LATENT]),
                    lse.reshape([b * sq, heads]),
                    sink,
                    idxs_flat,
                    softmax_scale=scale,
                    topk_length=_csa_compute_topk_length(idxs_flat),
                )
                self.assertEqual(
                    float(dq[:, :, hn:].abs().max()), 0.0, "bwd dq"
                )
                self.assertEqual(float(dkv[:, hn:].abs().max()), 0.0, "bwd dkv")


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(), "sub-512 latent tests require CUDA"
)
class TestSubLatentWithSubTileHeads(unittest.TestCase):
    """The ernielite exp2 layer: 24 heads AND a 256 latent, both padded."""

    @classmethod
    def setUpClass(cls):
        _skip_without_cudnn_sparse_bwd()

    def test_head_and_latent_padding_compose(self):
        for heads in (24, 32):
            for b, sq, skv, topk, inv, name in _SHAPES:
                with self.subTest(heads=heads, shape=name):
                    args = _make_inputs(b, sq, skv, heads, 256, topk, inv)
                    ref = _run(*args, backend="unfused", grad_seed=7)
                    got = _run(*args, backend="cudnn", grad_seed=7)
                    for key, ceiling in _REL_L2_CEILING.items():
                        self.assertLess(
                            _rel_l2(got[key], ref[key]),
                            ceiling,
                            f"{key} at heads={heads} shape={name}",
                        )

    def test_head_and_latent_padding_compose_with_sink_sweep(self):
        """Both paddings active *and* a degenerate sink, which is the case where
        a leak between the padded head rows and the padded latent columns would
        be easiest to miss: the padded heads carry a -1e30 sink while the real
        heads carry an extreme one."""
        for sink in (20.0, -1e4, "split"):
            with self.subTest(sink=sink):
                args = _make_inputs(1, 128, 512, 24, 256, 192, 0.2, sink=sink)
                ref = _run(*args, backend="unfused", grad_seed=7)
                got = _run(*args, backend="cudnn", grad_seed=7)
                for key, ceiling in _REL_L2_CEILING.items():
                    self.assertLess(
                        _rel_l2(got[key], ref[key]),
                        ceiling,
                        f"{key} at sink={sink}",
                    )


if __name__ == "__main__":
    unittest.main()
