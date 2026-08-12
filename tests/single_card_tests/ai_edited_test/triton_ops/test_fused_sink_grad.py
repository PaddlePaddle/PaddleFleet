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

"""``fused_sink_grad`` vs the eager epilogue it replaces in ``mqa_sparse_attn``."""

import unittest
from unittest import mock

import numpy as np
import paddle


def _eager_sink_grad(out, do, lse, sink, h):
    """Verbatim copy of the eager branch in ``_MQASparseAttention.backward``."""
    out_h = out[:, :, :h, :].astype("float32")
    do_h = do[:, :, :h, :].astype("float32")
    delta = (out_h * do_h).sum(axis=-1)
    sink_real = sink[:h].astype("float32").reshape([1, 1, h])
    lse_h = lse[:, :, :h].astype("float32")
    lse_full = paddle.logaddexp(lse_h, sink_real)
    p_sink = paddle.exp(sink_real - lse_full)
    return (-(delta * p_sink).sum(axis=[0, 1])).contiguous().cast("float32")


def _make_inputs(b, s, h, h_pad, d_v, dtype="bfloat16", seed=0):
    paddle.seed(seed)
    out = paddle.randn([b, s, h_pad, d_v]).cast(dtype)
    do = paddle.randn([b, s, h_pad, d_v]).cast(dtype)
    if h_pad > h:
        out[:, :, h:, :] = 0
        do[:, :, h:, :] = 0
    lse = paddle.randn([b, s, h_pad]).cast("float32") * 2.0
    sink = paddle.randn([h_pad]).cast("float32")
    if h_pad > h:
        sink[h:] = -1e30
    return out, do, lse, sink


def _try_use_cuda_device():
    """Return True only when a real CUDA device is usable for kernels.

    Same probe as ``tests/single_card_tests/transformer/hybrid_mla_utils.py`` and
    ``test_hca_csa_independent_rope.py``. A CUDA build is not enough here: these
    tests JIT-compile and launch Triton kernels. ``set_device`` and the ``place``
    check are inside the ``try`` because driver or context initialisation can
    raise, which must skip rather than error.
    """
    if not paddle.is_compiled_with_cuda():
        return False
    if paddle.device.cuda.device_count() == 0:
        return False
    try:
        paddle.set_device("gpu:0")
        place = str(paddle.empty([1]).place).lower()
    except Exception:
        return False
    return paddle.get_device().startswith("gpu") and (
        "gpu" in place or "cuda" in place
    )


_CUDA_OK = _try_use_cuda_device()
_REQUIRES_CUDA = unittest.skipUnless(_CUDA_OK, "requires a usable CUDA device")


def _dsa_skip_reason():
    """``None`` when ``mqa_sparse_attn`` can run here, else why it cannot.

    Only ``ImportError`` is swallowed (the ops package may be absent); anything
    else propagates instead of being silently turned into a skip.
    ``is_dsa_available`` already reports SM100 / FlashMLA / cuDNN-frontend gaps as
    ``False`` rather than raising.
    """
    if not _CUDA_OK:
        return "requires a usable CUDA device"
    try:
        from paddlefleet.cudnn_ops.block_sparse_mqa_dsa import is_dsa_available
    except ImportError as exc:
        return f"paddlefleet.cudnn_ops import failed: {exc}"
    if not is_dsa_available():
        return "requires SM100+ FlashMLA sparse fwd + cuDNN DSA bwd kernels"
    return None


_DSA_SKIP_REASON = _dsa_skip_reason()
_REQUIRES_DSA = unittest.skipIf(_DSA_SKIP_REASON is not None, _DSA_SKIP_REASON)


@_REQUIRES_CUDA
class TestFusedSinkGrad(unittest.TestCase):
    def _assert_close(self, got, ref, tol=1e-6):
        """Compare to eager, normalised by the gradient vector's own scale.

        Per-element ``rtol`` is the wrong judge here. ``d_sink`` spans orders of
        magnitude across heads (a head whose sink barely competes in the softmax
        gets a near-zero gradient), and such a lane's per-element relative error
        blows up to 1.5e-5 while its absolute contribution to the parameter
        update stays around 1e-9. What matters is the error relative to the
        vector's own scale: 2.1e-7, i.e. ~1.8 fp32 ulp, coming purely from the
        fp32 summation order of ``Delta = sum_dv(out * do)`` -- fusing only the
        cast+mul and leaving ``sum`` to paddle is bitwise identical, which pins
        the summation as the sole source.
        """
        g, r = got.numpy(), ref.numpy()
        rel = np.abs(g - r).max() / max(np.abs(r).max(), 1e-30)
        self.assertLess(rel, tol, f"max|diff| / max|ref| = {rel:.3e}")

    def _run(self, b, s, h, h_pad, d_v, dtype="bfloat16", seed=0):
        from paddlefleet.triton_ops.fused_sink_grad import fused_sink_grad

        paddle.seed(seed)
        out = paddle.randn([b, s, h_pad, d_v]).cast(dtype)
        do = paddle.randn([b, s, h_pad, d_v]).cast(dtype)
        # Padded heads carry no signal and must not reach the result.
        if h_pad > h:
            out[:, :, h:, :] = 0
            do[:, :, h:, :] = 0
        lse = paddle.randn([b, s, h_pad]).cast("float32") * 2.0
        sink = paddle.randn([h_pad]).cast("float32")
        if h_pad > h:
            sink[h:] = -1e30

        got = fused_sink_grad(out, do, lse, sink, h)
        ref = _eager_sink_grad(out, do, lse, sink, h)
        self.assertEqual(list(got.shape), [h])
        self.assertEqual(got.dtype, paddle.float32)
        self._assert_close(got, ref)
        return got

    def test_production_shape(self):
        """b=1, s=8192, h=64, d_qk=576/d_v=512: the online config's shape."""
        self._run(1, 8192, 64, 64, 512)

    def test_padded_heads(self):
        """h < 64 (e.g. TP>1): padded heads must not contribute."""
        self._run(1, 512, 16, 64, 512)

    def test_batch_gt_one(self):
        self._run(4, 256, 8, 8, 512)

    def test_ragged_rows(self):
        """n_rows not a multiple of BLOCK_N, d_v not a multiple of BLOCK_D."""
        self._run(1, 1000, 8, 8, 100)

    def test_fp16(self):
        self._run(1, 256, 8, 8, 512, dtype="float16")

    def test_empty_rows_contribute_nothing(self):
        """``lse = -inf`` rows (all-``-1`` token_indices) have out == 0."""
        from paddlefleet.triton_ops.fused_sink_grad import fused_sink_grad

        b, s, h, d_v = 1, 128, 8, 512
        out = paddle.randn([b, s, h, d_v]).cast("bfloat16")
        do = paddle.randn([b, s, h, d_v]).cast("bfloat16")
        lse = paddle.randn([b, s, h]).cast("float32")
        sink = paddle.randn([h]).cast("float32")
        out[:, ::4, :, :] = 0
        lse[:, ::4, :] = float("-inf")

        got = fused_sink_grad(out, do, lse, sink, h)
        ref = _eager_sink_grad(out, do, lse, sink, h)
        self.assertFalse(bool(paddle.isnan(got).any()))
        self._assert_close(got, ref)

    def test_deterministic(self):
        """No atomics: repeated calls must be bit-identical."""
        a = self._run(1, 1024, 32, 64, 512, seed=7)
        b = self._run(1, 1024, 32, 64, 512, seed=7)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_input_validation(self):
        """Shape/dtype contract violations must raise, not launch the kernel."""
        from paddlefleet.triton_ops.fused_sink_grad import fused_sink_grad

        out, do, lse, sink = _make_inputs(1, 8, 4, 4, 32)
        cases = [
            ("out/do shape", (out, do[:, :4], lse, sink, 4)),
            ("out/do dtype", (out, do.astype("float32"), lse, sink, 4)),
            ("lse/sink must be fp32", (out, do, lse.cast(out.dtype), sink, 4)),
            ("exceeds h_pad", (out, do, lse, sink, 8)),
            ("lse must be", (out, do, lse[:, :4], sink, 4)),
            ("sink must be", (out, do, lse, sink[:2], 4)),
        ]
        for msg, args in cases:
            with (
                self.subTest(msg=msg),
                self.assertRaisesRegex(ValueError, msg),
            ):
                fused_sink_grad(*args)


@_REQUIRES_DSA
class TestSinkGradFusionSwitch(unittest.TestCase):
    """``sink_grad_fusion`` must not change anything but the sink gradient path."""

    def test_switch_matches_eager_end_to_end(self):
        from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

        dk, dv, h, s, topk = 576, 512, 8, 64, 128
        paddle.seed(11)
        q0 = (paddle.randn([1, s, h, dk]) * 0.3).cast("bfloat16")
        kv0 = (paddle.randn([1, s, dk]) * 0.3).cast("bfloat16")
        sink0 = paddle.randn([h]).cast("float32")
        dout = (paddle.randn([1, s, h * dv]) * 0.5).cast("bfloat16")

        idx = np.full((1, s, topk), -1, dtype=np.int32)
        for i in range(s):
            n = min(i + 1, topk)
            idx[0, i, :n] = np.arange(i + 1 - n, i + 1)
        token_indices = paddle.to_tensor(idx)

        def run(fusion):
            q = q0.detach().clone()
            kv = kv0.detach().clone()
            sink = sink0.detach().clone()
            for t in (q, kv, sink):
                t.stop_gradient = False
            out = mqa_sparse_attn(
                q,
                kv,
                token_indices,
                dk**-0.5,
                dv,
                attn_sink=sink,
                sink_grad_fusion=fusion,
            )
            paddle.autograd.backward([out], [dout])
            return out, q.grad, kv.grad, sink.grad

        o_e, dq_e, dkv_e, ds_e = run(False)
        o_f, dq_f, dkv_f, ds_f = run(True)

        # Forward and dq are untouched by the switch and are bit-stable.
        np.testing.assert_array_equal(
            o_e.astype("float32").numpy(), o_f.astype("float32").numpy()
        )
        np.testing.assert_array_equal(
            dq_e.astype("float32").numpy(), dq_f.astype("float32").numpy()
        )
        # dkv comes back through atomics, so it is only close, not equal.
        np.testing.assert_allclose(
            dkv_e.astype("float32").numpy(),
            dkv_f.astype("float32").numpy(),
            rtol=1e-2,
            atol=1e-3,
        )
        self.assertFalse(bool(paddle.isnan(ds_f).any()))
        rel = np.abs(ds_f.numpy() - ds_e.numpy()).max() / max(
            np.abs(ds_e.numpy()).max(), 1e-30
        )
        self.assertLess(rel, 1e-6, f"max|diff| / max|ref| = {rel:.3e}")


class _FakeCtx:
    """Stand-in for the ``PyLayer`` ctx that ``backward`` reads.

    ``backward`` only needs ``saved_tensor()`` plus the plain attributes the
    forward stashed, so it can be driven directly on hardware that has no
    FlashMLA/DSA kernels (the CI runner is Hopper; the kernels are SM100+).
    """

    def __init__(self, saved, **attrs):
        self._saved = saved
        self.__dict__.update(attrs)

    def saved_tensor(self):
        return self._saved


class _RecordingBwd:
    """Stub for ``csa_sparse_attn_bwd_cudnn``: records args, returns dq/dkv.

    ``d_sink`` is returned as zeros, matching the real SM100 kernel (which
    allocates it but never writes it -- the reason the epilogue under test
    computes the sink gradient analytically).
    """

    def __init__(self, out_dtype=None):
        self.out_dtype = out_dtype
        self.calls = []

    def __call__(self, q, kv, o, do, lse, sink, gidx, **kwargs):
        self.calls.append(
            dict(
                q=q, kv=kv, o=o, do=do, lse=lse, sink=sink, gidx=gidx, **kwargs
            )
        )
        dtype = self.out_dtype or q.dtype
        # Distinct scalings so a wrong reshape/unpad cannot pass by symmetry;
        # powers of two so the fp32 -> bf16 round trip stays exact.
        dq = (q.astype("float32") * 2.0).cast(dtype)
        dkv = (kv.astype("float32") * 4.0).cast(dtype)
        d_sink = paddle.zeros([q.shape[1]], dtype="float32")
        return dq, dkv, d_sink


def _make_bwd_case(
    h=8,
    h_pad=64,
    b=2,
    s=16,
    d_k=576,
    d_v=512,
    topk=8,
    dtype="bfloat16",
    learnable_sink=True,
    fusion=False,
    seed=3,
    attn_sink_dtype="float32",
):
    """A ctx + grad_output pair equivalent to what a real forward would leave.

    ``attn_sink_dtype`` is the dtype of the ``attn_sink`` the forward received
    (what it stashes on ``ctx``), independent of the fp32 internal ``sink`` the
    kernels use -- the backward casts the sink gradient back to it.
    """
    paddle.seed(seed)
    q_pad = (paddle.randn([b, s, h_pad, d_k]) * 0.3).cast(dtype)
    kv = (paddle.randn([b, s, d_k]) * 0.3).cast(dtype)
    out = (paddle.randn([b, s, h_pad, d_v]) * 0.3).cast(dtype)
    lse = paddle.randn([b, s, h_pad]).cast("float32")
    if h_pad > h:
        out[:, :, h:, :] = 0
    if learnable_sink:
        sink = paddle.randn([h_pad]).cast("float32")
        if h_pad > h:
            sink[h:] = -1e30
    else:
        sink = paddle.full([h_pad], -1e30, dtype="float32")

    idx = np.full((b, s, topk), -1, dtype=np.int32)
    for j in range(s):
        n = min(j + 1, topk)
        idx[:, j, :n] = np.arange(j + 1 - n, j + 1)
    token_indices = paddle.to_tensor(idx)
    topk_len = paddle.to_tensor(
        np.minimum(np.arange(s) + 1, topk)
        .astype(np.int32)[None, :]
        .repeat(b, axis=0)
        .reshape([b * s])
    )

    ctx = _FakeCtx(
        (q_pad, kv, out, lse, token_indices, sink, topk_len),
        num_heads=h,
        d_v=d_v,
        sm_scale=float(d_k**-0.5),
        query_dtype=q_pad.dtype,
        kv_dtype=kv.dtype,
        attn_sink_dtype=paddle.empty([0], dtype=attn_sink_dtype).dtype,
        learnable_sink=learnable_sink,
        sink_grad_fusion=fusion,
        needs_grad=(True, True, learnable_sink),
    )
    grad_output = (paddle.randn([b, s, h * d_v]) * 0.5).cast(dtype)
    return ctx, grad_output


def _run_backward(ctx, grad_output, fake):
    """Drive ``_MQASparseAttention.backward`` with the DSA kernel stubbed out."""
    from paddlefleet import cudnn_ops
    from paddlefleet.fusions.mqa_sparse_attn import _MQASparseAttention

    with mock.patch.object(cudnn_ops, "csa_sparse_attn_bwd_cudnn", fake):
        return _MQASparseAttention.backward(ctx, grad_output)


@_REQUIRES_CUDA
class TestBackwardEpilogue(unittest.TestCase):
    """``_MQASparseAttention.backward`` without the SM100-only DSA kernel.

    ``TestSinkGradFusionSwitch`` above is the end-to-end check, but it needs
    SM100 and is skipped on CI hardware. These cases stub the one unavailable
    kernel and exercise the python epilogue -- head unpadding, the finite-sink
    LSE correction, the dq/dkv dtype guards and both sink-gradient paths -- which
    is where the logic that the switch can break actually lives.
    """

    def test_eager_epilogue(self):
        ctx, go = _make_bwd_case(fusion=False)
        fake = _RecordingBwd()
        grads = _run_backward(ctx, go, fake)

        q_pad, kv, out, lse, _, sink, _ = ctx._saved
        b, s, h_pad, d_k = q_pad.shape
        h, d_v = ctx.num_heads, ctx.d_v

        # 4 slots: query, kv, token_indices, attn_sink.
        self.assertEqual(len(grads), 4)
        dq, dkv, d_tok, d_sink = grads
        self.assertIsNone(d_tok)

        # dq is the stub's 2*q, unpadded back to the real heads; no cast, since
        # the backend already returns the forward's dtype.
        self.assertEqual(list(dq.shape), [b, s, h, d_k])
        self.assertEqual(dq.dtype, ctx.query_dtype)
        np.testing.assert_array_equal(
            dq.astype("float32").numpy(),
            (q_pad[:, :, :h, :].astype("float32") * 2.0).numpy(),
        )
        self.assertEqual(list(dkv.shape), [b, s, d_k])
        self.assertEqual(dkv.dtype, ctx.kv_dtype)
        np.testing.assert_array_equal(
            dkv.astype("float32").numpy(),
            (kv.astype("float32") * 4.0).numpy(),
        )

        # The analytic sink gradient, not the stub's zeros.
        do = paddle.concat(
            [
                go.reshape([b, s, h, d_v]),
                paddle.zeros([b, s, h_pad - h, d_v], dtype=go.dtype),
            ],
            axis=2,
        )
        ref = _eager_sink_grad(out, do, lse, sink, h)
        self.assertEqual(list(d_sink.shape), [h])
        self.assertEqual(d_sink.dtype, paddle.float32)
        np.testing.assert_allclose(
            d_sink.numpy(), ref.numpy(), rtol=1e-6, atol=1e-8
        )

    def test_fusion_epilogue_matches_eager(self):
        eager = _run_backward(*_make_bwd_case(fusion=False), _RecordingBwd())
        fused = _run_backward(*_make_bwd_case(fusion=True), _RecordingBwd())
        ds_e, ds_f = eager[3], fused[3]
        self.assertFalse(bool(paddle.isnan(ds_f).any()))
        rel = np.abs(ds_f.numpy() - ds_e.numpy()).max() / max(
            np.abs(ds_e.numpy()).max(), 1e-30
        )
        self.assertLess(rel, 1e-6, f"max|diff| / max|ref| = {rel:.3e}")

    def test_sink_grad_cast_to_attn_sink_dtype(self):
        """Both epilogues return the sink grad in the forward's ``attn_sink``
        dtype, not fp32. A float32 grad on a bf16 parameter breaks the PyLayer
        contract and the optimizer's master-weight path.
        """
        for fusion in (False, True):
            grads = _run_backward(
                *_make_bwd_case(fusion=fusion, attn_sink_dtype="bfloat16"),
                _RecordingBwd(),
            )
            d_sink = grads[3]
            self.assertEqual(d_sink.dtype, paddle.bfloat16, f"fusion={fusion}")
            self.assertFalse(
                bool(paddle.isnan(d_sink.astype("float32")).any()),
                f"fusion={fusion}",
            )

    def test_finite_sink_lse_correction(self):
        """Learnable sink on the ``d_qk != d_v`` branch: sink folded into LSE."""
        ctx, go = _make_bwd_case(learnable_sink=True)
        fake = _RecordingBwd()
        _run_backward(ctx, go, fake)

        q_pad, _, _, lse, _, sink, _ = ctx._saved
        b, s, h_pad, _ = q_pad.shape
        call = fake.calls[0]
        expected = paddle.logaddexp(lse, sink.reshape([1, 1, h_pad])).reshape(
            [b * s, h_pad]
        )
        np.testing.assert_allclose(
            call["lse"].numpy(), expected.numpy(), rtol=1e-6, atol=1e-6
        )
        # The sink argument is neutralised so the kernel cannot double-count it.
        np.testing.assert_array_equal(
            call["sink"].numpy(), np.full([h_pad], -1e30, dtype=np.float32)
        )

    def test_sinkless_keeps_kv_only_lse_and_three_grads(self):
        ctx, go = _make_bwd_case(learnable_sink=False)
        fake = _RecordingBwd()
        grads = _run_backward(ctx, go, fake)

        q_pad, _, _, lse, _, sink, _ = ctx._saved
        b, s, h_pad, _ = q_pad.shape
        self.assertEqual(len(grads), 3)  # no attn_sink slot
        call = fake.calls[0]
        np.testing.assert_array_equal(
            call["lse"].numpy(), lse.reshape([b * s, h_pad]).numpy()
        )
        np.testing.assert_array_equal(call["sink"].numpy(), sink.numpy())

    def test_casts_when_backend_returns_fp32(self):
        """The dtype guards must still convert a backend that widens to fp32."""
        ctx, go = _make_bwd_case()
        grads = _run_backward(ctx, go, _RecordingBwd(out_dtype="float32"))
        self.assertEqual(grads[0].dtype, ctx.query_dtype)
        self.assertEqual(grads[1].dtype, ctx.kv_dtype)

    def test_frozen_query_and_kv_get_no_grad(self):
        """``stop_gradient`` inputs must come back as ``None``, not zeros."""
        ctx, go = _make_bwd_case()
        ctx.needs_grad = (False, False, True)
        dq, dkv, d_tok, d_sink = _run_backward(ctx, go, _RecordingBwd())
        self.assertIsNone(dq)
        self.assertIsNone(dkv)
        self.assertIsNone(d_tok)
        self.assertIsNotNone(d_sink)


class _RecordingFwd:
    """Stub for ``flash_mla_sparse_attn``: records args, returns out/lse."""

    def __init__(self):
        self.calls = []

    def __call__(self, q_pad, kv, sink, token_indices, **kwargs):
        self.calls.append(
            dict(q_pad=q_pad, kv=kv, sink=sink, idx=token_indices, **kwargs)
        )
        b, s, h_pad, _ = q_pad.shape
        d_v = kwargs["d_v"]
        out = (q_pad[:, :, :, :d_v].astype("float32") * 0.5).cast(q_pad.dtype)
        lse = paddle.randn([b, s, h_pad]).cast("float32")
        lse_indexer = (
            lse.clone() if int(kwargs.get("indexer_topk", 0)) > 0 else None
        )
        return out, lse, lse_indexer


@_REQUIRES_CUDA
class TestApplyWithStubbedKernels(unittest.TestCase):
    """``mqa_sparse_attn`` end to end with both SM100 kernels stubbed out.

    Covers the python glue the DSA-gated tests cannot reach on CI hardware:
    head padding, sink padding, the ``_lse_indexer`` side channel and the
    autograd wiring (grad arity / dtypes) around the stubbed kernels.
    """

    def _run(self, h=8, indexer_topk=0, fusion=True, learnable_sink=True):
        from paddlefleet import cudnn_ops
        from paddlefleet.cudnn_ops.attn import csa_sparse_attn_fwd_cudnn
        from paddlefleet.fusions import mqa_sparse_attn as mod

        b, s, d_k, d_v, topk = 1, 16, 576, 512, 8
        paddle.seed(5)
        q = (paddle.randn([b, s, h, d_k]) * 0.3).cast("bfloat16")
        kv = (paddle.randn([b, s, d_k]) * 0.3).cast("bfloat16")
        sink = paddle.randn([h]).cast("float32") if learnable_sink else None
        for t in (q, kv) if sink is None else (q, kv, sink):
            t.stop_gradient = False
        dout = (paddle.randn([b, s, h * d_v]) * 0.5).cast("bfloat16")

        idx = np.full((b, s, topk), -1, dtype=np.int32)
        for j in range(s):
            n = min(j + 1, topk)
            idx[:, j, :n] = np.arange(j + 1 - n, j + 1)
        token_indices = paddle.to_tensor(idx)

        fwd, bwd = _RecordingFwd(), _RecordingBwd()
        with (
            mock.patch.object(
                csa_sparse_attn_fwd_cudnn, "flash_mla_sparse_attn", fwd
            ),
            mock.patch.object(cudnn_ops, "csa_sparse_attn_bwd_cudnn", bwd),
        ):
            ret = mod.mqa_sparse_attn(
                q,
                kv,
                token_indices,
                d_k**-0.5,
                d_v,
                attn_sink=sink,
                indexer_topk=indexer_topk,
                sink_grad_fusion=fusion,
            )
            out = ret[0] if indexer_topk > 0 else ret
            paddle.autograd.backward([out], [dout])
        return mod, out, ret, fwd, q, kv, sink

    def test_forward_pads_heads_and_sink(self):
        mod, out, _, fwd, q, kv, sink = self._run(h=8)
        b, s, h, d_k = q.shape
        self.assertEqual(list(out.shape), [b, s, h * 512])
        call = fwd.calls[0]
        # Query heads padded to the DSA-fixed 64 with zeros.
        self.assertEqual(list(call["q_pad"].shape), [b, s, 64, d_k])
        np.testing.assert_array_equal(
            call["q_pad"][:, :, h:, :].astype("float32").numpy(),
            np.zeros([b, s, 64 - h, d_k], dtype=np.float32),
        )
        # Sink: real logits first, ``-1e30`` on the padded heads.
        s_np = call["sink"].numpy()
        np.testing.assert_array_equal(s_np[:h], sink.numpy())
        np.testing.assert_array_equal(
            s_np[h:], np.full([64 - h], -1e30, dtype=np.float32)
        )
        # Per-query early-stop bound: last valid column + 1, clamped to >= 1.
        np.testing.assert_array_equal(
            call["topk_length"].numpy().reshape([-1]),
            np.minimum(np.arange(s) + 1, 8).astype(np.int32),
        )
        # Grads reached every differentiable input.
        for name, t in (("q", q), ("kv", kv), ("sink", sink)):
            self.assertIsNotNone(t.grad, f"{name} got no gradient")
            self.assertFalse(
                bool(paddle.isnan(t.grad.astype("float32")).any()), name
            )
        self.assertEqual(q.grad.dtype, q.dtype)
        self.assertEqual(sink.grad.dtype, paddle.float32)

    def test_indexer_side_channel_is_returned_and_cleared(self):
        mod, out, ret, _, _, _, _ = self._run(h=8, indexer_topk=4)
        self.assertEqual(len(ret), 2)
        self.assertEqual(list(ret[1].shape), [1, 16, 64])
        # The side channel must not outlive the call.
        self.assertIsNone(mod._MQASparseAttention._lse_indexer)

    def test_no_head_padding_when_h_is_64(self):
        _, out, _, fwd, q, _, sink = self._run(h=64)
        self.assertEqual(list(fwd.calls[0]["q_pad"].shape), [1, 16, 64, 576])
        np.testing.assert_array_equal(
            fwd.calls[0]["sink"].numpy(), sink.numpy()
        )
        self.assertIsNotNone(q.grad)

    def test_sinkless_uses_neg_sink(self):
        """``attn_sink=None`` must reach the kernel as a per-head ``-1e30``."""
        _, _, _, fwd, q, kv, _ = self._run(h=8, learnable_sink=False)
        np.testing.assert_array_equal(
            fwd.calls[0]["sink"].numpy(),
            np.full([64], -1e30, dtype=np.float32),
        )
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(kv.grad)

    def test_rejects_more_than_64_heads(self):
        with self.assertRaisesRegex(ValueError, "at most 64 query"):
            self._run(h=80)


if __name__ == "__main__":
    unittest.main()
