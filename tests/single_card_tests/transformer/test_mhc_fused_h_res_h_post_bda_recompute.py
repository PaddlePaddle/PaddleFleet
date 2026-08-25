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

"""Precision guard for the ``RecomputeWithoutOutput`` span that
``HyperConnectionTransformerLayer._fused_h_res_h_post_bda`` puts around
``HyperConnectionModule.fused_h_res_h_post_bda``.

The span only exists to save memory, so it is required to be *numerically
free*: it is compared against the matching no-span baseline **bitwise**
(``numpy.array_equal``), never with ``allclose``. A tolerance-based check would
let a real regression -- a replay that reads a stale tensor, a lost RNG state, a
dropped up-cast -- through as "small noise".

Two baselines are needed, because the half-layer already had a span before this
one:

  * the BDA span alone against no span at all;
  * the BDA span stacked on the mHC-aggregate span against the aggregate span
    alone -- that pair is the production shape before/after this change.

The aggregate span is itself *not* bitwise against the no-span path: its replay
reassociates a few fp32 sums inside the mHC projection. That predates this
change, so it is only bounded loosely, and it is checked at all so that the
bitwise assertions cannot pass vacuously.

Coverage: both BDA kernels (native / ``fused_h_post_bda``), bias present and
absent, fp32 and bf16, two stacked
half-layers so the second aggregate consumes the first BDA output (this is what
caught the ``h_res``/``h_post`` ``_clear_data()`` aliasing bug), the whole layer
through ``recompute_modules=['mhc_forward']``, and the span contract itself: a
span is created exactly when it can pay off, the discard is mandatory, and the
retained memory really does go down.
"""

import unittest
from unittest import mock

import numpy as np
import paddle

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.tensor_parallel.random import (
    RecomputeWithoutOutput,
    model_parallel_cuda_manual_seed,
)
from paddlefleet.transformer import hyper_connection
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
)
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

S, B, C, N = 8, 2, 64, 4


def _make_config(**overrides):
    """Minimal mHC-enabled config; mirrors the production mHC settings."""
    defaults = {
        "num_hidden_layers": 1,
        "hidden_size": C,
        "intermediate_size": 2 * C,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": C // 4,
        "use_bias": False,
        "hidden_dropout_prob": 0.0,
        "attention_dropout": 0.0,
        "normalization": "RMSNorm",
        "rms_norm_eps": 1e-5,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "recompute_modules": None,
        "block_attention_residuals": False,
        "bias_dropout_fusion": False,
        "apply_rope_fusion": False,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "softmax_type": "vanilla",
        "gated_linear_unit": False,
        "bias_activation_fusion": False,
        "gated_attention": False,
        "num_nextn_predict_layers": 0,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "softmax_scale": None,
        "multi_latent_attention": False,
        "rotary_interleaved": False,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        # mHC
        "enable_hyper_connections": True,
        "num_residual_streams": N,
        "mhc_sinkhorn_iterations": 5,
        "mhc_init_gating_factor": 0.01,
        "high_precision_mhc": True,
        "use_fused_mhc": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_layer(config, layer_number=1):
    model_parallel_cuda_manual_seed(42, tp_rank=0, ep_rank=0, etp_rank=0)
    paddle.seed(42)
    spec = get_gpt_layer_local_spec(config)
    return HyperConnectionTransformerLayer(
        config=config,
        sublayers_spec=spec.sublayers_spec,
        layer_number=layer_number,
    )


def _npy(t):
    return t.astype("float32").numpy().copy()


class _CompareMixin:
    def assert_bitwise(self, ref, got, name):
        """Exact equality. Never relax this to allclose -- see module docstring."""
        self.assertEqual(ref.shape, got.shape, f"{name}: shape differs")
        if not np.array_equal(ref, got):
            n = int((ref != got).sum())
            self.fail(
                f"{name}: {n}/{ref.size} elements differ, "
                f"max abs diff = {np.abs(ref - got).max():.6e}"
            )

    def assert_close(self, ref, got, name):
        """Loose bound, only used for the pre-existing mHC-aggregate span."""
        if not np.allclose(ref, got, rtol=1e-4, atol=1e-6):
            self.fail(
                f"{name}: max abs diff = {np.abs(ref - got).max():.6e} "
                f"(ref max abs = {np.abs(ref).max():.6e})"
            )

    def assert_no_nan(self, arr, name):
        self.assertTrue(np.isfinite(arr).all(), f"{name}: non-finite values")


def _half_layer(layer, hc, hidden_states, w, bias, wrap_aggregate, wrap_bda):
    """One mHC half-layer: aggregate -> (linear stand-in) -> fused BDA.

    Mirrors ``_forward_attention`` / ``_forward_mlp``, including the order of the
    two discard calls, with the layernorm + self-attn (or MLP) span replaced by a
    single matmul so that the test isolates the BDA span. Returns the half-layer
    output and the span the helper decided to create (``None`` if it did not).
    """
    original_residual = hidden_states
    ori_dtype = hidden_states.dtype

    if wrap_aggregate:
        agg_span = RecomputeWithoutOutput()
        aggregated, h_res, h_post = agg_span.recompute(
            hc, hidden_states, preserve_rng_state=False, share_grad_holder=True
        )
    else:
        agg_span = None
        aggregated, h_res, h_post = hc(hidden_states)
    aggregated = aggregated.to(ori_dtype)

    x = paddle.matmul(aggregated, w)

    hidden_states, bda_span = layer._fused_h_res_h_post_bda(
        hc, h_res, original_residual, h_post, (x, bias), wrap_bda
    )
    if agg_span is not None:
        agg_span.discard_output_and_register_recompute(hidden_states)
    hidden_states = layer._cast_and_discard_fused_bda(
        hidden_states, ori_dtype, bda_span
    )
    return hidden_states, bda_span


def _inputs(resid_np, w_np, bias_np, dtype):
    tensors = []
    for arr in (resid_np, w_np, bias_np):
        if arr is None:
            tensors.append(None)
            continue
        t = paddle.to_tensor(arr, dtype=dtype)
        t.stop_gradient = False
        tensors.append(t)
    return tensors


def _run(
    layer,
    hc,
    wrap_aggregate,
    wrap_bda,
    resid_np,
    w_np,
    bias_np,
    halves=2,
    dtype="float32",
):
    """Stack ``halves`` half-layers, backward, and collect every gradient.

    Returns ``(grads, n_spans)``; ``n_spans`` lets the caller assert that the
    span under test was actually created instead of silently comparing two
    identical no-span runs.
    """
    # Reseed so the dropout configurations draw the same mask in every run;
    # without this the comparison below could only ever be run with dropout off.
    paddle.seed(1234)
    resid, w, bias = _inputs(resid_np, w_np, bias_np, dtype)

    n_spans = 0
    h = resid
    for _ in range(halves):
        h, span = _half_layer(layer, hc, h, w, bias, wrap_aggregate, wrap_bda)
        n_spans += span is not None

    loss = h.astype("float32").sum()
    loss.backward()

    out = {"loss": _npy(loss), "resid": _npy(resid.grad), "w": _npy(w.grad)}
    if bias is not None:
        out["bias"] = _npy(bias.grad)
    for name, p in hc.named_parameters():
        if p.grad is not None:
            out[f"hc.{name}"] = _npy(p.grad)
    for p in layer.parameters():
        p.clear_gradient()
    return out, n_spans


def _random_inputs(with_bias, seed=0):
    rng = np.random.RandomState(seed)
    resid_np = rng.randn(S, B, N * C).astype("float32") * 0.02
    w_np = rng.randn(C, C).astype("float32") * 0.02
    bias_np = rng.randn(C).astype("float32") * 0.02 if with_bias else None
    return resid_np, w_np, bias_np


def _use_fused_kernel(test_case, hc):
    """Swap in the cuTile BDA kernel, or skip if cuTile is unavailable.

    The import alone is not a capability check: without ``cuda.tile``
    ``fused_mhc_kernels`` still binds ``fused_h_post_bda``, to a stub that
    raises at call time. Ask ``is_cutile_available()`` instead so the test skips
    rather than fails on a CUDA box without cuTile.
    """
    try:
        from paddlefleet.fusions.fused_mhc_kernels import (
            fused_h_post_bda,
            is_cutile_available,
        )
    except ImportError as exc:  # pragma: no cover - env dependent
        test_case.skipTest(f"fused_mhc_kernels unavailable: {exc}")
    if not is_cutile_available():  # pragma: no cover - env dependent
        test_case.skipTest("cuTile unavailable, fused_h_post_bda is a stub")
    hc._h_post_bda_op = fused_h_post_bda


@unittest.skipUnless(paddle.is_compiled_with_cuda(), "requires CUDA")
class TestFusedHResHPostBDASpanNumerics(_CompareMixin, unittest.TestCase):
    """The span must be bitwise-transparent on every kernel/config combination."""

    def _check(
        self,
        config_kw=None,
        use_fused_kernel=False,
        with_bias=True,
        dtype="float32",
        expect_span=True,
    ):
        config = _make_config(**(config_kw or {}))
        layer = _make_layer(config)
        layer.train()
        hc = layer.self_attention_hyper_connection
        if use_fused_kernel:
            _use_fused_kernel(self, hc)

        arrays = _random_inputs(with_bias)
        halves = 2

        def run(wrap_aggregate, wrap_bda, expected_spans):
            grads, n_spans = _run(
                layer, hc, wrap_aggregate, wrap_bda, *arrays, dtype=dtype
            )
            self.assertEqual(
                n_spans,
                expected_spans,
                "the helper did not create the expected number of BDA spans, "
                "so the comparison below would be vacuous",
            )
            return grads

        n_spans = halves if expect_span else 0
        plain = run(False, False, 0)
        only_bda = run(False, True, n_spans)
        only_agg = run(True, False, 0)
        both = run(True, True, n_spans)

        self.assertGreater(len(plain), 4, "no gradients collected")
        for key in plain:
            self.assert_no_nan(plain[key], key)
            for label, got in (("bda-span", only_bda), ("both-spans", both)):
                self.assertIn(key, got, f"{label}: missing grad {key}")
            # The span under test, alone and stacked on the aggregate span.
            self.assert_bitwise(plain[key], only_bda[key], f"bda-span {key}")
            self.assert_bitwise(only_agg[key], both[key], f"both-spans {key}")
            # Vacuity guard: the aggregate span does perturb the last fp32 bits,
            # so a run that silently skipped every span would not look like this.
            self.assert_close(
                plain[key], only_agg[key], f"aggregate-span {key}"
            )

    def test_native_kernel_with_bias(self):
        self._check(use_fused_kernel=False, with_bias=True)

    def test_native_kernel_without_bias(self):
        self._check(use_fused_kernel=False, with_bias=False)

    def test_fused_kernel_with_bias(self):
        self._check(use_fused_kernel=True, with_bias=True)

    def test_fused_kernel_without_bias(self):
        self._check(use_fused_kernel=True, with_bias=False)

    def test_bf16_native_kernel(self):
        self._check(
            {"params_dtype": paddle.bfloat16, "bf16": True},
            use_fused_kernel=False,
            dtype="bfloat16",
        )

    def test_bf16_fused_kernel(self):
        self._check(
            {"params_dtype": paddle.bfloat16, "bf16": True},
            use_fused_kernel=True,
            dtype="bfloat16",
        )

    def test_sequential_path_with_dropout_with_bias(self):
        # Active dropout takes fused_h_res_h_post_bda down the sequential path,
        # where the span hides that path's dropout mask instead of an up-cast,
        # and the replay has to restore the RNG state to reproduce the mask.
        # Bitwise equality is what proves the restore works.
        self._check({"hidden_dropout_prob": 0.1}, with_bias=True)

    def test_sequential_path_with_dropout_without_bias(self):
        self._check({"hidden_dropout_prob": 0.1}, with_bias=False)

    def test_accuracy_compatible_kernel_is_not_wrapped(self):
        # FLAGS_use_accuracy_compatible_kernel keeps the mHC input in the
        # incoming dtype, so there is no up-cast to hide and the helper must
        # decline (measured: 1040 KiB retained either way, minus the clones).
        # Only holds while dropout is off -- see the contract test for the
        # combination.
        with mock.patch.object(
            hyper_connection, "_ACCURACY_COMPATIBLE_KERNEL", True
        ):
            self._check(expect_span=False)


def _bda_inputs(layer, hc, dtype, s=S, b=B, c=C, seed=0):
    """Everything ``_fused_h_res_h_post_bda`` consumes, plus the residual."""
    rng = np.random.RandomState(seed)
    resid = paddle.to_tensor(
        rng.randn(s, b, N * c).astype("float32") * 0.02, dtype=dtype
    )
    resid.stop_gradient = False
    w = paddle.to_tensor(rng.randn(c, c).astype("float32") * 0.02, dtype=dtype)
    w.stop_gradient = False
    aggregated, h_res, h_post = hc(resid)
    x = paddle.matmul(aggregated.to(resid.dtype), w)
    return resid, w, h_res, h_post, x


@unittest.skipUnless(paddle.is_compiled_with_cuda(), "requires CUDA")
class TestFusedHResHPostBDASpanContract(_CompareMixin, unittest.TestCase):
    """The span/discard protocol itself, independent of the numbers."""

    def _prepare(self, **config_kw):
        config = _make_config(**config_kw)
        layer = _make_layer(config)
        layer.train()
        hc = layer.self_attention_hyper_connection
        return layer, hc

    def test_no_span_when_caller_disables_it(self):
        layer, hc = self._prepare()
        resid, _, h_res, h_post, x = _bda_inputs(layer, hc, "float32")
        _, span = layer._fused_h_res_h_post_bda(
            hc, h_res, resid, h_post, (x, None), False
        )
        self.assertIsNone(span)

    def test_span_created_when_it_pays_off(self):
        layer, hc = self._prepare()
        resid, _, h_res, h_post, x = _bda_inputs(layer, hc, "float32")
        _, span = layer._fused_h_res_h_post_bda(
            hc, h_res, resid, h_post, (x, None), True
        )
        self.assertIsInstance(span, RecomputeWithoutOutput)

    def test_no_span_with_accuracy_compatible_kernel(self):
        # The accuracy-compatible switch keeps mHC in the incoming dtype, so the
        # fast path has no up-cast to hide and the helper must veto the caller.
        layer, hc = self._prepare()
        resid, _, h_res, h_post, x = _bda_inputs(layer, hc, "float32")
        with mock.patch.object(
            hyper_connection, "_ACCURACY_COMPATIBLE_KERNEL", True
        ):
            out, span = layer._fused_h_res_h_post_bda(
                hc, h_res, resid, h_post, (x, None), True
            )
        self.assertIsNone(span)
        self.assertEqual(out.dtype, paddle.float32)

    def test_span_created_when_dropout_is_active(self):
        # Sequential path: no up-cast, but the dropout mask is worth hiding, so
        # the span stays (measured: 252 KiB per half-layer at [128, 2, 4*256]).
        layer, hc = self._prepare(hidden_dropout_prob=0.1)
        resid, _, h_res, h_post, x = _bda_inputs(layer, hc, "float32")
        _, span = layer._fused_h_res_h_post_bda(
            hc, h_res, resid, h_post, (x, None), True
        )
        self.assertIsInstance(span, RecomputeWithoutOutput)
        self.assertTrue(
            span.preserve_rng_state,
            "the dropout mask cannot be reproduced without the RNG state",
        )

    def test_span_created_for_accuracy_compatible_kernel_with_dropout(self):
        # Same reasoning for the other sequential-path trigger: the switch
        # removes the up-cast, not the mask.
        layer, hc = self._prepare(hidden_dropout_prob=0.1)
        resid, _, h_res, h_post, x = _bda_inputs(layer, hc, "float32")
        with mock.patch.object(
            hyper_connection, "_ACCURACY_COMPATIBLE_KERNEL", True
        ):
            _, span = layer._fused_h_res_h_post_bda(
                hc, h_res, resid, h_post, (x, None), True
            )
        self.assertIsInstance(span, RecomputeWithoutOutput)

    def test_span_created_in_eval_with_dropout_configured(self):
        # eval(): dropout is configured but does not run, so the call is back on
        # the fast path and the RNG snapshot is not needed.
        layer, hc = self._prepare(hidden_dropout_prob=0.1)
        layer.eval()
        resid, _, h_res, h_post, x = _bda_inputs(layer, hc, "float32")
        _, span = layer._fused_h_res_h_post_bda(
            hc, h_res, resid, h_post, (x, None), True
        )
        self.assertIsInstance(span, RecomputeWithoutOutput)
        self.assertFalse(span.preserve_rng_state)

    def test_discard_releases_the_fp32_result(self):
        # bf16 residual: the cast is a real cast, so the span owns the fp32
        # result outright and the discard must empty it while the returned bf16
        # tensor stays usable.
        layer, hc = self._prepare(params_dtype=paddle.bfloat16, bf16=True)
        resid, _, h_res, h_post, x = _bda_inputs(layer, hc, "bfloat16")
        out, span = layer._fused_h_res_h_post_bda(
            hc, h_res, resid, h_post, (x, None), True
        )
        self.assertEqual(out.dtype, paddle.float32)
        self.assertTrue(out._is_initialized())
        casted = layer._cast_and_discard_fused_bda(out, resid.dtype, span)
        self.assertEqual(casted.dtype, paddle.bfloat16)
        self.assertFalse(
            out._is_initialized(), "the fp32 result was not discarded"
        )
        self.assertTrue(casted._is_initialized())
        casted.astype("float32").sum().backward()
        self.assertTrue(np.isfinite(_npy(resid.grad)).all())

    def test_identity_cast_does_not_discard_live_data(self):
        # fp32 residual: Tensor.to(dtype) returns *self*, so a naive
        # ``discard(output.to(ori_dtype))`` would clear the tensor the rest of
        # the half-layer is still holding and the next layer would hit
        # "Tensor holds no memory". The helper must hand the span its own copy.
        layer, hc = self._prepare()
        resid, _, h_res, h_post, x = _bda_inputs(layer, hc, "float32")
        out, span = layer._fused_h_res_h_post_bda(
            hc, h_res, resid, h_post, (x, None), True
        )
        self.assertIs(
            out.to(resid.dtype), out, "premise of this test no longer holds"
        )
        casted = layer._cast_and_discard_fused_bda(out, resid.dtype, span)
        self.assertIsNot(casted, out)
        self.assertTrue(
            casted._is_initialized(), "the discard cleared live data"
        )
        # Consume it the way the next half-layer would.
        (casted * 2.0).astype("float32").sum().backward()
        self.assertTrue(np.isfinite(_npy(resid.grad)).all())

    def test_span_without_discard_breaks_backward(self):
        # Documents why the discard is mandatory rather than an optimization:
        # RecomputeWithoutOutput._recompute is the only place that populates
        # ctx.inputs/ctx.outputs for the backward, and only the discard
        # registers it.
        layer, hc = self._prepare(params_dtype=paddle.bfloat16, bf16=True)
        resid, _, h_res, h_post, x = _bda_inputs(layer, hc, "bfloat16")
        out, span = layer._fused_h_res_h_post_bda(
            hc, h_res, resid, h_post, (x, None), True
        )
        self.assertIsNotNone(span)
        leaked = out.to(resid.dtype)  # deliberately skip the discard
        with self.assertRaises(Exception) as caught:
            leaked.astype("float32").sum().backward()
        self.assertIn("RecomputeWithoutOutput", str(caught.exception))


@unittest.skipUnless(paddle.is_compiled_with_cuda(), "requires CUDA")
class TestFusedHResHPostBDASpanMemory(unittest.TestCase):
    """The span has no other purpose than this, so measure it."""

    S_MEM, B_MEM, C_MEM = 128, 2, 256

    def _retained(self, dtype, **config_kw):
        """Bytes still held after one half-layer's BDA, without / with the span."""
        s, b, c = self.S_MEM, self.B_MEM, self.C_MEM
        if dtype == "bfloat16":
            config_kw.update(params_dtype=paddle.bfloat16, bf16=True)
        config = _make_config(
            hidden_size=c,
            intermediate_size=2 * c,
            head_dim=c // 4,
            **config_kw,
        )
        layer = _make_layer(config)
        layer.train()
        hc = layer.self_attention_hyper_connection

        def retained(wrap_bda):
            resid, w, h_res, h_post, x = _bda_inputs(
                layer, hc, dtype, s=s, b=b, c=c
            )
            paddle.device.synchronize()
            base = paddle.device.memory_allocated()
            out, span = layer._fused_h_res_h_post_bda(
                hc, h_res, resid, h_post, (x, None), wrap_bda
            )
            out = layer._cast_and_discard_fused_bda(out, resid.dtype, span)
            paddle.device.synchronize()
            used = paddle.device.memory_allocated() - base
            self.assertEqual(span is not None, wrap_bda)
            del out, span, resid, w, h_res, h_post, x
            return used

        return retained(False), retained(True)

    def test_span_shrinks_the_retained_set(self):
        s, b, c = self.S_MEM, self.B_MEM, self.C_MEM
        without, with_span = self._retained("bfloat16")
        # The kernel's save_for_backward pins two fp32 up-casts when
        # high_precision_mhc is on: the [s, b, n, c] residual and the [s, b, c]
        # layer output. Those are what the span removes; the fp32 result itself
        # is released by the cast in both paths, so it does not show up here.
        upcast = s * b * (N + 1) * c * 4
        # What the span itself adds back: the h_res [s, b, n, n] / h_post
        # [s, b, n] clones it has to make because the mHC-aggregate span clears
        # the originals. Sized as fp32 so the bound holds for either dtype.
        overhead = s * b * (N * N + N) * 4
        self.assertGreaterEqual(
            without,
            upcast,
            f"baseline retains {without} B, less than the {upcast} B of "
            "up-casts the span is supposed to remove -- the premise is stale",
        )
        saved = without - with_span
        self.assertGreaterEqual(
            saved,
            upcast - overhead,
            f"span saved only {saved} B, expected at least "
            f"{upcast - overhead} B "
            f"(baseline {without} B, with span {with_span} B)",
        )

    def _assert_sequential_span_saves(self, **config_kw):
        """Active dropout, i.e. the sequential path: the span hides the mask.

        fp32 because the sequential path cannot run in bf16 today -- apply_h_res()
        bmm's an fp32 h_res (the mapping/sinkhorn output stays fp32 whatever
        high_precision_mhc says) against a bf16 residual and Paddle raises
        instead of promoting. Pre-existing, on the no-span path too.
        """
        s, b, c = self.S_MEM, self.B_MEM, self.C_MEM
        without, with_span = self._retained(
            "float32", hidden_dropout_prob=0.1, **config_kw
        )
        saved = without - with_span
        # Floor: the dropout mask is a byte per element of the [s, b, n*c]
        # output, less the h_res [s, b, n, n] / h_post [s, b, n] clones the span
        # adds back. Measured saving at this shape is 252 KiB.
        floor = s * b * N * c - s * b * (N * N + N) * 4
        self.assertGreaterEqual(
            saved,
            floor,
            f"span saved only {saved} B on the sequential path, expected at "
            f"least {floor} B (baseline {without} B, with span {with_span} B)",
        )

    def test_span_shrinks_the_retained_set_on_the_sequential_path(self):
        self._assert_sequential_span_saves()


def _span_spy(created, force_disable=False):
    """Patch the helper to record span creation, optionally forcing it off."""
    real = HyperConnectionTransformerLayer._fused_h_res_h_post_bda

    def wrapper(self, *args):
        if force_disable:
            args = (*args[:-1], False)
        output, span = real(self, *args)
        created.append(span is not None)
        return output, span

    return mock.patch.object(
        HyperConnectionTransformerLayer,
        "_fused_h_res_h_post_bda",
        wrapper,
    )


@unittest.skipUnless(paddle.is_compiled_with_cuda(), "requires CUDA")
class TestLayerLevelSpanNumerics(_CompareMixin, unittest.TestCase):
    """End-to-end through the real layer, driven by recompute_modules.

    Both spans hang off the single ``mhc_forward`` switch, so the reference run
    cannot be produced by config alone: it forces ``enable_recompute=False`` at
    the BDA helper while leaving the mHC-aggregate span in place, which is
    exactly the before/after of this change.
    """

    def _run_layer(self, layer, x_np, patch):
        x = paddle.to_tensor(x_np, dtype="float32")
        x.stop_gradient = False
        with patch:
            out = layer.forward({"hidden_states": x, "attention_mask": None})[
                "hidden_states"
            ]
        loss = out.astype("float32").sum()
        loss.backward()
        grads = {"loss": _npy(loss), "x": _npy(x.grad)}
        for name, p in layer.named_parameters():
            if p.grad is not None:
                grads[name] = _npy(p.grad)
        for p in layer.parameters():
            p.clear_gradient()
        return grads

    def test_recompute_modules_mhc_forward_is_bitwise(self):
        config = _make_config(
            recompute_granularity="selective",
            recompute_modules=["mhc_forward"],
        )
        layer = _make_layer(config)
        layer.train()
        self.assertTrue(
            layer.recompute_mhc_forward,
            "recompute_modules=['mhc_forward'] did not enable the switch",
        )
        rng = np.random.RandomState(7)
        x_np = rng.randn(S, B, N * C).astype("float32") * 0.02

        off, on = [], []
        ref = self._run_layer(layer, x_np, _span_spy(off, force_disable=True))
        got = self._run_layer(layer, x_np, _span_spy(on))

        self.assertEqual(off, [False, False], "reference run created a span")
        self.assertEqual(on, [True, True], "both half-layers must be wrapped")
        self.assertEqual(set(ref), set(got), "gradient sets differ")
        for key in ref:
            self.assert_no_nan(ref[key], key)
            self.assert_bitwise(ref[key], got[key], f"layer {key}")
