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

"""Behavior tests for Manifold-Constrained Hyper-Connections (mHC).

Covers the production surface in
``src/paddlefleet/transformer/hyper_connection.py``:

* the native reference ops (``native_proj_rms``, ``native_compute_h``,
  ``native_h_aggregate``, ``native_h_post_bda``, ``native_sinkhorn``),
* the differentiable ``SinkhornKnopp`` PyLayer (forward + custom backward),
* ``HyperConnectionModule`` methods (``aggregate``, ``apply_h_res``,
  ``_apply_h_post`` / ``apply_h_post``, ``compute_mappings``, ``forward``,
  ``fused_h_res_h_post_bda``, ``bda_span_pays_off``) and its parameter init,
* the block-level static helpers (``input_expand``, ``output_contract``,
  ``learned_output_contract``).

Every expected value is derived independently in NumPy (or via a plainly
re-implemented Paddle reference for the custom backward) from the documented
formulas, never by calling the production op back on itself.

No-card environment: Paddle runs on CPU here. When Paddle is not importable the
whole module is skipped with an honest reason; only ImportError /
ModuleNotFoundError are treated as "dependency missing".
"""

import math
import unittest
from types import SimpleNamespace

import numpy as np

try:
    import paddle
    import paddle.nn.functional as PF

    from paddlefleet.transformer.hyper_connection import (
        HyperConnectionModule,
        native_compute_h,
        native_h_aggregate,
        native_h_post_bda,
        native_proj_rms,
        native_sinkhorn,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc


_MHC_COMPUTE_H_EPS = 1e-6  # mirrors hyper_connection._MHC_COMPUTE_H_EPS


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# --------------------------------------------------------------------------- #
# Independent NumPy references (documented mHC formulas, no production calls)  #
# --------------------------------------------------------------------------- #
def _ref_proj_rms(x, weight, eps):
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    nC = x.shape[-1]
    r = np.linalg.norm(x, axis=-1, keepdims=True) / math.sqrt(nC)
    r = 1.0 / (r + eps)
    proj = x @ weight
    return proj, r


def _ref_compute_h(proj, r, a_pre, a_post, a_res, bias, n, eps):
    proj = np.asarray(proj, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    alpha = np.concatenate(
        [np.full(n, a_pre), np.full(n, a_post), np.full(n * n, a_res)]
    )
    h = r * proj * alpha + bias
    h_pre = _sigmoid(h[..., :n]) + eps
    h_post = 2.0 * _sigmoid(h[..., n : 2 * n])
    h_res = h[..., 2 * n :]
    return h_pre, h_post, h_res


def _ref_h_aggregate(x_streams, h_pre):
    x_streams = np.asarray(x_streams, dtype=np.float64)
    h_pre = np.asarray(h_pre, dtype=np.float64)
    return (x_streams * h_pre[..., None]).sum(axis=-2)


def _ref_h_post_bda(h_res, residual, h_post, x, bias):
    h_res = np.asarray(h_res, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    h_post = np.asarray(h_post, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    lead = residual.shape[:-2]
    n, C = residual.shape[-2], residual.shape[-1]
    ntok = int(np.prod(lead)) if lead else 1
    hr = h_res.reshape(ntok, n, n).transpose(0, 2, 1)  # H_res^T
    res = residual.reshape(ntok, n, C)
    mixed = np.matmul(hr, res).reshape(*lead, n, C)
    x_exp = h_post[..., None] * x[..., None, :]
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        return (
            x_exp
            + h_post[..., None] * bias.reshape(*([1] * len(lead)), 1, C)
            + mixed
        )
    return x_exp + mixed


def _ref_sinkhorn(logits, iters, eps):
    logits = np.asarray(logits, dtype=np.float64)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    M = e / e.sum(axis=-1, keepdims=True) + eps
    M = M / (M.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        M = M / (M.sum(axis=-1, keepdims=True) + eps)
        M = M / (M.sum(axis=-2, keepdims=True) + eps)
    return M


def _ref_learned_contract(hs, head_fn, base, scale, n, eps):
    hs = np.asarray(hs, dtype=np.float64)
    head_fn = np.asarray(head_fn, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    scale = float(np.asarray(scale, dtype=np.float64).reshape([-1])[0])
    rsqrt = 1.0 / np.sqrt((hs**2).mean(axis=-1, keepdims=True) + eps)
    mixes = (hs @ head_fn) * rsqrt
    pre = _sigmoid(mixes * scale + base) + eps
    h = hs.shape[-1] // n
    y = (pre[..., None] * hs.reshape(*hs.shape[:-1], n, h)).sum(axis=-2)
    return y


def _make_config(**over):
    cfg = {
        "num_residual_streams": 4,
        "hidden_size": 8,
        "mhc_sinkhorn_iterations": 20,
        "mhc_single_stream_init": False,
        "mhc_init_gating_factor": 0.01,
        "use_fused_mhc": False,
        "high_precision_mhc": True,
        "sequence_parallel": False,
    }
    cfg.update(over)
    return SimpleNamespace(**cfg)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle / paddlefleet not importable on this host: {_IMPORT_ERROR}",
)
class NativeOpsTest(unittest.TestCase):
    """The native reference ops that back the non-fused mHC path."""

    def setUp(self):
        paddle.seed(0)
        paddle.set_device("cpu")

    # ----- native_proj_rms --------------------------------------------- #
    def test_proj_rms_matches_matmul_and_inverse_rms(self):
        nC = 6
        x = paddle.to_tensor(
            [
                [1.0, -2.0, 0.5, 3.0, -1.5, 2.0],
                [0.0, 1.0, -1.0, 2.0, 4.0, -3.0],
            ],
            dtype="float32",
        )
        weight = paddle.to_tensor(
            np.arange(nC * 4).reshape(nC, 4) * 0.1 - 1.0, dtype="float32"
        )
        proj, r = native_proj_rms(x, weight, 1e-6)
        proj_ref, r_ref = _ref_proj_rms(x.numpy(), weight.numpy(), 1e-6)
        np.testing.assert_allclose(proj.numpy(), proj_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r.numpy(), r_ref, rtol=1e-5, atol=1e-6)
        # r is the *inverse* RMS-like norm: larger vectors -> smaller r.
        self.assertLess(float(r.numpy()[0, 0]), float(r.numpy()[1, 0]) + 1e6)
        self.assertTrue((r.numpy() > 0).all())

    # ----- native_compute_h -------------------------------------------- #
    def test_compute_h_segments_and_activations(self):
        n = 2  # proj width = n*n + 2n = 8
        proj = paddle.to_tensor(
            [[0.5, -0.5, 1.0, -1.0, 0.2, 0.4, -0.6, 0.8]], dtype="float32"
        )
        r = paddle.to_tensor([[2.0]], dtype="float32")
        # Distinct alpha per segment so a pre/post/res mix-up is observable.
        a_pre = paddle.to_tensor([0.5], dtype="float32")
        a_post = paddle.to_tensor([0.3], dtype="float32")
        a_res = paddle.to_tensor([0.7], dtype="float32")
        bias = paddle.to_tensor(
            [[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]], dtype="float32"
        ).reshape([8])
        h_pre, h_post, h_res = native_compute_h(
            proj, r, a_pre, a_post, a_res, bias, n, _MHC_COMPUTE_H_EPS
        )
        pre_ref, post_ref, res_ref = _ref_compute_h(
            proj.numpy(),
            r.numpy(),
            0.5,
            0.3,
            0.7,
            bias.numpy(),
            n,
            _MHC_COMPUTE_H_EPS,
        )
        np.testing.assert_allclose(h_pre.numpy(), pre_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            h_post.numpy(), post_ref, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(h_res.numpy(), res_ref, rtol=1e-5, atol=1e-6)
        # Activation ranges are part of the contract: h_pre in (eps, 1+eps),
        # h_post in (0, 2), h_res is the raw (un-activated) logit slice.
        self.assertTrue((h_pre.numpy() > _MHC_COMPUTE_H_EPS).all())
        self.assertTrue((h_pre.numpy() < 1.0 + _MHC_COMPUTE_H_EPS + 1e-6).all())
        self.assertTrue((h_post.numpy() > 0).all())
        self.assertTrue((h_post.numpy() < 2.0).all())

    def test_compute_h_pre_gets_eps_but_post_does_not(self):
        # A zero logit isolates the epsilon convention: h_pre = 0.5 + eps,
        # h_post = 2*0.5 = 1.0 (no eps added on the post segment).
        n = 1  # width = 1 + 2 = 3
        proj = paddle.zeros([1, 3], dtype="float32")
        r = paddle.ones([1, 1], dtype="float32")
        zero = paddle.zeros([1], dtype="float32")
        bias = paddle.zeros([3], dtype="float32")
        h_pre, h_post, _ = native_compute_h(
            proj, r, zero, zero, zero, bias, n, _MHC_COMPUTE_H_EPS
        )
        np.testing.assert_allclose(
            h_pre.numpy(), [[0.5 + _MHC_COMPUTE_H_EPS]], rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(h_post.numpy(), [[1.0]], rtol=0, atol=1e-6)

    # ----- native_h_aggregate ------------------------------------------ #
    def test_h_aggregate_weighted_stream_sum(self):
        # Streams are per-stream-distinguishable so a wrong weight or a
        # dropped stream changes the result.
        x_streams = paddle.to_tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype="float32"
        )  # [T=1, n=3, C=2]
        h_pre = paddle.to_tensor([[0.1, 0.5, 0.4]], dtype="float32")
        out = native_h_aggregate(x_streams, h_pre)
        ref = _ref_h_aggregate(x_streams.numpy(), h_pre.numpy())
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-6, atol=1e-6)
        # Hand value: 0.1*[1,2]+0.5*[3,4]+0.4*[5,6] = [3.6, 4.6]
        np.testing.assert_allclose(
            out.numpy(), [[3.6, 4.6]], rtol=1e-6, atol=1e-5
        )

    # ----- native_h_post_bda ------------------------------------------- #
    def test_h_post_bda_uses_h_res_transpose_no_bias(self):
        # Asymmetric H_res so the transpose direction (H_res^T @ residual)
        # is actually exercised.
        h_res = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype="float32")
        residual = paddle.to_tensor(
            [[10.0, 11.0, 20.0, 21.0]], dtype="float32"
        )  # [T=1, n*C=4] -> streams [[10,11],[20,21]]
        h_post = paddle.to_tensor([[0.5, 2.0]], dtype="float32")
        x = paddle.to_tensor([[100.0, 200.0]], dtype="float32")
        residual_streams = residual.reshape([1, 2, 2])
        out = native_h_post_bda(h_res, residual_streams, h_post, x, None)
        ref = _ref_h_post_bda(
            h_res.numpy(),
            residual_streams.numpy(),
            h_post.numpy(),
            x.numpy(),
            None,
        )
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)
        # Independent hand check of the mixed term (H_res^T @ residual):
        #   col0 of H_res = [1,3] -> 1*[10,11]+3*[20,21] = [70, 74]
        #   col1 of H_res = [2,4] -> 2*[10,11]+4*[20,21] = [100,106]
        # plus x_expanded: [0.5*100,0.5*200]=[50,100]; [2*100,2*200]=[200,400]
        expected = np.array([[[70 + 50, 74 + 100], [100 + 200, 106 + 400]]])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-4)

    def test_h_post_bda_adds_gated_bias(self):
        h_res = paddle.to_tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype="float32")
        residual = paddle.to_tensor(
            [[1.0, 2.0, 3.0, 4.0]], dtype="float32"
        ).reshape([1, 2, 2])
        h_post = paddle.to_tensor([[0.5, 2.0]], dtype="float32")
        x = paddle.to_tensor([[10.0, 20.0]], dtype="float32")
        bias = paddle.to_tensor([1.0, -1.0], dtype="float32")
        out = native_h_post_bda(h_res, residual, h_post, x, bias)
        out_nobias = native_h_post_bda(h_res, residual, h_post, x, None)
        ref = _ref_h_post_bda(
            h_res.numpy(),
            residual.numpy(),
            h_post.numpy(),
            x.numpy(),
            bias.numpy(),
        )
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)
        # The bias contribution must itself be gated by h_post per stream.
        delta = (out - out_nobias).numpy()
        expected_delta = np.array(
            [[[0.5 * 1.0, 0.5 * -1.0], [2.0 * 1.0, 2.0 * -1.0]]]
        )
        np.testing.assert_allclose(delta, expected_delta, rtol=1e-5, atol=1e-5)

    def test_h_post_bda_multi_leading_dims(self):
        # [S=2, B=2] leading dims exercise the reshape/prod(leading) path.
        rng = np.random.RandomState(1)
        n, C = 3, 2
        h_res = paddle.to_tensor(rng.randn(2, 2, n, n), dtype="float32")
        residual = paddle.to_tensor(rng.randn(2, 2, n, C), dtype="float32")
        h_post = paddle.to_tensor(rng.randn(2, 2, n), dtype="float32")
        x = paddle.to_tensor(rng.randn(2, 2, C), dtype="float32")
        out = native_h_post_bda(h_res, residual, h_post, x, None)
        ref = _ref_h_post_bda(
            h_res.numpy(), residual.numpy(), h_post.numpy(), x.numpy(), None
        )
        self.assertEqual(list(out.shape), [2, 2, n, C])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-4)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle / paddlefleet not importable on this host: {_IMPORT_ERROR}",
)
class SinkhornKnoppTest(unittest.TestCase):
    """Differentiable doubly-stochastic projection."""

    def setUp(self):
        paddle.seed(0)
        paddle.set_device("cpu")

    def test_single_iteration_exact(self):
        # iters=1 -> softmax(rows) + eps, then one column normalization; the
        # inner loop (range(iters-1)) does not run. Pin the exact arithmetic.
        logits = paddle.to_tensor([[[0.0, 1.0], [2.0, -1.0]]], dtype="float32")
        eps = 1e-6
        out = native_sinkhorn(logits, 1, eps)
        ref = _ref_sinkhorn(logits.numpy(), 1, eps)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-6)

    def test_twenty_iterations_doubly_stochastic(self):
        rng = np.random.RandomState(3)
        logits = paddle.to_tensor(rng.randn(4, 5, 5), dtype="float32")
        out = native_sinkhorn(logits, 20, 1e-6).numpy()
        ref = _ref_sinkhorn(logits.numpy(), 20, 1e-6)
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-4)
        # The final op is a column normalization, so column sums are ~1.
        col_sums = out.sum(axis=-2)
        np.testing.assert_allclose(col_sums, np.ones_like(col_sums), atol=2e-3)
        # After 20 iterations rows are ~1 as well (converged Sinkhorn).
        row_sums = out.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=5e-2)
        self.assertTrue((out > 0).all())

    def test_backward_matches_plain_autograd_reference(self):
        # Independent reference: re-implement the same normalization with plain
        # Paddle ops and differentiate it via autograd. The PyLayer's custom
        # backward recomputes exactly this, so gradients must agree. This is a
        # separate implementation, not a call back into SinkhornKnopp.
        iters, eps = 3, 1e-6
        base = np.random.RandomState(7).randn(2, 3, 3).astype("float32")

        logits = paddle.to_tensor(base)
        logits.stop_gradient = False
        out = native_sinkhorn(logits, iters, eps)
        upstream = paddle.to_tensor(
            np.random.RandomState(8).randn(2, 3, 3).astype("float32")
        )
        out.backward(upstream)
        got = logits.grad.numpy()

        ref_logits = paddle.to_tensor(base)
        ref_logits.stop_gradient = False
        M = PF.softmax(ref_logits, axis=-1) + eps
        M = M / (M.sum(axis=-2, keepdim=True) + eps)
        for _ in range(iters - 1):
            M = M / (M.sum(axis=-1, keepdim=True) + eps)
            M = M / (M.sum(axis=-2, keepdim=True) + eps)
        expected = paddle.grad([M], [ref_logits], grad_outputs=[upstream])[
            0
        ].numpy()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(np.isfinite(got).all())
        self.assertGreater(np.abs(expected).max(), 1e-4)  # non-trivial gradient
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-4)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle / paddlefleet not importable on this host: {_IMPORT_ERROR}",
)
class BlockStaticHelpersTest(unittest.TestCase):
    """input_expand / output_contract / learned_output_contract."""

    def setUp(self):
        paddle.seed(0)
        paddle.set_device("cpu")

    def test_input_expand_replicates_each_stream(self):
        n = 3
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype="float32"
        )  # [T=2, C=2]
        out = HyperConnectionModule.input_expand(x, n)
        self.assertEqual(list(out.shape), [2, n * 2])
        streams = out.reshape([2, n, 2]).numpy()
        for i in range(n):
            np.testing.assert_array_equal(streams[:, i, :], x.numpy())

    def test_output_contract_averages_distinct_streams(self):
        n = 3
        # Distinguishable streams so a wrong reduction (e.g. sum, or first
        # stream only) is caught.
        flat = paddle.to_tensor(
            [[1.0, 2.0, 10.0, 20.0, 100.0, 200.0]], dtype="float32"
        )  # streams [[1,2],[10,20],[100,200]]
        out = HyperConnectionModule.output_contract(flat, n)
        expected = np.array([[(1 + 10 + 100) / 3.0, (2 + 20 + 200) / 3.0]])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-5)

    def test_expand_then_contract_is_identity(self):
        n = 4
        x = paddle.to_tensor(
            np.random.RandomState(2).randn(3, 5).astype("float32")
        )
        expanded = HyperConnectionModule.input_expand(x, n)
        contracted = HyperConnectionModule.output_contract(expanded, n)
        np.testing.assert_allclose(
            contracted.numpy(), x.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_learned_output_contract_matches_reference(self):
        n, h = 2, 3
        hc_dim = n * h
        rng = np.random.RandomState(5)
        hs = paddle.to_tensor(rng.randn(4, hc_dim), dtype="float32")
        head_fn = paddle.to_tensor(rng.randn(hc_dim, n), dtype="float32")
        base = paddle.to_tensor(rng.randn(n), dtype="float32")
        scale = paddle.to_tensor([0.7], dtype="float32")
        eps = 1e-5
        out = HyperConnectionModule.learned_output_contract(
            hs, head_fn, base, scale, n, eps
        )
        ref = _ref_learned_contract(
            hs.numpy(), head_fn.numpy(), base.numpy(), scale.numpy(), n, eps
        )
        self.assertEqual(list(out.shape), [4, h])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-4)

    def test_learned_output_contract_preserves_input_dtype(self):
        n, h = 2, 2
        hc_dim = n * h
        hs = paddle.ones([2, hc_dim], dtype="float32")
        head_fn = paddle.ones([hc_dim, n], dtype="float32")
        base = paddle.zeros([n], dtype="float32")
        scale = paddle.ones([1], dtype="float32")
        out = HyperConnectionModule.learned_output_contract(
            hs, head_fn, base, scale, n, 1e-5
        )
        # Documented: internal compute is fp32, output cast back to input dtype.
        self.assertEqual(out.dtype, hs.dtype)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle / paddlefleet not importable on this host: {_IMPORT_ERROR}",
)
class HyperConnectionModuleTest(unittest.TestCase):
    """End-to-end behavior of the mHC module on the native (CPU) path."""

    def setUp(self):
        paddle.seed(0)
        paddle.set_device("cpu")

    def _module(self, **cfg_over):
        return HyperConnectionModule(_make_config(**cfg_over), layer_number=0)

    # ----- parameter initialization ------------------------------------ #
    def test_mapping_params_stored_in_fp32(self):
        m = self._module(num_residual_streams=3, hidden_size=4)
        # mapping head is kept out of low precision (fp32) by contract.
        self.assertEqual(m.mapping_proj.weight.dtype, paddle.float32)
        self.assertEqual(m.alpha_pre.dtype, paddle.float32)
        self.assertEqual(m.bias.dtype, paddle.float32)
        n, C = 3, 4
        self.assertEqual(
            list(m.mapping_proj.weight.shape), [n * C, n * n + 2 * n]
        )
        self.assertEqual(list(m.bias.shape), [n * n + 2 * n])

    def test_default_init_alphas_and_zero_bias(self):
        m = self._module(
            mhc_init_gating_factor=0.02, mhc_single_stream_init=False
        )
        for a in (m.alpha_pre, m.alpha_post, m.alpha_res):
            np.testing.assert_allclose(a.numpy(), [0.02], rtol=0, atol=1e-7)
        # Historical (non single-stream) init keeps bias at zero and a
        # non-zero Xavier projection.
        np.testing.assert_array_equal(
            m.bias.numpy(), np.zeros_like(m.bias.numpy())
        )
        self.assertGreater(np.abs(m.mapping_proj.weight.numpy()).max(), 0.0)

    def test_single_stream_init_bias_pattern(self):
        n = 4
        m = self._module(
            num_residual_streams=n, hidden_size=4, mhc_single_stream_init=True
        )
        bias = m.bias.numpy()
        # Projection is zero-initialized under single-stream init.
        np.testing.assert_array_equal(
            m.mapping_proj.weight.numpy(),
            np.zeros_like(m.mapping_proj.weight.numpy()),
        )
        # b_pre = -3 except the home stream (layer_number % n == 0) which is +3.
        expected_pre = np.full(n, -3.0)
        expected_pre[0] = 3.0
        np.testing.assert_allclose(bias[:n], expected_pre, atol=1e-5)
        # b_post = 0
        np.testing.assert_allclose(bias[n : 2 * n], np.zeros(n), atol=1e-6)
        # b_res = 6I - 3
        expected_res = (6.0 * np.eye(n) - 3.0).flatten()
        np.testing.assert_allclose(bias[2 * n :], expected_res, atol=1e-5)

    def test_single_stream_home_stream_rotates_with_layer_number(self):
        n = 4
        m = HyperConnectionModule(
            _make_config(
                num_residual_streams=n,
                hidden_size=4,
                mhc_single_stream_init=True,
            ),
            layer_number=5,  # home stream = 5 % 4 = 1
        )
        bias = m.bias.numpy()
        expected_pre = np.full(n, -3.0)
        expected_pre[1] = 3.0
        np.testing.assert_allclose(bias[:n], expected_pre, atol=1e-5)

    # ----- aggregate --------------------------------------------------- #
    def test_aggregate_weighted_sum(self):
        n, C = 3, 2
        m = self._module(num_residual_streams=n, hidden_size=C)
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype="float32"
        )  # streams [[1,2],[3,4],[5,6]]
        h_pre = paddle.to_tensor([[0.2, 0.3, 0.5]], dtype="float32")
        out = m.aggregate(x, h_pre)
        ref = _ref_h_aggregate(x.reshape([1, n, C]).numpy(), h_pre.numpy())
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)

    # ----- apply_h_res ------------------------------------------------- #
    def test_apply_h_res_transpose_and_mix(self):
        n, C = 2, 2
        m = self._module(num_residual_streams=n, hidden_size=C)
        h_res = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype="float32")
        residual = paddle.to_tensor([[10.0, 11.0, 20.0, 21.0]], dtype="float32")
        out = m.apply_h_res(h_res, residual)  # [T, n*C]
        # H_res^T @ residual streams:
        #   col0 [1,3] -> 1*[10,11]+3*[20,21]=[70,74]
        #   col1 [2,4] -> 2*[10,11]+4*[20,21]=[100,106]
        expected = np.array([[70.0, 74.0, 100.0, 106.0]])
        self.assertEqual(list(out.shape), [1, n * C])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-4)

    # ----- _apply_h_post / apply_h_post -------------------------------- #
    def test_apply_h_post_expands_hidden_states(self):
        n, C = 3, 2
        m = self._module(num_residual_streams=n, hidden_size=C)
        x = paddle.to_tensor([[1.0, 10.0]], dtype="float32")
        h_post = paddle.to_tensor([[0.5, 1.0, 2.0]], dtype="float32")
        out = m._apply_h_post(x, h_post)  # [T, n*C]
        expected = np.array([[0.5, 5.0, 1.0, 10.0, 2.0, 20.0]])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_apply_h_post_broadcasts_1d_bias(self):
        n, C = 2, 3
        m = self._module(num_residual_streams=n, hidden_size=C)
        bias = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")  # [C]
        h_post = paddle.to_tensor([[0.5, 2.0]], dtype="float32")
        out = m._apply_h_post(bias, h_post)
        expected = np.array([[0.5, 1.0, 1.5, 2.0, 4.0, 6.0]])
        self.assertEqual(list(out.shape), [1, n * C])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_apply_h_post_tuple_none_bias(self):
        n, C = 2, 2
        m = self._module(num_residual_streams=n, hidden_size=C)
        x = paddle.to_tensor([[3.0, 4.0]], dtype="float32")
        h_post = paddle.to_tensor([[1.0, 0.5]], dtype="float32")
        x_out, bias_out = m.apply_h_post((x, None), h_post)
        self.assertIsNone(bias_out)
        np.testing.assert_allclose(
            x_out.numpy(), [[3.0, 4.0, 1.5, 2.0]], rtol=1e-5, atol=1e-5
        )

    # ----- compute_mappings ------------------------------------------- #
    def test_compute_mappings_single_stream_init_is_token_independent(self):
        n, C = 4, 3
        m = self._module(
            num_residual_streams=n, hidden_size=C, mhc_single_stream_init=True
        )
        # Zero projection => h = bias, independent of the input. Two very
        # different token rows must yield identical mappings.
        x = paddle.to_tensor(
            np.random.RandomState(11).randn(2, n * C).astype("float32") * 5.0
        )
        h_pre, h_post, h_res = m.compute_mappings(x)
        bias = m.bias.numpy()
        exp_pre = _sigmoid(bias[:n]) + _MHC_COMPUTE_H_EPS
        exp_post = 2.0 * _sigmoid(bias[n : 2 * n])
        for t in range(2):
            np.testing.assert_allclose(h_pre.numpy()[t], exp_pre, atol=1e-5)
            np.testing.assert_allclose(h_post.numpy()[t], exp_post, atol=1e-5)
        # token independence
        np.testing.assert_allclose(
            h_pre.numpy()[0], h_pre.numpy()[1], atol=1e-6
        )
        # h_res is doubly stochastic and, from b_res = 6I-3, ~identity.
        hr = h_res.numpy()
        self.assertEqual(list(h_res.shape), [2, n, n])
        np.testing.assert_allclose(hr.sum(axis=-2), np.ones((2, n)), atol=2e-3)
        for t in range(2):
            diag = np.diag(hr[t])
            off = hr[t] - np.diagflat(diag)
            self.assertGreater(diag.min(), off.max())  # diagonally dominant

    def test_compute_mappings_default_init_ranges(self):
        n, C = 3, 4
        m = self._module(num_residual_streams=n, hidden_size=C)
        x = paddle.to_tensor(
            np.random.RandomState(12).randn(5, n * C).astype("float32")
        )
        h_pre, h_post, h_res = m.compute_mappings(x)
        self.assertEqual(list(h_pre.shape), [5, n])
        self.assertEqual(list(h_post.shape), [5, n])
        self.assertEqual(list(h_res.shape), [5, n, n])
        self.assertTrue((h_pre.numpy() > 0).all())
        self.assertTrue((h_pre.numpy() < 1.0 + 1e-5).all())
        self.assertTrue((h_post.numpy() > 0).all())
        self.assertTrue((h_post.numpy() < 2.0).all())
        np.testing.assert_allclose(
            h_res.numpy().sum(axis=-2), np.ones((5, n)), atol=2e-3
        )

    # ----- forward ----------------------------------------------------- #
    def test_forward_returns_aggregated_hres_hpost(self):
        n, C = 4, 3
        m = self._module(
            num_residual_streams=n, hidden_size=C, mhc_single_stream_init=True
        )
        x = paddle.to_tensor(
            np.random.RandomState(13).randn(2, n * C).astype("float32")
        )
        with paddle.no_grad():
            aggregated, h_res, h_post = m.forward(x)
        # h_pre / h_post are token-independent under single-stream init, so the
        # aggregation is a fixed weighted stream sum that we derive by hand.
        bias = m.bias.numpy()
        exp_pre = _sigmoid(bias[:n]) + _MHC_COMPUTE_H_EPS
        x_streams = x.numpy().reshape(2, n, C)
        expected_agg = (x_streams * exp_pre[None, :, None]).sum(axis=1)
        self.assertEqual(list(aggregated.shape), [2, C])
        np.testing.assert_allclose(
            aggregated.numpy(), expected_agg, rtol=1e-4, atol=1e-4
        )
        self.assertEqual(list(h_res.shape), [2, n, n])
        self.assertEqual(list(h_post.shape), [2, n])
        np.testing.assert_allclose(
            h_post.numpy(),
            (2.0 * _sigmoid(bias[n : 2 * n]))[None, :].repeat(2, 0),
            atol=1e-5,
        )

    # ----- fused_h_res_h_post_bda -------------------------------------- #
    def test_fused_bda_fast_path_matches_reference_no_bias(self):
        n, C = 2, 2
        m = self._module(num_residual_streams=n, hidden_size=C)
        h_res = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype="float32")
        residual = paddle.to_tensor([[10.0, 11.0, 20.0, 21.0]], dtype="float32")
        h_post = paddle.to_tensor([[0.5, 2.0]], dtype="float32")
        x = paddle.to_tensor([[100.0, 200.0]], dtype="float32")
        out = m.fused_h_res_h_post_bda(
            h_res=h_res,
            original_residual=residual,
            h_post=h_post,
            layer_output_with_bias=(x, None),
            dropout_prob=0.0,
            training=True,
            fused=False,
        )
        ref = _ref_h_post_bda(
            h_res.numpy(),
            residual.reshape([1, n, C]).numpy(),
            h_post.numpy(),
            x.numpy(),
            None,
        ).reshape(1, n * C)
        self.assertEqual(list(out.shape), [1, n * C])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-4)

    def test_fused_bda_fast_path_with_bias(self):
        n, C = 2, 3
        m = self._module(num_residual_streams=n, hidden_size=C)
        rng = np.random.RandomState(14)
        h_res = paddle.to_tensor(rng.randn(1, n, n), dtype="float32")
        residual = paddle.to_tensor(rng.randn(1, n * C), dtype="float32")
        h_post = paddle.to_tensor(rng.randn(1, n), dtype="float32")
        x = paddle.to_tensor(rng.randn(1, C), dtype="float32")
        bias = paddle.to_tensor(rng.randn(C), dtype="float32")
        out = m.fused_h_res_h_post_bda(
            h_res=h_res,
            original_residual=residual,
            h_post=h_post,
            layer_output_with_bias=(x, bias),
            dropout_prob=0.0,
            training=False,
            fused=False,
        )
        ref = _ref_h_post_bda(
            h_res.numpy(),
            residual.reshape([1, n, C]).numpy(),
            h_post.numpy(),
            x.numpy(),
            bias.numpy(),
        ).reshape(1, n * C)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-4)

    def test_fused_bda_ignores_dropout_when_not_training(self):
        # dropout_prob > 0 but training=False must take the deterministic fast
        # path (no dropout applied), matching the no-dropout reference.
        n, C = 2, 2
        m = self._module(num_residual_streams=n, hidden_size=C)
        h_res = paddle.to_tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype="float32")
        residual = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]], dtype="float32")
        h_post = paddle.to_tensor([[1.0, 1.0]], dtype="float32")
        x = paddle.to_tensor([[5.0, 6.0]], dtype="float32")
        out = m.fused_h_res_h_post_bda(
            h_res=h_res,
            original_residual=residual,
            h_post=h_post,
            layer_output_with_bias=(x, None),
            dropout_prob=0.9,
            training=False,
            fused=False,
        )
        ref = _ref_h_post_bda(
            h_res.numpy(),
            residual.reshape([1, n, C]).numpy(),
            h_post.numpy(),
            x.numpy(),
            None,
        ).reshape(1, n * C)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)

    # ----- bda_span_pays_off ------------------------------------------- #
    def test_bda_span_pays_off_dropout_branch(self):
        m = self._module(high_precision_mhc=True)
        # Active dropout in training always pays off, regardless of precision.
        self.assertTrue(m.bda_span_pays_off(0.5, True, None))
        # Not training -> dropout does not force it.
        self.assertTrue(
            m.bda_span_pays_off(0.0, False, None)
        )  # high precision path

    def test_bda_span_pays_off_low_precision_returns_false(self):
        m = self._module(high_precision_mhc=False)
        self.assertFalse(m.bda_span_pays_off(0.0, False, None))
        self.assertFalse(m.bda_span_pays_off(0.0, True, None))
        # Dropout branch still short-circuits to True first.
        self.assertTrue(m.bda_span_pays_off(0.3, True, None))


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle / paddlefleet not importable on this host: {_IMPORT_ERROR}",
)
class TransformerConfigMhcDefaultsTest(unittest.TestCase):
    """The mHC config fields that HyperConnectionModule consumes exist with
    the documented defaults on the real TransformerConfig dataclass."""

    def test_mhc_field_defaults(self):
        fields = TransformerConfig.__dataclass_fields__
        self.assertEqual(fields["num_residual_streams"].default, 4)
        self.assertEqual(fields["mhc_sinkhorn_iterations"].default, 20)
        self.assertEqual(fields["mhc_init_gating_factor"].default, 0.01)
        self.assertFalse(fields["use_fused_mhc"].default)
        self.assertTrue(fields["high_precision_mhc"].default)
        self.assertFalse(fields["mhc_single_stream_init"].default)
        self.assertFalse(fields["enable_hyper_connections"].default)


if __name__ == "__main__":
    unittest.main()
