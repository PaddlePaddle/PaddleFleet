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

"""``native_compute_h`` and the mHC operator dispatch, without cuTile.

The fused mHC tests are all gated behind ``is_cutile_available()``, so on a
runner without ``cuda.tile`` they skip and leave the mapping head untested.
Everything checked here runs on any GPU: the reference implementation
``native_compute_h`` -- which is also what the fused kernel is validated
against -- plus the dispatch that decides which implementation a module
instance gets and whether its operands are widened in Python or in-register.

The dispatch assertions are the cheap guard that matters: they are what catches
a configuration being silently routed to the wrong implementation, which no
amount of kernel-level testing would notice.
"""

import unittest

import paddle

import paddlefleet.transformer.hyper_connection as hc
from paddlefleet.tensor_parallel.random import get_cuda_rng_tracker
from paddlefleet.transformer.hyper_connection import (
    HyperConnectionModule,
    native_compute_h,
    native_h_aggregate,
    native_h_post_bda,
    native_proj_rms,
    native_sinkhorn,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

_S, _B, _N = 4, 2, 4
_P = _N * _N + 2 * _N
_C = 64
_EPS = 1e-6

# Both supported stream counts. n = 2 needs its own coverage: nothing in the
# init is n-agnostic by construction -- the three bias segments change length,
# the 6I - 3 block changes shape, and the ``% n`` rotation degenerates. At n = 2
# the sub-layer numbering 2*index / 2*index + 1 makes every attention sub-layer
# read stream 0 and every MLP sub-layer read stream 1, so the home stream stops
# depending on the layer at all -- a property worth pinning, and one that hides
# phase errors an n = 4 assertion would catch.
_NS = (2, _N)


def _config(**kw):
    """Minimal config that satisfies the mHC field validation."""
    base = {
        "num_hidden_layers": 2,
        "hidden_size": _C,
        "num_attention_heads": 4,
        "enable_hyper_connections": True,
        "num_residual_streams": _N,
    }
    base.update(kw)
    return TransformerConfig(**base)


def _module(**kw):
    """Build a module, tolerating a distributed env with world_size > 1.

    ``_init_weights`` forks the model-parallel RNG tracker whenever
    ``get_world_size() > 1``, which a single-process test has not seeded. CI
    runs with world_size 1 and never hits this; inside a multi-node job
    container the env vars say otherwise, and without the seed the constructor
    would raise before any assertion runs.
    """
    tracker = get_cuda_rng_tracker()
    if "model-parallel-rng" not in tracker.states_:
        tracker.add("model-parallel-rng", 1)
    return HyperConnectionModule(_config(**kw), layer_number=1)


def _inputs(dtype="float32"):
    paddle.seed(23)
    proj = paddle.randn([_S, _B, _P], dtype=dtype)
    # r is 1 / (||x|| / sqrt(K) + eps): positive and O(1)
    r = paddle.randn([_S, _B, 1], dtype=dtype).abs() + 1.0
    alpha_pre = paddle.randn([1], dtype=dtype)
    alpha_post = paddle.randn([1], dtype=dtype)
    alpha_res = paddle.randn([1], dtype=dtype)
    bias = paddle.randn([_P], dtype=dtype)
    return proj, r, alpha_pre, alpha_post, alpha_res, bias


class TestNativeComputeH(unittest.TestCase):
    """The reference mapping head: Eq. (5) of the mHC paper."""

    def test_matches_written_out_formula(self):
        """Bitwise against the expression the docstring claims to implement."""
        proj, r, a_pre, a_post, a_res, bias = _inputs()
        h_pre, h_post, h_res = native_compute_h(
            proj, r, a_pre, a_post, a_res, bias, _N, _EPS
        )

        alpha = paddle.concat(
            [
                a_pre.expand([_N]),
                a_post.expand([_N]),
                a_res.expand([_N * _N]),
            ],
            axis=-1,
        )
        u = r * proj * alpha + bias
        self.assertTrue(paddle.equal_all(h_pre, u[..., :_N].sigmoid() + _EPS))
        self.assertTrue(
            paddle.equal_all(h_post, u[..., _N : 2 * _N].sigmoid() * 2)
        )
        self.assertTrue(paddle.equal_all(h_res, u[..., 2 * _N :]))

    def test_shapes_and_ranges(self):
        proj, r, a_pre, a_post, a_res, bias = _inputs()
        h_pre, h_post, h_res = native_compute_h(
            proj, r, a_pre, a_post, a_res, bias, _N, _EPS
        )
        self.assertEqual(h_pre.shape, [_S, _B, _N])
        self.assertEqual(h_post.shape, [_S, _B, _N])
        self.assertEqual(h_res.shape, [_S, _B, _N * _N])
        # h_pre is sigmoid + eps, h_post is 2 * sigmoid; h_res is unactivated
        self.assertGreater(float(h_pre.min()), _EPS)
        self.assertLess(float(h_pre.max()), 1.0 + _EPS)
        self.assertGreater(float(h_post.min()), 0.0)
        self.assertLess(float(h_post.max()), 2.0)

    def test_backward_reaches_every_input(self):
        proj, r, a_pre, a_post, a_res, bias = _inputs()
        leaves = [proj, r, a_pre, a_post, a_res, bias]
        for t in leaves:
            t.stop_gradient = False
        h_pre, h_post, h_res = native_compute_h(
            proj, r, a_pre, a_post, a_res, bias, _N, _EPS
        )
        # weight the heads unequally so no gradient path cancels out
        (h_pre.sum() + 2.0 * h_post.sum() + 3.0 * h_res.sum()).backward()
        for name, t in zip(
            ("proj", "r", "alpha_pre", "alpha_post", "alpha_res", "bias"),
            leaves,
        ):
            self.assertIsNotNone(t.grad, name)
            self.assertEqual(t.grad.shape, t.shape, name)
            self.assertTrue(bool(t.grad.abs().sum() > 0), name)

    def test_fma_probe_rounds_once(self):
        """``_fma_probe`` must round ``t1 * alpha + bias`` exactly once.

        This is the discriminator the fused tests use to separate the kernel's
        FMA contraction from a real defect, so the branch needs to do what it
        claims: match an fp64 single-rounding reference bitwise, while the
        default path -- which rounds twice -- generally does not.
        """
        proj, r, a_pre, a_post, a_res, bias = _inputs()
        alpha = paddle.concat(
            [
                a_pre.expand([_N]),
                a_post.expand([_N]),
                a_res.expand([_N * _N]),
            ],
            axis=-1,
        )
        t1 = r * proj
        fused_mul_add = (
            t1.astype("float64") * alpha.astype("float64")
            + bias.astype("float64")
        ).astype(proj.dtype)
        expected = fused_mul_add[..., 2 * _N :]

        _, _, probe_res = native_compute_h(
            proj, r, a_pre, a_post, a_res, bias, _N, _EPS, _fma_probe=True
        )
        self.assertTrue(paddle.equal_all(probe_res, expected))

        _, _, plain_res = native_compute_h(
            proj, r, a_pre, a_post, a_res, bias, _N, _EPS
        )
        # same values to fp32 ULP, but reached with one rounding fewer
        self.assertTrue(bool((plain_res - probe_res).abs().max() < 1e-5 * _P))


class TestOperatorDispatch(unittest.TestCase):
    """Which implementation a module gets, and who widens its operands.

    No kernel runs here -- only the binding is inspected -- so this holds with
    or without cuTile installed.
    """

    def test_native_config_binds_reference_ops(self):
        m = _module(use_fused_mhc=False)
        self.assertIs(m._sinkhorn_op, native_sinkhorn)
        self.assertIs(m._h_aggregate_op, native_h_aggregate)
        self.assertIs(m._h_post_bda_op, native_h_post_bda)
        self.assertIs(m._proj_rms_op, native_proj_rms)
        self.assertIs(m._compute_h_op, native_compute_h)
        # a plain op composition cannot absorb a widening
        self.assertFalse(m._widen_in_kernel)

    def test_fused_config_binds_fused_ops(self):
        from paddlefleet.fusions.fused_mhc_kernels import (
            fused_compute_h,
            fused_h_aggregate,
            fused_h_post_bda,
            fused_proj_rms,
            fused_sinkhorn,
        )

        m = _module(use_fused_mhc=True)
        self.assertIs(m._sinkhorn_op, fused_sinkhorn)
        self.assertIs(m._h_aggregate_op, fused_h_aggregate)
        self.assertIs(m._h_post_bda_op, fused_h_post_bda)
        self.assertIs(m._proj_rms_op, fused_proj_rms)
        self.assertIs(m._compute_h_op, fused_compute_h)

    def test_widen_in_kernel_follows_high_precision_mhc(self):
        """``high_precision_mhc`` is what gives the kernel a widening to absorb.

        Only the enabled case is exercised: ``TransformerConfig`` now rejects
        ``high_precision_mhc=False`` together with ``enable_hyper_connections``,
        so a module with the widening switched off can no longer be built.
        """
        self.assertTrue(
            _module(
                use_fused_mhc=True, high_precision_mhc=True
            )._widen_in_kernel
        )

    def test_accuracy_compatible_mode_keeps_the_reference(self):
        """The Megatron-alignment mode must not reach the fused head.

        ``_projection_and_get_norm``, ``aggregate`` and the BDA site all fall
        back to the reference under this switch. The mapping head has to match:
        the fused head's FMA contraction alone is enough to break the alignment
        contract, whatever the tolerances elsewhere say.
        """
        prev = hc._ACCURACY_COMPATIBLE_KERNEL
        hc._ACCURACY_COMPATIBLE_KERNEL = True
        try:
            self.assertIs(
                _module(use_fused_mhc=True)._compute_h_op, native_compute_h
            )
        finally:
            hc._ACCURACY_COMPATIBLE_KERNEL = prev
        # and the switch is what made the difference, not the config
        self.assertIsNot(
            _module(use_fused_mhc=True)._compute_h_op, native_compute_h
        )


class TestBdaSpanPaysOff(unittest.TestCase):
    """The recompute-span cost/benefit predicate.

    It decides whether ``_fused_h_res_h_post_bda`` is wrapped in a
    ``RecomputeWithoutOutput`` span, so a wrong answer either wastes memory or
    pays for a replay that saves nothing. Only the ``high_precision_mhc=True``
    branches are reachable now that the config rejects the low-precision mHC
    combination.
    """

    def test_high_precision_pays(self):
        m = _module(use_fused_mhc=True, high_precision_mhc=True)
        self.assertTrue(m.bda_span_pays_off(0.0, training=True))

    def test_dropout_pays_too(self):
        """The mask is one byte per element of the [..., n*C] output."""
        m = _module(use_fused_mhc=True, high_precision_mhc=True)
        self.assertTrue(m.bda_span_pays_off(0.1, training=True))


class TestSingleStreamInit(unittest.TestCase):
    """``mhc_single_stream_init`` gates the mapping-head initialization.

    Off is the historical init and must stay what it was: a Xavier-uniform
    projection and a zero bias. On is the paper's: a zero projection, so
    h == bias at step 0, plus the A.6 bias.
    """

    def test_off_keeps_the_historical_init(self):
        m = _module()
        self.assertFalse(bool(paddle.all(m.mapping_proj.weight == 0)))
        self.assertTrue(bool(paddle.all(m.bias == 0)))

    def test_on_zeroes_the_projection(self):
        m = _module(mhc_single_stream_init=True)
        self.assertTrue(bool(paddle.all(m.mapping_proj.weight == 0)))

    def test_on_writes_the_a6_bias(self):
        """b_pre = -3 with +3 on the home stream, b_post = 0, b_res = 6I - 3."""
        for n in _NS:
            with self.subTest(n=n):
                # layer_number=1 -> home stream 1 % n
                m = _module(mhc_single_stream_init=True, num_residual_streams=n)
                b = m.bias.astype("float32").numpy()
                self.assertEqual(b.shape, (n * n + 2 * n,))
                expected_pre = [-3.0] * n
                expected_pre[1 % n] = 3.0
                self.assertEqual(b[:n].tolist(), expected_pre)
                self.assertEqual(b[n : 2 * n].tolist(), [0.0] * n)
                res = b[2 * n :].reshape(n, n)
                for i in range(n):
                    for j in range(n):
                        self.assertEqual(res[i][j], 3.0 if i == j else -3.0)

    def test_home_stream_rotates_with_the_sub_layer_index(self):
        """Consecutive sub-layers read stream 0, 1, ..., n-1, 0, ...

        Constructed without seeding the tracker on purpose: this init is
        deterministic, so unlike the Xavier branch it must not need the fork.

        The whole b_pre segment is compared rather than its argmax: at n = 2 an
        index written out of range leaves b_pre at [-3, -3], whose argmax is 0
        -- the expected answer for every even sub-layer index.
        """
        for n in _NS:
            with self.subTest(n=n):
                cfg = _config(
                    mhc_single_stream_init=True, num_residual_streams=n
                )
                for ln in range(n + 1):
                    b = HyperConnectionModule(cfg, layer_number=ln).bias
                    expected = [-3.0] * n
                    expected[ln % n] = 3.0
                    self.assertEqual(
                        b[:n].astype("float32").numpy().tolist(),
                        expected,
                        f"n={n} layer_number={ln}",
                    )
                    # the other two segments do not rotate
                    self.assertEqual(
                        b[n : 2 * n].astype("float32").numpy().tolist(),
                        [0.0] * n,
                        f"n={n} layer_number={ln}",
                    )

    def test_step_0_mappings_are_a_standard_residual_connection(self):
        """With W = 0 the mappings are the static ones, token-independent.

        This is the whole point of the zero init, and it is a property of the
        mappings, not of the bias: H_res only becomes ~= I after Sinkhorn. That
        last step is n-dependent -- the diagonal is
        e^3 / (e^3 + (n-1) e^-3), i.e. 0.9975 at n = 2 and 0.9926 at n = 4.
        """
        for n in _NS:
            with self.subTest(n=n):
                m = _module(mhc_single_stream_init=True, num_residual_streams=n)
                home = 1 % n  # layer_number=1
                paddle.seed(23)
                x = paddle.randn(
                    [_S, _B, n * _C], dtype=m.mapping_proj.weight.dtype
                )
                h_pre, h_post, h_res = m.compute_mappings(x)

                # token-independent: every position sees the same mapping
                for h in (h_pre, h_post, h_res):
                    self.assertLess(float((h - h[:1, :1]).abs().max()), 1e-6)
                # one-hot-ish read of the home stream, unit write, identity mix
                self.assertAlmostEqual(
                    float(h_pre[0, 0, home]), 0.9526, places=3
                )
                others = [float(h_pre[0, 0, i]) for i in range(n) if i != home]
                self.assertLess(max(others), 0.05)
                self.assertLess(float((h_post - 1.0).abs().max()), 1e-6)
                self.assertGreater(
                    float(paddle.diagonal(h_res[0, 0]).min()), 0.99
                )

    def test_the_zero_projection_still_gets_a_gradient(self):
        """dh/dW = r*alpha*x != 0, so the projection is not pinned at zero.

        A zero weight that also received no gradient would keep the mappings
        static forever, i.e. the "dynamic" half of mHC would never turn on.
        """
        m = _module(mhc_single_stream_init=True)
        paddle.seed(23)
        x = paddle.randn([_S, _B, _N * _C], dtype=m.mapping_proj.weight.dtype)
        h_pre, h_post, h_res = m.compute_mappings(x)
        # weight the heads unequally so no gradient path cancels out
        (h_pre.sum() + 2.0 * h_post.sum() + 3.0 * h_res.sum()).backward()
        g = m.mapping_proj.weight.grad
        self.assertIsNotNone(g)
        self.assertEqual(g.shape, m.mapping_proj.weight.shape)
        # ~7.6e-2 for this fixture, and the same order of magnitude as the
        # Xavier branch gets: a zero weight does not attenuate its own gradient
        self.assertGreater(float(g.abs().max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
