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
"""Behavior tests for the latent-MoE projection logic in
``paddlefleet.transformer.moe.moe_layer.MoELayer``.

Two real, CPU-executable production methods are driven directly:

  * ``MoELayer._project_to_latent`` -- the three-way branch that (a) returns the
    input untouched when latent MoE is off, (b) consumes and clears a cached
    AllGather-overlap projection when present, and (c) otherwise projects the
    hidden states down to the latent width via ``fc1_latent_proj``.
  * ``MoELayer.aux_loss_compute`` -- the tail that, under latent MoE, optionally
    applies ``latent_norm`` and then expands the routed output back to
    ``hidden_size`` via ``fc2_latent_proj`` before reshaping to the residual
    layout.

Expected values are derived independently: the ``nn.Linear`` projection is
recomputed with the plain ``x @ weight + bias`` formula (never by calling the
method under test), the cache-consume branch is checked against the exact cached
tensor (which is deliberately different from what ``fc1`` would produce), and the
norm-before-projection ordering is pinned with a deterministic norm stand-in and
a non-zero projection bias so that swapping the two operations is observable.

Only the projection control flow is exercised here; token dispatch, expert
compute and cross-rank collectives are separate concerns validated elsewhere.
The math runs on CPU and needs only ``paddle``/``numpy``; when those (or the
paddlefleet package) cannot be imported the module is skipped with an honest
reason rather than reported as passing.
"""

import types
import unittest

try:
    import numpy as np
    import paddle
    from paddle import nn

    from paddlefleet.transformer.moe.moe_layer import MoELayer

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without deps
    np = None
    paddle = None
    nn = None
    MoELayer = None
    _IMPORT_ERROR = exc


class _DeferralOffConfig:
    """Passive config carrying only what ``deferrable_linear_bare`` reads.

    With ``p2p_overlap_dw_calc`` unset and no interleaved-PP scheduler, the
    real ``_can_defer`` returns ``False`` so the projection layer is called
    inline. This is a configuration holder, not a mock of the code under test:
    the ``_project_to_latent`` / ``aux_loss_compute`` bodies run unchanged.
    """

    p2p_overlap_dw_calc = None
    tensor_model_parallel_size = 1
    pipeline_model_parallel_size = 1
    virtual_pipeline_model_parallel_size = None
    use_bias = False


def _linear_ref(x, layer):
    """Independent reference for ``paddle.nn.Linear``: ``y = x @ W + b``.

    Recomputed from the layer's own parameters with the documented affine
    formula, without invoking the production method under test.
    """
    y = paddle.matmul(x, layer.weight)
    if layer.bias is not None:
        y = y + layer.bias
    return y


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR!r}",
)
class TestProjectToLatent(unittest.TestCase):
    """Drive the real ``MoELayer._project_to_latent`` branch selection."""

    def setUp(self):
        paddle.set_device("cpu")
        paddle.seed(2026)
        self.hidden_size = 8
        self.latent_size = 4
        self.n_tokens = 6

    def _hidden(self):
        # Deterministic, position-distinguishable input.
        return paddle.arange(
            self.n_tokens * self.hidden_size, dtype="float32"
        ).reshape([self.n_tokens, self.hidden_size])

    def test_disabled_returns_input_untouched(self):
        stub = types.SimpleNamespace(use_latent_moe=False)
        hidden = self._hidden()

        out = MoELayer._project_to_latent(stub, hidden)

        # Latent MoE off: the exact same object flows through unmodified.
        self.assertIs(out, hidden)
        np.testing.assert_array_equal(out.numpy(), self._hidden().numpy())

    def test_projects_via_fc1_when_no_cache(self):
        fc1 = nn.Linear(self.hidden_size, self.latent_size)
        stub = types.SimpleNamespace(
            use_latent_moe=True,
            _latent_hidden=None,
            config=_DeferralOffConfig(),
            fc1_latent_proj=fc1,
        )
        hidden = self._hidden()
        expected = _linear_ref(hidden, fc1)

        out = MoELayer._project_to_latent(stub, hidden)

        # Compressed to the latent width via the real fc1 projection.
        self.assertEqual(out.shape, [self.n_tokens, self.latent_size])
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6
        )
        # No cached projection existed, so none is left dangling.
        self.assertIsNone(stub._latent_hidden)

    def test_consumes_and_clears_cached_projection(self):
        fc1 = nn.Linear(self.hidden_size, self.latent_size)
        # A cache with values fc1 could not have produced, so "used the cache"
        # and "re-projected" are distinguishable.
        cached = paddle.full(
            [self.n_tokens, self.latent_size], 7.0, dtype="float32"
        )
        stub = types.SimpleNamespace(
            use_latent_moe=True,
            _latent_hidden=cached,
            config=_DeferralOffConfig(),
            fc1_latent_proj=fc1,
        )
        hidden = self._hidden()

        out = MoELayer._project_to_latent(stub, hidden)

        # The cached AllGather-overlap projection is returned verbatim ...
        self.assertIs(out, cached)
        np.testing.assert_array_equal(
            out.numpy(), np.full([self.n_tokens, self.latent_size], 7.0)
        )
        # ... it is NOT re-derived from fc1(hidden) ...
        fresh = _linear_ref(hidden, fc1).numpy()
        self.assertFalse(np.allclose(out.numpy(), fresh))
        # ... and the cache slot is cleared so it cannot leak into a later step.
        self.assertIsNone(stub._latent_hidden)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR!r}",
)
class TestAuxLossComputeLatentProjection(unittest.TestCase):
    """Drive the real ``MoELayer.aux_loss_compute`` latent tail on CPU.

    ``training`` is ``False`` and ``shared_experts`` is ``None`` so the
    ``AddAuxiliaryLoss`` / shared-expert / ``ScatterOp`` branches are inert and
    the observed output is exactly the latent output-projection path.
    """

    def setUp(self):
        paddle.set_device("cpu")
        paddle.seed(2026)
        self.hidden_size = 8
        self.latent_size = 4
        self.n_tokens = 6

    def _base_stub(self, **overrides):
        stub = types.SimpleNamespace(
            use_latent_moe=True,
            latent_norm=None,
            config=_DeferralOffConfig(),
            training=False,
            router_aux_loss_coef=0.0,
            shared_experts=None,
            expert_model_parallel_size=1,
            sequence_parallel=False,
        )
        for key, value in overrides.items():
            setattr(stub, key, value)
        return stub

    def _latent_hidden(self):
        return paddle.arange(
            self.n_tokens * self.latent_size, dtype="float32"
        ).reshape([self.n_tokens, self.latent_size])

    def test_latent_output_expands_and_reshapes_to_residual(self):
        fc2 = nn.Linear(self.latent_size, self.hidden_size)
        stub = self._base_stub(fc2_latent_proj=fc2)

        hidden = self._latent_hidden()  # [6, 4] in latent space
        residuals = paddle.arange(
            2 * 3 * self.hidden_size, dtype="float32"
        ).reshape([2, 3, self.hidden_size])
        expected = _linear_ref(hidden, fc2).reshape([2, 3, self.hidden_size])

        out = MoELayer.aux_loss_compute(
            stub, (hidden, paddle.zeros([1]), None, residuals)
        )

        # Restored to hidden_size and folded back into the residual layout.
        self.assertEqual(out.shape, [2, 3, self.hidden_size])
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_latent_norm_is_applied_before_fc2(self):
        fc2 = nn.Linear(self.latent_size, self.hidden_size)
        # Non-zero projection bias makes norm-then-fc2 differ from fc2-then-norm.
        fc2.bias.set_value(
            paddle.full([self.hidden_size], 0.5, dtype="float32")
        )
        # Deterministic norm collaborator (a stand-in for the RMSNorm module,
        # not for aux_loss_compute itself): scales its input by 3.
        stub = self._base_stub(
            fc2_latent_proj=fc2, latent_norm=lambda h: h * 3.0
        )

        hidden = self._latent_hidden()
        residuals = paddle.zeros(
            [self.n_tokens, self.hidden_size], dtype="float32"
        )

        out = MoELayer.aux_loss_compute(
            stub, (hidden, paddle.zeros([1]), None, residuals)
        )

        expected_norm_first = _linear_ref(hidden * 3.0, fc2)
        np.testing.assert_allclose(
            out.numpy(), expected_norm_first.numpy(), rtol=1e-6, atol=1e-6
        )
        # If fc2 ran before the norm, the bias would be scaled too; confirm the
        # observed output does not match that (mis)ordering.
        wrong_order = _linear_ref(hidden, fc2) * 3.0
        self.assertFalse(np.allclose(out.numpy(), wrong_order.numpy()))

    def test_disabled_passes_output_through_and_reshapes(self):
        stub = self._base_stub(use_latent_moe=False)

        hidden = paddle.arange(
            self.n_tokens * self.hidden_size, dtype="float32"
        ).reshape([self.n_tokens, self.hidden_size])
        residuals = paddle.zeros([2, 3, self.hidden_size], dtype="float32")

        out = MoELayer.aux_loss_compute(
            stub, (hidden, paddle.zeros([1]), None, residuals)
        )

        # No projection when latent MoE is off: values preserved, only reshaped
        # into the residual layout.
        self.assertEqual(out.shape, [2, 3, self.hidden_size])
        np.testing.assert_array_equal(
            out.numpy(), hidden.reshape([2, 3, self.hidden_size]).numpy()
        )


if __name__ == "__main__":
    unittest.main()
