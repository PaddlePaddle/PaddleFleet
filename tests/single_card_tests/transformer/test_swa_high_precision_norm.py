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

"""Behavior tests for the swa_high_precision_norm feature.

Three production surfaces are exercised against independently derived
references (plain numpy RMS math, not the code under test):

- ``dsv4_hybrid_attention._q_rms_norm`` -- weightless RMS norm of the query,
  with a float32-compute + cast-back branch when ``high_precision_norm=True``.
- ``paddle_norm.RMSNorm.forward`` -- weighted RMS norm whose
  ``high_precision_norm`` / ``return_high_precision_norm`` flags control the
  compute precision and the returned dtype.
- ``transformer_config.TransformerConfig`` -- the ``swa_high_precision_norm``
  gate that is only legal together with ``experimental_attention_variant=
  'dsv4_hybrid'``.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.dsv4_hybrid_attention import _q_rms_norm
    from paddlefleet.transformer.paddle_norm import RMSNorm
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    _q_rms_norm = None
    RMSNorm = None
    TransformerConfig = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    None
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet not importable in this environment: {_IMPORT_ERROR!r}"
)


def _ref_rms(x_f64, eps, weight_f64=None):
    """Independent RMS-norm reference computed entirely in numpy float64.

    ``out = x / sqrt(mean(x**2, last_dim) + eps)`` optionally scaled by a
    per-channel weight. This mirrors the mathematical contract only; it never
    calls the code under test.
    """
    denom = np.sqrt(np.mean(x_f64**2, axis=-1, keepdims=True) + eps)
    out = x_f64 / denom
    if weight_f64 is not None:
        out = out * weight_f64
    return out


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "")
class TestQRmsNorm(unittest.TestCase):
    """dsv4_hybrid_attention._q_rms_norm: weightless RMS norm of the query."""

    def test_float32_high_precision_matches_reference(self):
        # eps is deliberately large relative to the unit-scale input so that
        # an implementation that dropped it would diverge by ~20%.
        eps = 0.5
        q_np = np.array(
            [[[[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, -2.0, 4.0]]]],
            dtype=np.float32,
        )
        q = paddle.to_tensor(q_np)
        out = _q_rms_norm(q, eps=eps, high_precision_norm=True)

        ref = _ref_rms(q_np.astype(np.float64), eps)
        self.assertEqual(out.dtype, paddle.float32)
        self.assertEqual(list(out.shape), [1, 1, 2, 4])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-6)

    def test_float32_low_precision_matches_same_reference(self):
        # With float32 input, .float() is a no-op so both branches must agree
        # with the same independent reference (covers the else branch).
        eps = 0.5
        q_np = np.array(
            [[[[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, -2.0, 4.0]]]],
            dtype=np.float32,
        )
        q = paddle.to_tensor(q_np)
        out = _q_rms_norm(q, eps=eps, high_precision_norm=False)

        ref = _ref_rms(q_np.astype(np.float64), eps)
        self.assertEqual(out.dtype, paddle.float32)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-6)

    def test_bfloat16_high_precision_computes_in_float32_and_casts_back(self):
        # The contract: cast up to float32, normalize, cast back to the input
        # dtype. Build the reference from the *bf16-rounded* input values (what
        # the float() upcast actually sees) so the check reflects the real
        # compute path, not the pre-rounding float32 values.
        eps = 1e-5
        rng = np.random.RandomState(0)
        q_np = rng.standard_normal([2, 3, 4, 8]).astype(np.float32)
        q_bf16 = paddle.to_tensor(q_np).astype(paddle.bfloat16)

        out = _q_rms_norm(q_bf16, eps=eps, high_precision_norm=True)
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertEqual(list(out.shape), [2, 3, 4, 8])

        q_used = q_bf16.astype(paddle.float32).numpy().astype(np.float64)
        ref = _ref_rms(q_used, eps)
        # Output is bf16 (~8 mantissa bits): compare with a bf16-scale tolerance.
        np.testing.assert_allclose(
            out.astype(paddle.float32).numpy(), ref, rtol=8e-3, atol=4e-3
        )


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "")
class TestRMSNormForward(unittest.TestCase):
    """paddle_norm.RMSNorm.forward: weighted RMS norm with precision flags."""

    HIDDEN = 16

    def _make_config(self, params_dtype):
        return TransformerConfig(
            num_hidden_layers=1,
            hidden_size=self.HIDDEN,
            num_attention_heads=4,
            normalization="RMSNorm",
            params_dtype=params_dtype,
        )

    def test_high_precision_float32_applies_weight_and_matches_reference(self):
        eps = 0.5
        rng = np.random.RandomState(1)
        x_np = rng.standard_normal([2, 4, self.HIDDEN]).astype(np.float32)
        # Distinguishable per-channel weight (not the all-ones default) so a
        # dropped/broadcast-wrong weight would be caught.
        w_np = (rng.standard_normal([self.HIDDEN]) * 0.5 + 1.0).astype(
            np.float32
        )

        norm = RMSNorm(self._make_config(paddle.float32), norm_eps=eps)
        norm.weight.set_value(paddle.to_tensor(w_np))

        out = norm(paddle.to_tensor(x_np), high_precision_norm=True)
        ref = _ref_rms(x_np.astype(np.float64), eps, w_np.astype(np.float64))
        self.assertEqual(out.dtype, paddle.float32)
        self.assertEqual(list(out.shape), [2, 4, self.HIDDEN])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)

    def test_low_precision_float32_matches_same_reference(self):
        # else branch (high_precision_norm=False) with matching dtypes: same
        # weighted RMS math, same independent reference.
        eps = 0.5
        rng = np.random.RandomState(2)
        x_np = rng.standard_normal([2, 4, self.HIDDEN]).astype(np.float32)
        w_np = (rng.standard_normal([self.HIDDEN]) * 0.5 + 1.0).astype(
            np.float32
        )

        norm = RMSNorm(self._make_config(paddle.float32), norm_eps=eps)
        norm.weight.set_value(paddle.to_tensor(w_np))

        out = norm(paddle.to_tensor(x_np), high_precision_norm=False)
        ref = _ref_rms(x_np.astype(np.float64), eps, w_np.astype(np.float64))
        self.assertEqual(out.dtype, paddle.float32)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)

    def test_return_high_precision_keeps_float32_from_bf16_params(self):
        # return_high_precision_norm overrides the return dtype to float32 even
        # though the weight is bfloat16; value must still match the reference
        # computed from the bf16-rounded input and weight.
        eps = 1e-3
        rng = np.random.RandomState(3)
        x_np = rng.standard_normal([2, 4, self.HIDDEN]).astype(np.float32)
        w_np = (rng.standard_normal([self.HIDDEN]) * 0.5 + 1.0).astype(
            np.float32
        )

        norm = RMSNorm(self._make_config(paddle.bfloat16), norm_eps=eps)
        norm.weight.set_value(paddle.to_tensor(w_np).astype(paddle.bfloat16))

        x_bf16 = paddle.to_tensor(x_np).astype(paddle.bfloat16)
        out = norm(
            x_bf16, high_precision_norm=True, return_high_precision_norm=True
        )
        self.assertEqual(out.dtype, paddle.float32)

        x_used = x_bf16.astype(paddle.float32).numpy().astype(np.float64)
        w_used = norm.weight.astype(paddle.float32).numpy().astype(np.float64)
        ref = _ref_rms(x_used, eps, w_used)
        # Inputs are exact bf16 values; the compute + returned value are
        # float32, so a modest tolerance isolates only kernel rounding.
        np.testing.assert_allclose(out.numpy(), ref, rtol=3e-3, atol=2e-3)

    def test_default_return_dtype_follows_weight_dtype(self):
        # Without return_high_precision_norm, the returned dtype is the weight
        # dtype (bfloat16 here), and the value matches the reference rounded to
        # bf16.
        eps = 1e-3
        rng = np.random.RandomState(4)
        x_np = rng.standard_normal([2, 4, self.HIDDEN]).astype(np.float32)
        w_np = (rng.standard_normal([self.HIDDEN]) * 0.5 + 1.0).astype(
            np.float32
        )

        norm = RMSNorm(self._make_config(paddle.bfloat16), norm_eps=eps)
        norm.weight.set_value(paddle.to_tensor(w_np).astype(paddle.bfloat16))

        x_bf16 = paddle.to_tensor(x_np).astype(paddle.bfloat16)
        out = norm(
            x_bf16, high_precision_norm=True, return_high_precision_norm=False
        )
        self.assertEqual(out.dtype, paddle.bfloat16)

        x_used = x_bf16.astype(paddle.float32).numpy().astype(np.float64)
        w_used = norm.weight.astype(paddle.float32).numpy().astype(np.float64)
        ref = _ref_rms(x_used, eps, w_used)
        np.testing.assert_allclose(
            out.astype(paddle.float32).numpy(), ref, rtol=8e-3, atol=4e-3
        )


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "")
class TestSwaHighPrecisionNormConfigGate(unittest.TestCase):
    """TransformerConfig.__post_init__: swa_high_precision_norm gating."""

    def test_true_without_dsv4_hybrid_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            "swa_high_precision_norm=True is only supported when",
        ):
            TransformerConfig(
                hidden_size=128,
                num_attention_heads=4,
                swa_high_precision_norm=True,
                experimental_attention_variant=None,
            )

    def test_false_without_dsv4_hybrid_is_allowed(self):
        # Same shape as the raising case, flag flipped off: proves the guard is
        # conditional on the flag, not on the variant alone.
        config = TransformerConfig(
            hidden_size=128,
            num_attention_heads=4,
            swa_high_precision_norm=False,
            experimental_attention_variant=None,
        )
        self.assertFalse(config.swa_high_precision_norm)

    def test_true_with_dsv4_hybrid_is_exempted(self):
        # Same flag value that raised above; only the variant (and its required
        # csa_compress_ratios) changed, so the exemption branch is what lets it
        # through. A window-only ratio list keeps the config minimal: no
        # indexer, no MLA, no extra dsv4 requirements fire.
        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            swa_high_precision_norm=True,
            experimental_attention_variant="dsv4_hybrid",
            csa_compress_ratios=[0],
        )
        self.assertTrue(config.swa_high_precision_norm)
        self.assertEqual(config.experimental_attention_variant, "dsv4_hybrid")


if __name__ == "__main__":
    unittest.main()
