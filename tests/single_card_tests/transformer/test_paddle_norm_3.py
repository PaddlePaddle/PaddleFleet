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

"""Behavior tests for paddlefleet.transformer.paddle_norm.

Numeric expectations are derived from an independent numpy implementation of
each normalization formula, never from the fused paddle ops under test:

  RMSNorm   : x / sqrt(mean(x^2, -1) + eps) * weight
  LayerNorm : (x - mean) / sqrt(var + eps) * weight + bias   (population var)
  L2Norm    : x / sqrt(mean(x^2, -1) + eps)                   (weightless)

These are GPU tests: the fused ``rms_norm`` kernel is registered for the GPU
backend only, so every norm forward here must run on a real CUDA device. When
paddle/paddlefleet cannot be imported the heavy imports are guarded and every
test class is skipped with an honest reason; when no CUDA GPU is available the
device-dependent classes skip in setUp. No import failure is ever swallowed
into a fake pass.
"""

import unittest
from unittest import mock

import numpy as np

try:
    import paddle
    from paddle.distributed.fleet.meta_parallel import LayerSpec

    from paddlefleet.transformer import paddle_norm as paddle_norm_mod
    from paddlefleet.transformer.paddle_norm import (
        L2Norm,
        LayerNorm,
        RMSNorm,
        WrappedPaddleNorm,
        WrappedPaddleNormPipe,
        get_norm_extra_args,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    None
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet unavailable (CPU-only): {_IMPORT_ERROR!r}"
)

# --- Fixed, distinguishable inputs (positions/values all unique) ------------
X_NP = np.array(
    [[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 4.0]], dtype=np.float32
)
W_NP = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
B_NP = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
UP_NP = np.array(
    [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -0.25]], dtype=np.float32
)
# Three uniquely-valued rows so an axis-0 split can be tracked per chunk.
XM_NP = np.array(
    [[2.0, -1.0, 0.5, 3.0], [10.0, 20.0, -30.0, 40.0], [-5.0, 5.0, -5.0, 5.0]],
    dtype=np.float32,
)
RMS_EPS = 1e-5
L2_EPS = 1e-6


def _rms_norm_ref(x, w, eps):
    """Independent RMSNorm reference in float64."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * w


def _layer_norm_ref(x, w, b, eps):
    """Independent LayerNorm reference (population variance) in float64."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * np.asarray(
        w, dtype=np.float64
    ) + np.asarray(b, dtype=np.float64)


def _l2_norm_ref(x, eps):
    """Independent L2Norm reference in float64 (weightless)."""
    x = np.asarray(x, dtype=np.float64)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps)


def _make_config(**overrides):
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "normalization": "RMSNorm",
        "rms_norm_eps": RMS_EPS,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class _GpuNormTestBase(unittest.TestCase):
    def setUp(self):
        # rms_norm has a GPU-only CUDA kernel, so these norm tests must run on
        # a real GPU. Skip honestly when CUDA is unavailable; otherwise select
        # the GPU device and restore the previous device in tearDown.
        self._orig_device = paddle.device.get_device()
        if not paddle.is_compiled_with_cuda():
            self.skipTest("rms_norm requires a CUDA GPU; none available")
        paddle.set_device("gpu")

    def tearDown(self):
        paddle.set_device(self._orig_device)

    def _set_weight(self, layer, values):
        layer.weight.set_value(
            paddle.to_tensor(np.asarray(values, dtype=np.float32))
        )

    def _set_bias(self, layer, values):
        layer.bias.set_value(
            paddle.to_tensor(np.asarray(values, dtype=np.float32))
        )


class TestRMSNormNumeric(_GpuNormTestBase):
    def test_forward_matches_independent_reference_with_weight(self):
        norm = RMSNorm(_make_config(), normalized_shape=4, norm_eps=RMS_EPS)
        self._set_weight(norm, W_NP)  # non-trivial weight must be applied
        out = norm(paddle.to_tensor(X_NP))
        expected = _rms_norm_ref(X_NP, W_NP, RMS_EPS)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_forward_and_backward_against_reference(self):
        norm = RMSNorm(_make_config(), normalized_shape=4, norm_eps=RMS_EPS)
        self._set_weight(norm, W_NP)

        x = paddle.to_tensor(X_NP)
        x.stop_gradient = False
        upstream = paddle.to_tensor(UP_NP)
        out = norm(x)
        out.backward(upstream)

        # Independent reference implemented with elementwise paddle ops, i.e.
        # a different code path than the fused rms_norm op under test.
        x_ref = paddle.to_tensor(X_NP)
        x_ref.stop_gradient = False
        w_ref = paddle.to_tensor(W_NP)
        w_ref.stop_gradient = False
        var = (x_ref * x_ref).mean(axis=-1, keepdim=True)
        ref = x_ref * paddle.rsqrt(var + RMS_EPS) * w_ref
        ref.backward(paddle.to_tensor(UP_NP))

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(norm.weight.grad)
        self.assertIsNotNone(x_ref.grad)
        self.assertIsNotNone(w_ref.grad)
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            x.grad.numpy(), x_ref.grad.numpy(), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            norm.weight.grad.numpy(), w_ref.grad.numpy(), rtol=1e-4, atol=1e-5
        )
        # Guard against a degenerate all-zero gradient masking the comparison.
        self.assertGreater(np.abs(w_ref.grad.numpy()).max(), 1e-3)


class TestLayerNormNumeric(_GpuNormTestBase):
    def test_forward_matches_independent_reference_with_weight_and_bias(self):
        norm = LayerNorm(_make_config(), normalized_shape=4, norm_eps=RMS_EPS)
        self._set_weight(norm, W_NP)  # non-trivial affine must be applied
        self._set_bias(norm, B_NP)
        out = norm(paddle.to_tensor(X_NP))
        expected = _layer_norm_ref(X_NP, W_NP, B_NP, RMS_EPS)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)


class TestL2NormNumeric(_GpuNormTestBase):
    def test_forward_matches_independent_reference(self):
        norm = L2Norm(hidden_size=4)  # default eps == 1e-6
        out = norm(paddle.to_tensor(X_NP))
        expected = _l2_norm_ref(X_NP, L2_EPS)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_zero_input_returns_exact_zeros(self):
        norm = L2Norm(hidden_size=4)
        out = norm(paddle.zeros([2, 4], dtype="float32"))
        np.testing.assert_array_equal(
            out.numpy(), np.zeros([2, 4], dtype=np.float32)
        )

    def test_eps_is_consumed_for_small_magnitude_input(self):
        # With tiny values the eps term dominates the denominator, so the
        # output must track the reference computed with that specific eps.
        small = np.array([[1e-4, -2e-4, 3e-4, -1e-4]], dtype=np.float32)
        norm = L2Norm(hidden_size=4, eps=1e-2)
        out = norm(paddle.to_tensor(small))
        expected = _l2_norm_ref(small, 1e-2)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-7)
        # A wrong (default) eps would give a materially different result.
        wrong = _l2_norm_ref(small, 1e-6)
        self.assertFalse(np.allclose(out.numpy(), wrong, rtol=1e-3, atol=1e-6))


class TestWrappedPaddleNormSelection(_GpuNormTestBase):
    def test_selects_rmsnorm_and_normalizes(self):
        norm = WrappedPaddleNorm(config=_make_config(), hidden_size=4)
        self.assertIsInstance(norm, RMSNorm)
        self._set_weight(norm, W_NP)
        out = norm(paddle.to_tensor(X_NP))
        np.testing.assert_allclose(
            out.numpy(),
            _rms_norm_ref(X_NP, W_NP, RMS_EPS),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_selects_layernorm_and_normalizes(self):
        norm = WrappedPaddleNorm(
            config=_make_config(normalization="LayerNorm"), hidden_size=4
        )
        self.assertIsInstance(norm, LayerNorm)
        self._set_weight(norm, W_NP)
        self._set_bias(norm, B_NP)
        out = norm(paddle.to_tensor(X_NP))
        np.testing.assert_allclose(
            out.numpy(),
            _layer_norm_ref(X_NP, W_NP, B_NP, RMS_EPS),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_unsupported_normalization_raises_with_message(self):
        with self.assertRaisesRegex(Exception, "Only RMSNorm"):
            WrappedPaddleNorm(
                config=_make_config(normalization="InvalidNorm"), hidden_size=4
            )

    def test_input_is_parallel_true_marks_weight(self):
        with mock.patch.object(
            paddle_norm_mod, "mark_as_sequence_parallel_parameter"
        ) as marker:
            norm = WrappedPaddleNorm(
                config=_make_config(), hidden_size=4, input_is_parallel=True
            )
        marker.assert_called_once()
        (arg,), _ = marker.call_args
        self.assertIs(arg, norm.weight)

    def test_input_is_parallel_false_does_not_mark(self):
        with mock.patch.object(
            paddle_norm_mod, "mark_as_sequence_parallel_parameter"
        ) as marker:
            WrappedPaddleNorm(
                config=_make_config(), hidden_size=4, input_is_parallel=False
            )
        marker.assert_not_called()

    def test_input_is_parallel_default_resolves_false_on_single_rank(self):
        # tp==1 and sequence_parallel forced False => default is False.
        with mock.patch.object(
            paddle_norm_mod, "mark_as_sequence_parallel_parameter"
        ) as marker:
            WrappedPaddleNorm(config=_make_config(), hidden_size=4)
        marker.assert_not_called()


class TestWrappedPaddleNormPipe(_GpuNormTestBase):
    def test_no_mtp_normalizes_and_preserves_other_keys(self):
        pipe = WrappedPaddleNormPipe(config=_make_config(), hidden_size=4)
        self._set_weight(pipe.norm, W_NP)
        mask = paddle.to_tensor(
            np.arange(16, dtype=np.float32).reshape([1, 1, 4, 4])
        )
        out = pipe(
            {"hidden_states": paddle.to_tensor(X_NP), "attention_mask": mask}
        )
        np.testing.assert_allclose(
            out["hidden_states"].numpy(),
            _rms_norm_ref(X_NP, W_NP, RMS_EPS),
            rtol=1e-5,
            atol=1e-6,
        )
        # Sibling keys are passed through unchanged, by value not just presence.
        self.assertIn("attention_mask", out)
        np.testing.assert_array_equal(
            out["attention_mask"].numpy(), mask.numpy()
        )

    def test_mtp_splits_axis0_normalizes_first_chunk_only(self):
        # num_nextn=2 => split into 3 axis-0 chunks; only chunk 0 is normalized,
        # chunks 1 and 2 are forwarded unchanged (Megatron final_layernorm MTP
        # contract), then concatenated back in order.
        config = _make_config(
            num_nextn_predict_layers=2, mtp_load_weight_only=False
        )
        pipe = WrappedPaddleNormPipe(config=config, hidden_size=4)
        self._set_weight(pipe.norm, W_NP)
        out = pipe({"hidden_states": paddle.to_tensor(XM_NP)})["hidden_states"]

        expected_row0 = _rms_norm_ref(XM_NP[0:1], W_NP, RMS_EPS)
        np.testing.assert_allclose(
            out.numpy()[0:1], expected_row0, rtol=1e-5, atol=1e-6
        )
        # Rows 1 and 2 are untouched passthrough (weight is intentionally != 1
        # to prove no normalization/scaling was applied to them).
        np.testing.assert_array_equal(out.numpy()[1:2], XM_NP[1:2])
        np.testing.assert_array_equal(out.numpy()[2:3], XM_NP[2:3])

    def test_mtp_load_weight_only_normalizes_whole_tensor(self):
        # mtp_load_weight_only=True disables the split: every row is normalized.
        config = _make_config(
            num_nextn_predict_layers=2, mtp_load_weight_only=True
        )
        pipe = WrappedPaddleNormPipe(config=config, hidden_size=4)
        self._set_weight(pipe.norm, W_NP)
        out = pipe({"hidden_states": paddle.to_tensor(XM_NP)})["hidden_states"]

        expected = _rms_norm_ref(XM_NP, W_NP, RMS_EPS)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # Contrast with the split path: here rows 1 and 2 ARE changed.
        self.assertFalse(np.allclose(out.numpy()[1:2], XM_NP[1:2]))
        self.assertFalse(np.allclose(out.numpy()[2:3], XM_NP[2:3]))


class TestGetNormExtraArgs(_GpuNormTestBase):
    def test_wrapped_paddle_norm_via_layerspec_uses_hidden_size_eps(self):
        config = _make_config()
        args = get_norm_extra_args(
            LayerSpec(WrappedPaddleNorm), config, 128, 1e-5, False
        )
        self.assertEqual(
            args,
            {
                "config": config,
                "input_is_parallel": False,
                "hidden_size": 128,
                "eps": 1e-5,
            },
        )

    def test_wrapped_paddle_norm_direct_uses_hidden_size_eps(self):
        config = _make_config()
        args = get_norm_extra_args(WrappedPaddleNorm, config, 256, 2e-5, True)
        self.assertEqual(
            args,
            {
                "config": config,
                "input_is_parallel": True,
                "hidden_size": 256,
                "eps": 2e-5,
            },
        )

    def test_other_norm_uses_normalized_shape_norm_eps(self):
        config = _make_config()
        args = get_norm_extra_args(RMSNorm, config, 64, 1e-6, True)
        self.assertEqual(
            args,
            {
                "config": config,
                "input_is_parallel": True,
                "normalized_shape": 64,
                "norm_eps": 1e-6,
            },
        )


if __name__ == "__main__":
    unittest.main()
