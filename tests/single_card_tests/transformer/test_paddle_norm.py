# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from unittest import mock

import numpy as np

# Heavy backend imports are optional: on a CPU box without the paddle wheel the
# whole module is honestly skipped rather than faked. Only ImportError /
# ModuleNotFoundError are treated as "dependency missing"; any other error must
# surface as a real failure.
try:
    import paddle
    from paddle.distributed.fleet.meta_parallel import LayerSpec, ScheduleNode

    from paddlefleet.transformer.paddle_norm import (
        FusedRMSNorm,
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

_SKIP_REASON = f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR}"
_PADDLE_AVAILABLE = _IMPORT_ERROR is None

_MARK_TARGET = (
    "paddlefleet.transformer.paddle_norm.mark_as_sequence_parallel_parameter"
)


def _make_config(**overrides):
    """Build a minimal, CPU-friendly TransformerConfig for norm layers."""
    kwargs = {
        "hidden_size": 8,
        "num_attention_heads": 4,
        "normalization": "RMSNorm",
        "rms_norm_eps": 1e-5,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


# --- Independent numpy references (do NOT call the production layers) --------


def _rmsnorm_ref(x, weight, eps):
    """RMSNorm definition: x / sqrt(mean(x^2) + eps) * weight, in float64."""
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(mean_sq + eps) * weight


def _layernorm_ref(x, weight, bias, eps):
    """LayerNorm definition with biased variance, in float64."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # biased (divide by N)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def _nonuniform(shape, start=0.5, step=0.37):
    """Fixed, position-distinguishable tensor so swaps/misorders are visible."""
    n = int(np.prod(shape))
    vals = start + step * np.arange(n, dtype=np.float64)
    # Introduce sign variety so a dropped sign is observable.
    vals[::3] *= -1.0
    return vals.reshape(shape)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestRMSNorm(unittest.TestCase):
    """RMSNorm forward/backward numerics and sequence-parallel wiring."""

    def test_forward_matches_reference_and_consumes_weight(self):
        # A non-trivial weight makes "weight was ignored" observable: with the
        # default all-ones weight (as in the coverage test) that bug hides.
        config = _make_config()
        norm = RMSNorm(config)
        weight_np = _nonuniform([config.hidden_size], start=0.2, step=0.11)
        norm.weight.set_value(paddle.to_tensor(weight_np, dtype="float32"))

        x_np = _nonuniform([2, 3, config.hidden_size])
        out = norm(paddle.to_tensor(x_np, dtype="float32"))

        ref = _rmsnorm_ref(x_np, weight_np, config.rms_norm_eps)
        self.assertEqual(out.shape, [2, 3, config.hidden_size])
        self.assertEqual(out.dtype, paddle.float32)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)

    def test_eps_is_consumed(self):
        # Same input/weight, two very different eps must yield different output;
        # each must match its own independent reference.
        config = _make_config()
        x_np = _nonuniform([2, config.hidden_size])
        weight_np = np.ones([config.hidden_size], dtype=np.float64)

        small = RMSNorm(config, norm_eps=1e-5)
        big = RMSNorm(config, norm_eps=0.5)
        out_small = small(paddle.to_tensor(x_np, dtype="float32")).numpy()
        out_big = big(paddle.to_tensor(x_np, dtype="float32")).numpy()

        np.testing.assert_allclose(
            out_small, _rmsnorm_ref(x_np, weight_np, 1e-5), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            out_big, _rmsnorm_ref(x_np, weight_np, 0.5), rtol=1e-4, atol=1e-5
        )
        self.assertGreater(np.abs(out_small - out_big).max(), 1e-2)

    def test_forward_backward_matches_independent_autograd(self):
        # Drive a real backward and compare dx, dw against an independent paddle
        # autograd graph built from the RMSNorm math (not the production layer).
        config = _make_config()
        x_np = _nonuniform([2, 3, config.hidden_size])
        weight_np = _nonuniform([config.hidden_size], start=0.3, step=0.09)
        upstream_np = _nonuniform(
            [2, 3, config.hidden_size], start=1.0, step=0.05
        )
        eps = config.rms_norm_eps

        norm = RMSNorm(config)
        norm.weight.set_value(paddle.to_tensor(weight_np, dtype="float32"))
        xp = paddle.to_tensor(x_np, dtype="float32")
        xp.stop_gradient = False
        out = norm(xp)
        out.backward(paddle.to_tensor(upstream_np, dtype="float32"))

        x_ref = paddle.to_tensor(x_np, dtype="float32")
        x_ref.stop_gradient = False
        w_ref = paddle.to_tensor(weight_np, dtype="float32")
        w_ref.stop_gradient = False
        mean_sq = (x_ref * x_ref).mean(axis=-1, keepdim=True)
        ref = x_ref * paddle.rsqrt(mean_sq + eps) * w_ref
        ref.backward(paddle.to_tensor(upstream_np, dtype="float32"))

        self.assertIsNotNone(xp.grad)
        self.assertIsNotNone(norm.weight.grad)
        np.testing.assert_allclose(
            out.numpy(), ref.numpy(), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            xp.grad.numpy(), x_ref.grad.numpy(), rtol=1e-3, atol=1e-4
        )
        np.testing.assert_allclose(
            norm.weight.grad.numpy(), w_ref.grad.numpy(), rtol=1e-3, atol=1e-4
        )
        # Gradients must be genuinely non-trivial for the check to have teeth.
        self.assertGreater(np.abs(w_ref.grad.numpy()).max(), 1e-3)

    def test_custom_normalized_shape_and_eps_recorded(self):
        config = _make_config()
        norm = RMSNorm(config, normalized_shape=6, norm_eps=1e-6)
        self.assertEqual(norm.normalized_shape, 6)
        self.assertAlmostEqual(norm.variance_epsilon, 1e-6)
        self.assertEqual(list(norm.weight.shape), [6])

    def test_enable_sequence_parallel_marks_weight(self):
        config = _make_config()
        with mock.patch(_MARK_TARGET) as marker:
            norm = RMSNorm(config, input_is_parallel=False)
            marker.assert_not_called()  # False path must not mark
            norm.enable_sequence_parallel()
            marker.assert_called_once_with(norm.weight)

    def test_input_is_parallel_true_marks_on_construction(self):
        config = _make_config()
        with mock.patch(_MARK_TARGET) as marker:
            norm = RMSNorm(config, input_is_parallel=True)
            marker.assert_called_once_with(norm.weight)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestLayerNorm(unittest.TestCase):
    """LayerNorm forward/backward numerics and default init."""

    def test_default_weight_ones_bias_zeros(self):
        config = _make_config(normalization="LayerNorm")
        norm = LayerNorm(config)
        np.testing.assert_array_equal(
            norm.weight.numpy(), np.ones([config.hidden_size], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            norm.bias.numpy(), np.zeros([config.hidden_size], dtype=np.float32)
        )

    def test_forward_matches_reference_with_weight_and_bias(self):
        config = _make_config(normalization="LayerNorm")
        norm = LayerNorm(config)
        weight_np = _nonuniform([config.hidden_size], start=0.4, step=0.07)
        bias_np = _nonuniform([config.hidden_size], start=-0.3, step=0.05)
        norm.weight.set_value(paddle.to_tensor(weight_np, dtype="float32"))
        norm.bias.set_value(paddle.to_tensor(bias_np, dtype="float32"))

        x_np = _nonuniform([2, 3, config.hidden_size])
        out = norm(paddle.to_tensor(x_np, dtype="float32"))

        ref = _layernorm_ref(x_np, weight_np, bias_np, config.rms_norm_eps)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)

    def test_forward_backward_matches_independent_autograd(self):
        config = _make_config(normalization="LayerNorm")
        x_np = _nonuniform([2, 3, config.hidden_size])
        weight_np = _nonuniform([config.hidden_size], start=0.5, step=0.06)
        bias_np = _nonuniform([config.hidden_size], start=0.1, step=0.03)
        upstream_np = _nonuniform(
            [2, 3, config.hidden_size], start=1.0, step=0.04
        )
        eps = config.rms_norm_eps

        norm = LayerNorm(config)
        norm.weight.set_value(paddle.to_tensor(weight_np, dtype="float32"))
        norm.bias.set_value(paddle.to_tensor(bias_np, dtype="float32"))
        xp = paddle.to_tensor(x_np, dtype="float32")
        xp.stop_gradient = False
        out = norm(xp)
        out.backward(paddle.to_tensor(upstream_np, dtype="float32"))

        x_ref = paddle.to_tensor(x_np, dtype="float32")
        x_ref.stop_gradient = False
        w_ref = paddle.to_tensor(weight_np, dtype="float32")
        w_ref.stop_gradient = False
        b_ref = paddle.to_tensor(bias_np, dtype="float32")
        b_ref.stop_gradient = False
        mu = x_ref.mean(axis=-1, keepdim=True)
        var = (x_ref - mu).pow(2).mean(axis=-1, keepdim=True)
        ref = (x_ref - mu) * paddle.rsqrt(var + eps) * w_ref + b_ref
        ref.backward(paddle.to_tensor(upstream_np, dtype="float32"))

        for name, actual, expected in (
            ("out", out, ref),
            ("dx", xp.grad, x_ref.grad),
            ("dw", norm.weight.grad, w_ref.grad),
            ("db", norm.bias.grad, b_ref.grad),
        ):
            self.assertIsNotNone(actual, name)
            self.assertIsNotNone(expected, name)
            np.testing.assert_allclose(
                actual.numpy(),
                expected.numpy(),
                rtol=1e-3,
                atol=1e-4,
                err_msg=name,
            )


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestFusedRMSNorm(unittest.TestCase):
    """FusedRMSNorm is an RMSNorm subclass with a simplified forward."""

    def test_is_subclass_of_rms_norm(self):
        self.assertTrue(issubclass(FusedRMSNorm, RMSNorm))

    def test_forward_matches_reference_and_consumes_weight(self):
        config = _make_config()
        norm = FusedRMSNorm(config)
        weight_np = _nonuniform([config.hidden_size], start=0.25, step=0.13)
        norm.weight.set_value(paddle.to_tensor(weight_np, dtype="float32"))

        x_np = _nonuniform([2, 3, config.hidden_size])
        out = norm(paddle.to_tensor(x_np, dtype="float32"))

        ref = _rmsnorm_ref(x_np, weight_np, config.rms_norm_eps)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestWrappedPaddleNorm(unittest.TestCase):
    """WrappedPaddleNorm.__new__ dispatches to the right norm class."""

    def test_creates_rms_norm(self):
        config = _make_config(normalization="RMSNorm")
        norm = WrappedPaddleNorm(config, hidden_size=6)
        self.assertIsInstance(norm, RMSNorm)
        self.assertNotIsInstance(norm, LayerNorm)
        self.assertEqual(norm.normalized_shape, 6)

    def test_creates_layer_norm(self):
        config = _make_config(normalization="LayerNorm")
        norm = WrappedPaddleNorm(config, hidden_size=6)
        self.assertIsInstance(norm, LayerNorm)
        self.assertIsNotNone(norm.bias)

    def test_unsupported_normalization_raises(self):
        config = _make_config(normalization="UnsupportedNorm")
        with self.assertRaisesRegex(Exception, "Only RMSNorm"):
            WrappedPaddleNorm(config, hidden_size=6)

    def test_passes_hidden_size_and_eps(self):
        config = _make_config()
        norm = WrappedPaddleNorm(config, hidden_size=5, eps=1e-6)
        self.assertEqual(norm.normalized_shape, 5)
        self.assertAlmostEqual(norm.variance_epsilon, 1e-6)

    def test_input_is_parallel_none_not_parallel_does_not_mark(self):
        # sequence_parallel False and tp==1 -> derived input_is_parallel False.
        config = _make_config(
            normalization="RMSNorm",
            sequence_parallel=False,
            tensor_model_parallel_size=1,
        )
        with mock.patch(_MARK_TARGET) as marker:
            WrappedPaddleNorm(config, hidden_size=6, input_is_parallel=None)
            marker.assert_not_called()

    def test_input_is_parallel_none_derives_true_from_sequence_parallel(self):
        # The derivation reads config.sequence_parallel; when True the weight is
        # marked as a sequence-parallel parameter during construction.
        config = _make_config(normalization="RMSNorm")
        config.sequence_parallel = True  # local fresh config; no shared state
        with mock.patch(_MARK_TARGET) as marker:
            norm = WrappedPaddleNorm(
                config, hidden_size=6, input_is_parallel=None
            )
            marker.assert_called_once_with(norm.weight)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestWrappedPaddleNormPipe(unittest.TestCase):
    """Pipeline norm: split/normalize main, pass MTP embeddings through."""

    def _build(self, **cfg_overrides):
        config = _make_config(**cfg_overrides)
        pipe = WrappedPaddleNormPipe(config, hidden_size=config.hidden_size)
        # Non-trivial norm weight so "weight ignored" is observable.
        weight_np = _nonuniform([config.hidden_size], start=0.3, step=0.1)
        pipe.norm.weight.set_value(paddle.to_tensor(weight_np, dtype="float32"))
        return config, pipe, weight_np

    def test_forward_without_mtp_normalizes_and_preserves_other_keys(self):
        config, pipe, weight_np = self._build(num_nextn_predict_layers=0)
        x_np = _nonuniform([2, 4, config.hidden_size])
        sentinel = paddle.to_tensor([7, 8, 9])
        out = pipe(
            {
                "hidden_states": paddle.to_tensor(x_np, dtype="float32"),
                "labels": sentinel,
            }
        )
        ref = _rmsnorm_ref(x_np, weight_np, config.rms_norm_eps)
        np.testing.assert_allclose(
            out["hidden_states"].numpy(), ref, rtol=1e-4, atol=1e-5
        )
        # Unrelated payload must pass through untouched.
        np.testing.assert_array_equal(out["labels"].numpy(), [7, 8, 9])

    def test_forward_with_mtp_normalizes_main_passes_through_mtp(self):
        # num_nextn=1, mtp_load_weight_only=False, non-experimental defaults:
        # input = concat([main, mtp], axis=0); only main is normalized, the MTP
        # slice is forwarded unchanged. This catches split/concat mistakes and a
        # bug that would (wrongly) normalize the MTP embeddings too.
        config, pipe, weight_np = self._build(
            num_nextn_predict_layers=1, mtp_load_weight_only=False
        )
        main_np = _nonuniform([2, 4, config.hidden_size], start=0.5, step=0.2)
        mtp_np = _nonuniform([2, 4, config.hidden_size], start=-2.0, step=0.3)
        concat_np = np.concatenate([main_np, mtp_np], axis=0)

        out = pipe(
            {"hidden_states": paddle.to_tensor(concat_np, dtype="float32")}
        )
        result = out["hidden_states"].numpy()

        self.assertEqual(list(result.shape), [4, 4, config.hidden_size])
        main_ref = _rmsnorm_ref(main_np, weight_np, config.rms_norm_eps)
        np.testing.assert_allclose(result[:2], main_ref, rtol=1e-4, atol=1e-5)
        # MTP half is passed through verbatim (not normalized).
        np.testing.assert_allclose(
            result[2:], mtp_np.astype(np.float32), rtol=0, atol=0
        )

    def test_forward_with_mtp_load_weight_only_skips_split(self):
        # mtp_load_weight_only=True disables the split guard: the *entire* input
        # is normalized in one shot (no concat semantics).
        config, pipe, weight_np = self._build(
            num_nextn_predict_layers=1, mtp_load_weight_only=True
        )
        x_np = _nonuniform([2, 4, config.hidden_size])
        out = pipe({"hidden_states": paddle.to_tensor(x_np, dtype="float32")})
        ref = _rmsnorm_ref(x_np, weight_np, config.rms_norm_eps)
        self.assertEqual(
            list(out["hidden_states"].shape), [2, 4, config.hidden_size]
        )
        np.testing.assert_allclose(
            out["hidden_states"].numpy(), ref, rtol=1e-4, atol=1e-5
        )

    def test_build_schedule_node(self):
        config, pipe, _ = self._build()
        node = pipe.build_schedule_node()
        self.assertIsInstance(node, ScheduleNode)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestL2Norm(unittest.TestCase):
    """L2Norm performs mean-of-squares (RMS-style) normalization."""

    def test_construction(self):
        norm = L2Norm(hidden_size=16, eps=1e-8)
        self.assertEqual(norm.hidden_size, 16)
        self.assertAlmostEqual(norm.eps, 1e-8)

    def test_forward_matches_mean_of_squares_reference(self):
        # Non-uniform input (unlike the coverage test's all-equal vector, which
        # cannot distinguish mean-of-squares from true unit-L2 normalization).
        norm = L2Norm(hidden_size=4, eps=1e-6)
        x_np = _nonuniform([2, 3, 4], start=0.7, step=0.4)
        out = norm(paddle.to_tensor(x_np, dtype="float32")).numpy()

        ones = np.ones([4], dtype=np.float64)
        ref = _rmsnorm_ref(x_np, ones, 1e-6)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)

        # Documented contract: per-vector mean of squared outputs is ~1. A true
        # unit-L2 normalization would instead give mean-of-squares == 1/dim.
        mean_sq = np.mean(out.astype(np.float64) ** 2, axis=-1)
        np.testing.assert_allclose(
            mean_sq, np.ones_like(mean_sq), rtol=1e-3, atol=1e-3
        )
        self.assertGreater(abs(mean_sq.mean() - 1.0 / 4), 0.1)

    def test_forward_preserves_dtype(self):
        norm = L2Norm(hidden_size=8)
        out = norm(paddle.to_tensor(_nonuniform([2, 8]), dtype="float32"))
        self.assertEqual(out.dtype, paddle.float32)


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestGetNormExtraArgs(unittest.TestCase):
    """Argument-name mapping between WrappedPaddleNorm and direct norm classes."""

    def test_wrapped_paddle_norm_maps_hidden_size_and_eps(self):
        config = _make_config()
        extra = get_norm_extra_args(WrappedPaddleNorm, config, 64, 1e-5, False)
        self.assertIs(extra["config"], config)
        self.assertEqual(extra["hidden_size"], 64)
        self.assertAlmostEqual(extra["eps"], 1e-5)
        self.assertFalse(extra["input_is_parallel"])
        self.assertNotIn("normalized_shape", extra)
        self.assertNotIn("norm_eps", extra)

    def test_direct_class_maps_normalized_shape_and_norm_eps(self):
        config = _make_config()
        extra = get_norm_extra_args(RMSNorm, config, 64, 1e-5, True)
        self.assertIs(extra["config"], config)
        self.assertEqual(extra["normalized_shape"], 64)
        self.assertAlmostEqual(extra["norm_eps"], 1e-5)
        self.assertTrue(extra["input_is_parallel"])
        self.assertNotIn("hidden_size", extra)
        self.assertNotIn("eps", extra)

    def test_layer_spec_wrapping_wrapped_paddle_norm(self):
        # A real LayerSpec exercises the isinstance(LayerSpec) branch and the
        # `.layer` lookup; a plain mock (as in the coverage test) never enters it.
        config = _make_config()
        spec = LayerSpec(layer=WrappedPaddleNorm)
        extra = get_norm_extra_args(spec, config, 32, 1e-6, False)
        self.assertEqual(extra["hidden_size"], 32)
        self.assertAlmostEqual(extra["eps"], 1e-6)
        self.assertNotIn("normalized_shape", extra)

    def test_layer_spec_wrapping_direct_class(self):
        config = _make_config()
        spec = LayerSpec(layer=RMSNorm)
        extra = get_norm_extra_args(spec, config, 32, 1e-6, False)
        self.assertEqual(extra["normalized_shape"], 32)
        self.assertAlmostEqual(extra["norm_eps"], 1e-6)
        self.assertNotIn("hidden_size", extra)


if __name__ == "__main__":
    unittest.main()
