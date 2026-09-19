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

import os
import sys
import unittest

# Bootstrap: make the repo `src/` importable when the test is run standalone
# (CI normally puts it on PYTHONPATH; this keeps the file runnable directly).
_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle
    from paddle.distributed.fleet.meta_parallel import LayerSpec, ScheduleNode

    from paddlefleet.transformer.paddle_norm import (
        FusedRMSNorm,
        L2Norm,
        LayerNorm,
        RMSNorm,
        RMSNormTriton,
        WrappedPaddleNorm,
        WrappedPaddleNormPipe,
        WrappedRMSNormTriton,
        get_norm_extra_args,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig

    PADDLE_AVAILABLE = True
    _IMPORT_ERROR = ""
except (
    ImportError,
    ModuleNotFoundError,
) as exc:  # CPU-only env may lack paddle
    PADDLE_AVAILABLE = False
    _IMPORT_ERROR = repr(exc)


_SKIP_REASON = (
    "paddle / paddlefleet not importable in this CPU-only environment: "
    + _IMPORT_ERROR
)

# --- Independent numpy references (float64) -------------------------------
# These reimplement the math from scratch so a bug shared with the production
# path cannot hide. Inputs are saved and the expected value is derived here,
# never by calling the norm under test.


def _rms_norm_ref(x, weight, eps):
    # RMSNorm: x / sqrt(mean(x**2, last dim) + eps) * weight. No mean removal.
    x64 = np.asarray(x, dtype=np.float64)
    w64 = np.asarray(weight, dtype=np.float64)
    ms = np.mean(np.square(x64), axis=-1, keepdims=True)
    return (x64 / np.sqrt(ms + eps)) * w64


def _layer_norm_ref(x, weight, bias, eps):
    # LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias, biased variance.
    x64 = np.asarray(x, dtype=np.float64)
    mean = np.mean(x64, axis=-1, keepdims=True)
    var = np.mean(np.square(x64 - mean), axis=-1, keepdims=True)
    normed = (x64 - mean) / np.sqrt(var + eps)
    return normed * np.asarray(weight, dtype=np.float64) + np.asarray(
        bias, dtype=np.float64
    )


def _l2_norm_ref(x, eps):
    # L2Norm: x / sqrt(mean(x**2, last dim) + eps) (mean-square normalization).
    x64 = np.asarray(x, dtype=np.float64)
    ms = np.mean(np.square(x64), axis=-1, keepdims=True)
    return x64 / np.sqrt(ms + eps)


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "num_attention_heads": 4,
        "normalization": "RMSNorm",
        "rms_norm_eps": 1e-5,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


# Tolerances for comparing float32 kernels against a float64 reference.
_RTOL = 1e-4
_ATOL = 1e-5


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestRMSNorm(unittest.TestCase):
    def test_forward_matches_independent_reference(self):
        # Fixed, non-uniform input and weight so a dropped weight multiply or
        # a mean-subtraction bug (LayerNorm-style) would change the result.
        eps = 1e-5
        config = _make_config(hidden_size=8, rms_norm_eps=eps)
        norm = RMSNorm(config=config)
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 3, 8)).astype(np.float32)
        w_np = np.array(
            [0.5, 1.5, -2.0, 0.25, 3.0, -1.0, 0.75, 2.5], dtype=np.float32
        )
        norm.weight.set_value(paddle.to_tensor(w_np))

        out = norm(paddle.to_tensor(x_np))
        expected = _rms_norm_ref(x_np, w_np, eps)
        self.assertEqual(
            out.dtype, paddle.float32
        )  # return_dtype = weight.dtype
        self.assertEqual(list(out.shape), [2, 3, 8])
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )

    def test_custom_eps_is_consumed(self):
        # A large eps materially changes the denominator; assert the exact
        # value tracks the requested eps rather than the config default.
        config = _make_config(hidden_size=8, rms_norm_eps=1e-5)
        big_eps = 0.5
        norm = RMSNorm(config=config, norm_eps=big_eps)
        self.assertEqual(norm.variance_epsilon, big_eps)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((2, 8)).astype(np.float32)

        out = norm(paddle.to_tensor(x_np)).numpy()
        expected = _rms_norm_ref(x_np, np.ones(8), big_eps)
        np.testing.assert_allclose(out, expected, rtol=_RTOL, atol=_ATOL)
        # The default-eps reference must NOT match, proving eps is consumed.
        wrong = _rms_norm_ref(x_np, np.ones(8), 1e-5)
        self.assertGreater(np.abs(out - wrong).max(), 1e-3)

    def test_custom_normalized_shape_normalizes_over_last_dim(self):
        config = _make_config(hidden_size=8)
        norm = RMSNorm(config=config, normalized_shape=4)
        self.assertEqual(norm.normalized_shape, 4)
        self.assertEqual(list(norm.weight.shape), [4])
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((2, 4)).astype(np.float32)

        out = norm(paddle.to_tensor(x_np)).numpy()
        expected = _rms_norm_ref(x_np, np.ones(4), config.rms_norm_eps)
        np.testing.assert_allclose(out, expected, rtol=_RTOL, atol=_ATOL)

    def test_high_precision_norm_path_matches_reference(self):
        # Exercises the high_precision_norm branch (float32 upcast). On CPU the
        # weight is already float32, so this asserts the branch stays numerically
        # correct, not a dtype change.
        eps = 1e-5
        config = _make_config(hidden_size=8, rms_norm_eps=eps)
        norm = RMSNorm(config=config)
        rng = np.random.default_rng(3)
        x_np = rng.standard_normal((2, 8)).astype(np.float32)

        out = norm(
            paddle.to_tensor(x_np),
            high_precision_norm=True,
            return_high_precision_norm=True,
        )
        self.assertEqual(out.dtype, paddle.float32)
        expected = _rms_norm_ref(x_np, np.ones(8), eps)
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestLayerNorm(unittest.TestCase):
    def test_forward_matches_independent_reference(self):
        # Distinguishable weight AND bias so a missing bias-add or a swapped
        # mean/var term would be caught.
        eps = 1e-5
        config = _make_config(normalization="LayerNorm", rms_norm_eps=eps)
        norm = LayerNorm(config=config)
        rng = np.random.default_rng(10)
        x_np = rng.standard_normal((2, 3, 8)).astype(np.float32)
        w_np = np.array(
            [0.5, 1.5, -2.0, 0.25, 3.0, -1.0, 0.75, 2.5], dtype=np.float32
        )
        b_np = np.array(
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], dtype=np.float32
        )
        norm.weight.set_value(paddle.to_tensor(w_np))
        norm.bias.set_value(paddle.to_tensor(b_np))

        out = norm(paddle.to_tensor(x_np)).numpy()
        expected = _layer_norm_ref(x_np, w_np, b_np, eps)
        np.testing.assert_allclose(out, expected, rtol=_RTOL, atol=_ATOL)

    def test_bias_is_applied(self):
        # With ones weight and zero input the output equals the bias exactly,
        # isolating the bias term (a dropped bias would give zeros).
        eps = 1e-5
        config = _make_config(normalization="LayerNorm", rms_norm_eps=eps)
        norm = LayerNorm(config=config)
        b_np = np.array(
            [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0], dtype=np.float32
        )
        norm.bias.set_value(paddle.to_tensor(b_np))
        x = paddle.zeros([1, 8], dtype="float32")

        out = norm(x).numpy()
        # (0 - 0)/sqrt(0 + eps) * 1 + bias == bias
        np.testing.assert_allclose(out[0], b_np, rtol=_RTOL, atol=_ATOL)


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestFusedRMSNorm(unittest.TestCase):
    def test_forward_matches_independent_reference(self):
        eps = 1e-5
        config = _make_config(hidden_size=8, rms_norm_eps=eps)
        norm = FusedRMSNorm(config=config)
        rng = np.random.default_rng(20)
        x_np = rng.standard_normal((2, 3, 8)).astype(np.float32)
        w_np = np.array(
            [1.0, -1.0, 2.0, 0.5, -0.5, 3.0, -2.0, 0.25], dtype=np.float32
        )
        norm.weight.set_value(paddle.to_tensor(w_np))

        out = norm(paddle.to_tensor(x_np))
        expected = _rms_norm_ref(x_np, w_np, eps)
        self.assertEqual(out.dtype, paddle.float32)
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestL2Norm(unittest.TestCase):
    def test_forward_matches_independent_reference(self):
        eps = 1e-6
        norm = L2Norm(hidden_size=8, eps=eps)
        rng = np.random.default_rng(30)
        x_np = rng.standard_normal((2, 3, 8)).astype(np.float32)

        out = norm(paddle.to_tensor(x_np)).numpy()
        expected = _l2_norm_ref(x_np, eps)
        np.testing.assert_allclose(out, expected, rtol=_RTOL, atol=_ATOL)
        # Supplementary invariant: mean of squared outputs is ~1 per row.
        mean_sq = np.mean(np.square(out.astype(np.float64)), axis=-1)
        np.testing.assert_allclose(
            mean_sq, np.ones_like(mean_sq), rtol=1e-3, atol=1e-3
        )

    def test_custom_eps_is_consumed(self):
        # Large eps shrinks outputs measurably; assert the exact value tracks it.
        big_eps = 0.5
        norm = L2Norm(hidden_size=8, eps=big_eps)
        rng = np.random.default_rng(31)
        x_np = rng.standard_normal((2, 8)).astype(np.float32)

        out = norm(paddle.to_tensor(x_np)).numpy()
        expected = _l2_norm_ref(x_np, big_eps)
        np.testing.assert_allclose(out, expected, rtol=_RTOL, atol=_ATOL)
        wrong = _l2_norm_ref(x_np, 1e-6)
        self.assertGreater(np.abs(out - wrong).max(), 1e-3)


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestWrappedPaddleNormFactory(unittest.TestCase):
    def test_dispatches_to_rmsnorm_and_forwards_params(self):
        config = _make_config(normalization="RMSNorm")
        norm = WrappedPaddleNorm(config=config, hidden_size=8, eps=1e-6)
        self.assertIsInstance(norm, RMSNorm)
        self.assertNotIsInstance(norm, LayerNorm)
        # hidden_size -> normalized_shape, eps -> norm_eps (name conversion).
        self.assertEqual(norm.normalized_shape, 8)
        self.assertEqual(norm.variance_epsilon, 1e-6)

    def test_dispatches_to_layernorm(self):
        config = _make_config(normalization="LayerNorm")
        norm = WrappedPaddleNorm(config=config, hidden_size=8, eps=1e-6)
        self.assertIsInstance(norm, LayerNorm)
        self.assertIsNotNone(norm.bias)
        self.assertEqual(norm.normalized_shape, 8)
        self.assertEqual(norm.variance_epsilon, 1e-6)

    def test_unknown_normalization_raises(self):
        config = _make_config(normalization="GroupNorm")
        with self.assertRaises(Exception) as ctx:
            WrappedPaddleNorm(config=config, hidden_size=8)
        self.assertIn("RMSNorm", str(ctx.exception))


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestGetNormExtraArgs(unittest.TestCase):
    def test_wrapped_paddle_norm_branch(self):
        config = _make_config()
        extra = get_norm_extra_args(WrappedPaddleNorm, config, 128, 1e-5, False)
        self.assertEqual(
            set(extra), {"config", "input_is_parallel", "hidden_size", "eps"}
        )
        self.assertIs(extra["config"], config)
        self.assertFalse(extra["input_is_parallel"])
        self.assertEqual(extra["hidden_size"], 128)
        self.assertEqual(extra["eps"], 1e-5)

    def test_other_norm_branch_uses_different_keys(self):
        config = _make_config()
        extra = get_norm_extra_args(RMSNorm, config, 128, 1e-5, True)
        self.assertEqual(
            set(extra),
            {"config", "input_is_parallel", "normalized_shape", "norm_eps"},
        )
        self.assertIs(extra["config"], config)
        self.assertTrue(extra["input_is_parallel"])
        self.assertEqual(extra["normalized_shape"], 128)
        self.assertEqual(extra["norm_eps"], 1e-5)

    def test_layer_spec_is_unwrapped_to_its_layer(self):
        # A LayerSpec(WrappedPaddleNorm) must route through the WrappedPaddleNorm
        # branch (hidden_size/eps), proving the .layer unwrap happens.
        config = _make_config()
        spec = LayerSpec(WrappedPaddleNorm)
        extra = get_norm_extra_args(spec, config, 64, 2e-5, False)
        self.assertEqual(
            set(extra), {"config", "input_is_parallel", "hidden_size", "eps"}
        )
        self.assertEqual(extra["hidden_size"], 64)
        self.assertEqual(extra["eps"], 2e-5)


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestWrappedRMSNormTriton(unittest.TestCase):
    def test_construction_converts_param_names(self):
        # Factory maps build-spec params (hidden_size, eps) onto RMSNorm params
        # (normalized_shape, norm_eps) and returns an RMSNormTriton instance.
        config = _make_config()
        norm = WrappedRMSNormTriton(config=config, hidden_size=64, eps=1e-6)
        self.assertIsInstance(norm, RMSNormTriton)
        self.assertEqual(norm.normalized_shape, 64)
        self.assertEqual(norm.variance_epsilon, 1e-6)
        self.assertEqual(list(norm.weight.shape), [64])


@unittest.skipUnless(PADDLE_AVAILABLE, _SKIP_REASON)
class TestWrappedPaddleNormPipe(unittest.TestCase):
    def test_forward_without_mtp_normalizes_and_preserves_other_keys(self):
        eps = 1e-5
        config = _make_config(hidden_size=8, rms_norm_eps=eps)
        pipe = WrappedPaddleNormPipe(config=config, hidden_size=8)
        w_np = np.array(
            [0.5, 1.5, -2.0, 0.25, 3.0, -1.0, 0.75, 2.5], dtype=np.float32
        )
        pipe.norm.weight.set_value(paddle.to_tensor(w_np))

        rng = np.random.default_rng(40)
        x_np = rng.standard_normal((2, 4, 8)).astype(np.float32)
        marker = paddle.to_tensor([7.0, 8.0])
        result = pipe(
            {"hidden_states": paddle.to_tensor(x_np), "position_ids": marker}
        )

        expected = _rms_norm_ref(x_np, w_np, eps)
        np.testing.assert_allclose(
            result["hidden_states"].numpy(), expected, rtol=_RTOL, atol=_ATOL
        )
        # Unrelated dict entries are forwarded untouched.
        self.assertIn("position_ids", result)
        np.testing.assert_array_equal(
            result["position_ids"].numpy(), np.array([7.0, 8.0])
        )

    def test_forward_with_mtp_normalizes_main_only(self):
        # With num_nextn_predict_layers=2 the input is [main, mtp0, mtp1]
        # concatenated along dim 0. Only the main chunk is normalized; the two
        # MTP chunks must pass through unchanged. Each chunk carries a distinct
        # offset so a mis-split or a wrongly-normalized MTP chunk is visible.
        eps = 1e-5
        config = _make_config(
            hidden_size=8,
            rms_norm_eps=eps,
            num_nextn_predict_layers=2,
            mtp_load_weight_only=False,
        )
        pipe = WrappedPaddleNormPipe(config=config, hidden_size=8)
        w_np = np.array(
            [1.0, -1.0, 2.0, 0.5, -0.5, 3.0, -2.0, 0.25], dtype=np.float32
        )
        pipe.norm.weight.set_value(paddle.to_tensor(w_np))

        rng = np.random.default_rng(41)
        main = rng.standard_normal((1, 4, 8)).astype(np.float32)
        mtp0 = rng.standard_normal((1, 4, 8)).astype(np.float32) + 100.0
        mtp1 = rng.standard_normal((1, 4, 8)).astype(np.float32) - 100.0
        x_np = np.concatenate([main, mtp0, mtp1], axis=0)

        result = pipe({"hidden_states": paddle.to_tensor(x_np)})
        out = result["hidden_states"].numpy()
        self.assertEqual(list(result["hidden_states"].shape), [3, 4, 8])

        expected_main = _rms_norm_ref(main, w_np, eps)
        np.testing.assert_allclose(
            out[0:1], expected_main, rtol=_RTOL, atol=_ATOL
        )
        # MTP chunks are passed through unchanged (not normalized).
        np.testing.assert_allclose(out[1:2], mtp0, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(out[2:3], mtp1, rtol=_RTOL, atol=_ATOL)

    def test_build_schedule_node_returns_schedule_node(self):
        config = _make_config(hidden_size=8)
        pipe = WrappedPaddleNormPipe(config=config, hidden_size=8)
        node = pipe.build_schedule_node()
        self.assertIsInstance(node, ScheduleNode)


if __name__ == "__main__":
    unittest.main()
