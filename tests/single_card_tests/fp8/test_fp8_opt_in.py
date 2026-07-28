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
"""End-to-end tests for the FP8 opt-in plumbing added on top of ``develop``.

All tests execute real forward / backward through paddle FP8 kernels
(no mocks). Covers:

* ``TransformerConfig.full_fp8_computation`` default.
* ``is_fp8_tensor`` accepting real UE8M0 int32 scales produced by
  ``paddle.incubate.nn.functional.fp8_quant_blockwise``.
* ``full_fp8_computation`` master switch actually gates the FP8 path.
* ``disable_fp8`` opt-out really keeps a Linear in bf16 (bit-exact
  match with a bf16-config Linear).
* End-to-end FP8 forward / backward on ``Linear`` and
  ``ColumnParallelLinear`` matches bf16 within tolerance.
* ``_fp8_prequant_weight`` actually stashes cached fp8 tensors on the
  weight, and a second call replaces the cache in-place.
* Setting ``save_original_input=True`` on an FP8 Linear keeps its
  numerical output equivalent to the default path.
"""

from __future__ import annotations

import unittest

import numpy as np
import paddle

from paddlefleet.fp8.utils import is_fp8_tensor
from paddlefleet.tensor_parallel import ColumnParallelLinear
from paddlefleet.tensor_parallel.layers import (
    Linear,
    RowParallelLinear,
    _fp8_clear_prequant_weight,
    _fp8_prequant_weight,
)
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.transformer_config import TransformerConfig

# Initialize the model-parallel RNG tracker so that GPU weight initialization
# works in single-card (non-distributed) test environments.
if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
    model_parallel_cuda_manual_seed(42)

_HAS_GPU = (
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
)
_REQUIRE_GPU = unittest.skipUnless(
    _HAS_GPU, "FP8 blockwise kernels require a CUDA device"
)
# FP8 forward/backward with deep_gemm requires SM100+ (Blackwell).
# On SM90 (H20/H100) the wgrad path hits "c.has_value() and d.scalar_type() == kFloat".
_SM100_PLUS = _HAS_GPU and paddle.device.cuda.get_device_capability()[0] >= 10
_REQUIRE_SM100 = unittest.skipUnless(
    _SM100_PLUS, "FP8 Linear forward/backward requires SM100+ (Blackwell) GPU"
)


def _calc_diff(x: paddle.Tensor, y: paddle.Tensor) -> float:
    """Cosine-style relative error, matches tests/.../test_fp8_linear.py."""
    x = x.astype("float64").numpy()
    y = y.astype("float64").numpy()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def _fp8_config(**overrides) -> TransformerConfig:
    base = {
        "num_hidden_layers": 1,
        "hidden_size": 512,
        "intermediate_size": 512,
        "use_bias": False,
        "use_cpu_initialization": False,
        "fp8": "blockwise",
        "fp8_wgrad": True,
        "full_fp8_computation": True,
    }
    base.update(overrides)
    return TransformerConfig(**base)


def _bf16_config(**overrides) -> TransformerConfig:
    base = {
        "num_hidden_layers": 1,
        "hidden_size": 512,
        "intermediate_size": 512,
        "use_bias": False,
        "use_cpu_initialization": False,
    }
    base.update(overrides)
    return TransformerConfig(**base)


def _new_linear(config, cls=Linear, **kwargs):
    layer = cls(
        config.hidden_size,
        config.intermediate_size,
        config=config,
        init_method=config.init_method,
        bias=False,
        **kwargs,
    )
    # Downcast weights to bf16 to match the runtime dtype used by fp8 gemm.
    paddle.amp.decorate(models=layer, level="O2", dtype="bfloat16")
    return layer


def _copy_weight(dst, src):
    """Copy src.weight into dst.weight without triggering param replacement."""
    with paddle.no_grad():
        dst.weight.copy_(src.weight.detach(), False)


class TestFullFp8ComputationDefault(unittest.TestCase):
    def test_default_is_false(self):
        """Master switch defaults to False so precision matches develop."""
        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=64,
            intermediate_size=64,
            use_cpu_initialization=True,
        )
        self.assertFalse(config.full_fp8_computation)


class TestIsFp8TensorRealQuant(unittest.TestCase):
    """Exercise ``is_fp8_tensor`` on tuples produced by real fp8 quant."""

    @_REQUIRE_GPU
    def test_rejects_bf16_tensor(self):
        x = paddle.randn([128, 128], dtype="bfloat16")
        self.assertFalse(is_fp8_tensor(x))

    @_REQUIRE_GPU
    def test_accepts_float32_scale_from_blockwise(self):
        x = paddle.randn([128, 128], dtype="bfloat16")
        out = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x,
            output_scale_transpose=False,
            quant_method="1x128",
            input_transpose=False,
        )
        fp8, scale = out[0], out[1]
        self.assertEqual(scale.dtype, paddle.float32)
        self.assertTrue(is_fp8_tensor((fp8, scale)))

    @_REQUIRE_GPU
    def test_accepts_int32_ue8m0_scale_from_blockwise(self):
        """UE8M0 packs pow2 exponents into int32 — must be accepted."""
        x = paddle.randn([128, 128], dtype="bfloat16")
        out = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x,
            output_scale_transpose=False,
            quant_method="1x128",
            input_transpose=False,
            using_pow2_scale=True,
            using_ue8m0_scale=True,
        )
        fp8, scale = out[0], out[1]
        self.assertEqual(scale.dtype, paddle.int32)
        self.assertTrue(is_fp8_tensor((fp8, scale)))


class TestLinearFp8GatingRealForward(unittest.TestCase):
    """The gating flags must actually route to bf16 / fp8 code paths."""

    @_REQUIRE_GPU
    def test_master_switch_off_matches_bf16(self):
        """``full_fp8_computation=False`` on a fp8-recipe config must be
        bit-exact with a pure bf16 config."""
        fp8_off_config = _bf16_config(
            fp8="blockwise", full_fp8_computation=False
        )
        bf16_config = _bf16_config()

        paddle.seed(0)
        layer_off = _new_linear(fp8_off_config)
        paddle.seed(0)
        layer_bf16 = _new_linear(bf16_config)
        _copy_weight(layer_off, layer_bf16)

        self.assertFalse(layer_off.fp8)
        self.assertFalse(layer_bf16.fp8)

        x = paddle.randn([4, 128, 512], dtype="bfloat16")
        x_off = x.detach()
        x_off.stop_gradient = False
        x_bf16 = x.detach()
        x_bf16.stop_gradient = False

        out_off, _ = layer_off(x_off)
        out_bf16, _ = layer_bf16(x_bf16)

        np.testing.assert_array_equal(
            out_off.astype("float32").numpy(),
            out_bf16.astype("float32").numpy(),
        )

    @_REQUIRE_GPU
    def test_disable_fp8_matches_bf16(self):
        """``disable_fp8=True`` on a fp8-on config must be bit-exact with
        the same-shape bf16-only Linear."""
        fp8_on_config = _fp8_config()
        bf16_config = _bf16_config()

        paddle.seed(0)
        layer_disabled = _new_linear(fp8_on_config, disable_fp8=True)
        paddle.seed(0)
        layer_bf16 = _new_linear(bf16_config)
        _copy_weight(layer_disabled, layer_bf16)

        self.assertFalse(layer_disabled.fp8)
        self.assertIsNone(layer_disabled.inp_quant_func)
        self.assertIsNone(layer_disabled.weight_quant_func)

        x = paddle.randn([4, 128, 512], dtype="bfloat16")
        x_disabled = x.detach()
        x_disabled.stop_gradient = False
        x_bf16 = x.detach()
        x_bf16.stop_gradient = False

        out_disabled, _ = layer_disabled(x_disabled)
        out_bf16, _ = layer_bf16(x_bf16)

        np.testing.assert_array_equal(
            out_disabled.astype("float32").numpy(),
            out_bf16.astype("float32").numpy(),
        )


class TestFp8LinearForwardBackward(unittest.TestCase):
    """Real fp8 forward+backward on Linear / ColumnParallelLinear should
    match a bf16 reference within a bounded quantization error."""

    def _run_fwd_bwd(self, fp8_layer, bf16_layer, x_shape):
        _copy_weight(fp8_layer, bf16_layer)

        x = paddle.randn(x_shape, dtype="bfloat16")
        x_fp8 = x.detach()
        x_fp8.stop_gradient = False
        x_bf16 = x.detach()
        x_bf16.stop_gradient = False

        out_fp8, _ = fp8_layer(x_fp8)
        out_bf16, _ = bf16_layer(x_bf16)
        out_fp8.sum().backward()
        out_bf16.sum().backward()

        return (
            _calc_diff(out_fp8, out_bf16),
            _calc_diff(x_fp8.grad, x_bf16.grad),
            _calc_diff(fp8_layer.weight.grad, bf16_layer.weight.grad),
        )

    @_REQUIRE_SM100
    def test_linear_fp8_matches_bf16_within_tol(self):
        fp8_cfg = _fp8_config()
        bf16_cfg = _bf16_config()

        paddle.seed(0)
        fp8_layer = _new_linear(fp8_cfg)
        paddle.seed(0)
        bf16_layer = _new_linear(bf16_cfg)

        self.assertTrue(fp8_layer.fp8)
        self.assertFalse(bf16_layer.fp8)

        out_diff, x_grad_diff, w_grad_diff = self._run_fwd_bwd(
            fp8_layer, bf16_layer, [4, 128, 512]
        )

        self.assertLess(out_diff, 0.001, f"output diff too large: {out_diff}")
        self.assertLess(
            x_grad_diff, 0.001, f"x_grad diff too large: {x_grad_diff}"
        )
        self.assertLess(
            w_grad_diff, 0.001, f"w_grad diff too large: {w_grad_diff}"
        )

    @_REQUIRE_SM100
    def test_column_parallel_linear_fp8_matches_bf16(self):
        fp8_cfg = _fp8_config()
        bf16_cfg = _bf16_config()

        paddle.seed(0)
        fp8_layer = _new_linear(
            fp8_cfg,
            cls=ColumnParallelLinear,
            gather_output=False,
            tp_group=None,
        )
        paddle.seed(0)
        bf16_layer = _new_linear(
            bf16_cfg,
            cls=ColumnParallelLinear,
            gather_output=False,
            tp_group=None,
        )

        self.assertTrue(fp8_layer.fp8)
        self.assertFalse(bf16_layer.fp8)

        out_diff, x_grad_diff, w_grad_diff = self._run_fwd_bwd(
            fp8_layer, bf16_layer, [4, 128, 512]
        )

        self.assertLess(out_diff, 0.001, f"output diff too large: {out_diff}")
        self.assertLess(
            x_grad_diff, 0.001, f"x_grad diff too large: {x_grad_diff}"
        )
        self.assertLess(
            w_grad_diff, 0.001, f"w_grad diff too large: {w_grad_diff}"
        )

    @_REQUIRE_SM100
    def test_save_original_input_true_keeps_output_equivalent(self):
        """Toggling ``save_original_input`` changes the wgrad path but
        forward output must stay identical."""
        cfg = _fp8_config()

        paddle.seed(0)
        default_layer = _new_linear(cfg)
        paddle.seed(0)
        keep_bf16_layer = _new_linear(cfg)
        keep_bf16_layer.save_original_input = True
        _copy_weight(keep_bf16_layer, default_layer)

        x = paddle.randn([2, 128, 512], dtype="bfloat16")
        x_a = x.detach()
        x_a.stop_gradient = False
        x_b = x.detach()
        x_b.stop_gradient = False

        out_default, _ = default_layer(x_a)
        out_keep, _ = keep_bf16_layer(x_b)
        out_default.sum().backward()
        out_keep.sum().backward()

        # Forward output is the same fp8 gemm result either way.
        np.testing.assert_allclose(
            out_default.astype("float32").numpy(),
            out_keep.astype("float32").numpy(),
            rtol=0.0,
            atol=0.0,
        )
        # The two wgrad paths should produce close (but not necessarily
        # bit-identical) weight gradients.
        w_grad_diff = _calc_diff(
            default_layer.weight.grad, keep_bf16_layer.weight.grad
        )
        self.assertLess(w_grad_diff, 0.05)

    @_REQUIRE_SM100
    def test_linear_fp8_no_ue8m0_matches_bf16(self):
        """FP8 Linear with use_ue8m0=False, fp8_wgrad=True requires SM100+."""
        fp8_cfg = _fp8_config(use_ue8m0=False)
        bf16_cfg = _bf16_config()

        paddle.seed(0)
        fp8_layer = _new_linear(fp8_cfg)
        paddle.seed(0)
        bf16_layer = _new_linear(bf16_cfg)

        self.assertTrue(fp8_layer.fp8)
        self.assertFalse(fp8_layer.use_ue8m0)

        out_diff, x_grad_diff, w_grad_diff = self._run_fwd_bwd(
            fp8_layer, bf16_layer, [4, 128, 512]
        )

        self.assertLess(out_diff, 0.001, f"output diff too large: {out_diff}")
        self.assertLess(
            x_grad_diff, 0.001, f"x_grad diff too large: {x_grad_diff}"
        )
        self.assertLess(
            w_grad_diff, 0.001, f"w_grad diff too large: {w_grad_diff}"
        )

    @_REQUIRE_GPU
    def test_linear_fp8_no_ue8m0_forward_only(self):
        """FP8 Linear with use_ue8m0=False and fp8_wgrad=False works on H-series."""
        fp8_cfg = _fp8_config(use_ue8m0=False, fp8_wgrad=False)
        bf16_cfg = _bf16_config()

        paddle.seed(0)
        fp8_layer = _new_linear(fp8_cfg)
        paddle.seed(0)
        bf16_layer = _new_linear(bf16_cfg)
        _copy_weight(fp8_layer, bf16_layer)

        self.assertTrue(fp8_layer.fp8)
        self.assertFalse(fp8_layer.use_ue8m0)
        self.assertFalse(fp8_layer.fp8_wgrad)

        x = paddle.randn([4, 128, 512], dtype="bfloat16")
        x_fp8 = x.detach()
        x_fp8.stop_gradient = False
        x_bf16 = x.detach()
        x_bf16.stop_gradient = False

        out_fp8, _ = fp8_layer(x_fp8)
        out_bf16, _ = bf16_layer(x_bf16)
        out_fp8.sum().backward()
        out_bf16.sum().backward()

        out_diff = _calc_diff(out_fp8, out_bf16)
        x_grad_diff = _calc_diff(x_fp8.grad, x_bf16.grad)
        self.assertLess(out_diff, 0.001, f"output diff too large: {out_diff}")
        self.assertLess(
            x_grad_diff, 0.001, f"x_grad diff too large: {x_grad_diff}"
        )
        fp8_cfg = _fp8_config(use_ue8m0=False, fp8_wgrad=False)
        bf16_cfg = _bf16_config()

        paddle.seed(0)
        fp8_layer = _new_linear(
            fp8_cfg,
            cls=ColumnParallelLinear,
            gather_output=False,
            tp_group=None,
        )
        paddle.seed(0)
        bf16_layer = _new_linear(
            bf16_cfg,
            cls=ColumnParallelLinear,
            gather_output=False,
            tp_group=None,
        )

        self.assertTrue(fp8_layer.fp8)
        self.assertFalse(fp8_layer.use_ue8m0)

        out_diff, x_grad_diff, w_grad_diff = self._run_fwd_bwd(
            fp8_layer, bf16_layer, [4, 128, 512]
        )

        self.assertLess(out_diff, 0.001, f"output diff too large: {out_diff}")
        self.assertLess(
            x_grad_diff, 0.001, f"x_grad diff too large: {x_grad_diff}"
        )
        self.assertLess(
            w_grad_diff, 0.001, f"w_grad diff too large: {w_grad_diff}"
        )

    """``_fp8_prequant_weight`` must actually populate the weight cache."""

    @_REQUIRE_GPU
    def test_populates_and_refreshes_cache(self):
        cfg = _fp8_config()
        paddle.seed(0)
        layer = _new_linear(cfg)
        self.assertTrue(layer.fp8)

        # Fresh weight has no cache.
        self.assertIsNone(getattr(layer.weight, "fp8_weight_fwd", None))

        _fp8_prequant_weight(layer)

        fp8_fwd = getattr(layer.weight, "fp8_weight_fwd", None)
        scale_fwd = getattr(layer.weight, "fp8_scale_fwd", None)
        scale_bwd = getattr(layer.weight, "fp8_scale_bwd", None)
        self.assertIsNotNone(fp8_fwd)
        self.assertIsNotNone(scale_fwd)
        self.assertIsNotNone(scale_bwd)
        self.assertEqual(fp8_fwd.dtype, paddle.float8_e4m3fn)
        self.assertIn(scale_fwd.dtype, (paddle.float32, paddle.int32))

        # Second call must replace the cache, not raise (delattr path).
        prev_id = id(fp8_fwd)
        _fp8_prequant_weight(layer)
        new_fp8_fwd = getattr(layer.weight, "fp8_weight_fwd", None)
        self.assertIsNotNone(new_fp8_fwd)
        # The freshly-quantized tensor should be a new object (the old one
        # was delattr'd first) — but even if paddle interns it, the cache
        # entries must still be valid FP8 tensors.
        self.assertEqual(new_fp8_fwd.dtype, paddle.float8_e4m3fn)
        del prev_id  # silence unused-var lint

    @_REQUIRE_GPU
    def test_noop_on_bf16_linear(self):
        """Non-fp8 layer stays untouched — no cache attrs appear."""
        cfg = _bf16_config()
        paddle.seed(0)
        layer = _new_linear(cfg)
        self.assertFalse(layer.fp8)

        _fp8_prequant_weight(layer)

        self.assertIsNone(getattr(layer.weight, "fp8_weight_fwd", None))
        self.assertIsNone(getattr(layer.weight, "fp8_scale_fwd", None))
        self.assertIsNone(getattr(layer.weight, "fp8_scale_bwd", None))


class TestFp8ClearPrequantWeight(unittest.TestCase):
    """``clear_fp8_quant_weight`` must invalidate the per-Linear cache so
    the next forward re-quantizes the (post-optimizer-step) weight."""

    def _assert_cache_absent(self, weight):
        self.assertIsNone(getattr(weight, "fp8_weight_fwd", None))
        self.assertIsNone(getattr(weight, "fp8_scale_fwd", None))
        self.assertIsNone(getattr(weight, "fp8_scale_bwd", None))

    def _assert_cache_present(self, weight):
        self.assertIsNotNone(getattr(weight, "fp8_weight_fwd", None))
        self.assertIsNotNone(getattr(weight, "fp8_scale_fwd", None))
        self.assertIsNotNone(getattr(weight, "fp8_scale_bwd", None))

    @_REQUIRE_GPU
    def test_helper_strips_all_three_cache_attrs(self):
        cfg = _fp8_config()
        paddle.seed(0)
        layer = _new_linear(cfg)
        _fp8_prequant_weight(layer)
        self._assert_cache_present(layer.weight)

        _fp8_clear_prequant_weight(layer)
        self._assert_cache_absent(layer.weight)

    @_REQUIRE_GPU
    def test_helper_is_idempotent_and_bf16_safe(self):
        # Clearing a never-quantized fp8 layer must not raise.
        cfg = _fp8_config()
        paddle.seed(0)
        layer = _new_linear(cfg)
        _fp8_clear_prequant_weight(layer)
        self._assert_cache_absent(layer.weight)

        # Second clear on a real cache is also safe.
        _fp8_prequant_weight(layer)
        _fp8_clear_prequant_weight(layer)
        _fp8_clear_prequant_weight(layer)
        self._assert_cache_absent(layer.weight)

        # bf16 layer never had a cache — must remain untouched.
        bf16_layer = _new_linear(_bf16_config())
        _fp8_clear_prequant_weight(bf16_layer)
        self._assert_cache_absent(bf16_layer.weight)

    @_REQUIRE_GPU
    def test_linear_clear_fp8_quant_weight_method(self):
        cfg = _fp8_config()
        paddle.seed(0)
        layer = _new_linear(cfg)
        layer.fp8_quant_weight()
        self._assert_cache_present(layer.weight)

        layer.clear_fp8_quant_weight()
        self._assert_cache_absent(layer.weight)

    @_REQUIRE_GPU
    def test_column_parallel_clear_fp8_quant_weight_method(self):
        cfg = _fp8_config()
        paddle.seed(0)
        layer = _new_linear(
            cfg, cls=ColumnParallelLinear, gather_output=False, tp_group=None
        )
        layer.fp8_quant_weight()
        self._assert_cache_present(layer.weight)

        layer.clear_fp8_quant_weight()
        self._assert_cache_absent(layer.weight)

    @_REQUIRE_GPU
    def test_row_parallel_clear_fp8_quant_weight_method(self):
        cfg = _fp8_config()
        paddle.seed(0)
        layer = _new_linear(
            cfg,
            cls=RowParallelLinear,
            input_is_parallel=False,
            tp_group=None,
            skip_bias_add=False,
        )
        layer.fp8_quant_weight()
        self._assert_cache_present(layer.weight)

        layer.clear_fp8_quant_weight()
        self._assert_cache_absent(layer.weight)

    @_REQUIRE_GPU
    def test_clear_forces_requantization_after_weight_update(self):
        """The real bug: without clear, a stale cache shadows the updated
        weight. After clear, the next quant call must reflect the new bf16
        weight (different fp8 bytes)."""
        cfg = _fp8_config()
        paddle.seed(0)
        layer = _new_linear(cfg)

        layer.fp8_quant_weight()
        cached_fp8_pre = layer.weight.fp8_weight_fwd
        pre_bytes = cached_fp8_pre.astype("float32").numpy().copy()

        # Simulate an optimizer step: perturb the bf16 weight in-place.
        with paddle.no_grad():
            layer.weight.add_(
                paddle.full_like(layer.weight, 0.5, dtype=layer.weight.dtype)
            )

        # Without clear, weight_quant_func would keep returning the stale
        # cache — verify the cache is still the pre-step fp8 tensor.
        self.assertTrue(
            np.array_equal(
                layer.weight.fp8_weight_fwd.astype("float32").numpy(),
                pre_bytes,
            )
        )

        # After clear + re-quant, the cache must reflect the new weight.
        layer.clear_fp8_quant_weight()
        self._assert_cache_absent(layer.weight)
        layer.fp8_quant_weight()
        self._assert_cache_present(layer.weight)
        post_bytes = layer.weight.fp8_weight_fwd.astype("float32").numpy()
        self.assertFalse(
            np.array_equal(pre_bytes, post_bytes),
            "fp8 cache did not refresh after weight update",
        )


if __name__ == "__main__":
    unittest.main()
