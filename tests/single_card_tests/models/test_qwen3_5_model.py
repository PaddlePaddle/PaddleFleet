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

"""No-card unit tests for the Qwen3.5 RMSNorm slice.

These tests exercise the CPU-runnable numeric contracts of the Qwen3.5 model
module: the 1-centered ``Qwen3_5RMSNorm`` forward, the ``_HFBroadcastScale``
PyLayer forward/backward reductions, the ``Qwen3_5RMSNormPipe`` dict I/O plus
MTP tensor splitting, and the ``Qwen3_5VisionSublayersSpec`` dataclass. All
expected values are hand-derived with an independent NumPy reference (never by
re-invoking the production formula).

This slice is disjoint from the multi-card TP+SP loss-alignment test
(``tests/multi_card_tests/tensor_parallel/test_qwen3_5_model.py``), which
validates end-to-end distributed loss rather than these local layers.
"""

import dataclasses
import os
import sys
import unittest

import numpy as np

# Make the in-tree ``src/paddlefleet`` importable when the package is not
# pip-installed. Harmless if it is already on the path.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Honest capability probe: only genuine import failures (no Paddle / package
# not on path) become a skip. Any other error must surface as a real failure.
_SKIP_REASON = None
try:
    import paddle
    from paddle.distributed.fleet.meta_parallel import ScheduleNode

    from paddlefleet.models.qwen3_5.qwen3_5_model import (
        Qwen3_5RMSNorm,
        Qwen3_5RMSNormPipe,
        Qwen3_5VisionSublayersSpec,
        _HFBroadcastScale,
    )
    from paddlefleet.transformer.transformer_config import TransformerConfig
except (ImportError, ModuleNotFoundError) as exc:
    _SKIP_REASON = f"paddle/paddlefleet not importable: {exc}"

_HAS_PADDLE = _SKIP_REASON is None


def _make_config(**overrides):
    """Build a small real ``TransformerConfig`` on CPU."""
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "use_cpu_initialization": True,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _rmsnorm_reference(x, eps, weight=None):
    """Independent NumPy RMSNorm: x / sqrt(mean(x^2, -1) + eps) * (1 + weight).

    Computed in float64 to avoid sharing the production float32 accumulation.
    """
    xf = np.asarray(x, dtype=np.float64)
    var = np.mean(xf * xf, axis=-1, keepdims=True)
    normed = xf / np.sqrt(var + eps)
    if weight is None:
        return normed
    return normed * (1.0 + np.asarray(weight, dtype=np.float64))


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON or "paddle unavailable")
class TestQwen3_5RMSNormConstruction(unittest.TestCase):
    """Constructor conventions and parameter initialization."""

    def test_hidden_size_convention(self):
        cfg = _make_config()
        norm = Qwen3_5RMSNorm(cfg, hidden_size=32, eps=1e-5)
        self.assertEqual(norm.normalized_shape, 32)
        self.assertEqual(norm.variance_epsilon, 1e-5)
        self.assertEqual(list(norm.weight.shape), [32])

    def test_normalized_shape_convention(self):
        # The SelfAttention._build_norm else-branch calling convention.
        cfg = _make_config()
        norm = Qwen3_5RMSNorm(cfg, normalized_shape=48, norm_eps=1e-6)
        self.assertEqual(norm.normalized_shape, 48)
        self.assertEqual(norm.variance_epsilon, 1e-6)
        self.assertEqual(list(norm.weight.shape), [48])

    def test_eps_from_config_default(self):
        # Neither eps nor norm_eps given -> falls back to config.rms_norm_eps;
        # neither hidden_size nor normalized_shape -> config.hidden_size.
        cfg = _make_config(rms_norm_eps=3e-7)
        norm = Qwen3_5RMSNorm(cfg)
        self.assertEqual(norm.normalized_shape, cfg.hidden_size)
        self.assertEqual(norm.variance_epsilon, 3e-7)

    def test_head_major_grad_flag(self):
        cfg = _make_config()
        default_norm = Qwen3_5RMSNorm(cfg, hidden_size=16)
        self.assertFalse(default_norm.head_major_grad)
        hm = Qwen3_5RMSNorm(cfg, hidden_size=16, head_major_grad=True)
        self.assertTrue(hm.head_major_grad)

    def test_weight_initialized_to_zero(self):
        # 1-centered parameterization: weight starts at 0 so the initial
        # forward is a pure RMSNorm (scale == 1).
        cfg = _make_config()
        norm = Qwen3_5RMSNorm(cfg, hidden_size=24)
        np.testing.assert_array_equal(
            norm.weight.numpy(), np.zeros([24], dtype=norm.weight.numpy().dtype)
        )

    def test_input_is_parallel_marks_weight(self):
        cfg = _make_config()
        norm = Qwen3_5RMSNorm(cfg, hidden_size=16, input_is_parallel=True)
        # mark_as_sequence_parallel_parameter tags the parameter.
        self.assertTrue(getattr(norm.weight, "sequence_parallel", False))
        # A norm without the flag must NOT be marked (guards default path).
        plain = Qwen3_5RMSNorm(cfg, hidden_size=16)
        self.assertFalse(getattr(plain.weight, "sequence_parallel", False))


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON or "paddle unavailable")
class TestQwen3_5RMSNormForward(unittest.TestCase):
    """Forward numeric contract against an independent reference."""

    def test_forward_matches_independent_rmsnorm(self):
        cfg = _make_config()
        eps = 1e-5
        norm = Qwen3_5RMSNorm(cfg, hidden_size=4, eps=eps)
        x = np.array(
            [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 2.0, -3.0]], dtype=np.float32
        )
        out = norm(paddle.to_tensor(x)).numpy()
        # weight is 0 -> scale factor (1 + 0) == 1.
        ref = _rmsnorm_reference(x, eps, weight=np.zeros(4))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_weight_zero_is_identity_scale(self):
        # Explicitly pin the "1-centered" contract: with weight 0 the output
        # equals the bare normalized tensor, i.e. no scaling toward zero.
        cfg = _make_config()
        eps = 1e-5
        norm = Qwen3_5RMSNorm(cfg, hidden_size=3, eps=eps)
        x = np.array([[2.0, -2.0, 4.0]], dtype=np.float32)
        out = norm(paddle.to_tensor(x)).numpy()
        bare = _rmsnorm_reference(x, eps)
        np.testing.assert_allclose(out, bare, rtol=1e-5, atol=1e-6)

    def test_forward_preserves_dtype(self):
        # Computation happens in float32 then casts back to the input dtype.
        cfg = _make_config()
        eps = 1e-5
        norm = Qwen3_5RMSNorm(cfg, hidden_size=4, eps=eps)
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        out = norm(paddle.to_tensor(x).astype("float16"))
        self.assertEqual(out.dtype, paddle.float16)
        ref = _rmsnorm_reference(x, eps, weight=np.zeros(4))
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, rtol=3e-3, atol=3e-3
        )


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON or "paddle unavailable")
class TestHFBroadcastScale(unittest.TestCase):
    """Forward/backward of the bit-exact 1-centered scale PyLayer."""

    def test_forward_one_centered_scale(self):
        normed = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        weight = np.array([0.5, -1.0], dtype=np.float32)
        out = _HFBroadcastScale.apply(
            paddle.to_tensor(normed), paddle.to_tensor(weight), False
        ).numpy()
        # out = normed * (1 + weight) = normed * [1.5, 0.0]
        expected = normed * (1.0 + weight)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
        # Guard the "+1": a plain normed*weight would give a different column 0.
        self.assertFalse(np.allclose(out, normed * weight))

    def test_forward_zero_weight_identity(self):
        normed = np.array([[5.0, -7.0, 2.0]], dtype=np.float32)
        weight = np.zeros(3, dtype=np.float32)
        out = _HFBroadcastScale.apply(
            paddle.to_tensor(normed), paddle.to_tensor(weight), False
        ).numpy()
        np.testing.assert_allclose(out, normed, rtol=1e-6, atol=1e-6)

    def test_backward_2d_reduction(self):
        normed = paddle.to_tensor(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        weight = paddle.to_tensor(np.array([0.5, -1.0], dtype=np.float32))
        normed.stop_gradient = False
        weight.stop_gradient = False
        grad_out = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        out = _HFBroadcastScale.apply(normed, weight, False)
        out.backward(paddle.to_tensor(grad_out))

        # grad_normed = grad_out * (1 + weight)
        expected_grad_normed = grad_out * (1.0 + np.array([0.5, -1.0]))
        np.testing.assert_allclose(
            normed.grad.numpy(), expected_grad_normed, rtol=1e-6, atol=1e-6
        )
        # grad_weight = column reduction of (grad_out * normed) = [1+3, 2+4]
        expected_grad_weight = np.array([4.0, 6.0], dtype=np.float64)
        np.testing.assert_allclose(
            weight.grad.numpy(), expected_grad_weight, rtol=1e-6, atol=1e-6
        )

    def test_backward_head_major_total(self):
        # 4D [b=1, s=2, h=2, d=2]. head_major transposes to [b, h, s, d]
        # before the column reduction. In exact arithmetic the summed total is
        # order-independent, so integer inputs verify the reduction *total* and
        # that grad_normed uses (1 + weight); the FP row-order effect that
        # head_major actually guards is a bf16-rounding concern not observable
        # with these clean values (documented, not asserted).
        normed_np = np.arange(1, 9, dtype=np.float32).reshape([1, 2, 2, 2])
        weight_np = np.array([0.1, 0.2], dtype=np.float32)
        grad_np = np.ones([1, 2, 2, 2], dtype=np.float32)

        # Independent reference: sum grad*normed over every leading position.
        prod = grad_np * normed_np
        expected_grad_weight = prod.reshape([-1, 2]).sum(axis=0)  # [16, 20]
        np.testing.assert_array_equal(expected_grad_weight, [16.0, 20.0])
        expected_grad_normed = grad_np * (1.0 + weight_np)

        for head_major in (False, True):
            normed = paddle.to_tensor(normed_np)
            weight = paddle.to_tensor(weight_np)
            normed.stop_gradient = False
            weight.stop_gradient = False
            out = _HFBroadcastScale.apply(normed, weight, head_major)
            out.backward(paddle.to_tensor(grad_np))
            np.testing.assert_allclose(
                weight.grad.numpy(), expected_grad_weight, rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                normed.grad.numpy(),
                expected_grad_normed,
                rtol=1e-6,
                atol=1e-6,
            )


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON or "paddle unavailable")
class TestQwen3_5RMSNormPipe(unittest.TestCase):
    """Pipeline wrapper dict I/O and MTP tensor splitting."""

    def test_forward_non_mtp_normalizes_whole(self):
        cfg = _make_config()  # num_nextn_predict_layers defaults to 0
        eps = 1e-5
        pipe = Qwen3_5RMSNormPipe(cfg, hidden_size=4, eps=eps)
        x = np.array(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        result = pipe({"hidden_states": paddle.to_tensor(x)})
        ref = _rmsnorm_reference(x, eps, weight=np.zeros(4))
        np.testing.assert_allclose(
            result["hidden_states"].numpy(), ref, rtol=1e-5, atol=1e-6
        )

    def test_forward_preserves_other_keys(self):
        cfg = _make_config()
        pipe = Qwen3_5RMSNormPipe(cfg, hidden_size=4, eps=1e-5)
        mask = np.array([[1.0], [0.0], [1.0]], dtype=np.float32)
        result = pipe(
            {
                "hidden_states": paddle.to_tensor(
                    np.ones([3, 4], dtype=np.float32)
                ),
                "attention_mask": paddle.to_tensor(mask),
            }
        )
        self.assertIn("attention_mask", result)
        # Passthrough must preserve content, not just the key.
        np.testing.assert_array_equal(result["attention_mask"].numpy(), mask)

    def test_forward_mtp_splits_and_passes_through(self):
        # With num_nextn_predict_layers=2 the input is split along axis 0 into
        # 3 equal chunks; only the FIRST chunk is normalized, the rest pass
        # through unchanged and the concat order is preserved.
        cfg = _make_config()
        eps = 1e-5
        pipe = Qwen3_5RMSNormPipe(cfg, hidden_size=4, eps=eps)
        pipe.config.num_nextn_predict_layers = 2  # local, per-test config

        x = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [10.0, 20.0, 30.0, 40.0],
                [5.0, 6.0, 7.0, 8.0],
                [-1.0, -2.0, -3.0, -4.0],
                [9.0, 9.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )
        result = pipe({"hidden_states": paddle.to_tensor(x)})[
            "hidden_states"
        ].numpy()

        # First chunk (rows 0-1) normalized; rows 2-5 identical to input.
        first_ref = _rmsnorm_reference(x[:2], eps, weight=np.zeros(4))
        np.testing.assert_allclose(result[:2], first_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(result[2:], x[2:])
        # And the first chunk must actually differ from its raw input.
        self.assertFalse(np.allclose(result[:2], x[:2]))

    def test_build_schedule_node_type(self):
        cfg = _make_config()
        pipe = Qwen3_5RMSNormPipe(cfg, hidden_size=4, eps=1e-5)
        node = pipe.build_schedule_node()
        self.assertIsInstance(node, ScheduleNode)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON or "paddle unavailable")
class TestQwen3_5VisionSublayersSpec(unittest.TestCase):
    """The vision sublayers spec dataclass."""

    def test_defaults_all_none(self):
        spec = Qwen3_5VisionSublayersSpec()
        self.assertIsNone(spec.embedding)
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.transformer_layers)
        self.assertIsNone(spec.tail_empty_layers)
        self.assertIsNone(spec.merger)

    def test_field_assignment(self):
        emb = object()
        merger = object()
        heads = [object(), object()]
        spec = Qwen3_5VisionSublayersSpec(
            embedding=emb, transformer_layers=heads, merger=merger
        )
        self.assertIs(spec.embedding, emb)
        self.assertIs(spec.merger, merger)
        self.assertIs(spec.transformer_layers, heads)
        # Unset fields stay None.
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.tail_empty_layers)

    def test_dataclass_fields(self):
        self.assertTrue(dataclasses.is_dataclass(Qwen3_5VisionSublayersSpec))
        names = [f.name for f in dataclasses.fields(Qwen3_5VisionSublayersSpec)]
        self.assertEqual(
            names,
            [
                "embedding",
                "head_empty_layers",
                "transformer_layers",
                "tail_empty_layers",
                "merger",
            ],
        )


if __name__ == "__main__":
    unittest.main()
