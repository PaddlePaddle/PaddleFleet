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

import unittest
from dataclasses import fields

import numpy as np

# Heavy backend: Paddle (and therefore paddlefleet) may be absent on a CPU-only
# runner. Import failures are the only thing swallowed here; any other error
# surfaces so a real API/regression break is not silently reported as "skipped".
try:
    import paddle
    from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
        WeightGradStore,
    )

    from paddlefleet.transformer.dw_overlap import DeferredWeightGradLinear
    from paddlefleet.transformer.multi_latent_attention import (
        MLASelfAttentionSublayersSpec,
        _ec_compatible_rope_apply,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    _IMPORT_ERROR = exc

_HEAVY_OK = paddle is not None
_SKIP_REASON = (
    f"paddle/paddlefleet not importable on this runner: {_IMPORT_ERROR!r}"
)


def _reference_ec_rope(
    q_pe, k_pe, seq_len, rope_base=1000000.0, position_offset=0, positions=None
):
    """Independent NumPy re-derivation of ``_ec_compatible_rope_apply``.

    EC-style RoPE treats each adjacent pair ``(a, b)`` of the head dimension as
    a complex number ``a + b*i`` and multiplies it by ``exp(i * pos * freq)``,
    where ``freq[j] = base ** (-2j / D)``. This reference does the algebra in
    float64 purely from that definition -- it never calls the production RoPE --
    so a wrong pairing, a swapped sin/cos sign, or a mis-indexed frequency all
    diverge from it. Shapes: ``q_pe`` is ``[B, S, H, D]``, ``k_pe`` ``[B, S, 1, D]``.
    """
    q_pe = np.asarray(q_pe, dtype=np.float64)
    k_pe = np.asarray(k_pe, dtype=np.float64)
    head_dim = q_pe.shape[-1]
    freqs = 1.0 / (
        rope_base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    )
    if positions is None:
        positions = np.arange(
            position_offset, position_offset + seq_len, dtype=np.float64
        )
    else:
        positions = np.asarray(positions, dtype=np.float64)
    angle = np.outer(positions, freqs)  # [S, D/2]
    cos = np.cos(angle)[None, :, None, :]  # [1, S, 1, D/2]
    sin = np.sin(angle)[None, :, None, :]

    def _rotate(x):
        b, s, h, d = x.shape
        pairs = x.reshape(b, s, h, d // 2, 2)
        a = pairs[..., 0]
        c = pairs[..., 1]
        out_real = a * cos - c * sin
        out_imag = a * sin + c * cos
        return np.stack([out_real, out_imag], axis=-1).reshape(b, s, h, d)

    return _rotate(q_pe), _rotate(k_pe)


@unittest.skipUnless(_HEAVY_OK, _SKIP_REASON)
class TestECCompatibleRopeApply(unittest.TestCase):
    """Numerical behavior of ``_ec_compatible_rope_apply`` (CP world size 1)."""

    def setUp(self):
        paddle.set_device("cpu")
        # Distinguishable, non-degenerate inputs: unique per position/head/dim,
        # spanning positive and negative values so a pair swap or sign flip in
        # the complex rotation cannot be masked by symmetry.
        b, s, h, d = 1, 3, 2, 4
        self.seq_len = s
        q = (np.arange(b * s * h * d, dtype=np.float32) * 0.1) - 1.0
        self.q_np = q.reshape(b, s, h, d)
        k = (np.arange(b * s * 1 * d, dtype=np.float32) * 0.13) - 0.7
        self.k_np = k.reshape(b, s, 1, d)

    def test_matches_independent_complex_reference(self):
        q_pe = paddle.to_tensor(self.q_np)
        k_pe = paddle.to_tensor(self.k_np)
        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, self.seq_len)
        q_ref, k_ref = _reference_ec_rope(self.q_np, self.k_np, self.seq_len)
        np.testing.assert_allclose(q_out.numpy(), q_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(k_out.numpy(), k_ref, rtol=1e-5, atol=1e-6)

    def test_position_zero_is_identity(self):
        # pos 0 -> angle 0 -> cos=1, sin=0, so the first sequence slot must be
        # returned unrotated. This pins the boundary that a wrong pairing would
        # already perturb.
        q_pe = paddle.to_tensor(self.q_np)
        k_pe = paddle.to_tensor(self.k_np)
        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, self.seq_len)
        np.testing.assert_allclose(
            q_out.numpy()[:, 0], self.q_np[:, 0], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            k_out.numpy()[:, 0], self.k_np[:, 0], rtol=1e-6, atol=1e-6
        )
        # A later position must actually be rotated (guards against a no-op).
        self.assertFalse(
            np.allclose(q_out.numpy()[:, 2], self.q_np[:, 2], atol=1e-4)
        )

    def test_custom_rope_base_changes_frequencies(self):
        q_pe = paddle.to_tensor(self.q_np)
        k_pe = paddle.to_tensor(self.k_np)
        base = 500000.0
        q_out, k_out = _ec_compatible_rope_apply(
            q_pe, k_pe, self.seq_len, rope_base=base
        )
        q_ref, k_ref = _reference_ec_rope(
            self.q_np, self.k_np, self.seq_len, rope_base=base
        )
        np.testing.assert_allclose(q_out.numpy(), q_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(k_out.numpy(), k_ref, rtol=1e-5, atol=1e-6)
        # The base genuinely matters: default base gives a different rotation.
        q_default, _ = _ec_compatible_rope_apply(q_pe, k_pe, self.seq_len)
        self.assertFalse(
            np.allclose(q_out.numpy(), q_default.numpy(), atol=1e-4)
        )

    def test_position_offset_shifts_positions(self):
        offset = 5
        q_pe = paddle.to_tensor(self.q_np)
        k_pe = paddle.to_tensor(self.k_np)
        q_out, k_out = _ec_compatible_rope_apply(
            q_pe, k_pe, self.seq_len, position_offset=offset
        )
        q_ref, k_ref = _reference_ec_rope(
            self.q_np, self.k_np, self.seq_len, position_offset=offset
        )
        np.testing.assert_allclose(q_out.numpy(), q_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(k_out.numpy(), k_ref, rtol=1e-5, atol=1e-6)

    def test_explicit_1d_position_ids_override_sequential(self):
        # A 1-D position_ids tensor must be consumed verbatim as the rotation
        # angles' positions, not the default [0, 1, ..., S-1].
        pos = np.array([2.0, 0.0, 5.0], dtype=np.float32)
        q_pe = paddle.to_tensor(self.q_np)
        k_pe = paddle.to_tensor(self.k_np)
        q_out, k_out = _ec_compatible_rope_apply(
            q_pe,
            k_pe,
            self.seq_len,
            position_ids=paddle.to_tensor(pos),
        )
        q_ref, k_ref = _reference_ec_rope(
            self.q_np, self.k_np, self.seq_len, positions=pos
        )
        np.testing.assert_allclose(q_out.numpy(), q_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(k_out.numpy(), k_ref, rtol=1e-5, atol=1e-6)
        # position_ids[1] == 0 -> that slot is identity, distinguishing it from
        # the sequential default where slot 1 would be rotated by angle 1*freq.
        np.testing.assert_allclose(
            q_out.numpy()[:, 1], self.q_np[:, 1], rtol=1e-6, atol=1e-6
        )

    def test_output_dtype_matches_input_and_values_correct(self):
        # dtype is a real contract (the caller casts back), but assert it
        # together with numerical correctness rather than on its own.
        q_pe = paddle.to_tensor(self.q_np).astype("float32")
        k_pe = paddle.to_tensor(self.k_np).astype("float32")
        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, self.seq_len)
        self.assertEqual(q_out.dtype, q_pe.dtype)
        self.assertEqual(k_out.dtype, k_pe.dtype)
        q_ref, k_ref = _reference_ec_rope(self.q_np, self.k_np, self.seq_len)
        np.testing.assert_allclose(q_out.numpy(), q_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(k_out.numpy(), k_ref, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(_HEAVY_OK, _SKIP_REASON)
class TestDeferredWeightGradLinear(unittest.TestCase):
    """``DeferredWeightGradLinear``: eager forward/dx, deferred dW into main_grad."""

    def setUp(self):
        paddle.set_device("cpu")
        # WeightGradStore is a process-global; make sure a prior test did not
        # leave it enabled, and restore that on the way out.
        self._prev_enabled = WeightGradStore.enabled
        WeightGradStore.enabled = False
        self.addCleanup(setattr, WeightGradStore, "enabled", self._prev_enabled)

        # Arbitrary leading batch dims [2, 3]; in=4, out=5. Distinguishable,
        # non-square, non-degenerate values so a transpose or reduction error
        # cannot survive the comparison.
        rng = np.random.default_rng(0)
        self.x_np = rng.standard_normal((2, 3, 4)).astype("float32")
        self.w_np = rng.standard_normal((4, 5)).astype("float32")
        self.g_np = rng.standard_normal((2, 3, 5)).astype("float32")

    def _make_weight(self):
        weight = paddle.create_parameter(
            shape=list(self.w_np.shape),
            dtype="float32",
            default_initializer=paddle.nn.initializer.Assign(self.w_np),
        )
        weight.main_grad = None
        return weight

    def test_forward_matches_matmul_over_leading_dims(self):
        x = paddle.to_tensor(self.x_np)
        weight = self._make_weight()
        out = DeferredWeightGradLinear.apply(x, weight)
        # Independent reference: F.linear(x, W) == x @ W for W laid out [in, out].
        ref = self.x_np @ self.w_np
        self.assertEqual(list(out.shape), [2, 3, 5])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-6, atol=1e-6)

    def test_backward_eager_dx_and_deferred_dw(self):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        weight = self._make_weight()
        upstream = paddle.to_tensor(self.g_np)

        out = DeferredWeightGradLinear.apply(x, weight)
        out.backward(upstream)

        # dx is computed eagerly: dx = g @ W^T, over the full leading batch.
        dx_ref = self.g_np @ self.w_np.T
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(x.grad.numpy(), dx_ref, rtol=1e-5, atol=1e-6)

        # dW is *deferred*: it must not land in the autograd .grad slot, and
        # main_grad must still be empty until the WeightGradStore is drained.
        self.assertIsNone(weight.grad)
        self.assertIsNone(weight.main_grad)

        # Drain the queued thunk the way the PP scheduler does.
        WeightGradStore.flush()
        WeightGradStore.pop()

        # dW = x_2d^T @ g_2d, flattened over the leading batch dims -> [in, out],
        # accumulated into main_grad (fp32).
        x_2d = self.x_np.reshape(-1, self.x_np.shape[-1])
        g_2d = self.g_np.reshape(-1, self.g_np.shape[-1])
        dw_ref = x_2d.T @ g_2d
        self.assertIsNotNone(weight.main_grad)
        self.assertEqual(list(weight.main_grad.shape), [4, 5])
        np.testing.assert_allclose(
            weight.main_grad.numpy(), dw_ref, rtol=1e-5, atol=1e-5
        )

    def test_main_grad_accumulates_across_backwards(self):
        # main_grad is added into, not overwritten: two identical backwards must
        # double it (this is why the store uses add_, not assignment).
        weight = self._make_weight()
        x_2d = self.x_np.reshape(-1, self.x_np.shape[-1])
        g_2d = self.g_np.reshape(-1, self.g_np.shape[-1])
        dw_ref = x_2d.T @ g_2d

        for _ in range(2):
            x = paddle.to_tensor(self.x_np, stop_gradient=False)
            upstream = paddle.to_tensor(self.g_np)
            out = DeferredWeightGradLinear.apply(x, weight)
            out.backward(upstream)
            WeightGradStore.flush()
            WeightGradStore.pop()

        np.testing.assert_allclose(
            weight.main_grad.numpy(), 2.0 * dw_ref, rtol=1e-5, atol=1e-5
        )

    def test_frozen_weight_produces_no_weight_grad(self):
        # stop_gradient weight: dx still flows, but no dW is queued, so main_grad
        # stays empty and no .grad appears.
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        weight = self._make_weight()
        weight.stop_gradient = True
        upstream = paddle.to_tensor(self.g_np)

        out = DeferredWeightGradLinear.apply(x, weight)
        out.backward(upstream)

        dx_ref = self.g_np @ self.w_np.T
        np.testing.assert_allclose(x.grad.numpy(), dx_ref, rtol=1e-5, atol=1e-6)
        self.assertIsNone(weight.grad)
        self.assertIsNone(weight.main_grad)


@unittest.skipUnless(_HEAVY_OK, _SKIP_REASON)
class TestMLASelfAttentionSublayersSpec(unittest.TestCase):
    """Schema contract of the MLA sublayer spec dataclass."""

    EXPECTED_FIELDS = (
        "q_a_layernorm",
        "kv_a_layernorm",
        "q_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "core_attention",
        "o_proj",
        "gate_proj",
    )

    def test_exact_field_set_and_none_defaults(self):
        names = tuple(f.name for f in fields(MLASelfAttentionSublayersSpec))
        # Exact set and order pins the schema consumers rely on; a dropped or
        # renamed sublayer slot fails here.
        self.assertEqual(names, self.EXPECTED_FIELDS)
        spec = MLASelfAttentionSublayersSpec()
        for name in self.EXPECTED_FIELDS:
            self.assertIsNone(
                getattr(spec, name), f"{name} default should be None"
            )

    def test_fields_bind_by_name_not_position(self):
        # Distinguishable sentinels per slot catch a field-order/assignment swap
        # that None-vs-not-None checks would miss.
        sentinels = {name: object() for name in self.EXPECTED_FIELDS}
        spec = MLASelfAttentionSublayersSpec(**sentinels)
        for name in self.EXPECTED_FIELDS:
            self.assertIs(getattr(spec, name), sentinels[name])


if __name__ == "__main__":
    unittest.main()
