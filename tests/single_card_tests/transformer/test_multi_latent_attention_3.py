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
"""Behavior tests for MLA RoPE and its sublayer spec dataclass.

Targets in ``paddlefleet.transformer.multi_latent_attention``:

* ``_ec_compatible_rope_apply`` -- applies ErnieCore-style rotary embedding via
  interleaved-pair complex multiplication. Each even/odd channel pair ``(a, b)``
  at position ``p`` and frequency ``inv_freq[j] = rope_base**(-2j/D)`` becomes
  ``(a*cos - b*sin, a*sin + b*cos)`` with ``theta = p * inv_freq[j]``; the same
  rotation broadcasts over the head axis. Expected values are derived from an
  independent NumPy implementation (no reuse of the module under test), and the
  ``position_offset`` / ``position_ids`` branches are exercised with
  distinguishable positions so a swap or an ignored argument is rejected.
* ``MLASelfAttentionSublayersSpec`` -- a dataclass whose ten sublayer fields
  default to ``None`` and store exactly the object assigned to each named field
  (distinct sentinels catch a field-to-field swap).

CPU-only: these paths need no CUDA. Heavy imports are guarded so the file skips
honestly (never falsely passes) when the paddle stack is unavailable.
"""

import dataclasses
import unittest

import numpy as np

_IMPORT_ERROR = None
try:
    import paddle

    from paddlefleet.transformer.multi_latent_attention import (
        MLASelfAttentionSublayersSpec,
        _ec_compatible_rope_apply,
    )
except (ImportError, ModuleNotFoundError) as exc:  # no paddle stack here
    _IMPORT_ERROR = exc


# Field names are an independent statement of the spec's public contract; they
# are compared against the dataclass definition rather than derived from it.
_EXPECTED_SPEC_FIELDS = (
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


def _numpy_rope_reference(x, positions, rope_base):
    """Independent interleaved-pair RoPE.

    ``x`` is ``[B, S, H, D]`` (float64), ``positions`` is ``[S]``. Returns the
    rotated array with the same layout. This does not call any paddle op or the
    function under test.
    """
    b_sz, s_len, n_head, dim = x.shape
    half = dim // 2
    idx = np.arange(half, dtype=np.float64)
    inv_freq = rope_base ** (-(2.0 * idx) / dim)  # [half]
    theta = positions[:, None] * inv_freq[None, :]  # [S, half]
    cos = np.cos(theta)[None, :, None, :]  # broadcast over B and H
    sin = np.sin(theta)[None, :, None, :]
    even = x[..., 0::2]  # [B, S, H, half]
    odd = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR!r}",
)
class TestEcCompatibleRopeApply(unittest.TestCase):
    """Numeric behavior of ``_ec_compatible_rope_apply`` on CPU."""

    B, S, H, D = 2, 3, 2, 8
    ROPE_BASE = 1000000.0

    def _make_inputs(self):
        # Fixed, fully distinguishable content so channel/head/position mixups
        # change at least one element.
        q_np = np.arange(
            self.B * self.S * self.H * self.D, dtype=np.float64
        ).reshape(self.B, self.S, self.H, self.D)
        k_np = (
            np.arange(self.B * self.S * 1 * self.D, dtype=np.float64).reshape(
                self.B, self.S, 1, self.D
            )
            + 100.0
        )
        return q_np, k_np

    def test_forward_matches_independent_numpy_reference(self):
        q_np, k_np = self._make_inputs()
        positions = np.arange(self.S, dtype=np.float64)  # default offset 0
        q_ref = _numpy_rope_reference(q_np, positions, self.ROPE_BASE)
        k_ref = _numpy_rope_reference(k_np, positions, self.ROPE_BASE)

        q_pe = paddle.to_tensor(q_np.astype("float32"))
        k_pe = paddle.to_tensor(k_np.astype("float32"))
        q_out, k_out = _ec_compatible_rope_apply(
            q_pe, k_pe, self.S, rope_base=self.ROPE_BASE
        )

        self.assertEqual(q_out.shape, list(q_pe.shape))
        self.assertEqual(k_out.shape, list(k_pe.shape))
        np.testing.assert_allclose(q_out.numpy(), q_ref, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(k_out.numpy(), k_ref, rtol=1e-5, atol=1e-4)
        # Position 0 induces zero rotation: that row must be untouched, which a
        # spurious constant rotation would break.
        np.testing.assert_allclose(
            q_out.numpy()[:, 0], q_np[:, 0], rtol=1e-6, atol=1e-6
        )

    def test_position_offset_shifts_rotation(self):
        q_np, k_np = self._make_inputs()
        offset = 5
        positions = np.arange(offset, offset + self.S, dtype=np.float64)
        q_ref = _numpy_rope_reference(q_np, positions, self.ROPE_BASE)

        q_pe = paddle.to_tensor(q_np.astype("float32"))
        k_pe = paddle.to_tensor(k_np.astype("float32"))
        q_out, _ = _ec_compatible_rope_apply(
            q_pe, k_pe, self.S, rope_base=self.ROPE_BASE, position_offset=offset
        )
        np.testing.assert_allclose(q_out.numpy(), q_ref, rtol=1e-5, atol=1e-4)

        # The offset must actually be consumed: with a nonzero offset the first
        # row is now rotated (position 5), unlike the offset-0 case.
        q_out0, _ = _ec_compatible_rope_apply(
            q_pe, k_pe, self.S, rope_base=self.ROPE_BASE, position_offset=0
        )
        self.assertFalse(
            np.allclose(
                q_out.numpy()[:, 0], q_out0.numpy()[:, 0], rtol=1e-4, atol=1e-4
            )
        )

    def test_explicit_position_ids_override_sequence(self):
        q_np, k_np = self._make_inputs()
        # Non-monotonic 1D position_ids so ordering/consumption is observable.
        pos_list = [2, 0, 1]
        positions = np.asarray(pos_list, dtype=np.float64)
        q_ref = _numpy_rope_reference(q_np, positions, self.ROPE_BASE)

        q_pe = paddle.to_tensor(q_np.astype("float32"))
        k_pe = paddle.to_tensor(k_np.astype("float32"))
        position_ids = paddle.to_tensor(pos_list, dtype="int64")
        q_out, _ = _ec_compatible_rope_apply(
            q_pe,
            k_pe,
            self.S,
            rope_base=self.ROPE_BASE,
            position_ids=position_ids,
        )
        np.testing.assert_allclose(q_out.numpy(), q_ref, rtol=1e-5, atol=1e-4)
        # Row 1 uses position 0 -> identity; distinct from the sequential case
        # where row 1 uses position 1.
        np.testing.assert_allclose(
            q_out.numpy()[:, 1], q_np[:, 1], rtol=1e-6, atol=1e-6
        )

    def test_output_dtype_preserved(self):
        q_np, k_np = self._make_inputs()
        q_pe = paddle.to_tensor(q_np.astype("float32"))
        k_pe = paddle.to_tensor(k_np.astype("float32"))
        q_out, k_out = _ec_compatible_rope_apply(
            q_pe, k_pe, self.S, rope_base=self.ROPE_BASE
        )
        self.assertEqual(q_out.dtype, q_pe.dtype)
        self.assertEqual(k_out.dtype, k_pe.dtype)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR!r}",
)
class TestMLASelfAttentionSublayersSpec(unittest.TestCase):
    """Contract of the MLA sublayer spec dataclass."""

    def test_is_dataclass_with_expected_fields(self):
        self.assertTrue(dataclasses.is_dataclass(MLASelfAttentionSublayersSpec))
        actual = tuple(
            f.name for f in dataclasses.fields(MLASelfAttentionSublayersSpec)
        )
        self.assertEqual(actual, _EXPECTED_SPEC_FIELDS)

    def test_all_fields_default_to_none(self):
        spec = MLASelfAttentionSublayersSpec()
        for name in _EXPECTED_SPEC_FIELDS:
            self.assertIsNone(getattr(spec, name))

    def test_fields_store_their_own_assigned_object(self):
        # Distinct sentinel per field: a field-to-field swap in the dataclass
        # definition would surface as a mismatch here.
        markers = {name: f"marker::{name}" for name in _EXPECTED_SPEC_FIELDS}
        spec = MLASelfAttentionSublayersSpec(**markers)
        for name in _EXPECTED_SPEC_FIELDS:
            self.assertEqual(getattr(spec, name), markers[name])


if __name__ == "__main__":
    unittest.main()
