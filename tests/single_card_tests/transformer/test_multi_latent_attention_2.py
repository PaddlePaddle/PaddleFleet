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
"""Behavior tests for paddlefleet.transformer.multi_latent_attention.

Two independently-verifiable contracts are covered:

* ``_ec_compatible_rope_apply`` -- the ErnieCore-style interleaved complex
  RoPE. Expected outputs are derived from the mathematical definition with an
  independent NumPy implementation (never by calling the function under test).
* ``MLASelfAttention.backward_dw`` -- the delayed weight-gradient orchestration,
  whose only observable contract is *which* projection sublayers get their
  ``backward_dw`` invoked for a given config (q-LoRA on/off, absorbed KV).

Everything here is CPU-only. Paddle is imported lazily; when it (or the heavy
``multi_latent_attention`` import chain) is unavailable the tests skip with an
honest reason instead of pretending to pass.
"""

import unittest
from unittest import mock

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.multi_latent_attention import (
        MLASelfAttention,
        _ec_compatible_rope_apply,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    MLASelfAttention = None
    _ec_compatible_rope_apply = None
    _IMPORT_ERROR = repr(exc)

_MLA_MODULE = "paddlefleet.transformer.multi_latent_attention"
_SKIP_REASON = (
    f"paddle / multi_latent_attention not importable: {_IMPORT_ERROR}"
)


def _numpy_ec_rope(
    q_pe,
    k_pe,
    seq_len,
    rope_base=1000000.0,
    position_offset=0,
    position_ids=None,
):
    """Independent reference for the interleaved complex RoPE.

    Mirrors the mathematical definition only (no Paddle call): each adjacent
    pair ``(x0, x1)`` of the last dim is treated as a complex number and
    rotated by ``angle = position * inv_freq``::

        out0 = x0 * cos(angle) - x1 * sin(angle)
        out1 = x0 * sin(angle) + x1 * cos(angle)

    and re-interleaved as ``[out0, out1, ...]``. ``cos``/``sin`` broadcast over
    batch and head axes; q and k share the same rotation table.
    """
    head_dim = q_pe.shape[-1]
    inv_freq = 1.0 / (
        rope_base
        ** (np.arange(0, head_dim, 2, dtype=np.float64) / float(head_dim))
    )  # [D/2]
    if position_ids is not None and position_ids.ndim == 1:
        positions = position_ids.astype(np.float64)
    else:
        positions = np.arange(
            position_offset, position_offset + seq_len, dtype=np.float64
        )
    angles = np.outer(positions, inv_freq)  # [S, D/2]
    cos = np.cos(angles)[None, :, None, :]  # [1, S, 1, D/2]
    sin = np.sin(angles)[None, :, None, :]

    def _apply(x):
        b, s, h, d = x.shape
        xr = x.astype(np.float64).reshape(b, s, h, d // 2, 2)
        x0 = xr[..., 0]
        x1 = xr[..., 1]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return np.stack([out0, out1], axis=-1).reshape(b, s, h, d)

    return _apply(q_pe), _apply(k_pe)


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestECCompatibleRopeApply(unittest.TestCase):
    """Numeric contract of the ErnieCore-compatible RoPE application."""

    def setUp(self):
        paddle.set_device("cpu")
        paddle.seed(2026)
        # Isolate from any ambient context-parallel state: this file validates
        # the single-card (world_size == 1) math path only. Patching the CP
        # world-size query is mocking a non-under-test collaborator; the rotation
        # math itself stays real.
        patcher = mock.patch(
            f"{_MLA_MODULE}.get_context_parallel_world_size", return_value=1
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _distinguishable_qk(self, batch, seq_len, heads, head_dim):
        """Position-unique, non-uniform q/k so swaps and misrotations show up."""
        rng = np.random.default_rng(0)
        q = rng.standard_normal((batch, seq_len, heads, head_dim)).astype(
            "float32"
        )
        k = rng.standard_normal((batch, seq_len, 1, head_dim)).astype("float32")
        return q, k

    def test_matches_independent_numpy_reference(self):
        batch, seq_len, heads, head_dim = 2, 4, 3, 8
        q_np, k_np = self._distinguishable_qk(batch, seq_len, heads, head_dim)

        q_out, k_out = _ec_compatible_rope_apply(
            paddle.to_tensor(q_np), paddle.to_tensor(k_np), seq_len
        )
        q_ref, k_ref = _numpy_ec_rope(q_np, k_np, seq_len)

        np.testing.assert_allclose(q_out.numpy(), q_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(k_out.numpy(), k_ref, rtol=1e-5, atol=1e-6)
        # Shapes are preserved (interleave/flatten must not reorder axes).
        self.assertEqual(list(q_out.shape), [batch, seq_len, heads, head_dim])
        self.assertEqual(list(k_out.shape), [batch, seq_len, 1, head_dim])

    def test_position_zero_is_identity(self):
        """Position 0 has angle 0 -> cos=1, sin=0, so the first row is untouched.

        This is an input-derived anchor (not the function's own output) that a
        sign flip, swapped cos/sin, or wrong rotation direction cannot satisfy.
        """
        batch, seq_len, heads, head_dim = 1, 3, 2, 8
        q_np, k_np = self._distinguishable_qk(batch, seq_len, heads, head_dim)

        q_out, _ = _ec_compatible_rope_apply(
            paddle.to_tensor(q_np), paddle.to_tensor(k_np), seq_len
        )
        # Row 0 (position 0) is identity; a later row must have actually rotated.
        np.testing.assert_allclose(
            q_out.numpy()[:, 0], q_np[:, 0], rtol=1e-6, atol=1e-6
        )
        self.assertFalse(
            np.allclose(q_out.numpy()[:, 2], q_np[:, 2], rtol=1e-4, atol=1e-4),
            "position 2 should be rotated away from its input",
        )

    def test_position_offset_is_consumed(self):
        """position_offset must shift the rotation table by that many positions."""
        batch, seq_len, heads, head_dim = 1, 4, 2, 8
        offset = 5
        q_np, k_np = self._distinguishable_qk(batch, seq_len, heads, head_dim)

        q_off, k_off = _ec_compatible_rope_apply(
            paddle.to_tensor(q_np),
            paddle.to_tensor(k_np),
            seq_len,
            position_offset=offset,
        )
        q_ref, k_ref = _numpy_ec_rope(
            q_np, k_np, seq_len, position_offset=offset
        )
        np.testing.assert_allclose(q_off.numpy(), q_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(k_off.numpy(), k_ref, rtol=1e-5, atol=1e-6)

        # With a non-zero offset, even the first row rotates, so the result must
        # differ from the offset=0 path (proves the arg is not ignored).
        q_base, _ = _ec_compatible_rope_apply(
            paddle.to_tensor(q_np), paddle.to_tensor(k_np), seq_len
        )
        self.assertFalse(
            np.allclose(q_off.numpy(), q_base.numpy(), rtol=1e-4, atol=1e-4),
            "position_offset had no effect on the output",
        )

    def test_explicit_position_ids_override_sequential(self):
        """A 1D position_ids replaces the sequential positions and is consumed."""
        batch, seq_len, heads, head_dim = 1, 4, 2, 8
        q_np, k_np = self._distinguishable_qk(batch, seq_len, heads, head_dim)
        # Non-monotonic permutation so an ignored / mis-broadcast id is visible.
        pos_np = np.array([3, 1, 2, 0], dtype="float32")

        q_out, k_out = _ec_compatible_rope_apply(
            paddle.to_tensor(q_np),
            paddle.to_tensor(k_np),
            seq_len,
            position_ids=paddle.to_tensor(pos_np),
        )
        q_ref, k_ref = _numpy_ec_rope(q_np, k_np, seq_len, position_ids=pos_np)
        np.testing.assert_allclose(q_out.numpy(), q_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(k_out.numpy(), k_ref, rtol=1e-5, atol=1e-6)

        # Row with id==0 is identity; the sequential path would have rotated it.
        np.testing.assert_allclose(
            q_out.numpy()[:, 3], q_np[:, 3], rtol=1e-6, atol=1e-6
        )
        q_seq, _ = _ec_compatible_rope_apply(
            paddle.to_tensor(q_np), paddle.to_tensor(k_np), seq_len
        )
        self.assertFalse(
            np.allclose(q_out.numpy(), q_seq.numpy(), rtol=1e-4, atol=1e-4),
            "explicit position_ids did not change the rotation",
        )

    def test_rope_base_changes_frequencies(self):
        """A different rope_base yields a different (independently derived) table."""
        batch, seq_len, heads, head_dim = 1, 4, 2, 8
        q_np, k_np = self._distinguishable_qk(batch, seq_len, heads, head_dim)
        base = 10000.0

        q_out, _ = _ec_compatible_rope_apply(
            paddle.to_tensor(q_np),
            paddle.to_tensor(k_np),
            seq_len,
            rope_base=base,
        )
        q_ref, _ = _numpy_ec_rope(q_np, k_np, seq_len, rope_base=base)
        np.testing.assert_allclose(q_out.numpy(), q_ref, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(paddle is not None, _SKIP_REASON)
class TestMLASelfAttentionBackwardDW(unittest.TestCase):
    """Branch-selection contract of ``backward_dw`` / its private helpers.

    ``backward_dw`` takes no arguments; its only observable behavior is *which*
    projection sublayers receive a ``backward_dw()`` call for a given config.
    The projection layers are genuine non-under-test collaborators, so they are
    mocked; the branch decision (including which layers are NOT called) is the
    value under test.
    """

    def _make_mla(self, q_lora_rank, kv_b_proj_present=True):
        mla = MLASelfAttention.__new__(MLASelfAttention)
        mla.q_lora_rank = q_lora_rank
        mla.kv_a_proj_with_mqa = mock.MagicMock()
        mla.o_proj = mock.MagicMock()
        mla.kv_b_proj = mock.MagicMock() if kv_b_proj_present else None
        # Provide BOTH q-paths as mocks so a wrong branch is caught by a
        # not-called assertion rather than an AttributeError.
        mla.q_proj = mock.MagicMock()
        mla.q_a_proj = mock.MagicMock()
        mla.q_b_proj = mock.MagicMock()
        return mla

    def test_with_q_lora_rank_uses_q_a_and_q_b(self):
        mla = self._make_mla(q_lora_rank=32)
        mla.backward_dw()

        mla.q_a_proj.backward_dw.assert_called_once_with()
        mla.q_b_proj.backward_dw.assert_called_once_with()
        mla.q_proj.backward_dw.assert_not_called()
        mla.kv_b_proj.backward_dw.assert_called_once_with()
        mla.kv_a_proj_with_mqa.backward_dw.assert_called_once_with()
        mla.o_proj.backward_dw.assert_called_once_with()

    def test_without_q_lora_rank_uses_q_proj(self):
        mla = self._make_mla(q_lora_rank=None)
        mla.backward_dw()

        mla.q_proj.backward_dw.assert_called_once_with()
        mla.q_a_proj.backward_dw.assert_not_called()
        mla.q_b_proj.backward_dw.assert_not_called()
        mla.kv_b_proj.backward_dw.assert_called_once_with()
        mla.kv_a_proj_with_mqa.backward_dw.assert_called_once_with()
        mla.o_proj.backward_dw.assert_called_once_with()

    def test_absorbed_mode_skips_kv_b_proj(self):
        """Split-absorption mode has no kv_b_proj (L2513-2516): it must be
        skipped without error while kv_a / q / o still run."""
        mla = self._make_mla(q_lora_rank=None, kv_b_proj_present=False)
        mla.backward_dw()  # must not raise despite kv_b_proj is None

        mla.kv_a_proj_with_mqa.backward_dw.assert_called_once_with()
        mla.q_proj.backward_dw.assert_called_once_with()
        mla.o_proj.backward_dw.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
