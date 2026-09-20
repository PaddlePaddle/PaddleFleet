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
"""Behavior tests for paddlefleet.transformer.attention.

Focus is on the CPU-executable numeric surface referenced by the coverage
source: the EC-style complex 3D MRoPE rotation, the gated-attention apply,
the base-class recompute contract, and the VHA premix/postmix transforms.
Every numeric expectation is derived from an INDEPENDENT reference (rotation
formulation / explicit einsum / M = I + V U^T), never by calling the
production function under test.

Heavy imports (paddle + the attention module) are guarded so that a missing
runtime is reported honestly as a skip; only ImportError/ModuleNotFoundError
is treated as "dependency absent" so that real API breaks still surface.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.attention import (
        Attention,
        CrossAttentionSublayersSpec,
        SelfAttentionSublayersSpec,
        SelfAttentionVHA,
        SelfAttentionVHASublayersSpec,
        _apply_ec_complex_3d_mrope,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle/paddlefleet/numpy not importable in this environment: "
    f"{_IMPORT_ERROR}"
    if _IMPORT_ERROR is not None
    else ""
)


def _reference_ec_mrope(
    query, key, position_ids, head_dim, rope_theta, mrope_section
):
    """Independent numpy reference for _apply_ec_complex_3d_mrope.

    Implements the rotation directly (real cos/sin form) instead of the
    production complex-polar multiply, so a common bug would not hide on both
    sides. Also reproduces the position_ids incremental padding contract.
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    pos = np.asarray(position_ids).astype(np.float64)
    seq_q = q.shape[1]
    seq_p = pos.shape[1]
    if seq_p < seq_q:
        last = pos[:, -1:, :]
        pads = [last + (i + 1) for i in range(seq_q - seq_p)]
        pos = np.concatenate([pos, *pads], axis=1)

    point_num = head_dim // 2
    axis = []
    for idx, n in enumerate(mrope_section[1:]):
        axis += [idx + 1] * n
    repeat = (point_num - mrope_section[0]) // sum(mrope_section[1:])
    axis = axis * repeat + [0] * mrope_section[0]
    axis = np.array(axis, dtype=np.int64)
    assert axis.shape[0] == point_num

    selected = pos[:, :, axis]  # [B, S, point_num]
    comp = np.arange(point_num)
    inv_freq = rope_theta ** (-(2.0 * comp) / head_dim)  # [point_num]
    angle = selected * inv_freq[None, None, :]  # [B, S, point_num]
    cos = np.cos(angle)[:, :, None, :]  # [B, S, 1, point_num]
    sin = np.sin(angle)[:, :, None, :]

    def rot(x):
        b, s, h, d = x.shape
        xr = x.reshape(b, s, h, d // 2, 2)
        even = xr[..., 0]
        odd = xr[..., 1]
        out = np.stack(
            [even * cos - odd * sin, even * sin + odd * cos], axis=-1
        )
        return out.reshape(b, s, h, d)

    return rot(q), rot(k)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestApplyEcComplex3dMrope(unittest.TestCase):
    """Numeric behavior of _apply_ec_complex_3d_mrope on CPU."""

    def setUp(self):
        paddle.set_device("cpu")

    def _fixed_qk(self, batch, seq, heads, head_dim):
        # Distinguishable, non-degenerate q/k so pair swaps / axis mixups show.
        n = batch * seq * heads * head_dim
        base = (
            np.arange(n, dtype=np.float32).reshape(batch, seq, heads, head_dim)
            % 7
        ) - 3.0
        q = base + 0.25
        k = -base * 0.5 + 1.0
        return q.astype(np.float32), k.astype(np.float32)

    def _distinct_positions(self, batch, seq):
        # Each of the 3 MRoPE axes gets a distinct value per position so that
        # a wrong axis selection cannot reproduce the reference angles.
        pos = np.zeros((batch, seq, 3), dtype=np.int64)
        for b in range(batch):
            for s in range(seq):
                pos[b, s, 0] = s + 1
                pos[b, s, 1] = s + 11
                pos[b, s, 2] = s + 21
        return pos

    def test_rotation_matches_independent_reference(self):
        batch, seq, heads, head_dim = 1, 3, 2, 8
        mrope_section = [2, 1, 1]
        rope_theta = 10000.0  # non-default: catches ignored rope_theta
        q_np, k_np = self._fixed_qk(batch, seq, heads, head_dim)
        pos_np = self._distinct_positions(batch, seq)

        q_out, k_out = _apply_ec_complex_3d_mrope(
            paddle.to_tensor(q_np),
            paddle.to_tensor(k_np),
            paddle.to_tensor(pos_np),
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )
        ref_q, ref_k = _reference_ec_mrope(
            q_np, k_np, pos_np, head_dim, rope_theta, mrope_section
        )
        self.assertEqual(list(q_out.shape), [batch, seq, heads, head_dim])
        self.assertEqual(list(k_out.shape), [batch, seq, heads, head_dim])
        np.testing.assert_allclose(q_out.numpy(), ref_q, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(k_out.numpy(), ref_k, rtol=1e-5, atol=1e-5)
        # Sanity: rotation must actually change the inputs (angles non-zero
        # because positions are non-zero), otherwise an identity bug passes.
        self.assertGreater(float(np.abs(q_out.numpy() - q_np).max()), 1e-3)

    def test_axis_selection_is_position_axis_specific(self):
        # Reference selects axis order [1, 2, 0, 0] for mrope_section=[2,1,1]
        # with point_num=4. Feed positions where only one axis is non-zero to
        # prove the production picks that exact axis per frequency component.
        batch, seq, heads, head_dim = 1, 2, 1, 8
        mrope_section = [2, 1, 1]
        rope_theta = 10000.0
        q_np, k_np = self._fixed_qk(batch, seq, heads, head_dim)

        # Only axis 1 carries a position; axes 0 and 2 are zero.
        pos_np = np.zeros((batch, seq, 3), dtype=np.int64)
        pos_np[0, 0, 1] = 5
        pos_np[0, 1, 1] = 9

        q_out, _ = _apply_ec_complex_3d_mrope(
            paddle.to_tensor(q_np),
            paddle.to_tensor(k_np),
            paddle.to_tensor(pos_np),
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )
        ref_q, _ = _reference_ec_mrope(
            q_np, k_np, pos_np, head_dim, rope_theta, mrope_section
        )
        np.testing.assert_allclose(q_out.numpy(), ref_q, rtol=1e-5, atol=1e-5)

        # Only frequency component 0 (which maps to axis 1) should rotate;
        # components mapped to the zero axes must be untouched.
        out = q_out.numpy()[0, 0, 0]
        # component 0 -> pair (0,1) rotated; components 2,3 -> pairs (4..7)
        # use axis 0 (zero) so must equal the input pairs exactly.
        np.testing.assert_allclose(out[4:8], q_np[0, 0, 0, 4:8], atol=1e-6)
        self.assertGreater(abs(out[0] - q_np[0, 0, 0, 0]), 1e-3)

    def test_position_ids_padding_extends_incrementally(self):
        # position_ids shorter than query: production pads with
        # last_pos + (i + 1) along every axis. Verify by comparing to a call
        # that supplies the already-extended positions explicitly (no padding
        # branch), and to the independent reference.
        batch, seq, heads, head_dim = 1, 6, 2, 8
        mrope_section = [2, 1, 1]
        rope_theta = 1000000.0  # default value
        q_np, k_np = self._fixed_qk(batch, seq, heads, head_dim)

        short = self._distinct_positions(batch, seq)[:, : seq - 2, :].copy()
        # Build the manually-extended positions the padding logic should form.
        extended = short.copy()
        last = short[:, -1:, :]
        for i in range(2):
            extended = np.concatenate([extended, last + (i + 1)], axis=1)

        q_pad, k_pad = _apply_ec_complex_3d_mrope(
            paddle.to_tensor(q_np),
            paddle.to_tensor(k_np),
            paddle.to_tensor(short),
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )
        q_full, k_full = _apply_ec_complex_3d_mrope(
            paddle.to_tensor(q_np),
            paddle.to_tensor(k_np),
            paddle.to_tensor(extended),
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )
        ref_q, ref_k = _reference_ec_mrope(
            q_np, k_np, short, head_dim, rope_theta, mrope_section
        )
        np.testing.assert_allclose(q_pad.numpy(), q_full.numpy(), atol=1e-6)
        np.testing.assert_allclose(k_pad.numpy(), k_full.numpy(), atol=1e-6)
        np.testing.assert_allclose(q_pad.numpy(), ref_q, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(k_pad.numpy(), ref_k, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGateApply(unittest.TestCase):
    """Numeric behavior of Attention._gate_apply (sigmoid gating)."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_gate_apply_equals_sigmoid_scaled(self):
        core_np = np.array([[[1.0, -2.0, 3.0, -4.0]]], dtype=np.float32)
        gate_np = np.array([[[0.0, 1.0, -1.0, 2.0]]], dtype=np.float32)
        # _gate_apply does not use self; call on the class with self=None.
        out = Attention._gate_apply(
            None, paddle.to_tensor(core_np), paddle.to_tensor(gate_np)
        )
        expected = core_np * (1.0 / (1.0 + np.exp(-gate_np)))
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)
        # gate=0 -> factor 0.5 exactly; ensures the gate is truly consumed.
        self.assertAlmostEqual(float(out.numpy()[0, 0, 0]), 0.5, places=6)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestRecomputeContract(unittest.TestCase):
    """Base Attention.set_for_recompute_input_layernorm contract."""

    def test_base_raises_not_implemented(self):
        # Real exception contract; not swallowed. self is unused by the method.
        with self.assertRaises(NotImplementedError):
            Attention.set_for_recompute_input_layernorm(None)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestVHATransforms(unittest.TestCase):
    """Numeric behavior of VHA premix/postmix transforms.

    A bare instance is built with __new__ and the exact data attributes each
    method reads (weights + dims) are set directly. This bypasses only the
    heavy layer construction; the premix einsum expansion and postmix
    M = I + V U^T GEMM under test run for real and are checked against an
    independent numpy reference.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_premix_expands_kv_groups_to_heads(self):
        batch, seq, groups, q_head_dim, head_dim, nkv = 2, 3, 2, 4, 5, 3
        num_heads = nkv * groups
        obj = SelfAttentionVHA.__new__(SelfAttentionVHA)
        obj.num_attention_heads = num_heads
        obj.head_dim = head_dim

        w_np = (
            np.arange(nkv * q_head_dim * head_dim, dtype=np.float32).reshape(
                nkv, q_head_dim, head_dim
            )
            % 5
            - 2.0
        ).astype(np.float32)
        q_np = (
            np.arange(
                batch * seq * groups * q_head_dim, dtype=np.float32
            ).reshape(batch, seq, groups, q_head_dim)
            % 6
            - 2.5
        ).astype(np.float32)
        obj.vha_premix_weight = paddle.to_tensor(w_np)

        out = obj._apply_vha_premix(paddle.to_tensor(q_np))

        # Independent reference: out[b,t,k,g,d] = sum_r q[b,t,g,r]*w[k,r,d],
        # then head index h = k*groups + g.
        ref5 = np.einsum("btgr,krd->btkgd", q_np, w_np)
        ref = ref5.reshape(batch, seq, num_heads, head_dim)
        self.assertEqual(list(out.shape), [batch, seq, num_heads, head_dim])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)

    def test_postmix_applies_identity_plus_lowrank(self):
        batch, seq, nh, d, rank = 2, 2, 4, 3, 2
        obj = SelfAttentionVHA.__new__(SelfAttentionVHA)
        obj.num_attention_heads = nh
        obj.v_head_dim = d

        u_np = (
            np.arange(nh * rank, dtype=np.float32).reshape(nh, rank) % 3 - 1.0
        )
        v_np = (
            np.arange(nh * rank, dtype=np.float32).reshape(nh, rank) % 4 - 1.5
        )
        u_np = u_np.astype(np.float32)
        v_np = v_np.astype(np.float32)
        attn_np = (
            np.arange(batch * seq * nh * d, dtype=np.float32).reshape(
                batch, seq, nh * d
            )
            % 7
            - 3.0
        ).astype(np.float32)
        obj.vha_postmix_U = paddle.to_tensor(u_np)
        obj.vha_postmix_V = paddle.to_tensor(v_np)

        out = obj._apply_vha_postmix(paddle.to_tensor(attn_np))

        # Independent reference: M = V @ U^T + I; out = M @ mixed per token.
        mixed = attn_np.reshape(batch * seq, nh, d)
        M = v_np @ u_np.T + np.eye(nh, dtype=np.float32)
        ref = np.matmul(M[None, :, :], mixed).reshape(batch, seq, nh * d)
        self.assertEqual(list(out.shape), [batch, seq, nh * d])
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-5)

    def test_postmix_is_identity_when_v_zero(self):
        # V = 0 => M = I => output equals input exactly, independent of U.
        batch, seq, nh, d, rank = 1, 2, 4, 3, 2
        obj = SelfAttentionVHA.__new__(SelfAttentionVHA)
        obj.num_attention_heads = nh
        obj.v_head_dim = d
        u_np = np.linspace(-1.0, 1.0, nh * rank, dtype=np.float32).reshape(
            nh, rank
        )
        attn_np = (
            np.arange(batch * seq * nh * d, dtype=np.float32).reshape(
                batch, seq, nh * d
            )
            - 5.0
        ).astype(np.float32)
        obj.vha_postmix_U = paddle.to_tensor(u_np)
        obj.vha_postmix_V = paddle.to_tensor(np.zeros((nh, rank), np.float32))

        out = obj._apply_vha_postmix(paddle.to_tensor(attn_np))
        np.testing.assert_allclose(out.numpy(), attn_np, rtol=1e-6, atol=1e-6)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSublayerSpecDefaults(unittest.TestCase):
    """Dataclass default-value contracts for the sublayer specs."""

    def test_self_attention_spec_defaults_are_none(self):
        spec = SelfAttentionSublayersSpec()
        self.assertEqual(
            [
                spec.qkv_proj,
                spec.core_attention,
                spec.o_proj,
                spec.q_norm,
                spec.k_norm,
                spec.gate_proj,
            ],
            [None] * 6,
        )
        # Assignment reaches the field (guards against slots / bad defaults).
        marker = object()
        spec.core_attention = marker
        self.assertIs(spec.core_attention, marker)

    def test_cross_attention_spec_defaults_are_none(self):
        spec = CrossAttentionSublayersSpec()
        self.assertEqual(
            [spec.linear_q, spec.linear_kv, spec.core_attention, spec.o_proj],
            [None] * 4,
        )

    def test_vha_spec_defaults_are_none(self):
        spec = SelfAttentionVHASublayersSpec()
        self.assertEqual(
            [
                spec.q_proj,
                spec.k_proj,
                spec.v_proj,
                spec.shared_kv_proj,
                spec.gate_proj,
                spec.qkv_proj,
                spec.core_attention,
                spec.o_proj,
                spec.q_norm,
                spec.k_norm,
            ],
            [None] * 10,
        )


if __name__ == "__main__":
    unittest.main()
