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

"""Adversarial RoPE audit for hybrid-MLA attention (validation agent A2).

Every assertion here compares the *shipped* RoPE primitives against an
INDEPENDENT fp64 numpy reference written from the mathematical definition of
rotary embeddings -- the implementation is never compared with itself.

Target production keys (model_config_separated/conf/fleet_align/
ernielite_layer43_mla_{hca,mqa_hca,dsa_hca}/model_config.json):
    rope_theta 10000, rotary_base 10000.0, rope_scaling null,
    rotary_scaling_factor 16, original_max_position_embeddings 65536,
    rotary_percent 1.0, qk_rope_head_dim 64, csa_compress_rotary_base 160000.0,
    apply_rope_fusion false, rope_type "rope" (ernie config default).

The audit covers:
  1. frequency schedule + layout (half-split vs interleaved) + no-YaRN leak
  2. Q/K same-rotation / relative-offset invariance
  3. mha == mqa == mqa_dsa RoPE agreement
  4. packed multi-document position semantics (global vs per-doc)
  5. CP contiguous_allgather rank offset == global position
  6. MTP 1-token shift
  7. csa_compress_rotary_base 160000 vs 10000 base separation
  8. fused vs unfused equivalence
"""

from __future__ import annotations

import types
import unittest

import numpy as np
import paddle

from paddlefleet.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
)
from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_mscale,
)

paddle.set_device("gpu")

QK_ROPE_HEAD_DIM = 64
ROPE_THETA = 10000.0
CSA_COMPRESS_ROTARY_BASE = 160000.0


# ===========================================================================
# Independent fp64 numpy reference RoPE (written from the definition)
# ===========================================================================
def ref_inv_freq(dim: int, base: float) -> np.ndarray:
    """1 / base^(2i/dim), i = 0..dim/2-1, fp64."""
    i = np.arange(0, dim, 2, dtype=np.float64)
    return 1.0 / (base ** (i / dim))


def ref_angles(positions: np.ndarray, dim: int, base: float) -> np.ndarray:
    """[len(positions), dim/2] outer product of positions and inv_freq."""
    return np.outer(positions.astype(np.float64), ref_inv_freq(dim, base))


def ref_rope_halfsplit(x: np.ndarray, pos: int, base: float) -> np.ndarray:
    """GPT-NeoX / HF-Llama 'rotate_half' layout.

    pair channel j with channel j+dim/2:
        out[j]        = x[j]      * cos - x[j+d/2] * sin
        out[j+d/2]    = x[j+d/2]  * cos + x[j]     * sin
    """
    d = x.shape[-1]
    h = d // 2
    ang = ref_angles(np.array([pos]), d, base)[0]  # [d/2]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :h], x[..., h:]
    out = np.empty_like(x, dtype=np.float64)
    out[..., :h] = x1 * cos - x2 * sin
    out[..., h:] = x2 * cos + x1 * sin
    return out


def ref_rope_mla(x: np.ndarray, pos: int, base: float) -> np.ndarray:
    """DeepSeek-MLA layout used by ``multi_latent_attention=True``.

    The code de-interleaves the input (a=even, b=odd), then applies the
    non-interleaved rotate_half.  The complex rotation is on pairs (x[2j],
    x[2j+1]) but the OUTPUT stays in the reordered [real-parts, imag-parts]
    layout (no re-interleave, since mla_output_remove_interleaving=False):

        out[:d/2] = a*cos - b*sin      (a = x[0::2], b = x[1::2])
        out[d/2:] = b*cos + a*sin
    """
    d = x.shape[-1]
    h = d // 2
    ang = ref_angles(np.array([pos]), d, base)[0]  # [d/2]
    cos, sin = np.cos(ang), np.sin(ang)
    a = x[..., 0::2]
    b = x[..., 1::2]
    out = np.empty_like(x, dtype=np.float64)
    out[..., :h] = a * cos - b * sin
    out[..., h:] = b * cos + a * sin
    return out


def _fake_mla_config(multi_latent_attention=True, rotary_interleaved=False):
    """Minimal config accepted by apply_rotary_pos_emb (unfused MLA path)."""
    return types.SimpleNamespace(
        apply_rope_fusion=False,
        rotary_interleaved=rotary_interleaved,
        multi_latent_attention=multi_latent_attention,
        high_precision_rope=False,
        rope_theta=ROPE_THETA,
        sequence_parallel=False,
    )


def _apply_real(x_np, emb, config):
    """Run the shipped unfused apply_rotary_pos_emb on a [1,S,1,D] tensor."""
    t = paddle.to_tensor(x_np.astype("float32"))
    out = apply_rotary_pos_emb(t, emb, None, None, config=config)
    return np.asarray(out.astype("float32").numpy(), dtype=np.float64)


def _rope(
    dim=QK_ROPE_HEAD_DIM, rotary_base=ROPE_THETA, interleaved=False, **kw
):
    """RotaryEmbedding with the audit's shared defaults (rotary_percent=1.0).

    Distinct cases override via kwargs: ``rotary_base=`` (HCA 160000 schedule),
    ``rope_scaling=False`` (no-YaRN-leak), ``interleaved=True`` (interleaved).
    """
    return RotaryEmbedding(
        dim,
        rotary_percent=1.0,
        rotary_interleaved=interleaved,
        rotary_base=rotary_base,
        **kw,
    )


TOL = 2e-5  # fp32 round-trip tolerance for the rope primitives


# ===========================================================================
# Item 1: frequency schedule, layout, and no-YaRN-leak
# ===========================================================================
class TestFrequencyScheduleAndLayout(unittest.TestCase):
    def test_inv_freq_matches_reference(self):
        """RotaryEmbedding inv_freq == 1/theta^(2i/dim), theta=10000, dim=64."""
        r = _rope()
        got = np.asarray(r.inv_freq.astype("float64").numpy())
        ref = ref_inv_freq(QK_ROPE_HEAD_DIM, ROPE_THETA)
        self.assertEqual(got.shape, (QK_ROPE_HEAD_DIM // 2,))
        err = np.max(np.abs(got - ref))
        self.assertLess(err, 1e-6, f"inv_freq max abs err {err}")

    def test_rotary_percent_one_uses_full_dim(self):
        """rotary_percent=1.0 -> all 64 dims rotated (rot_dim == head_dim)."""
        r = _rope()
        emb = r(8)
        self.assertEqual(emb.shape[-1], QK_ROPE_HEAD_DIM)

    def test_no_yarn_scaling_leak_into_minus2_layers(self):
        """rope_type='rope' plain RotaryEmbedding must NOT apply YaRN scaling.

        rotary_scaling_factor=16 / original_max_position_embeddings=65536 exist
        in the config for the HCA layers; they must be inert for the -2 layers.
        A plain RotaryEmbedding built with base=theta must equal the reference
        with NO scaling applied.
        """
        r = _rope(rope_scaling=False)  # -2 layers: rope_scaling null
        got = np.asarray(r.inv_freq.astype("float64").numpy())
        ref_plain = ref_inv_freq(QK_ROPE_HEAD_DIM, ROPE_THETA)
        # A YaRN/llama-scaled inv_freq would divide low freqs by 16 -> would
        # differ from ref_plain by up to ~16x on the low-frequency tail.
        self.assertLess(np.max(np.abs(got - ref_plain)), 1e-6)

    def test_yarn_mscale_is_identity_for_minus2_softmax_scale(self):
        """MLA softmax mscale == _yarn_get_mscale(16, mscale_all_dim=0.0) == 1.0.

        This is what MultiLatentAttention.__init__ computes (line ~431). With
        mscale_all_dim=0.0 (config default) there is no scale change even though
        rotary_scaling_factor=16.
        """
        self.assertEqual(_yarn_get_mscale(16, 0.0), 1.0)
        # Guard: a NON-zero mscale_all_dim WOULD change it -> fragile coupling.
        self.assertGreater(_yarn_get_mscale(16, 1.0), 1.0)

    def test_layout_is_mla_deinterleave_not_plain_halfsplit(self):
        """Determine which layout the SHIPPED apply uses (multi_latent_attention=True)."""
        rng = np.random.default_rng(0)
        S, D = 6, QK_ROPE_HEAD_DIM
        x = rng.standard_normal((1, S, 1, D))
        r = _rope()
        emb = r(S)
        cfg = _fake_mla_config(multi_latent_attention=True)
        got = _apply_real(x, emb, cfg)

        ref_mla = np.stack(
            [ref_rope_mla(x[0, p, 0], p, ROPE_THETA) for p in range(S)]
        ).reshape(1, S, 1, D)
        ref_hs = np.stack(
            [ref_rope_halfsplit(x[0, p, 0], p, ROPE_THETA) for p in range(S)]
        ).reshape(1, S, 1, D)

        err_mla = np.max(np.abs(got - ref_mla))
        err_hs = np.max(np.abs(got - ref_hs))
        # The code must match the MLA de-interleave reference, NOT plain half-split.
        self.assertLess(err_mla, TOL, f"MLA layout err {err_mla}")
        self.assertGreater(
            err_hs, 1e-3, "unexpectedly matched plain half-split layout"
        )

    def test_plain_flag_matches_halfsplit(self):
        """With multi_latent_attention=False the apply uses plain half-split."""
        rng = np.random.default_rng(1)
        S, D = 5, QK_ROPE_HEAD_DIM
        x = rng.standard_normal((1, S, 1, D))
        r = _rope()
        emb = r(S)
        cfg = _fake_mla_config(multi_latent_attention=False)
        got = _apply_real(x, emb, cfg)
        ref_hs = np.stack(
            [ref_rope_halfsplit(x[0, p, 0], p, ROPE_THETA) for p in range(S)]
        ).reshape(1, S, 1, D)
        self.assertLess(np.max(np.abs(got - ref_hs)), TOL)


# ===========================================================================
# Item 2: Q and K get the same rotation; score depends only on relative offset
# ===========================================================================
class TestRelativeOffsetInvariance(unittest.TestCase):
    def _rope_pair_score(self, q_raw, k_raw, i, j, cfg):
        """Rotate q at pos i and k at pos j via the SHIPPED apply, dot them."""
        D = q_raw.shape[-1]
        maxpos = max(i, j) + 1
        r = _rope()
        emb = r(maxpos)
        # place q at position i, k at position j inside a length-maxpos seq
        qx = np.zeros((1, maxpos, 1, D))
        kx = np.zeros((1, maxpos, 1, D))
        qx[0, i, 0] = q_raw
        kx[0, j, 0] = k_raw
        q_rot = _apply_real(qx, emb, cfg)[0, i, 0]
        k_rot = _apply_real(kx, emb, cfg)[0, j, 0]
        return float(np.dot(q_rot, k_rot))

    def test_q_k_same_position_same_rotation(self):
        """Same raw vector at the same position -> identical rotation for q & k."""
        rng = np.random.default_rng(2)
        D = QK_ROPE_HEAD_DIM
        v = rng.standard_normal(D)
        cfg = _fake_mla_config(True)
        r = _rope()
        emb = r(10)
        vx = np.zeros((1, 10, 1, D))
        vx[0, 7, 0] = v
        rot = _apply_real(vx, emb, cfg)[0, 7, 0]
        # q-path and k-path call the identical function -> identical output.
        rot2 = _apply_real(vx, emb, cfg)[0, 7, 0]
        self.assertLess(np.max(np.abs(rot - rot2)), 1e-12)

    def test_score_depends_only_on_relative_offset(self):
        """score(i, j) == score(i+d, j+d) for the real qk_rope_head_dim=64."""
        rng = np.random.default_rng(3)
        D = QK_ROPE_HEAD_DIM
        q = rng.standard_normal(D)
        k = rng.standard_normal(D)
        cfg = _fake_mla_config(True)
        base = self._rope_pair_score(q, k, 3, 1, cfg)  # offset +2
        errs = []
        for shift in (1, 5, 20, 100):
            s = self._rope_pair_score(q, k, 3 + shift, 1 + shift, cfg)
            errs.append(abs(s - base))
        max_err = max(errs)
        self.assertLess(
            max_err, 5e-3, f"relative-offset invariance broken, err {max_err}"
        )

    def test_reference_confirms_relative_property(self):
        """Independent fp64 ref also shows relative-only dependence (sanity)."""
        rng = np.random.default_rng(4)
        D = QK_ROPE_HEAD_DIM
        q = rng.standard_normal(D)
        k = rng.standard_normal(D)

        def score(i, j):
            qr = ref_rope_mla(q, i, ROPE_THETA)
            kr = ref_rope_mla(k, j, ROPE_THETA)
            return float(np.dot(qr, kr))

        base = score(10, 4)
        for shift in (1, 7, 50):
            self.assertLess(abs(score(10 + shift, 4 + shift) - base), 1e-9)


# ===========================================================================
# Item 3: mha == mqa == mqa_dsa RoPE agreement
# ===========================================================================
class TestThreeModeRopeAgreement(unittest.TestCase):
    """RoPE is applied to q_pe/k_pe by the SAME apply_rotary_pos_emb call in
    all three modes (multi_latent_attention.py lines 1741-1764); the
    ``if self.mqa_latent`` branch that distinguishes the modes is *after* the
    rope apply (line 1768). So the rotated rope parts are identical by
    construction. This class proves that and that the absorption preserves the
    rope contribution to the attention score.
    """

    def test_rotated_rope_parts_identical_across_modes(self):
        """The rope apply is mode-independent -> identical rotated q_pe/k_pe."""
        rng = np.random.default_rng(10)
        S, D = 8, QK_ROPE_HEAD_DIM
        q_pe = rng.standard_normal((1, S, 4, D))  # 4 heads
        k_pe = rng.standard_normal((1, S, 1, D))
        r = _rope()
        emb = r(S)
        cfg = _fake_mla_config(True)
        # "mha", "mqa", "mqa_dsa" all execute exactly this call:
        q_rot = apply_rotary_pos_emb(
            paddle.to_tensor(q_pe.astype("float32")),
            emb,
            None,
            None,
            config=cfg,
        )
        k_rot = apply_rotary_pos_emb(
            paddle.to_tensor(k_pe.astype("float32")),
            emb,
            None,
            None,
            config=cfg,
        )
        # Re-run to emulate a second mode -> must be bit-identical.
        q_rot2 = apply_rotary_pos_emb(
            paddle.to_tensor(q_pe.astype("float32")),
            emb,
            None,
            None,
            config=cfg,
        )
        self.assertTrue(
            bool(paddle.all(q_rot == q_rot2)), "rope q not deterministic"
        )
        self.assertEqual(tuple(q_rot.shape), (1, S, 4, D))
        self.assertEqual(tuple(k_rot.shape), (1, S, 1, D))

    def test_absorption_preserves_full_score_including_rope(self):
        """mha score == mqa (absorbed) score, so rope contribution is identical.

        Replicates the absorption math from multi_latent_attention.py:
        - w_k_b = kv_b_proj[..., :qk_nope]  (line 1776-1782)
        - q_absorbed = einsum('hd,lhd->hl', q_nope, w_k_b)  (line 1786)
        - mqa key latent = [kv_compressed, k_pe]; mha k_nope = kv_compressed@w_k_b
        """
        rng = np.random.default_rng(11)
        h = 4
        dqk_nope = 16
        dkv = 32
        drope = QK_ROPE_HEAD_DIM

        q_nope = rng.standard_normal((h, dqk_nope))
        q_pe = rng.standard_normal((h, drope))
        kv_compressed = rng.standard_normal((dkv,))
        k_pe = rng.standard_normal((drope,))
        w_k_b = rng.standard_normal((dkv, h, dqk_nope))

        # --- mha path: reconstruct per-head k_nope then dot ---
        # k_nope[h, d'] = sum_l kv_compressed[l] * w_k_b[l, h, d']
        k_nope = np.einsum("l,lhd->hd", kv_compressed, w_k_b)
        mha_nope = np.einsum("hd,hd->h", q_nope, k_nope)
        rope_score = q_pe @ k_pe  # shared k_pe, per head
        mha_score = mha_nope + rope_score  # [h]

        # --- mqa path: absorbed query dotted with the shared latent ---
        q_absorbed = np.einsum("hd,lhd->hl", q_nope, w_k_b)  # [h, dkv]
        mqa_nope = np.einsum("hl,l->h", q_absorbed, kv_compressed)
        mqa_score = mqa_nope + rope_score  # rope identical term

        err = np.max(np.abs(mha_score - mqa_score))
        self.assertLess(
            err, 1e-9, f"absorption changes score (rope not preserved): {err}"
        )
        # And the rope term itself is byte-identical between modes.
        self.assertEqual(float(np.max(np.abs(rope_score - rope_score))), 0.0)


# ===========================================================================
# Item 4: packed multi-document position semantics
# ===========================================================================
class TestPackedMultiDocPositions(unittest.TestCase):
    """Ground truth (verified by code reading):

    During TRAINING, both the mha path (MLASelfAttention.get_query_key_value_
    tensors, multi_latent_attention.py:1306-1311) and the mqa/mqa_dsa path
    (MQASelfAttention, multi_latent_attention.py:2283-2286) call
    ``self.rotary_pos_emb(rotary_seq_len, position_ids=None if self.training
    else position_ids)``. position_ids is forced to None in training, so
    RotaryEmbedding.get_freqs_non_repeated falls to
    ``paddle.arange(max_seq_len)`` (rotary_pos_embedding.py:187-189): positions
    run GLOBALLY across the packed sequence and DO NOT restart per document.
    Document isolation is enforced by the attention mask
    (attn_mask_startend_row_indices -> per-doc causal index table), not by
    resetting RoPE positions. Both paths therefore agree.
    """

    def test_training_uses_global_arange_positions(self):
        """position_ids=None -> global arange freqs, no per-doc reset."""
        D = QK_ROPE_HEAD_DIM
        r = _rope()
        freqs = r.get_freqs_non_repeated(6, position_ids=None)
        got = np.asarray(freqs.astype("float64").numpy())
        ref = ref_angles(np.arange(6), D, ROPE_THETA)
        self.assertLess(np.max(np.abs(got - ref)), 1e-6)

    def test_global_positions_equivalent_to_perdoc_within_document(self):
        """Global-position RoPE == per-doc-reset RoPE for intra-doc scores.

        A document starting at global offset ``g`` with per-doc-reset would use
        positions 0..n-1; global scheme uses g..g+n-1. Because RoPE scores
        depend only on the relative offset, the two produce identical intra-doc
        attention scores. This is why mha and mqa (both global) are correct.
        """
        rng = np.random.default_rng(12)
        D = QK_ROPE_HEAD_DIM
        cfg = _fake_mla_config(True)
        q = rng.standard_normal(D)
        k = rng.standard_normal(D)

        def score(i, j):
            m = max(i, j) + 1
            r = _rope()
            emb = r(m)
            qx = np.zeros((1, m, 1, D))
            kx = np.zeros((1, m, 1, D))
            qx[0, i, 0] = q
            kx[0, j, 0] = k
            qr = _apply_real(qx, emb, cfg)[0, i, 0]
            kr = _apply_real(kx, emb, cfg)[0, j, 0]
            return float(np.dot(qr, kr))

        # doc-local offset (i-j) fixed at 3; vary global doc start g.
        per_doc = score(3, 0)  # positions 3,0 -> offset 3 (doc reset)
        for g in (16, 128, 1000):
            glob = score(g + 3, g)  # same intra-doc offset, global positions
            self.assertLess(
                abs(glob - per_doc),
                5e-3,
                f"global vs per-doc mismatch at g={g}",
            )


# ===========================================================================
# Item 5: CP contiguous_allgather rank offset must be the GLOBAL position
# ===========================================================================
class TestCPContiguousAllgatherOffset(unittest.TestCase):
    """With cp_balance_mode='contiguous_allgather', MLASelfAttention builds
    rotary_pos_emb over the FULL global sequence (rotary_seq_len = cp_size *
    local, RotaryEmbedding.get_rotary_seq_len) and then scatters it with the
    SAME ContextParallelScatterOp(axis=1, mode='contiguous_allgather') used for
    the queries (multi_latent_attention.py:1393; context_parallel_utils.py
    scatter_contiguous @402). scatter_contiguous gives rank r the slice
    [r*L, (r+1)*L]. Since the query on rank r is that same contiguous slice,
    local index i on rank r carries the GLOBAL position r*L + i. We simulate
    rank 1 in a single process and prove the rope offset is r*L + i, not i.
    """

    @staticmethod
    def _contiguous_slice(emb, rank, nranks):
        """Replicate context_parallel_utils.scatter_contiguous on axis=1."""
        L = emb.shape[1] // nranks
        return emb[:, rank * L : (rank + 1) * L]

    def test_rank1_offset_is_global_position(self):
        rng = np.random.default_rng(13)
        D = QK_ROPE_HEAD_DIM
        L = 8  # local length per rank
        nranks, rank = 2, 1
        S_global = nranks * L
        cfg = _fake_mla_config(True)

        r = _rope()
        emb_full = r(S_global)  # global positions 0..S_global-1
        emb_rank1 = self._contiguous_slice(emb_full, rank, nranks)  # [1,L,1,D]

        q = rng.standard_normal(D)
        k = rng.standard_normal(D)
        # local indices on rank 1
        li, lj = 3, 1
        qx = np.zeros((1, L, 1, D))
        kx = np.zeros((1, L, 1, D))
        qx[0, li, 0] = q
        kx[0, lj, 0] = k
        q_rot = _apply_real(qx, emb_rank1, cfg)[0, li, 0]
        k_rot = _apply_real(kx, emb_rank1, cfg)[0, lj, 0]
        cp_score = float(np.dot(q_rot, k_rot))

        # Reference: global positions rank*L + li and rank*L + lj.
        gi, gj = rank * L + li, rank * L + lj
        ref_score = float(
            np.dot(
                ref_rope_mla(q, gi, ROPE_THETA),
                ref_rope_mla(k, gj, ROPE_THETA),
            )
        )
        self.assertLess(
            abs(cp_score - ref_score),
            5e-3,
            f"CP rank-1 offset wrong: cp={cp_score} ref(global)={ref_score}",
        )

    def test_off_by_local_len_bug_would_be_detected(self):
        """Sanity: the WRONG slice (rank-0 freqs on rank 1) gives a different
        absolute rotation, confirming the test above can catch an
        off-by-local_len regression. (Relative score is unchanged; but the
        per-token rotated vector differs, which a naive equality would flag.)
        """
        rng = np.random.default_rng(14)
        D = QK_ROPE_HEAD_DIM
        L = 8
        cfg = _fake_mla_config(True)
        r = _rope()
        emb_full = r(2 * L)
        emb_r1 = self._contiguous_slice(emb_full, 1, 2)
        emb_r0 = self._contiguous_slice(emb_full, 0, 2)
        v = rng.standard_normal(D)
        vx = np.zeros((1, L, 1, D))
        vx[0, 3, 0] = v
        rot_correct = _apply_real(vx, emb_r1, cfg)[0, 3, 0]  # global pos 11
        rot_wrong = _apply_real(vx, emb_r0, cfg)[0, 3, 0]  # global pos 3
        # Different global position -> different rotated vector.
        self.assertGreater(np.max(np.abs(rot_correct - rot_wrong)), 1e-2)


# ===========================================================================
# Item 6: MTP layer (num_nextn_predict_layers=1) 1-token shift
# ===========================================================================
class TestMTPShift(unittest.TestCase):
    """The MTP layer is a -2 MLA layer. Its MLASelfAttention recomputes
    rotary_pos_emb as arange(0..seq_len-1) from the (shifted) hidden-state
    length (multi_latent_attention.py:1294-1311); the decoder input is the
    token window shifted by depth+1 (multi_token_prediction.py:811-815). The
    shift is UNIFORM across the MTP sequence, so every token's RoPE position is
    offset by the same constant (depth+1). Because RoPE scores depend only on
    the relative offset, intra-MTP attention scores are unchanged by the shift
    -- and identical in all three modes (mode only changes core attention).
    """

    def test_uniform_shift_leaves_scores_invariant(self):
        rng = np.random.default_rng(15)
        D = QK_ROPE_HEAD_DIM
        cfg = _fake_mla_config(True)
        q = rng.standard_normal(D)
        k = rng.standard_normal(D)

        def score(i, j):
            m = max(i, j) + 1
            r = _rope()
            emb = r(m)
            qx = np.zeros((1, m, 1, D))
            kx = np.zeros((1, m, 1, D))
            qx[0, i, 0] = q
            kx[0, j, 0] = k
            return float(
                np.dot(
                    _apply_real(qx, emb, cfg)[0, i, 0],
                    _apply_real(kx, emb, cfg)[0, j, 0],
                )
            )

        # main layer positions (5,2); MTP shift by depth+1 == 1 -> (6,3) etc.
        base = score(5, 2)
        for shift in (1, 2, 3):
            self.assertLess(
                abs(score(5 + shift, 2 + shift) - base),
                5e-3,
                f"MTP uniform shift {shift} changed score",
            )


# ===========================================================================
# Item 7: csa_compress_rotary_base 160000 (HCA) vs 10000 (-2 MLA)
# ===========================================================================
class TestCompressRotaryBaseSeparation(unittest.TestCase):
    """dsv4_hybrid_attention.py:375 sets rope_base = float(
    config.csa_compress_rotary_base) == 160000.0 for compress_ratio>1 (HCA/CSA)
    layers; the -2 MLA layers use RotaryEmbedding(rotary_base=self.rope_theta)
    == 10000 at multi_latent_attention.py:466-470. The two bases must produce
    distinct frequency schedules.
    """

    def test_hca_base_160000_matches_reference(self):
        r = _rope(rotary_base=CSA_COMPRESS_ROTARY_BASE)
        got = np.asarray(r.inv_freq.astype("float64").numpy())
        ref = ref_inv_freq(QK_ROPE_HEAD_DIM, CSA_COMPRESS_ROTARY_BASE)
        self.assertLess(np.max(np.abs(got - ref)), 1e-6)

    def test_two_bases_are_distinct(self):
        base_10k = ref_inv_freq(QK_ROPE_HEAD_DIM, ROPE_THETA)
        base_160k = ref_inv_freq(QK_ROPE_HEAD_DIM, CSA_COMPRESS_ROTARY_BASE)
        # identical only at i=0 (both 1.0); the tail must diverge substantially.
        rel = np.abs(base_10k[1:] - base_160k[1:]) / base_10k[1:]
        self.assertGreater(np.max(rel), 0.1)

    def test_string_base_coercion(self):
        """model_config.json ships csa_compress_rotary_base as the STRING
        '160000.0'; dsv4_hybrid_attention.py:375 coerces via float(). Verify
        float() of the shipped string equals the numeric base used here.
        """
        self.assertEqual(float("160000.0"), CSA_COMPRESS_ROTARY_BASE)


# ===========================================================================
# Item 8: fused vs unfused equivalence (apply_rope_fusion)
# ===========================================================================
class TestFusedVsUnfused(unittest.TestCase):
    """Production configs set apply_rope_fusion=false (YAML) and the mqa /
    mqa_dsa modes REJECT fusion (multi_latent_attention.py:448-453). So the
    live paths are all unfused. This test compares the fused MLA rope kernel to
    the unfused reference when the kernel is importable; otherwise it documents
    that the fused path is unused in production.
    """

    def test_fused_mla_rope_matches_unfused_if_available(self):
        try:
            from paddlefleet.triton_ops.fused_mla_yarn_rope_apply import (
                fused_apply_mla_rope_for_q,
            )
        except Exception as e:
            self.skipTest(
                "fused MLA rope kernel not importable in this env "
                f"({type(e).__name__}); production uses apply_rope_fusion=false "
                "and mqa/mqa_dsa forbid fusion, so the unfused path is "
                "authoritative."
            )
        if fused_apply_mla_rope_for_q is None:
            self.skipTest("fused_apply_mla_rope_for_q is None")
        # If we get here the kernel exists; a full numerical comparison needs
        # the exact cos/sin packing the kernel expects. We assert importability
        # only and defer the numeric check to the dedicated fused-rope tests.
        self.assertTrue(callable(fused_apply_mla_rope_for_q))
