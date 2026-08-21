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

"""Tests for the fused HCA inverse RoPE + ungrouped VHA postmix.

`InvRopeVhaPostmixFusion` computes `out = M @ inv_rope(O)` without ever
materialising `inv_rope(O)`. It is only worth having if it is *bitwise*
identical to the unfused pair it replaces, so every check here is exact
equality with no tolerance:

  - forward output
  - activation gradient (grad wrt the attention output)
  - postmix weight gradients (grad wrt vha_postmix_U / vha_postmix_V)
  - `O` byte-identical after backward: the wgrad rebuilds the rotated tensor
    in place on the CSA-saved attention output and must restore it, and Paddle
    has no version counter that would catch a failure to do so
  - no full-width rotated intermediate is allocated

Two cuBLAS/Paddle properties the fusion leans on are asserted directly, so a
library upgrade fails the test instead of silently drifting the loss curve:

  - splitting a GEMM along its N (channel) axis is bitwise equal to slicing the
    full-width result -- the whole reason a channel split can be exact
  - the explicit dgrad/wgrad formulas used in the fused backward reproduce what
    Paddle's own matmul backward emits
"""

import unittest

import paddle

from paddlefleet.transformer.dsv4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
from paddlefleet.triton_ops import fused_apply_mla_rope_inplace
from paddlefleet.triton_ops.inv_rope_vha_postmix_fusion import (
    InvRopeVhaPostmixFusion,
    build_mla_rope_cos_sin,
    fused_inv_rope_vha_postmix,
    rope_full_out_of_place,
    rope_pe_to_compact,
    scatter_pe_slice_,
)

# ernielite_layer43_pretrain_non_absorbed_mqa_hca_dsa_sparse_loss.yaml at 64K
# context with context_parallel_size=4: b=1, sq=65536/4=16384, and
# num_attention_heads=64, v_head_dim=512, qk_pos_emb_head_dim=64 from
# model_config.json. vha_postmix_rank defaults to num_attention_heads // 4.
NH, VD, PE = 64, 512, 64
NOPE = VD - PE
RANK = NH // 4
PROD_BS = 16384
# Smaller default so the whole suite does not need ~10 GiB; the production
# shape gets its own test.
BS = 2048


def _check_equal(a: paddle.Tensor, b: paddle.Tensor, what: str = "") -> None:
    """Binary-exact equality check (no tolerance allowed)."""
    a32, b32 = a.astype("float32"), b.astype("float32")
    if bool(paddle.all(a32 == b32)):
        return
    diff = (a32 - b32).abs()
    raise AssertionError(
        f"{what or 'tensor'} not bitwise equal: "
        f"{int((a32 != b32).sum())}/{a.numel().item()} elements differ, "
        f"max|diff|={float(diff.max()):.6e}"
    )


def _make_inputs(bs: int, seed: int = 0):
    paddle.seed(seed)
    o = paddle.randn([bs, NH, VD], "bfloat16")
    u = (paddle.randn([NH, RANK], "float32") * 0.01).astype("bfloat16")
    v = (paddle.randn([NH, RANK], "float32") * 0.01).astype("bfloat16")
    freqs = paddle.randn([1, bs, 1, PE], "float32")
    freqs.stop_gradient = True
    d_out = paddle.randn([bs, NH, VD], "bfloat16")
    return o, u, v, freqs, d_out


def _make_m(u: paddle.Tensor, v: paddle.Tensor, nh: int = NH) -> paddle.Tensor:
    """Same construction as _apply_vha_postmix's ungrouped branch."""
    m = paddle.matmul(v, u, transpose_y=True)
    return m + paddle.eye(nh, dtype=m.dtype)


def _leaves(o, u, v):
    out = []
    for t in (o, u, v):
        leaf = t.detach()
        leaf.stop_gradient = False
        out.append(leaf)
    return out


def _run_reference(o, u, v, freqs, d_out):
    """Unfused path: full-width inverse RoPE, then the postmix GEMM."""
    bs = o.shape[0]
    o_l, u_l, v_l = _leaves(o, u, v)
    # non-leaf, mirroring core_attn_out being an op output
    x = o_l * 1.0
    roped = fused_apply_mla_rope_inplace(
        x.reshape([1, bs, NH, VD]),
        freqs,
        NOPE,
        1.0,
        inverse=True,
        clone_input=True,
    )
    out = paddle.matmul(_make_m(u_l, v_l), roped.reshape([bs, NH, VD]))
    out.backward(d_out.clone())
    return out.detach(), o_l.grad, u_l.grad, v_l.grad


def _run_fused(o, u, v, freqs, d_out):
    bs = o.shape[0]
    o_l, u_l, v_l = _leaves(o, u, v)
    x = o_l * 1.0
    cos, sin = build_mla_rope_cos_sin(
        freqs, 1, bs, PE, 1.0, True, paddle.bfloat16
    )
    out = InvRopeVhaPostmixFusion.apply(
        x.reshape([bs, NH, VD]), _make_m(u_l, v_l), cos, sin, NOPE, PE
    )
    out.backward(d_out.clone())
    return out.detach(), o_l.grad, u_l.grad, v_l.grad


class TestInvRopeVhaPostmixFusion(unittest.TestCase):
    def _compare(self, bs: int, seed: int = 0) -> None:
        args = _make_inputs(bs, seed)
        ref = _run_reference(*args)
        fused = _run_fused(*args)
        for name, a, b in zip(
            ("forward out", "grad_O", "grad_U", "grad_V"), fused, ref
        ):
            _check_equal(a, b, f"{name} (B*S={bs}, seed={seed})")

    def test_bitwise_default_shape(self) -> None:
        self._compare(BS)

    def test_bitwise_multiple_seeds(self) -> None:
        for seed in (1, 2, 3):
            self._compare(BS, seed=seed)

    def test_bitwise_production_shape(self) -> None:
        """64K context / CP=4 -> sq=16384, a 1 GiB attention output."""
        self._compare(PROD_BS)

    def test_bitwise_odd_token_count(self) -> None:
        """Token count that is not a multiple of anything in particular."""
        self._compare(1023)

    def test_o_restored_after_backward(self) -> None:
        """The saved attention output must come back untouched.

        Nothing downstream would complain if it did not: the CSA backward reads
        the same buffer to build `delta = rowsum(dO * O)` and Paddle silently
        accepts a mutated saved tensor. The fused backward builds its rotated
        operand into a fresh transposed buffer precisely so that this holds.
        """
        o, u, v, freqs, d_out = _make_inputs(BS)
        snapshot = o.clone()
        o_l, u_l, v_l = _leaves(o, u, v)
        cos, sin = build_mla_rope_cos_sin(
            freqs, 1, BS, PE, 1.0, True, paddle.bfloat16
        )
        # Pass the leaf's own storage through so the PyLayer saves a view of it.
        x = o_l.reshape([BS, NH, VD])
        out = InvRopeVhaPostmixFusion.apply(
            x, _make_m(u_l, v_l), cos, sin, NOPE, PE
        )
        out.backward(d_out.clone())
        _check_equal(o_l, snapshot, "O after backward")

    def test_frozen_postmix_skips_wgrad(self) -> None:
        """With U/V frozen only the activation gradient is produced.

        That path skips both transposed operands, so it must not allocate them
        either -- checked indirectly by the gradient still matching exactly.
        """
        o, u, v, freqs, d_out = _make_inputs(BS)
        cos, sin = build_mla_rope_cos_sin(
            freqs, 1, BS, PE, 1.0, True, paddle.bfloat16
        )

        o_r = o.detach()
        o_r.stop_gradient = False
        m_frozen = _make_m(u, v)  # built from non-leaf constants -> no grad
        m_frozen.stop_gradient = True
        roped = fused_apply_mla_rope_inplace(
            (o_r * 1.0).reshape([1, BS, NH, VD]),
            freqs,
            NOPE,
            1.0,
            inverse=True,
            clone_input=True,
        )
        paddle.matmul(m_frozen, roped.reshape([BS, NH, VD])).backward(
            d_out.clone()
        )

        o_f = o.detach()
        o_f.stop_gradient = False
        snapshot = o_f.clone()
        InvRopeVhaPostmixFusion.apply(
            (o_f * 1.0).reshape([BS, NH, VD]), m_frozen, cos, sin, NOPE, PE
        ).backward(d_out.clone())

        _check_equal(o_f.grad, o_r.grad, "grad_O with frozen postmix")
        _check_equal(o_f, snapshot, "O with frozen postmix")

    def test_no_wide_rotated_intermediate(self) -> None:
        """Forward must leave one full-width tensor live, not two.

        The unfused pair keeps the rotated attention output alive alongside its
        own result, which is exactly the copy this fusion exists to remove.
        """
        o, u, v, freqs, d_out = _make_inputs(BS)
        cos, sin = build_mla_rope_cos_sin(
            freqs, 1, BS, PE, 1.0, True, paddle.bfloat16
        )
        m = _make_m(u, v)
        wide = BS * NH * VD * 2
        with paddle.no_grad():
            flat = o.reshape([BS, NH, VD])
            # warm up the JIT and any lazy allocator growth
            InvRopeVhaPostmixFusion.apply(flat, m, cos, sin, NOPE, PE)
            fused_apply_mla_rope_inplace(
                o.reshape([1, BS, NH, VD]),
                freqs,
                NOPE,
                1.0,
                inverse=True,
                clone_input=True,
            )
            paddle.device.synchronize()

            before = paddle.device.cuda.memory_allocated()
            out = InvRopeVhaPostmixFusion.apply(flat, m, cos, sin, NOPE, PE)
            paddle.device.synchronize()
            delta = paddle.device.cuda.memory_allocated() - before
            del out

            before = paddle.device.cuda.memory_allocated()
            roped = fused_apply_mla_rope_inplace(
                o.reshape([1, BS, NH, VD]),
                freqs,
                NOPE,
                1.0,
                inverse=True,
                clone_input=True,
            )
            out_ref = paddle.matmul(m, roped.reshape([BS, NH, VD]))
            paddle.device.synchronize()
            delta_ref = paddle.device.cuda.memory_allocated() - before
            del roped, out_ref

        # Fused: only the output survives; the compact pe buffers (N/8 each)
        # are freed inside the call.
        self.assertEqual(
            delta,
            wide,
            f"fused forward left {delta} B live, expected exactly one "
            f"full-width output ({wide} B)",
        )
        # Unfused: rotated copy + output.
        self.assertEqual(delta_ref, 2 * wide)


class TestFusionAssumptions(unittest.TestCase):
    """Pin the library behaviour the fusion's exactness rests on."""

    def test_split_n_gemm_is_bitwise(self) -> None:
        """matmul(M, X[..., a:b]) == matmul(M, X)[..., a:b].

        The postmix GEMM contracts the head axis, so the channel axis is a pure
        N dimension and splitting it cannot reorder the accumulation. This is the
        *only* cuBLAS property the fusion still relies on (the weight gradient
        goes through `matmul_grad` precisely because a hand-rolled GEMM there
        turned out to be architecture-dependent), so sweep it widely: small and
        large head counts, powers of two and not, and both split points.
        """
        for bs, nh, vd, pe in (
            (BS, NH, VD, PE),
            (PROD_BS, NH, VD, PE),
            (1023, NH, VD, PE),
            (1, NH, VD, PE),
            (128, 4, 64, 32),
            (128, 4, 64, 16),
            (333, 8, 128, 32),
            (97, 32, 96, 32),
            (7, 3, 40, 8),
            (64, 16, 256, 64),
        ):
            nope = vd - pe
            rank = max(1, nh // 4)
            paddle.seed(bs + nh)
            m = _make_m(
                paddle.randn([nh, rank], "bfloat16"),
                paddle.randn([nh, rank], "bfloat16"),
                nh,
            )
            x = paddle.randn([bs, nh, vd], "bfloat16")
            tag = f"({bs},{nh},{vd},pe={pe})"
            with paddle.no_grad():
                full = paddle.matmul(m, x)
                _check_equal(
                    paddle.matmul(m, x[..., :nope]),
                    full[..., :nope],
                    f"nope split {tag}",
                )
                _check_equal(
                    paddle.matmul(m, x[..., nope:]),
                    full[..., nope:],
                    f"pe split {tag}",
                )
                # the pe operand is a compact buffer in the fused path, not a view
                _check_equal(
                    paddle.matmul(m, x[..., nope:].contiguous()),
                    full[..., nope:],
                    f"pe split from a compact operand {tag}",
                )

    def test_backward_formulas_match_matmul_grad(self) -> None:
        """`matmul_grad` and the explicit dgrad == autograd's own results.

        The fused backward takes both gradients from `matmul_grad` and computes
        the frozen-postmix dgrad explicitly, so pin both against autograd.
        """
        paddle.seed(0)
        u = paddle.randn([NH, RANK], "bfloat16")
        v = paddle.randn([NH, RANK], "bfloat16")
        x0 = paddle.randn([BS, NH, VD], "bfloat16")
        d_out = paddle.randn([BS, NH, VD], "bfloat16")

        m_l = _make_m(u, v).detach()
        m_l.stop_gradient = False
        x_l = x0.detach()
        x_l.stop_gradient = False
        paddle.matmul(m_l, x_l).backward(d_out.clone())

        m_d, x_d = m_l.detach(), x0.detach()
        with paddle.no_grad():
            _check_equal(
                paddle.matmul(m_d, d_out, transpose_x=True),
                x_l.grad,
                "explicit dgrad",
            )
            d_m, d_x = paddle._C_ops.matmul_grad(m_d, x_d, d_out, False, False)
            _check_equal(d_m, m_l.grad, "matmul_grad wgrad")
            _check_equal(d_x, x_l.grad, "matmul_grad dgrad")
            _check_equal(
                paddle.einsum("bhd,bkd->hk", d_out, x_d),
                m_l.grad,
                "einsum wgrad",
            )

    def test_pe_scatter_matches_paddle(self) -> None:
        """The vectorised pe scatter == Paddle's strided slice assignment."""
        paddle.seed(0)
        t = paddle.randn([BS, NH, VD], "bfloat16")
        compact = paddle.randn([BS, NH, PE], "bfloat16")
        with paddle.no_grad():
            a, b = t.clone(), t.clone()
            scatter_pe_slice_(a, compact, NOPE, PE)
            b[..., NOPE:] = compact
            _check_equal(a, b, "pe scatter")

    def test_compact_rope_matches_wide_rope(self) -> None:
        """rope_pe_to_compact == the standalone op's pe channels."""
        paddle.seed(0)
        t = paddle.randn([BS, NH, VD], "bfloat16")
        freqs = paddle.randn([1, BS, 1, PE], "float32")
        freqs.stop_gradient = True
        with paddle.no_grad():
            for inverse in (True, False):
                cos, sin = build_mla_rope_cos_sin(
                    freqs, 1, BS, PE, 1.0, inverse, paddle.bfloat16
                )
                ref = fused_apply_mla_rope_inplace(
                    t.reshape([1, BS, NH, VD]),
                    freqs,
                    NOPE,
                    1.0,
                    inverse=inverse,
                    clone_input=True,
                )
                _check_equal(
                    rope_pe_to_compact(t, cos, sin, NOPE, PE),
                    ref.reshape([BS, NH, VD])[..., NOPE:].contiguous(),
                    f"compact rope (inverse={inverse})",
                )


class _GateStub:
    """Minimal stand-in for the attention layer's gating attributes."""

    def __init__(self, **kw):
        self.config = type("C", (), {})()
        self.config.fuse_inv_rope_into_vha_postmix = True
        self.config.apply_rope_fusion = True
        self.config.high_precision_rope = False
        self.use_vha_postmix = True
        self.vha_postmix_grouped = False
        self.recompute_vha_postmix = False
        self.training = True
        for k, val in kw.items():
            if hasattr(self.config, k):
                setattr(self.config, k, val)
            else:
                setattr(self, k, val)


class TestFusionGating(unittest.TestCase):
    """Every rejected combination must fall back, not raise."""

    def _gate(self, in_full_recompute=False, **kw) -> bool:
        return DSv4HybridSelfAttention._can_fuse_inv_rope_postmix(
            _GateStub(**kw), in_full_recompute
        )

    def test_enabled_by_default_config(self) -> None:
        self.assertTrue(self._gate())

    def test_disabled_cases(self) -> None:
        for kw in (
            {"fuse_inv_rope_into_vha_postmix": False},
            {"use_vha_postmix": False},
            {"vha_postmix_grouped": True},
            {"apply_rope_fusion": False},
            {"high_precision_rope": True},
            {"recompute_vha_postmix": True},
        ):
            self.assertFalse(self._gate(**kw), f"should be disabled for {kw}")

    def test_nested_recompute_allowed_inside_full_recompute(self) -> None:
        self.assertTrue(
            self._gate(in_full_recompute=True, recompute_vha_postmix=True)
        )

    def test_eval_mode_ignores_postmix_recompute(self) -> None:
        self.assertTrue(self._gate(recompute_vha_postmix=True, training=False))


class _LayerStub:
    """Just enough of the attention layer to drive the two postmix methods."""

    def __init__(self, u, v, nh=NH, vd=VD):
        self.vha_postmix_U = u
        self.vha_postmix_V = v
        self.num_attention_heads = nh
        self.v_head_dim = vd
        self.vha_postmix_grouped = False
        self.o_local_groups = 8


class TestWiring(unittest.TestCase):
    """`_apply_inv_rope_vha_postmix` must match RoPE + `_apply_vha_postmix`."""

    def test_method_matches_unfused_pair(self) -> None:
        b, sq = 1, BS
        o, u, v, freqs, d_out = _make_inputs(b * sq)

        o_r, u_r, v_r = _leaves(o, u, v)
        roped = fused_apply_mla_rope_inplace(
            (o_r * 1.0).reshape([b, sq, NH, VD]),
            freqs,
            NOPE,
            1.0,
            inverse=True,
            clone_input=True,
        )
        ref = DSv4HybridSelfAttention._apply_vha_postmix(
            _LayerStub(u_r, v_r), roped
        )
        ref.backward(d_out.reshape([b, sq, NH * VD]).clone())

        o_f, u_f, v_f = _leaves(o, u, v)
        got = DSv4HybridSelfAttention._apply_inv_rope_vha_postmix(
            _LayerStub(u_f, v_f),
            (o_f * 1.0).reshape([b, sq, NH, VD]),
            freqs,
            NOPE,
            PE,
            1.0,
        )
        self.assertEqual(list(got.shape), [b, sq, NH * VD])
        self.assertEqual(list(got.shape), list(ref.shape))
        got.backward(d_out.reshape([b, sq, NH * VD]).clone())

        _check_equal(got.detach(), ref.detach(), "wired forward")
        _check_equal(o_f.grad, o_r.grad, "wired grad_O")
        _check_equal(u_f.grad, u_r.grad, "wired grad_U")
        _check_equal(v_f.grad, v_r.grad, "wired grad_V")


class TestSmallHeadCounts(unittest.TestCase):
    """End-to-end bitwise equality at small head counts.

    The production config has nh=64, but small head counts are where cuBLAS
    algorithm selection gets unstable: an earlier revision hand-rolled the
    weight-gradient GEMM over head-major operands and CI caught it diverging at
    ``(bs=128, nh=4, d=64)`` on sm90 while it was bitwise equal on sm10.3. The
    weight gradient now goes through ``matmul_grad`` on every path, so this must
    hold on every architecture -- powers of two and not.
    """

    def _run(self, fused, bs, nh, vd, pe, seed):
        nope = vd - pe
        rank = max(1, nh // 4)
        paddle.seed(seed)
        o = paddle.randn([bs, nh, vd], "bfloat16")
        u = (paddle.randn([nh, rank], "float32") * 0.05).astype("bfloat16")
        v = (paddle.randn([nh, rank], "float32") * 0.05).astype("bfloat16")
        freqs = paddle.randn([1, bs, 1, pe], "float32")
        freqs.stop_gradient = True
        d_out = paddle.randn([bs, nh, vd], "bfloat16")

        o_l, u_l, v_l = _leaves(o, u, v)
        m = paddle.matmul(v_l, u_l, transpose_y=True) + paddle.eye(
            nh, dtype="bfloat16"
        )
        x = o_l * 1.0
        if fused:
            cos, sin = build_mla_rope_cos_sin(
                freqs, 1, bs, pe, 1.0, True, paddle.bfloat16
            )
            out = InvRopeVhaPostmixFusion.apply(
                x.reshape([bs, nh, vd]), m, cos, sin, nope, pe
            )
        else:
            roped = fused_apply_mla_rope_inplace(
                x.reshape([1, bs, nh, vd]),
                freqs,
                nope,
                1.0,
                inverse=True,
                clone_input=True,
            )
            out = paddle.matmul(m, roped.reshape([bs, nh, vd]))
        snapshot = o_l.clone()
        out.backward(d_out.clone())
        return out.detach(), o_l.grad, u_l.grad, v_l.grad, o_l, snapshot

    def test_bitwise(self) -> None:
        for bs, nh, vd, pe, seed in (
            (128, 4, 64, 32, 0),  # the shape CI flagged on sm90
            (128, 4, 64, 16, 1),
            (7, 3, 40, 8, 0),  # not a power of two
            (64, 3, 40, 8, 1),
            (1023, 3, 40, 8, 2),
            (333, 8, 128, 32, 0),
            (97, 32, 96, 32, 0),
            (64, 16, 256, 64, 0),
        ):
            tag = f"(bs={bs}, nh={nh}, d={vd}, pe={pe})"
            ref = self._run(False, bs, nh, vd, pe, seed)
            got = self._run(True, bs, nh, vd, pe, seed)
            for name, a, b in zip(
                ("out", "grad_O", "grad_U", "grad_V"), got, ref
            ):
                _check_equal(a, b, f"{name} {tag}")
            _check_equal(got[4], got[5], f"O untouched {tag}")


class TestArgumentValidation(unittest.TestCase):
    """Every helper rejects malformed input instead of reading out of bounds."""

    def setUp(self) -> None:
        paddle.seed(0)
        self.t = paddle.randn([32, NH, VD], "bfloat16")
        self.freqs = paddle.randn([1, 32, 1, PE], "float32")
        self.freqs.stop_gradient = True
        self.cos, self.sin = build_mla_rope_cos_sin(
            self.freqs, 1, 32, PE, 1.0, True, paddle.bfloat16
        )

    def test_rank_must_be_three(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[B\*S, H, D\]"):
            rope_pe_to_compact(
                self.t.reshape([1, 32, NH, VD]), self.cos, self.sin, NOPE, PE
            )

    def test_channel_split_must_add_up(self) -> None:
        with self.assertRaisesRegex(ValueError, r"nope_dim \+ pe_dim"):
            rope_pe_to_compact(self.t, self.cos, self.sin, NOPE + 1, PE)

    def test_cos_sin_must_be_contiguous(self) -> None:
        with self.assertRaisesRegex(ValueError, "cos/sin must be contiguous"):
            rope_pe_to_compact(self.t, self.cos[..., :-4], self.sin, NOPE, PE)

    def test_cos_sin_width_checked(self) -> None:
        narrow = self.cos[..., :-4].contiguous()
        with self.assertRaisesRegex(ValueError, "cos/sin last dim"):
            rope_pe_to_compact(self.t, narrow, self.sin, NOPE, PE)

    def test_freqs_rank_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, r"freqs must be \[B,S,1,D\]"):
            build_mla_rope_cos_sin(
                self.freqs[0], 1, 32, PE, 1.0, True, paddle.bfloat16
            )

    def test_freqs_seqlen_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "mismatches"):
            build_mla_rope_cos_sin(
                self.freqs, 1, 33, PE, 1.0, True, paddle.bfloat16
            )

    def test_freqs_batch_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 1 or"):
            build_mla_rope_cos_sin(
                paddle.randn([3, 32, 1, PE], "float32"),
                2,
                32,
                PE,
                1.0,
                True,
                paddle.bfloat16,
            )

    def test_freqs_broadcast_over_batch(self) -> None:
        """B=1 freqs are broadcast, matching a manually tiled version."""
        cos_b, sin_b = build_mla_rope_cos_sin(
            self.freqs, 2, 32, PE, 1.0, True, paddle.bfloat16
        )
        tiled = self.freqs.broadcast_to([2, 32, 1, PE])
        cos_t, sin_t = build_mla_rope_cos_sin(
            tiled, 2, 32, PE, 1.0, True, paddle.bfloat16
        )
        _check_equal(cos_b, cos_t, "broadcast cos")
        _check_equal(sin_b, sin_t, "broadcast sin")

    def test_scatter_shape_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "compact must be"):
            scatter_pe_slice_(
                self.t.clone(),
                paddle.randn([32, NH, PE + 4], "bfloat16"),
                NOPE,
                PE,
            )


class TestOutOfPlaceRopeHelper(unittest.TestCase):
    """``rope_full_out_of_place`` rebuilds the operand the wgrad needs."""

    def test_matches_standalone_op(self) -> None:
        for bs, nh, vd, pe in (
            (BS, NH, VD, PE),
            (97, 8, 128, 32),
            (7, 3, 40, 8),
        ):
            nope = vd - pe
            paddle.seed(bs)
            t = paddle.randn([bs, nh, vd], "bfloat16")
            before = t.clone()
            freqs = paddle.randn([1, bs, 1, pe], "float32")
            freqs.stop_gradient = True
            with paddle.no_grad():
                for inverse in (True, False):
                    cos, sin = build_mla_rope_cos_sin(
                        freqs, 1, bs, pe, 1.0, inverse, paddle.bfloat16
                    )
                    ref = fused_apply_mla_rope_inplace(
                        t.reshape([1, bs, nh, vd]),
                        freqs,
                        nope,
                        1.0,
                        inverse=inverse,
                        clone_input=True,
                    ).reshape([bs, nh, vd])
                    _check_equal(
                        rope_full_out_of_place(t, cos, sin, nope, pe),
                        ref,
                        f"rope_full_out_of_place ({bs},{nh},{vd},{pe},"
                        f"inverse={inverse})",
                    )
            _check_equal(t, before, "input unchanged by rope_full_out_of_place")


class TestEntryPoint(unittest.TestCase):
    """``fused_inv_rope_vha_postmix`` is what the attention layer calls."""

    def test_matches_unfused_pair(self) -> None:
        sq = 64
        o, u, v, freqs, d_out = _make_inputs(sq)
        g = d_out.reshape([1, sq, NH * VD])

        o_r, u_r, v_r = _leaves(o, u, v)
        roped = fused_apply_mla_rope_inplace(
            (o_r * 1.0).reshape([1, sq, NH, VD]),
            freqs,
            NOPE,
            1.0,
            inverse=True,
            clone_input=True,
        )
        ref = paddle.matmul(
            _make_m(u_r, v_r), roped.reshape([sq, NH, VD])
        ).reshape([1, sq, NH * VD])
        ref.backward(g.clone())

        o_f, u_f, v_f = _leaves(o, u, v)
        got = fused_inv_rope_vha_postmix(
            (o_f * 1.0).reshape([1, sq, NH, VD]), freqs, u_f, v_f, NOPE, PE
        )
        self.assertEqual(list(got.shape), [1, sq, NH * VD])
        got.backward(g.clone())

        _check_equal(got.detach(), ref.detach(), "entry point forward")
        _check_equal(o_f.grad, o_r.grad, "entry point grad_O")
        _check_equal(u_f.grad, u_r.grad, "entry point grad_U")
        _check_equal(v_f.grad, v_r.grad, "entry point grad_V")

    def test_accepts_non_contiguous_input(self) -> None:
        """A sliced attention output must be handled, not silently mis-read."""
        sq = 64
        o, u, v, freqs, _ = _make_inputs(sq)
        wide = paddle.concat([o, o], axis=-1)  # [sq, NH, 2 * VD]
        view = wide[..., :VD].reshape([1, sq, NH, VD])
        self.assertFalse(view.is_contiguous())
        with paddle.no_grad():
            got = fused_inv_rope_vha_postmix(view, freqs, u, v, NOPE, PE)
            want = fused_inv_rope_vha_postmix(
                o.reshape([1, sq, NH, VD]), freqs, u, v, NOPE, PE
            )
        _check_equal(got, want, "non-contiguous input")


if __name__ == "__main__":
    unittest.main()
