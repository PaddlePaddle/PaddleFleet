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

"""Tests for the Triton-fused rotate_half RoPE (DSA indexer, MLA q / k_pe).

Verifies `fused_apply_rope_half` against the unfused reference in
`DSAIndexer._apply_rope` (dsa_attention.py) and MLA's eager branch:

  - Forward bit-exact with the slice + rotate_half + concat baseline, i.e. the
    output is the *assembled* tensor and the caller's concat can go.
  - Out of place: input untouched, result in a fresh buffer, so nothing depends
    on whether the caller's tensor may be replayed by recompute.
  - Backward grads bit-exact with autograd through the reference.
  - q (CP-local) and k (all-gathered) shapes, including s_q != s_k.
"""

import os
import subprocess
import sys
import textwrap
import unittest

import paddle

from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddlefleet.triton_ops import (
    fused_apply_rope_half,
    fused_rope_cat_key,
)

# DSA indexer shapes: index_head_dim=128, rope=hybrid_mla_qk_rope_head_dim=64
B, S, H, D = 1, 2048, 64, 128
PE_DIM = 64
NOPE_DIM = D - PE_DIM


def _reference_forward(
    x: paddle.Tensor,
    freqs: paddle.Tensor,
    pe_dim: int,
    mscale: float = 1.0,
    pe_offset: int = 0,
) -> paddle.Tensor:
    """Eager rotate_half body, matching both layouts.

    ``pe_offset == 0`` is dsa_attention.py:_apply_rope's [pe | nope] split;
    ``pe_offset == nope_dim`` is multi_latent_attention.py's [nope | pe] one.
    """
    lead = x[..., :pe_offset]
    x_pe = x[..., pe_offset : pe_offset + pe_dim]
    trail = x[..., pe_offset + pe_dim :]
    x_pe = _apply_rotary_pos_emb_bshd(
        x_pe,
        freqs,
        rotary_interleaved=False,
        multi_latent_attention=False,
        mscale=mscale,
    )
    return paddle.concat([lead, x_pe, trail], axis=-1)


def _check_equal(a: paddle.Tensor, b: paddle.Tensor):
    """Binary-exact equality check (no tolerance allowed)."""
    assert paddle.all(a == b), f"tensor not equal:\nA: {a}\nB: {b}"


class TestFusedRopeHalf(unittest.TestCase):
    def setUp(self) -> None:
        paddle.seed(0)

    def _run_case(
        self,
        b: int,
        s: int,
        h: int,
        freqs: paddle.Tensor,
        mscale: float = 1.0,
        head_dim: int = D,
        pe_offset: int = 0,
    ) -> None:
        x = paddle.randn([b, s, h, head_dim], "bfloat16")
        x.stop_gradient = False
        untouched = list(range(0, pe_offset)) + list(
            range(pe_offset + PE_DIM, head_dim)
        )

        # ---- reference ----
        x_ref = x.detach()
        x_ref.stop_gradient = False
        out_ref = _reference_forward(x_ref, freqs, PE_DIM, mscale, pe_offset)
        out_grad = paddle.randn_like(out_ref)
        out_ref.backward(out_grad)
        grad_ref = x_ref.grad

        # ---- fused ----
        x_fused = x.detach()
        x_fused.stop_gradient = False
        before = x_fused.clone()
        ptr_before = x_fused.data_ptr()
        out_fused = fused_apply_rope_half(
            x_fused, freqs, PE_DIM, mscale, pe_offset=pe_offset
        )

        # ---- out-of-place invariants ----
        self.assertNotEqual(out_fused.data_ptr(), x_fused.data_ptr())
        self.assertEqual(x_fused.data_ptr(), ptr_before)
        _check_equal(x_fused, before)
        self.assertTrue(out_fused.is_contiguous())
        if untouched:
            # Copying these is what lets the caller drop its reassembly concat:
            # the output already *is* the assembled tensor.
            _check_equal(out_fused[..., untouched], before[..., untouched])

        # ---- forward / backward parity ----
        _check_equal(out_fused, out_ref)
        out_fused.backward(out_grad)
        _check_equal(x_fused.grad, grad_ref)

    def test_mla_q_shape_pe_last(self) -> None:
        """MLA q: [b, s, 64, 192+64] with the rope block trailing.

        multi_latent_attention.py rotates ``q[..., qk_nope_head_dim:]`` and then
        concatenates the halves back; the kernel must reproduce that whole
        tensor, leading channels included.
        """
        nope, s = 192, 512
        freqs = paddle.randn([B, s, 1, PE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, s, 64, freqs, head_dim=nope + PE_DIM, pe_offset=nope)

    def test_mla_k_pos_emb_shape(self) -> None:
        """MLA k_pos_emb: [b, s, 1, 64], pe only (pe_offset stays 0)."""
        s = 512
        freqs = paddle.randn([B, s, 1, PE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, s, 1, freqs, head_dim=PE_DIM)

    def test_q_shape(self) -> None:
        """Indexer q: [b, s_local, n_heads, head_dim]."""
        freqs = paddle.randn([B, S, 1, PE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, S, H, freqs)

    def test_k_shape_single_head(self) -> None:
        """Indexer k after unsqueeze(2): [b, s, 1, head_dim]."""
        freqs = paddle.randn([B, S, 1, PE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, S, 1, freqs)

    def test_cp_global_k_longer_than_q(self) -> None:
        """CP: k is all-gathered (s_global) while q stays local (s_local).

        The two live in the same layer, so the kernel must key only off the
        sequence axis of the tensor it is given.
        """
        s_local, cp = 256, 4
        s_global = s_local * cp
        freqs_global = paddle.randn([B, s_global, 1, PE_DIM])
        freqs_global.stop_gradient = True
        # q: local rows [position_offset, position_offset + s_local)
        offset = 2 * s_local
        freqs_q = freqs_global[:, offset : offset + s_local]
        self.assertFalse(freqs_q.is_contiguous())
        self._run_case(B, s_local, H, freqs_q)
        # k: full global range, single head
        self._run_case(B, s_global, 1, freqs_global)

    def test_mscale(self) -> None:
        """YaRN-style mscale on fp32 freqs, which the wrapper supports."""
        freqs = paddle.randn([B, S, 1, PE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, S, H, freqs, mscale=1.37)

    def test_bf16_freqs(self) -> None:
        """The real indexer hands over bf16 freqs once the module is cast.

        ``tl.cos`` needs fp32, and paddle evaluates cos/sin of a bf16 input in
        fp32 before rounding once, so upcasting inside the kernel wrapper is
        bit-exact at mscale == 1.0 (the ``rope_type == "rope"`` case).
        """
        freqs = paddle.randn([B, S, 1, PE_DIM]).astype("bfloat16")
        freqs.stop_gradient = True
        self.assertEqual(freqs.dtype, paddle.bfloat16)
        self._run_case(B, S, H, freqs)
        self._run_case(B, S, 1, freqs)

    def test_bf16_freqs_with_mscale_is_refused(self) -> None:
        """mscale != 1 on bf16 freqs is the one combination that cannot match.

        Eager rounds cos to bf16 *before* multiplying by mscale; the kernel
        scales in fp32. Measured to differ, so the wrapper rejects it. fp32
        freqs (production) has no such restriction -- see ``test_mscale``.
        """
        freqs = paddle.randn([B, 128, 1, PE_DIM]).astype("bfloat16")
        freqs.stop_gradient = True
        t = paddle.randn([B, 128, H, D], "bfloat16")
        with self.assertRaisesRegex(ValueError, "needs fp32 freqs"):
            fused_apply_rope_half(t, freqs, PE_DIM, mscale=1.37)

    def test_odd_head_count_and_mid_block(self) -> None:
        """H not a power of two, rope block neither leading nor trailing.

        Exercises the head mask and both sides of the copy mask at once.
        """
        s = 128
        freqs = paddle.randn([B, s, 1, PE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, s, 12, freqs, head_dim=192, pe_offset=64)

    def test_strided_input_view(self) -> None:
        """A last-dim-contiguous *view* is a legal input.

        ``q_pos_emb = q[..., qk_nope_head_dim:]`` -- the absorbed-MQA call site
        -- is exactly this: last-dim stride 1, row stride 256.
        """
        s = 256
        freqs = paddle.randn([B, s, 1, PE_DIM])
        freqs.stop_gradient = True
        q = paddle.randn([B, s, H, 256], "bfloat16")
        view = q[..., 192:]
        self.assertEqual(view.stride(-1), 1)
        out = fused_apply_rope_half(view, freqs, PE_DIM)
        _check_equal(out, _reference_forward(view, freqs, PE_DIM))


class TestFusedRopeCatKey(unittest.TestCase):
    """``fused_rope_cat_key`` vs the eager rope + concat it replaces.

    Absorbed-MQA shapes: kv_compressed [b, s, 512] (already normalised) and
    k_pos_emb [b, s, 1, 64], producing key [b, s, 1, 576] plus the rotated pe
    on its own -- the two things the eager snippet leaves behind.
    """

    LATENT = 512

    def setUp(self) -> None:
        paddle.seed(0)

    def _reference(self, kv, kpe, freqs):
        """Returns (key, rotated k_pe), matching the fused signature."""
        kpe = _apply_rotary_pos_emb_bshd(
            kpe,
            freqs,
            rotary_interleaved=False,
            multi_latent_attention=False,
            mscale=1.0,
        )
        return paddle.concat([kv.unsqueeze(-2), kpe], axis=-1), kpe

    def _run(self, b, s, use_k_pe):
        """``use_k_pe`` also backprops through the second output.

        Off is the production path (``MQALatentAttention`` takes ``k_pe`` and
        never reads it, so paddle hands back no gradient for it); on exercises
        the branch that adds the two incoming pe gradients.
        """
        kv0 = paddle.randn([b, s, self.LATENT], "bfloat16")
        kpe0 = paddle.randn([b, s, 1, PE_DIM], "bfloat16")
        freqs = paddle.randn([1, s, 1, PE_DIM])
        freqs.stop_gradient = True

        outs = []
        grads = {}
        for fused in (False, True):
            kv = kv0.detach()
            kpe = kpe0.detach()
            kv.stop_gradient = False
            kpe.stop_gradient = False
            if fused:
                key, k_pe = fused_rope_cat_key(
                    kv, kpe, freqs, self.LATENT, PE_DIM
                )
            else:
                key, k_pe = self._reference(kv, kpe, freqs)
            if not grads:
                grads["key"] = paddle.randn_like(key)
                grads["k_pe"] = paddle.randn_like(k_pe)
            loss = (key.astype("float32") * grads["key"]).sum()
            if use_k_pe:
                loss = loss + (k_pe.astype("float32") * grads["k_pe"]).sum()
            loss.backward()
            outs.append((key.detach(), k_pe.detach(), kv.grad, kpe.grad))

        (
            (k_ref, pe_ref, dkv_ref, dkpe_ref),
            (
                k_f,
                pe_f,
                dkv_f,
                dkpe_f,
            ),
        ) = outs
        self.assertEqual(list(k_f.shape), [b, s, 1, self.LATENT + PE_DIM])
        self.assertEqual(list(pe_f.shape), [b, s, 1, PE_DIM])
        self.assertTrue(pe_f.is_contiguous())
        _check_equal(k_f, k_ref)
        _check_equal(pe_f, pe_ref)
        # The second output must hold exactly what key's tail holds.
        _check_equal(pe_f, k_f[..., self.LATENT :])
        _check_equal(dkv_f, dkv_ref)
        _check_equal(dkpe_f, dkpe_ref)

    def test_forward_backward(self) -> None:
        self._run(1, 512, use_k_pe=False)

    def test_forward_backward_with_k_pe_consumed(self) -> None:
        self._run(1, 512, use_k_pe=True)

    def test_value_block_is_copied_verbatim(self) -> None:
        """The leading latent channels are the absorbed value: copy, no math."""
        kv = paddle.randn([1, 256, self.LATENT], "bfloat16")
        kpe = paddle.randn([1, 256, 1, PE_DIM], "bfloat16")
        freqs = paddle.randn([1, 256, 1, PE_DIM])
        freqs.stop_gradient = True
        key, _ = fused_rope_cat_key(kv, kpe, freqs, self.LATENT, PE_DIM)
        _check_equal(key[..., 0, : self.LATENT], kv)

    def test_3d_k_pos_emb(self) -> None:
        """``k_pos_emb`` may arrive un-unsqueezed."""
        kv = paddle.randn([1, 256, self.LATENT], "bfloat16")
        kpe = paddle.randn([1, 256, PE_DIM], "bfloat16")
        freqs = paddle.randn([1, 256, 1, PE_DIM])
        freqs.stop_gradient = True
        key, k_pe = fused_rope_cat_key(kv, kpe, freqs, self.LATENT, PE_DIM)
        ref, pe_ref = self._reference(kv, kpe.unsqueeze(-2), freqs)
        _check_equal(key, ref)
        _check_equal(k_pe, pe_ref)


class TestOptimizedModeValidation(unittest.TestCase):
    """The public entries must reject bad input even under ``python -O``.

    ``python -O`` strips ``assert``. If the dtype/shape/mscale checks in
    ``fused_apply_rope_half`` / ``fused_rope_cat_key`` (and their PyLayers) were
    asserts, an optimized production run would let fp16 into the bf16-rounding
    ``_mul_round_bf16`` path or a wrong shape into Triton address arithmetic and
    silently corrupt training. This spawns a real ``-O`` interpreter and
    requires each bad input to raise ``ValueError``; an ``assert`` would raise
    nothing there, which the child reports as a failure.

    Validation happens before any kernel launch, so the child never reaches
    Triton; it still imports paddlefleet on the default (GPU) device, since
    ``paddlefleet_ops`` queries the CUDA device capability at import time.
    """

    def test_bad_inputs_raise_valueerror_under_O(self) -> None:
        child = textwrap.dedent(
            """
            import sys
            import paddle

            from paddlefleet.triton_ops import (
                fused_apply_rope_half,
                fused_rope_cat_key,
            )

            if not sys.flags.optimize:
                raise SystemExit("child not running under python -O")

            PE = 8
            LATENT = 16

            def expect_valueerror(fn, label):
                try:
                    fn()
                except ValueError:
                    return
                except Exception as exc:  # noqa: BLE001
                    raise SystemExit(
                        f"{label}: expected ValueError, got "
                        f"{type(exc).__name__}: {exc}"
                    )
                raise SystemExit(
                    f"{label}: no exception (assert stripped under -O)"
                )

            def half_fp16():
                t = paddle.randn([1, 4, 2, PE]).astype("float16")
                freqs = paddle.randn([1, 4, 1, PE])
                fused_apply_rope_half(t, freqs, PE)

            def half_bad_shape():
                t = paddle.randn([1, 4, 2, PE]).astype("bfloat16")
                freqs = paddle.randn([1, 4, 1, PE + 4])
                fused_apply_rope_half(t, freqs, PE)

            def half_mscale():
                t = paddle.randn([1, 4, 2, PE]).astype("bfloat16")
                freqs = paddle.randn([1, 4, 1, PE]).astype("bfloat16")
                fused_apply_rope_half(t, freqs, PE, mscale=1.37)

            def cat_fp16():
                kv = paddle.randn([1, 4, LATENT]).astype("float16")
                kpe = paddle.randn([1, 4, 1, PE]).astype("float16")
                freqs = paddle.randn([1, 4, 1, PE])
                fused_rope_cat_key(kv, kpe, freqs, LATENT, PE)

            def cat_dtype_mismatch():
                kv = paddle.randn([1, 4, LATENT]).astype("bfloat16")
                kpe = paddle.randn([1, 4, 1, PE]).astype("float16")
                freqs = paddle.randn([1, 4, 1, PE])
                fused_rope_cat_key(kv, kpe, freqs, LATENT, PE)

            def cat_mscale():
                kv = paddle.randn([1, 4, LATENT]).astype("bfloat16")
                kpe = paddle.randn([1, 4, 1, PE]).astype("bfloat16")
                freqs = paddle.randn([1, 4, 1, PE]).astype("bfloat16")
                fused_rope_cat_key(kv, kpe, freqs, LATENT, PE, mscale=1.37)

            expect_valueerror(half_fp16, "fused_apply_rope_half fp16")
            expect_valueerror(half_bad_shape, "fused_apply_rope_half shape")
            expect_valueerror(half_mscale, "fused_apply_rope_half mscale")
            expect_valueerror(cat_fp16, "fused_rope_cat_key fp16")
            expect_valueerror(cat_dtype_mismatch, "fused_rope_cat_key dtype")
            expect_valueerror(cat_mscale, "fused_rope_cat_key mscale")
            print("OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-O", "-c", child],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        self.assertEqual(
            result.returncode,
            0,
            f"child failed under -O:\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
