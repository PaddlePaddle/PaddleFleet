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

"""Tests for the in-place Triton-fused MLA RoPE.

Verifies, against the unfused PaddleFleet reference path, that
`fused_apply_mla_rope_inplace`:

  - Forward output equivalent (bit-exact match) to the slice + rope +
    concat baseline used in dsv4_hybrid_attention.py.
  - Truly in-place when clone_input=False: q.data_ptr() preserved and not a
    single byte of extra device memory allocated.
  - With clone_input=True: the input is left completely untouched and the
    result is bit-identical to the clone_input=False result.
  - Backward grads match the autograd reference.
"""

import unittest

import paddle

from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddlefleet.triton_ops import fused_apply_mla_rope_inplace
from paddlefleet.triton_ops.mla_rope_inplace_fusion import (
    RoPEMLAInplaceFusion,
    _fused_cos_sin,
)

# Shapes from DeepSeek-V4-Flash
B, S, H, D = 1, 4096, 64, 512
NOPE_DIM = 448
ROPE_DIM = D - NOPE_DIM  # 64


def _reference_forward(
    q: paddle.Tensor,
    freqs: paddle.Tensor,
    inverse: bool = False,
) -> paddle.Tensor:
    """Slice + rope + concat baseline (current implementation)."""
    q_nope = q[..., :NOPE_DIM]
    q_pe = q[..., NOPE_DIM:]
    q_pe = _apply_rotary_pos_emb_bshd(
        q_pe,
        freqs,
        mscale=1.0,
        rotary_interleaved=False,
        multi_latent_attention=True,
        inverse=inverse,
        mla_output_remove_interleaving=True,
    )
    return paddle.concat([q_nope, q_pe], axis=-1)


def _check_equal(a: paddle.Tensor, b: paddle.Tensor):
    """Binary-exact equality check (no tolerance allowed)."""
    assert paddle.all(a == b), f"tensor not equal:\nA: {a}\nB: {b}"


class TestFusedMLARopeQPeInplace(unittest.TestCase):
    def setUp(self) -> None:
        paddle.seed(0)

    def _run_case(
        self,
        b: int,
        s: int,
        freqs: paddle.Tensor,
        inverse: bool = False,
        clone_input: bool = False,
    ) -> None:
        """Shared driver: q is contiguous bf16; freqs supplied by caller."""
        x = paddle.randn([b, s, H, D], "bfloat16")
        x.stop_gradient = False

        # ---- reference path ----
        x_ref = x.detach()
        x_ref.stop_gradient = False
        q_ref = x_ref.clone()  # non-leaf
        out_ref = _reference_forward(q_ref, freqs, inverse=inverse)
        out_grad = paddle.randn_like(out_ref)
        out_ref.backward(out_grad)
        grad_ref = x_ref.grad

        # ---- fused path ----
        x_fused = x.detach()
        x_fused.stop_gradient = False
        q_fused = x_fused.clone()  # non-leaf — safe target for in-place kernel
        self.assertTrue(q_fused.is_contiguous())
        input_before = q_fused.clone()
        ptr_before = q_fused.data_ptr()
        out_fused = fused_apply_mla_rope_inplace(
            q_fused, freqs, NOPE_DIM, inverse=inverse, clone_input=clone_input
        )

        # ---- storage invariants ----
        if clone_input:
            # A fresh buffer, and the input must survive completely intact —
            # this is what the attention backward relies on.
            self.assertIsNot(out_fused, q_fused)
            self.assertNotEqual(out_fused.data_ptr(), ptr_before)
            _check_equal(q_fused, input_before)
        else:
            self.assertIs(out_fused, q_fused)
            self.assertEqual(out_fused.data_ptr(), ptr_before)
        # The nope channels of the result always carry the input's values.
        _check_equal(out_fused[..., :NOPE_DIM], input_before[..., :NOPE_DIM])

        # ---- forward parity ----
        _check_equal(out_fused, out_ref)

        # ---- backward parity ----
        out_fused.backward(out_grad)
        grad_fused = x_fused.grad

        _check_equal(grad_fused[..., :NOPE_DIM], grad_ref[..., :NOPE_DIM])
        _check_equal(grad_fused[..., NOPE_DIM:], grad_ref[..., NOPE_DIM:])

    def test_forward_backward(self) -> None:
        """Test the normal case."""
        freqs = paddle.randn([B, S, 1, ROPE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, S, freqs)

    def test_forward_backward_clone_input(self) -> None:
        """Same as above but with clone_input=True (o inv-rope call site)."""
        freqs = paddle.randn([B, S, 1, ROPE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, S, freqs, clone_input=True)

    def test_freqs_noncontiguous_b_gt_1(self) -> None:
        """Test multi-batch and non-contiguous freqs."""
        b = 2
        s = 128  # smaller to keep test fast
        # Build oversize freqs and slice along seq (non-contig stride),
        # then unsqueeze the singleton head dim from a slice as well.
        rope_len = s + 17  # arbitrary position_offset
        freqs_full = paddle.randn([b, rope_len, ROPE_DIM])
        freqs_full.stop_gradient = True
        freqs = freqs_full[:, 17 : 17 + s, :].unsqueeze(2)  # [b, s, 1, D]
        # Sanity: this slice is non-contiguous.
        self.assertFalse(freqs.is_contiguous())
        self._run_case(b, s, freqs)

    def test_inverse(self) -> None:
        """Test inverse rope."""
        freqs = paddle.randn([B, S, 1, ROPE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, S, freqs, inverse=True)

    def test_inverse_clone_input(self) -> None:
        """Inverse rope with clone_input=True: the production o path."""
        freqs = paddle.randn([B, S, 1, ROPE_DIM])
        freqs.stop_gradient = True
        self._run_case(B, S, freqs, inverse=True, clone_input=True)

    def test_clone_input_matches_inplace_bitwise(self) -> None:
        """clone_input must not change a single bit of the result.

        Runs both modes on identical inputs and compares outputs and grads
        bit-for-bit, so the out-of-place kernel path cannot silently diverge
        from the in-place one (e.g. by reordering the bf16 rounding).
        """
        s = 256
        freqs = paddle.randn([B, s, 1, ROPE_DIM])
        freqs.stop_gradient = True
        x = paddle.randn([B, s, H, D], "bfloat16")
        out_grad = paddle.randn([B, s, H, D], "bfloat16")

        results = {}
        for clone_input in (False, True):
            leaf = x.detach()
            leaf.stop_gradient = False
            q = leaf.clone()
            out = fused_apply_mla_rope_inplace(
                q, freqs, NOPE_DIM, inverse=True, clone_input=clone_input
            )
            out.backward(out_grad.clone())
            results[clone_input] = (out.clone(), leaf.grad.clone())

        _check_equal(results[False][0], results[True][0])
        _check_equal(results[False][1], results[True][1])

    def test_inplace_allocates_nothing(self) -> None:
        """clone_input=False must cost zero extra device memory.

        Measured around `RoPEMLAInplaceFusion.apply` rather than the public
        wrapper, so the cos/sin buffers `_fused_cos_sin` allocates do not
        pollute the reading. clone_input=True is checked in the same way to
        confirm it allocates exactly one output tensor and nothing more.
        """
        s = 256
        freqs = paddle.randn([B, s, 1, ROPE_DIM])
        freqs.stop_gradient = True
        cos, sin = _fused_cos_sin(freqs, 1.0, False, paddle.bfloat16)
        nbytes = B * s * H * D * 2  # bf16

        with paddle.no_grad():
            t = paddle.randn([B, s, H, D], "bfloat16")
            # Warm up the JIT / any lazy allocator growth first.
            RoPEMLAInplaceFusion.apply(t, cos, sin, NOPE_DIM, ROPE_DIM, False)
            RoPEMLAInplaceFusion.apply(t, cos, sin, NOPE_DIM, ROPE_DIM, True)
            paddle.device.synchronize()

            before = paddle.device.cuda.memory_allocated()
            out_ip = RoPEMLAInplaceFusion.apply(
                t, cos, sin, NOPE_DIM, ROPE_DIM, False
            )
            paddle.device.synchronize()
            delta_inplace = paddle.device.cuda.memory_allocated() - before
            self.assertIs(out_ip, t)

            before = paddle.device.cuda.memory_allocated()
            out_oop = RoPEMLAInplaceFusion.apply(
                t, cos, sin, NOPE_DIM, ROPE_DIM, True
            )
            paddle.device.synchronize()
            delta_clone = paddle.device.cuda.memory_allocated() - before

        self.assertEqual(
            delta_inplace,
            0,
            f"clone_input=False allocated {delta_inplace} bytes; the in-place "
            "path must not allocate",
        )
        self.assertEqual(
            delta_clone,
            nbytes,
            f"clone_input=True allocated {delta_clone} bytes, expected exactly "
            f"one output tensor ({nbytes})",
        )
        del out_oop

    def test_fused_cos_sin(self) -> None:
        freqs = paddle.randn([B, S, 1, ROPE_DIM])
        dtype = paddle.bfloat16
        mscale = 0.5

        cos_ref = (paddle.cos(freqs) * mscale).to(dtype)
        sin_ref = (paddle.sin(freqs) * mscale).to(dtype)

        cos_fused, sin_fused = _fused_cos_sin(
            freqs, mscale, inverse=False, dtype=dtype
        )

        _check_equal(cos_ref, cos_fused)
        _check_equal(sin_ref, sin_fused)


if __name__ == "__main__":
    unittest.main()
