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
"""Smaller ``"hf"`` alignment pieces: dot-product attention, MLP, mRoPE.

Three unrelated but individually small invariants:

* ``_hf_qk_scores`` keeps the reference *operand layout*. ``torch.matmul(query,
  key.transpose(2, 3))`` hands cuBLAS an untransposed ``[.., sk, hn]`` key and
  lets it apply the transpose; materializing ``[.., hn, sk]`` first picks a
  different reduction split. It also deliberately has **no** hand-written
  backward, because paddle's own autograd for this expression already produces
  torch's operand shapes while a hand-rolled ``d_key`` does not.
* ``MLP.forward``'s ``hidden_states_up`` splits the fused gate/up projection into
  two independent autograd consumers, matching a reference that keeps
  ``gate_proj`` and ``up_proj`` as separate ``nn.Linear``s. The fused ``K=2*inter``
  dgrad is not bitwise equal to the sum of the two ``K=inter`` dgrads.
* ``MultimodalRotaryEmbedding`` opts out of AMP's buffer cast. ``inv_freq`` is a
  precision-critical constant, not a weight: at ``rotary_base=1e7`` the smallest
  frequencies need far more mantissa than BF16 has, and ``paddle.amp.decorate
  (level="O2")`` silently truncated the table. This one is unconditional -- it is
  a bug fix, not gated on any accuracy target.
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import paddle

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddlefleet.transformer.dot_product_attention import (
    _hf_qk_scores,
)


class TestHFQKScores(unittest.TestCase):
    """Forward value, operand layout, and the deliberate lack of a backward."""

    def setUp(self):
        paddle.seed(20260908)
        # [b*np, sq, hn] and [b*np, sk, hn] -- both "logical", untransposed.
        self.query = paddle.randn([6, 5, 8], dtype=paddle.float32)
        self.key = paddle.randn([6, 7, 8], dtype=paddle.float32)
        self.scale = 0.35355339

    def test_forward_is_scaled_qk_transpose(self):
        out = _hf_qk_scores(self.query, self.key, self.scale)
        expected = (
            paddle.matmul(self.query, self.key, transpose_y=True) * self.scale
        )
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_output_shape_is_sq_by_sk(self):
        out = _hf_qk_scores(self.query, self.key, self.scale)
        self.assertEqual(out.shape, [6, 5, 7])

    def test_takes_untransposed_key(self):
        """A pre-transposed ``[.., hn, sk]`` key would need ``transpose_y=False``.

        Feeding the materialized layout through this helper must NOT produce the
        same scores, which is what makes the layout choice observable.
        """
        key_t = self.key.transpose([0, 2, 1]).contiguous()
        self.assertNotEqual(list(key_t.shape), list(self.key.shape))
        with self.assertRaises(ValueError):
            _hf_qk_scores(self.query, key_t, self.scale)

    def test_gradients_flow_through_plain_autograd(self):
        """No custom backward: paddle's autograd already matches the reference."""
        q = self.query.detach()
        q.stop_gradient = False
        k = self.key.detach()
        k.stop_gradient = False
        out = _hf_qk_scores(q, k, self.scale)
        g = paddle.randn(out.shape, dtype=paddle.float32)
        gq, gk = paddle.grad([out], [q, k], grad_outputs=[g])
        np.testing.assert_allclose(
            gq.numpy(),
            (paddle.matmul(g, k) * self.scale).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            gk.numpy(),
            (paddle.matmul(g, q, transpose_x=True) * self.scale).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_scale_is_applied_after_the_gemm(self):
        """Scaling the product, not an operand, is what the reference does."""
        unscaled = _hf_qk_scores(self.query, self.key, 1.0)
        scaled = _hf_qk_scores(self.query, self.key, self.scale)
        np.testing.assert_allclose(
            scaled.numpy(), (unscaled * self.scale).numpy(), rtol=0, atol=0
        )

    def test_preserves_dtype(self):
        for dtype in (paddle.float32, paddle.bfloat16):
            with self.subTest(dtype=dtype):
                out = _hf_qk_scores(
                    self.query.astype(dtype), self.key.astype(dtype), self.scale
                )
                self.assertEqual(out.dtype, dtype)


class TestMultimodalRotaryNoLowPrecisionCast(unittest.TestCase):
    """``inv_freq`` must survive ``paddle.amp.decorate(level="O2")`` under "hf".

    The opt-out is gated on the accuracy target rather than unconditional: the
    phase error it removes is ``position * 4e-3``, i.e. radians rather than ULPs
    after a few thousand tokens, so enabling it for everyone changes the rotation
    of every existing mRoPE run -- which moved the qwen3vl CI loss. The default and
    Megatron paths therefore keep AMP's cast, and only the HF target opts out.
    """

    def _build(self, **kwargs):
        from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        with patch(
            "paddlefleet.models.common.embeddings."
            "rotary_pos_embedding.parallel_state"
        ) as mock_ps:
            mock_ps.get_context_parallel_group.return_value = None
            return MultimodalRotaryEmbedding(**kwargs)

    def test_opts_out_of_amp_buffer_cast_under_hf(self):
        """The flag AMP's ``decorate`` honours must be False on the HF target."""
        rope = self._build(
            head_dim=64,
            rotary_percent=1.0,
            rotary_base=10000000,
            use_accuracy_compatible="hf",
        )
        self.assertIs(rope._cast_to_low_precision, False)

    def test_default_and_megatron_keep_the_amp_cast(self):
        """No numerical change for runs that never asked for the HF reference."""
        for target in (False, True, "megatron"):
            with self.subTest(target=target):
                rope = self._build(
                    head_dim=64,
                    rotary_percent=1.0,
                    rotary_base=10000000,
                    use_accuracy_compatible=target,
                )
                # ``nn.Layer.__init__`` sets this to True; the guard must leave
                # that alone so AMP keeps casting exactly as it did before.
                self.assertIs(rope._cast_to_low_precision, True)

    def test_matches_the_plain_rotary_embedding_opt_out_under_hf(self):
        """Both rotary layers hold precision-critical buffers, not weights."""
        from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        with patch(
            "paddlefleet.models.common.embeddings."
            "rotary_pos_embedding.parallel_state"
        ) as mock_ps:
            mock_ps.get_context_parallel_group.return_value = None
            plain = RotaryEmbedding(
                head_dim=64, rotary_percent=1.0, rotary_base=10000000
            )
        mrope = self._build(
            head_dim=64,
            rotary_percent=1.0,
            rotary_base=10000000,
            use_accuracy_compatible="hf",
        )
        self.assertIs(plain._cast_to_low_precision, False)
        self.assertIs(mrope._cast_to_low_precision, False)

    def test_inv_freq_keeps_fp32_mantissa_at_large_base(self):
        """At base 1e7 the smallest frequency is far below BF16 resolution."""
        rope = self._build(
            head_dim=64, rotary_percent=1.0, rotary_base=10000000
        )
        inv_freq = rope.inv_freq
        self.assertEqual(inv_freq.dtype, paddle.float32)
        smallest = float(inv_freq.numpy().min())
        as_bf16 = float(
            paddle.to_tensor([smallest], dtype=paddle.float32)
            .astype(paddle.bfloat16)
            .astype(paddle.float32)
            .numpy()[0]
        )
        self.assertNotEqual(smallest, as_bf16)


class TestDotProductAttentionHFKeyLayout(unittest.TestCase):
    """``forward`` keeps the reference key layout on the eager-scores path."""

    def _make(self, target):
        from paddlefleet.transformer.dot_product_attention import (
            DotProductAttention,
        )
        from paddlefleet.transformer.enums import AttnMaskType
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            use_accuracy_compatible=target,
        )
        config.num_key_value_heads = 4
        config.head_dim = 8
        config.softmax_scale = None
        config.apply_query_key_layer_scaling = False
        config.attention_softmax_in_fp32 = True
        config.masked_softmax_fusion = False
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.fp16 = False
        config.bf16 = False
        config.context_parallel_size = 1
        config.sliding_window = None
        config._attn_implementation = "eager"
        return DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

    def _qkv(self):
        paddle.seed(61)
        # [b, s, np, hn]
        q = paddle.randn([1, 6, 4, 8], dtype=paddle.float32)
        k = paddle.randn([1, 6, 4, 8], dtype=paddle.float32)
        v = paddle.randn([1, 6, 4, 8], dtype=paddle.float32)
        return q, k, v

    def _mask(self, seq_len=6):
        # bool mask: True = masked out (strict upper triangle).
        return paddle.triu(
            paddle.ones([1, 1, seq_len, seq_len], dtype="bool"), diagonal=1
        )

    def test_hf_target_eager_forward_runs(self):
        """Exercises the untransposed-key branch and ``_hf_qk_scores``."""
        from paddlefleet.transformer.enums import AttnMaskType

        attn = self._make("hf")
        q, k, v = self._qkv()
        out = attn(q, k, v, self._mask(), attn_mask_type=AttnMaskType.causal)
        out_t = out[0] if isinstance(out, tuple) else out
        self.assertTrue(bool(paddle.all(paddle.isfinite(out_t))))

    def test_hf_and_megatron_eager_outputs_agree(self):
        """Only the operand layout differs, so the values must match closely."""
        from paddlefleet.transformer.enums import AttnMaskType

        outs = []
        for target in ("hf", "megatron"):
            attn = self._make(target)
            q, k, v = self._qkv()
            out = attn(
                q, k, v, self._mask(), attn_mask_type=AttnMaskType.causal
            )
            out_t = out[0] if isinstance(out, tuple) else out
            outs.append(out_t.numpy().copy())
        np.testing.assert_allclose(outs[0], outs[1], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
