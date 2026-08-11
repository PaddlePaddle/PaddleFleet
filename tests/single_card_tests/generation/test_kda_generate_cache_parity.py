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

"""Parity between KDA full-sequence and prefill+decode-step paths.

Two levels of verification:

1. **Layer level**: ``KimiDeltaAttention.forward(use_cache=True)`` prefill
   followed by single-token decode steps must reproduce
   ``forward(use_cache=False)`` over the full sequence, token for token.
2. **Generate level**: ``generate(no_cache=True)`` vs ``generate(no_cache=False)``
   must produce identical greedy tokens.
"""

from __future__ import annotations

import unittest

import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlefleet.generation.greedy_generator import DynamicKVCache
from paddlefleet.transformer.kimi_delta_attention import (
    KimiDeltaAttention,
    KimiDeltaAttentionSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

# ---- Local stand-in layers (no fleet / TP required) ----


class _NoBiasLinear(nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias_attr=False)

    def forward(self, x):
        return self.linear(x), None

    def backward_dw(self):
        pass


class _SimpleRMSNorm(nn.Layer):
    def __init__(self, normalized_shape, eps=1e-5, norm_eps=None, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[normalized_shape],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = norm_eps if norm_eps is not None else eps

    def forward(self, x):
        x_float = x.astype(paddle.float32)
        rms = paddle.rsqrt(
            x_float.pow(2).mean(axis=-1, keepdim=True) + self.eps
        )
        return (x_float * rms * self.weight.astype(paddle.float32)).astype(
            x.dtype
        )


class _FakeGroup:
    ranks = [0]
    nranks = 1


class _FakePGCollection:
    def __init__(self):
        self.tp = _FakeGroup()


HIDDEN_SIZE = 64
NUM_KEY_HEADS = 4
NUM_VALUE_HEADS = 4
KEY_HEAD_DIM = 16
VALUE_HEAD_DIM = 16
CONV_KERNEL_DIM = 4
GATE_LOWER_BOUND = -5.0


def _build_kda(
    use_full_rank_gate=True,
    num_key_heads=NUM_KEY_HEADS,
    num_value_heads=NUM_VALUE_HEADS,
):
    """Build a single-GPU KDA layer with the paddle-native path forced on."""
    config = TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=num_key_heads,
        num_hidden_layers=1,
        hidden_act=F.silu,
        rms_norm_eps=1e-5,
        normalization="RMSNorm",
        sequence_parallel=False,
        deterministic_mode=True,  # disables fused kernels -> paddle native
    )
    spec = KimiDeltaAttentionSublayersSpec(
        in_proj=_NoBiasLinear,
        f_a_proj=_NoBiasLinear,
        f_b_proj=_NoBiasLinear,
        g_a_proj=_NoBiasLinear,
        g_b_proj=_NoBiasLinear,
        out_norm=_SimpleRMSNorm,
        out_proj=_NoBiasLinear,
    )
    return KimiDeltaAttention(
        config=config,
        sublayers_spec=spec,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=_FakePGCollection(),
        conv_kernel_dim=CONV_KERNEL_DIM,
        key_head_dim=KEY_HEAD_DIM,
        value_head_dim=VALUE_HEAD_DIM,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        gate_lora_rank=VALUE_HEAD_DIM,
        use_full_rank_gate=use_full_rank_gate,
        gate_lower_bound=GATE_LOWER_BOUND,
    )


def _make_cache(num_layers=1):
    """The real ``DynamicKVCache`` -- KDA only touches its kda_* slots."""
    return DynamicKVCache(num_layers=num_layers)


class _KdaLayerParityTests:
    """Prefill+stepwise decode must match the full-sequence forward."""

    use_full_rank_gate = True
    num_key_heads = NUM_KEY_HEADS
    num_value_heads = NUM_VALUE_HEADS
    prefill_len = 20  # not a chunk multiple (chunk_size defaults to 64)
    decode_steps = 6
    batch = 2

    def setUp(self):
        paddle.seed(0)
        self.kda = _build_kda(
            use_full_rank_gate=self.use_full_rank_gate,
            num_key_heads=self.num_key_heads,
            num_value_heads=self.num_value_heads,
        )
        self.kda.eval()
        total = self.prefill_len + self.decode_steps
        self.hidden = paddle.randn([self.batch, total, HIDDEN_SIZE])

    def test_prefill_then_decode_matches_full(self):
        with paddle.no_grad():
            full_out, _ = self.kda(self.hidden, use_cache=False)

            cache = _make_cache()
            prefill = self.hidden[:, : self.prefill_len, :]
            cached_outs = []
            out, _ = self.kda(
                prefill,
                past_key_values=cache,
                layer_idx=0,
                use_cache=True,
            )
            cached_outs.append(out)
            for t in range(
                self.prefill_len, self.prefill_len + self.decode_steps
            ):
                step = self.hidden[:, t : t + 1, :]
                out, _ = self.kda(
                    step,
                    past_key_values=cache,
                    layer_idx=0,
                    use_cache=True,
                )
                cached_outs.append(out)
            cached = paddle.concat(cached_outs, axis=1)

        self.assertEqual(list(cached.shape), list(full_out.shape))
        assert paddle.allclose(cached, full_out, atol=1e-4, rtol=1e-4).item(), (
            (cached - full_out).abs().max().item()
        )

    def test_state_shape_and_dtype(self):
        cache = _make_cache()
        with paddle.no_grad():
            self.kda(
                self.hidden[:, : self.prefill_len, :],
                past_key_values=cache,
                layer_idx=0,
                use_cache=True,
            )
        state, conv_state = cache.get_kda_state(0)
        self.assertEqual(
            list(state.shape),
            [self.batch, self.num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM],
        )
        self.assertEqual(state.dtype, paddle.float32)
        conv_dim = KEY_HEAD_DIM * self.num_key_heads * 2 + (
            VALUE_HEAD_DIM * self.num_value_heads
        )
        self.assertEqual(
            list(conv_state.shape),
            [self.batch, conv_dim, CONV_KERNEL_DIM - 1],
        )
        self.assertEqual(conv_state.dtype, paddle.float32)


class TestKdaLayerParityFullRankGate(_KdaLayerParityTests, unittest.TestCase):
    use_full_rank_gate = True


class TestKdaLayerParityLowRankGate(_KdaLayerParityTests, unittest.TestCase):
    use_full_rank_gate = False


class TestKdaLayerParityGVA(_KdaLayerParityTests, unittest.TestCase):
    """Grouped value attention: hv > h."""

    num_key_heads = 2
    num_value_heads = 4


class TestKdaLayerParityChunkMultiple(_KdaLayerParityTests, unittest.TestCase):
    """Prefill length that is an exact chunk multiple."""

    prefill_len = 64


class TestKdaLayerParityShortPrefill(_KdaLayerParityTests, unittest.TestCase):
    """Prefill shorter than the conv window (exercises the zero pad)."""

    prefill_len = 2


class TestKdaDecodeGuards(unittest.TestCase):
    """The unsupported-feature guards must fire on the cache path."""

    def setUp(self):
        paddle.seed(0)
        self.kda = _build_kda()

    def _assert_cache_rejected(self, message, **kwargs):
        with self.assertRaisesRegex(NotImplementedError, message):
            self.kda(
                paddle.randn([1, 1, HIDDEN_SIZE]),
                past_key_values=_make_cache(),
                layer_idx=0,
                use_cache=True,
                **kwargs,
            )

    def test_tensor_parallel_rejected(self):
        self.kda.tp_size = 2
        self._assert_cache_rejected("tensor parallel")

    def test_sequence_parallel_rejected(self):
        self.kda.config.sequence_parallel = True
        self._assert_cache_rejected("sequence parallel")

    def test_context_parallel_rejected(self):
        self.kda.cp_size = 2
        self._assert_cache_rejected("context parallel")

    def test_cu_seqlens_rejected(self):
        self._assert_cache_rejected(
            "variable-length",
            cu_seqlens=paddle.to_tensor([0, 1], dtype="int32"),
        )

    def test_startend_mask_rejected(self):
        self._assert_cache_rejected(
            "variable-length",
            attn_mask_startend_row_indices=paddle.ones(
                [1, 1, 1, 1], dtype="int32"
            ),
        )

    def test_multi_token_decode_rejected(self):
        cache = _make_cache()
        hidden = paddle.randn([1, 8, HIDDEN_SIZE])
        with paddle.no_grad():
            self.kda(hidden, past_key_values=cache, layer_idx=0, use_cache=True)
        # Second call with >1 token while a state already exists must raise.
        with self.assertRaises(NotImplementedError), paddle.no_grad():
            self.kda(
                paddle.randn([1, 3, HIDDEN_SIZE]),
                past_key_values=cache,
                layer_idx=0,
                use_cache=True,
            )

    def test_packed_seq_rejected(self):
        with self.assertRaises(NotImplementedError):
            self.kda(
                paddle.randn([1, 4, HIDDEN_SIZE]),
                packed_seq_params=object(),
            )


if __name__ == "__main__":
    unittest.main()
