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

"""Unit tests for the PrefixLMTritonCore host-side logic.

The Triton kernel launch itself is exercised on GPU by the parity tests; here
it is replaced with a stub so the surrounding permute / reshape / guard logic
can be validated deterministically on CPU.
"""

import types
import unittest
from unittest import mock

import paddle

from paddlefleet.transformer import (
    prefix_lm_triton_core as core_mod,
)
from paddlefleet.transformer.prefix_lm_triton_core import (
    PrefixLMTritonCore,
)


def _cfg(head_dim=None, hidden_size=32, num_attention_heads=4, **over):
    base = {
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
    }
    base.update(over)
    return types.SimpleNamespace(**base)


def _psp(layout=None):
    if layout is None:
        layout = {
            "segment_starts": (0,),
            "n_contexts": (2,),
            "n_queries": (2,),
            "pad_len": 0,
        }
    return types.SimpleNamespace(prefix_lm_layout=layout)


class TestInit(unittest.TestCase):
    def test_head_dim_from_config(self):
        core = PrefixLMTritonCore(_cfg(head_dim=8))
        self.assertEqual(core.head_dim, 8)
        self.assertAlmostEqual(core.softmax_scale, 1.0 / (8**0.5))

    def test_head_dim_derived(self):
        core = PrefixLMTritonCore(
            _cfg(head_dim=None, hidden_size=32, num_attention_heads=4)
        )
        self.assertEqual(core.head_dim, 8)

    def test_softmax_scale_override(self):
        core = PrefixLMTritonCore(_cfg(head_dim=8), softmax_scale=0.5)
        self.assertAlmostEqual(core.softmax_scale, 0.5)

    def test_block_sizes_default(self):
        core = PrefixLMTritonCore(_cfg(head_dim=8))
        self.assertEqual(core.block_m, 64)
        self.assertEqual(core.block_n, 64)
        self.assertEqual(core.fwd_warps, 4)
        self.assertEqual(core.plan_cache_size, 64)

    def test_tuning_from_config(self):
        core = PrefixLMTritonCore(
            _cfg(
                head_dim=8,
                hyperencoder_triton_block_m=32,
                hyperencoder_triton_block_n=16,
                hyperencoder_triton_fwd_warps=8,
                hyperencoder_triton_fwd_stages=3,
                hyperencoder_triton_bwd_warps=2,
                hyperencoder_triton_bwd_stages=1,
                hyperencoder_triton_plan_cache_size=0,
            )
        )
        self.assertEqual(core.block_m, 32)
        self.assertEqual(core.block_n, 16)
        self.assertEqual(core.fwd_warps, 8)
        self.assertEqual(core.fwd_stages, 3)
        self.assertEqual(core.bwd_warps, 2)
        self.assertEqual(core.bwd_stages, 1)
        self.assertEqual(core.plan_cache_size, 0)

    def test_context_parallel_rejected(self):
        cp = types.SimpleNamespace(nranks=2)
        pg = types.SimpleNamespace(cp=cp)
        with self.assertRaises(RuntimeError):
            PrefixLMTritonCore(_cfg(head_dim=8), pg_collection=pg)

    def test_context_parallel_size_one_ok(self):
        cp = types.SimpleNamespace(nranks=1)
        pg = types.SimpleNamespace(cp=cp)
        core = PrefixLMTritonCore(_cfg(head_dim=8), pg_collection=pg)
        self.assertEqual(core.context_parallel_size, 1)


class TestResolveLayout(unittest.TestCase):
    def test_non_dict_raises(self):
        core = PrefixLMTritonCore(_cfg(head_dim=8))
        with self.assertRaises(RuntimeError):
            core._resolve_layout(4, _psp(layout="not-a-dict"))

    def test_none_params_raises(self):
        core = PrefixLMTritonCore(_cfg(head_dim=8))
        with self.assertRaises(RuntimeError):
            core._resolve_layout(4, None)

    def test_cache_miss_then_hit(self):
        core = PrefixLMTritonCore(_cfg(head_dim=8))
        sentinel = object()
        with mock.patch.object(
            core_mod, "layout_from_hyperbody_dict", return_value=sentinel
        ) as builder:
            first = core._resolve_layout(4, _psp())
            second = core._resolve_layout(4, _psp())
        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        # Second call must be served from the module cache, not rebuilt.
        builder.assert_called_once()


class TestForwardGuards(unittest.TestCase):
    def setUp(self):
        self.core = PrefixLMTritonCore(_cfg(head_dim=8))

    def test_attention_bias_rejected(self):
        q = paddle.randn([1, 4, 4, 8])
        with self.assertRaises(ValueError):
            self.core.forward(q, q, q, attention_bias=q)

    def test_attention_mask_rejected(self):
        q = paddle.randn([1, 4, 4, 8])
        with self.assertRaises(ValueError):
            self.core.forward(q, q, q, attention_mask=paddle.ones([1, 1, 4, 4]))

    def test_row_indices_rejected(self):
        q = paddle.randn([1, 4, 4, 8])
        with self.assertRaises(ValueError):
            self.core.forward(
                q,
                q,
                q,
                attn_mask_startend_row_indices=paddle.zeros([1, 1, 4, 4]),
            )

    def test_non_4d_rejected(self):
        q = paddle.randn([4, 4, 8])
        with self.assertRaises(ValueError):
            self.core.forward(q, q, q)

    def test_gqa_rejected(self):
        q = paddle.randn([1, 4, 4, 8])  # 4 heads
        k = paddle.randn([1, 4, 2, 8])  # 2 heads
        with self.assertRaises(ValueError):
            self.core.forward(q, k, k, packed_seq_params=_psp())


class TestForwardHappyPath(unittest.TestCase):
    def test_permute_reshape_around_kernel(self):
        core = PrefixLMTritonCore(_cfg(head_dim=8))
        b, s, n, d = 1, 4, 4, 8
        q = paddle.randn([b, s, n, d])

        def fake_kernel(qt, kt, vt, layout, scale, block_m, block_n, **kw):
            # kernel operates on [B, N, S, D]; echo the shape back.
            self.assertEqual(qt.shape, [b, n, s, d])
            # tuning knobs are forwarded from config as keyword args.
            self.assertIn("fwd_warps", kw)
            self.assertIn("plan_cache_size", kw)
            return paddle.zeros([b, n, s, d], dtype=qt.dtype)

        with (
            mock.patch.object(
                core_mod, "triton_prefix_lm_attention", side_effect=fake_kernel
            ),
            mock.patch.object(
                core_mod, "layout_from_hyperbody_dict", return_value=object()
            ),
        ):
            out = core.forward(q, q, q, packed_seq_params=_psp())
        # [B, N, S, D] -> [B, S, N*D]
        self.assertEqual(out.shape, [b, s, n * d])


if __name__ == "__main__":
    unittest.main()
