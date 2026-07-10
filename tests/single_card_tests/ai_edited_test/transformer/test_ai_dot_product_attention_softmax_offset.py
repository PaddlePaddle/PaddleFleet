# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
Tests for the softmax_offset SDPA branch in DotProductAttention.

Covers dot_product_attention.py:
    if self.softmax_offset is not None:
        attn_output = scaled_dot_product_attention_with_softmax_offset(
            query, key, value_for_sdpa,
            attn_mask_kv=attn_mask_kv,
            is_causal=is_causal,
            softmax_offset=self.softmax_offset,
            q_head_dim=q_head_dim,
        )
"""

import os
import sys

# Add the tests root so sibling imports work.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

# Compatibility shim: local src/ of paddlefleet imports
# `get_fa_version` from `paddlefleet_ops.flash_mask_facade`, which may not be
# present in older installed paddlefleet_ops builds. Inject a stub before
# paddlefleet is imported. The tests below mock out any actual attention
# dispatch, so the stub is never invoked.
try:
    import paddlefleet_ops.flash_mask_facade as _fm_facade

    if not hasattr(_fm_facade, "get_fa_version"):

        def _get_fa_version_stub(*args, **kwargs):
            return 3

        _fm_facade.get_fa_version = _get_fa_version_stub
except ImportError:
    pass

import unittest
from unittest.mock import patch

import paddle

from paddlefleet.transformer import dot_product_attention as dpa_module
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

_HAS_SDPA_SOFTMAX_OFFSET_FN = hasattr(
    dpa_module, "scaled_dot_product_attention_with_softmax_offset"
)


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "softmax_scale": None,
        "use_bias": True,
        "recompute_granularity": None,
        "recompute_modules": None,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "rms_norm_eps": 1e-5,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "window_attn_skip_freq": None,
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "attention_dropout": 0.0,
        "softmax_type": "vanilla",
        "fa_version": None,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


@unittest.skipUnless(
    _HAS_SDPA_SOFTMAX_OFFSET_FN,
    "scaled_dot_product_attention_with_softmax_offset not available in this "
    "build of paddlefleet; the SDPA softmax_offset branch is only present in "
    "newer versions.",
)
class TestSDPASoftmaxOffsetBranch(unittest.TestCase):
    """
    Tests SDPA path selection between
      scaled_dot_product_attention_with_softmax_offset  (softmax_offset != None)
    and
      paddle.nn.functional.scaled_dot_product_attention (softmax_offset is None).
    """

    def _make_attn(self, softmax_type="off-by-one", **cfg_overrides):
        config = _make_config(softmax_type=softmax_type, **cfg_overrides)
        return DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

    @patch(
        "paddlefleet.transformer.dot_product_attention.scaled_dot_product_attention_with_softmax_offset"
    )
    @patch(
        "paddlefleet.transformer.dot_product_attention.paddle.nn.functional.scaled_dot_product_attention"
    )
    def test_calls_softmax_offset_fn_when_offset_not_none(
        self, mock_sdpa, mock_offset_fn
    ):
        """softmax_type='off-by-one' => softmax_offset fn is invoked; default SDPA is not."""
        attn = self._make_attn(softmax_type="off-by-one")
        self.assertIsNotNone(attn.softmax_offset)

        mock_offset_fn.return_value = paddle.randn([1, 4, 4, 32]).astype(
            "bfloat16"
        )

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        attn(query, key, value, None, attn_mask_startend_row_indices=None)

        mock_offset_fn.assert_called_once()
        mock_sdpa.assert_not_called()

        call_args = mock_offset_fn.call_args
        # positional args: query, key, value_for_sdpa
        self.assertEqual(len(call_args.args), 3)
        kwargs = call_args.kwargs
        self.assertIn("attn_mask_kv", kwargs)
        self.assertIn("is_causal", kwargs)
        self.assertIn("softmax_offset", kwargs)
        self.assertIn("q_head_dim", kwargs)
        self.assertIn("dropout_p", kwargs)
        self.assertIn("training", kwargs)
        self.assertIs(kwargs["softmax_offset"], attn.softmax_offset)
        self.assertEqual(kwargs["q_head_dim"], 32)
        # training path: is_causal=True, attention_mask=None -> attn_mask_kv=None
        self.assertTrue(kwargs["is_causal"])
        self.assertIsNone(kwargs["attn_mask_kv"])
        # dropout_p is sourced from config.attention_dropout (0.0 in _make_config)
        self.assertEqual(kwargs["dropout_p"], 0.0)
        # training reflects the module's Layer.training flag (True by default)
        self.assertEqual(kwargs["training"], attn.training)

    @patch(
        "paddlefleet.transformer.dot_product_attention.scaled_dot_product_attention_with_softmax_offset"
    )
    @patch(
        "paddlefleet.transformer.dot_product_attention.paddle.nn.functional.scaled_dot_product_attention"
    )
    def test_calls_default_sdpa_when_offset_none(
        self, mock_sdpa, mock_offset_fn
    ):
        """softmax_type='vanilla' => softmax_offset is None => default SDPA path is taken."""
        attn = self._make_attn(softmax_type="vanilla")
        self.assertIsNone(attn.softmax_offset)

        mock_sdpa.return_value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        attn(query, key, value, None, attn_mask_startend_row_indices=None)

        mock_sdpa.assert_called_once()
        mock_offset_fn.assert_not_called()

    @patch(
        "paddlefleet.transformer.dot_product_attention.scaled_dot_product_attention_with_softmax_offset"
    )
    def test_forwards_attention_mask_as_attn_mask_kv(self, mock_offset_fn):
        """A non-None attention_mask is forwarded as attn_mask_kv to the offset fn."""
        attn = self._make_attn(softmax_type="off-by-one")
        mock_offset_fn.return_value = paddle.randn([1, 4, 4, 32]).astype(
            "bfloat16"
        )

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        attention_mask = paddle.zeros([1, 1, 4, 4], dtype="bfloat16")

        attn(
            query,
            key,
            value,
            attention_mask,
            attn_mask_startend_row_indices=None,
        )

        kwargs = mock_offset_fn.call_args.kwargs
        self.assertIs(kwargs["attn_mask_kv"], attention_mask)
        self.assertTrue(kwargs["is_causal"])

    @patch(
        "paddlefleet.transformer.dot_product_attention.scaled_dot_product_attention_with_softmax_offset"
    )
    def test_learnable_softmax_offset_is_used(self, mock_offset_fn):
        """softmax_type='learnable' also routes through the offset fn."""
        attn = self._make_attn(softmax_type="learnable")
        self.assertIsNotNone(attn.softmax_offset)
        self.assertTrue(
            isinstance(attn.softmax_offset, paddle.Tensor)
            and not attn.softmax_offset.stop_gradient
        )

        mock_offset_fn.return_value = paddle.randn([1, 4, 4, 32]).astype(
            "bfloat16"
        )

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        attn(query, key, value, None, attn_mask_startend_row_indices=None)

        mock_offset_fn.assert_called_once()
        kwargs = mock_offset_fn.call_args.kwargs
        self.assertIs(kwargs["softmax_offset"], attn.softmax_offset)


class TestSoftmaxOffsetFnDropoutTraining(unittest.TestCase):
    """
    Directly exercises scaled_dot_product_attention_with_softmax_offset to
    guard the dropout_p / training branch (regression test: previously used
    an undefined `self.training`).
    """

    @unittest.skipUnless(
        _HAS_SDPA_SOFTMAX_OFFSET_FN,
        "scaled_dot_product_attention_with_softmax_offset not available.",
    )
    def test_dropout_uses_training_argument_no_nameerror(self):
        from paddlefleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        # small tensors, MHA path (groups == 1)
        q = paddle.randn([1, 4, 2, 8]).astype("float32")
        k = paddle.randn([1, 4, 2, 8]).astype("float32")
        v = paddle.randn([1, 4, 2, 8]).astype("float32")
        offset = paddle.zeros([2], dtype="float32")

        # training=False + dropout_p>0: dropout is a no-op, must not raise.
        out_eval = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            softmax_offset=offset,
            q_head_dim=8,
            is_causal=True,
            dropout_p=0.5,
            training=False,
        )
        self.assertEqual(out_eval.shape, [1, 4, 2, 8])

        # training=True + dropout_p>0: dropout path executed, must not raise.
        out_train = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            softmax_offset=offset,
            q_head_dim=8,
            is_causal=True,
            dropout_p=0.5,
            training=True,
        )
        self.assertEqual(out_train.shape, [1, 4, 2, 8])


@unittest.skipUnless(
    _HAS_SDPA_SOFTMAX_OFFSET_FN,
    "scaled_dot_product_attention_with_softmax_offset not available.",
)
class TestSoftmaxOffsetFnMaskBranches(unittest.TestCase):
    """
    Exercises the two attn_mask_kv branches in
    scaled_dot_product_attention_with_softmax_offset:

      if attn_mask_kv.dtype == paddle.bool:
          scores = paddle.where(attn_mask_kv, -inf, scores)   # bool: True = masked
      else:
          scores = scores + attn_mask_kv.cast("float32")     # additive
    """

    def _run(self, mask):
        from paddlefleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        paddle.seed(0)
        q = paddle.randn([1, 2, 1, 4]).astype("float32")  # [B, Q, Hq, dq]
        k = paddle.randn([1, 3, 1, 4]).astype("float32")  # [B, K, Hkv, dk]
        v = paddle.randn([1, 3, 1, 4]).astype("float32")  # [B, K, Hkv, dv]
        offset = paddle.full([1], -1e9, dtype="float32")  # neutralize sink

        out = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            attn_mask_kv=mask,
            is_causal=False,
            softmax_offset=offset,
            q_head_dim=4,
        )
        return out, v

    def test_bool_mask_zeros_out_masked_positions(self):
        """
        With a bool mask that masks all but one key position, the attention
        output must equal V at that unmasked position (softmax collapses to
        a one-hot).
        """
        # mask shape [B, H, Q, K] = [1,1,2,3]. True = masked (per code).
        mask = paddle.to_tensor(
            [[[[True, True, False], [False, True, True]]]], dtype="bool"
        )
        out, v = self._run(mask)

        # For Q=0: only key 2 is unmasked -> out[0, 0, 0] == v[0, 2, 0]
        # For Q=1: only key 0 is unmasked -> out[0, 1, 0] == v[0, 0, 0]
        expected = paddle.stack([v[0, 2, 0], v[0, 0, 0]], axis=0)
        actual = out[0, :, 0]  # [Q=2, dv=4]

        self.assertTrue(
            paddle.allclose(actual, expected, atol=1e-5).item(),
            msg=f"bool-mask path collapsed weights incorrectly: {actual} vs {expected}",
        )

    def test_additive_mask_uses_same_path_as_before(self):
        """
        A float additive mask that assigns -inf to a position must zero out
        that position's contribution, same as the bool mask would.
        """
        neg_inf = float("-inf")
        # mask everything except key 2 for Q=0 and everything except key 0 for Q=1
        mask = paddle.to_tensor(
            [[[[neg_inf, neg_inf, 0.0], [0.0, neg_inf, neg_inf]]]],
            dtype="float32",
        )
        out, v = self._run(mask)

        expected = paddle.stack([v[0, 2, 0], v[0, 0, 0]], axis=0)
        actual = out[0, :, 0]

        self.assertTrue(
            paddle.allclose(actual, expected, atol=1e-5).item(),
            msg=f"additive-mask path collapsed weights incorrectly: {actual} vs {expected}",
        )

    def test_bool_and_additive_mask_produce_same_output(self):
        """Bool mask and its equivalent additive (-inf) mask must match."""
        bool_mask = paddle.to_tensor(
            [[[[True, False, True], [False, False, True]]]], dtype="bool"
        )
        neg_inf = float("-inf")
        add_mask = paddle.to_tensor(
            [[[[neg_inf, 0.0, neg_inf], [0.0, 0.0, neg_inf]]]],
            dtype="float32",
        )
        out_bool, _ = self._run(bool_mask)
        out_add, _ = self._run(add_mask)

        self.assertTrue(
            paddle.allclose(out_bool, out_add, atol=1e-5).item(),
            msg="bool and additive masks should produce the same output",
        )


@unittest.skipUnless(
    _HAS_SDPA_SOFTMAX_OFFSET_FN,
    "scaled_dot_product_attention_with_softmax_offset not available.",
)
class TestSoftmaxOffsetFnRowMaxWithSink(unittest.TestCase):
    """
    Guards the numerical-stability change:
        row_max = paddle.maximum(scores.max(axis=-1, keepdim=True), sink)

    When the sink far exceeds all scores, the softmax weights on real keys
    must approach zero (all mass absorbed by the virtual sink token).
    """

    def test_large_sink_dominates_weights(self):
        from paddlefleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        paddle.seed(0)
        q = paddle.randn([1, 2, 1, 4]).astype("float32")
        k = paddle.randn([1, 3, 1, 4]).astype("float32")
        v = paddle.randn([1, 3, 1, 4]).astype("float32")

        # Sink much larger than any score -> exp(sink - row_max) dominates
        # and all attention weights on real tokens approach zero, so the
        # output approaches zero.
        offset = paddle.full([1], 50.0, dtype="float32")

        out = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            softmax_offset=offset,
            q_head_dim=4,
        )

        self.assertTrue(
            paddle.all(paddle.abs(out) < 1e-6).item(),
            msg=f"large sink should absorb all weight; got out={out}",
        )

    def test_small_sink_matches_plain_softmax(self):
        """With a very negative sink, output must equal plain softmax(QK)V."""
        from paddlefleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        paddle.seed(0)
        q = paddle.randn([1, 2, 1, 4]).astype("float32")
        k = paddle.randn([1, 3, 1, 4]).astype("float32")
        v = paddle.randn([1, 3, 1, 4]).astype("float32")

        offset = paddle.full([1], -1e9, dtype="float32")
        out = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            softmax_offset=offset,
            q_head_dim=4,
        )

        # Manual reference: plain softmax(QK^T / sqrt(d)) @ V (no sink).
        # q,k,v are [B, Q, H, D]; transpose to [B, H, Q, D] to matmul.
        qh = q.transpose([0, 2, 1, 3])
        kh = k.transpose([0, 2, 1, 3])
        vh = v.transpose([0, 2, 1, 3])
        scale = 4**-0.5
        scores = paddle.matmul(qh, kh.transpose([0, 1, 3, 2])) * scale
        weights = paddle.nn.functional.softmax(scores, axis=-1)
        ref = paddle.matmul(weights, vh).transpose([0, 2, 1, 3])

        self.assertTrue(
            paddle.allclose(out, ref, atol=1e-5).item(),
            msg=f"small sink should reduce to plain softmax; out={out} ref={ref}",
        )


if __name__ == "__main__":
    unittest.main()
