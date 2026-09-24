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

"""MLA's ``mla_o_proj`` point: the wiring must not change numerics.

The mechanism itself is covered by ``test_auto_refined_recompute.py``.
"""

import unittest

import paddle
from paddle.distributed.fleet.utils import recompute

from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

_SEED = 42
_BATCH, _SEQ, _HIDDEN = 2, 4, 128

_RR_ON = {
    "recompute_granularity": "full",
    "recompute_method": "uniform",
    "recompute_num_layers": 1,
    "recompute_modules": ["mla_o_proj"],
}


class _Projection(paddle.nn.Layer):
    """A Fleet projection under ``use_bias=False``: returns ``(out, None)``."""

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(
            in_features, out_features, bias_attr=False
        )

    def forward(self, x):
        return self.linear(x), None


class _RMSNorm(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_size = kwargs.get("normalized_shape", kwargs.get("hidden_size"))
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.eps = kwargs.get("norm_eps", kwargs.get("eps", 1e-5))

    def forward(self, x):
        scale = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def _config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": _HIDDEN,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "softmax_scale": None,
        "use_bias": False,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "recompute_modules": None,
        "apply_rope_fusion": False,
        "rotary_interleaved": False,
        "multi_latent_attention": True,
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
        "kv_lora_rank": 32,
        "q_lora_rank": 64,
        "qk_nope_head_dim": 24,
        "qk_rope_head_dim": 8,
        "v_head_dim": 32,
        "rope_type": "rope",
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _build(**overrides):
    spec = MLASelfAttentionSublayersSpec(
        q_proj=_Projection,
        q_a_proj=_Projection,
        q_b_proj=_Projection,
        kv_a_proj_with_mqa=_Projection,
        kv_b_proj=_Projection,
        core_attention=DotProductAttention,
        o_proj=_Projection,
        q_a_layernorm=_RMSNorm,
        kv_a_layernorm=_RMSNorm,
    )
    paddle.seed(_SEED)
    attn = MLASelfAttention(
        config=_config(**overrides), sublayers_spec=spec, layer_number=1
    )
    attn.train()
    inner = attn.o_proj.forward
    attn.o_proj_calls = 0

    def counted(x):
        attn.o_proj_calls += 1
        return inner(x)

    attn.o_proj.forward = counted
    return attn


def _run(attn):
    paddle.seed(_SEED)
    x = paddle.randn([_BATCH, _SEQ, _HIDDEN])
    x.stop_gradient = False
    output = recompute(lambda h: attn(h, attention_mask=None)[0], x)
    # Position weighting: a plain sum would hide a row mix-up by cancelling it.
    weights = paddle.arange(1, output.shape[-1] + 1, dtype=output.dtype)
    (output * weights).sum().backward()
    grads = {
        name: param.grad.detach()
        for name, param in attn.named_parameters()
        if param.grad is not None
    }
    return output.detach(), x.grad.detach(), grads


class TestDecision(unittest.TestCase):
    def test_listing_the_point_under_full_recompute_enables_it(self):
        attn = _build(**_RR_ON)
        self.assertTrue(attn.use_rr_o_proj)
        self.assertEqual(attn._o_proj_rr.name, "mla_o_proj")

    def test_not_listing_the_point_leaves_it_off(self):
        attn = _build(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            recompute_modules=["flash_attn"],
        )
        self.assertFalse(attn.use_rr_o_proj)
        self.assertIsNone(attn._o_proj_rr)

    def test_a_bias_vetoes_the_point(self):
        attn = _build(use_bias=True, **_RR_ON)
        self.assertFalse(attn.use_rr_o_proj)


class TestNumericalEquivalence(unittest.TestCase):
    def test_matches_the_plain_recompute_path_bitwise(self):
        base = _build()
        refined = _build(**_RR_ON)
        refined.set_state_dict(base.state_dict())

        want = _run(base)
        got = _run(refined)

        # o_proj runs twice without RR (first pass + recompute) and once with.
        self.assertEqual(base.o_proj_calls, 2)
        self.assertEqual(refined.o_proj_calls, 1)
        self.assertEqual(refined._o_proj_rr.pending, 0)

        names = ("output", "input grad")
        for name, expected, actual in zip(
            names, want[:2], got[:2], strict=True
        ):
            self.assertTrue(
                paddle.equal_all(expected, actual).item(),
                f"{name} mismatch: max diff "
                f"{(expected - actual).abs().max().item()}",
            )
        self.assertEqual(sorted(want[2]), sorted(got[2]))
        for key in want[2]:
            self.assertTrue(
                paddle.equal_all(want[2][key], got[2][key]).item(),
                f"grad of {key} mismatch: max diff "
                f"{(want[2][key] - got[2][key]).abs().max().item()}",
            )


class TestNoBackwardPathQueuesNothing(unittest.TestCase):
    """A forward with no backward must not leave a frame behind.

    The boundary reads only ``tracer._has_grad``, so an inference forward is
    indistinguishable from a first recompute pass. Without the ``self.training``
    guard each eval forward would queue a frame no backward consumes, and the
    next training step would replay that stale frame instead of its own.
    """

    def test_eval_forwards_queue_no_frames(self):
        attn = _build(**_RR_ON)
        attn.eval()

        x = paddle.randn([_BATCH, _SEQ, _HIDDEN])
        with paddle.no_grad():
            for _ in range(3):
                attn(x, attention_mask=None)

        self.assertEqual(attn._o_proj_rr.pending, 0)
        # Plain path, so o_proj really ran each time rather than being skipped.
        self.assertEqual(attn.o_proj_calls, 3)

    def test_training_step_after_eval_still_balances(self):
        attn = _build(**_RR_ON)

        attn.eval()
        x = paddle.randn([_BATCH, _SEQ, _HIDDEN])
        with paddle.no_grad():
            attn(x, attention_mask=None)

        attn.train()
        _run(attn)
        self.assertEqual(attn._o_proj_rr.pending, 0)


if __name__ == "__main__":
    unittest.main()
