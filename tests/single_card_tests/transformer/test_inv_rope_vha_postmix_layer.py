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

"""Layer-level bitwise check for ``fuse_inv_rope_into_vha_postmix``.

``test_inv_rope_vha_postmix_fusion`` pins the fused op in isolation. This runs a
real ``DSv4HybridSelfAttention`` end to end with the flag off and on and requires
the layer output, the input gradient and *every* parameter gradient to be
bitwise identical, so nothing about the surrounding layer (grouped o-proj, gate,
CSA backward, RoPE freq construction) can quietly change when the flag flips.

The postmix is deliberately moved off its identity initialisation (``V`` is
zero-initialised, which would make every postmix gradient exactly zero and the
weight-gradient comparison vacuous).
"""

import unittest

import paddle
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

initialize_fleet(strategy=paddle.distributed.fleet.DistributedStrategy())

_SEED = 42
HIDDEN = 256
NH, VD, PE = 8, 32, 16
NOPE = VD - PE
RANK = 2


def _make_config(**overrides):
    kwargs = {
        "num_hidden_layers": 4,
        "hidden_size": HIDDEN,
        "num_attention_heads": NH,
        "params_dtype": paddle.bfloat16,
        "bf16": True,
        "use_bias": False,
        "multi_latent_attention": True,
        "experimental_attention_variant": "dsv4_hybrid",
        "q_lora_rank": 64,
        "kv_lora_rank": NOPE,
        "qk_nope_head_dim": NOPE,
        "qk_rope_head_dim": PE,
        "qk_pos_emb_head_dim": PE,
        "v_head_dim": VD,
        "hybrid_mla_q_lora_rank": 1536,
        "hybrid_mla_kv_lora_rank": 512,
        "hybrid_mla_qk_nope_head_dim": 192,
        "hybrid_mla_qk_rope_head_dim": 64,
        "hybrid_mla_v_head_dim": 256,
        "hybrid_mla_num_attention_heads": 64,
        "hybrid_mla_num_key_value_heads": 64,
        "o_groups": 4,
        "o_lora_rank": 32,
        "rope_type": "rope",
        "rotary_base": 10000.0,
        "rotary_percent": 1.0,
        "normalization": "RMSNorm",
        "use_qk_norm": True,
        # all-128 => every layer is an HCA layer, which is where the inverse
        # RoPE + postmix pair lives
        "csa_compress_ratios": [128, 128, 128, 128],
        "csa_window_size": 16,
        "dsa_index_n_heads": 4,
        "dsa_index_head_dim": 32,
        "dsa_index_topk": 8,
        "dsa_indexer_loss_coeff": 1.0,
        "dsa_indexer_use_sparse_loss": False,
        "dsa_indexer_rotary_interleaved": False,
        "apply_rope_fusion": True,
        "attention_dropout": 0.0,
        "attention_softmax_in_fp32": True,
        "masked_softmax_fusion": False,
        "softmax_type": "vanilla",
        "csa_indexer_backend": "unfused",
        "csa_sparse_attn_backend": "unfused",
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "csa_dense_mode": False,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "rms_norm_eps": 1e-5,
        "use_vha_attention": True,
        "vha_postmix_rank": RANK,
    }
    kwargs.update(overrides)
    config = TransformerConfig(**kwargs)
    config.dtype = "bfloat16"
    return config


def _force_fuse_flag(config):
    """Turn the flag on behind ``__post_init__``'s back.

    The config validation rejects the combinations below outright, because a flag
    that can never fire is a configuration mistake rather than a mode. The
    layer-level fallback is the second line of defence and is still worth
    pinning, so it is reached by setting the flag after validation.
    """
    config.fuse_inv_rope_into_vha_postmix = True
    return config


def _build(config, layer_number=0):
    """Build the layer with bf16 parameters (the fused RoPE path is bf16 only)."""
    model_parallel_cuda_manual_seed(_SEED)
    prev = paddle.get_default_dtype()
    paddle.set_default_dtype("bfloat16")
    try:
        spec = get_attention_spec(
            config=config,
            attention_layer_type="dsv4_hybrid_attention",
            attn_mask_type=AttnMaskType.causal,
        )
        return build_spec_layer(spec, config=config, layer_number=layer_number)
    finally:
        paddle.set_default_dtype(prev)


def _check_equal(a, b, what):
    a32, b32 = a.astype("float32"), b.astype("float32")
    if bool(paddle.all(a32 == b32)):
        return
    diff = (a32 - b32).abs()
    raise AssertionError(
        f"{what} not bitwise equal: "
        f"{int((a32 != b32).sum())}/{a.numel().item()} elements differ, "
        f"max|diff|={float(diff.max()):.6e}"
    )


def _postmix_init(seed):
    """Non-identity U/V so the postmix parameter gradients are non-trivial."""
    paddle.seed(seed)
    u = (paddle.randn([NH, RANK], "float32") * 0.05).astype("bfloat16")
    v = (paddle.randn([NH, RANK], "float32") * 0.05).astype("bfloat16")
    return u, v


def _run_layer(fuse, sq, layer_number=0, seed=0, expect_gate=None):
    config = _make_config(fuse_inv_rope_into_vha_postmix=fuse)
    attn = _build(config, layer_number=layer_number)
    gate = attn._can_fuse_inv_rope_postmix(False)
    if expect_gate is not None and gate is not expect_gate:
        raise AssertionError(
            f"fuse={fuse} should give gate={expect_gate}, got {gate}"
        )
    u, v = _postmix_init(_SEED + 1)
    attn.vha_postmix_U.set_value(u)
    attn.vha_postmix_V.set_value(v)
    attn.train()

    paddle.seed(seed)
    x = paddle.randn([1, sq, HIDDEN], "bfloat16")
    x.stop_gradient = False
    g_out = paddle.randn([1, sq, HIDDEN], "bfloat16")

    out, _bias = attn(x, attention_mask=None)
    out.backward(g_out)
    grads = {
        name: (None if p.grad is None else p.grad.clone())
        for name, p in attn.named_parameters()
    }
    return out.detach(), x.grad.clone(), grads


class TestInvRopePostmixLayerBitwise(unittest.TestCase):
    def _compare(self, sq, layer_number=0, seed=0):
        ref = _run_layer(False, sq, layer_number, seed, expect_gate=False)
        got = _run_layer(True, sq, layer_number, seed, expect_gate=True)
        tag = f"sq={sq} layer={layer_number} seed={seed}"

        _check_equal(got[0], ref[0], f"layer output ({tag})")
        _check_equal(got[1], ref[1], f"grad hidden_states ({tag})")

        self.assertEqual(sorted(got[2]), sorted(ref[2]))
        checked = 0
        for name in sorted(ref[2]):
            a, b = got[2][name], ref[2][name]
            self.assertEqual(
                a is None, b is None, f"grad presence differs for {name}"
            )
            if a is None:
                continue
            _check_equal(a, b, f"grad {name} ({tag})")
            checked += 1
        # The postmix parameters must be among the compared gradients and must
        # actually carry signal, otherwise the weight-gradient half of this test
        # proves nothing.
        for name in ("vha_postmix_U", "vha_postmix_V"):
            self.assertIsNotNone(ref[2][name], f"{name} got no gradient")
            self.assertGreater(
                float(ref[2][name].astype("float32").abs().sum()),
                0.0,
                f"{name} gradient is all zero; test is vacuous",
            )
        self.assertGreaterEqual(checked, 8, "suspiciously few parameter grads")

    def test_determinism_precondition(self) -> None:
        """Two unfused runs must agree, or nothing else here is meaningful."""
        a = _run_layer(False, 32, expect_gate=False)
        b = _run_layer(False, 32, expect_gate=False)
        _check_equal(a[0], b[0], "unfused output, run twice")
        _check_equal(a[1], b[1], "unfused grad hidden_states, run twice")
        for name in sorted(a[2]):
            if a[2][name] is not None:
                _check_equal(a[2][name], b[2][name], f"unfused grad {name}")

    def test_bitwise_sq32(self) -> None:
        self._compare(32)

    def test_bitwise_sq128(self) -> None:
        self._compare(128)

    def test_bitwise_odd_seq(self) -> None:
        self._compare(97)

    def test_bitwise_other_seed(self) -> None:
        self._compare(64, seed=7)

    def test_bitwise_other_layer(self) -> None:
        self._compare(64, layer_number=2)

    def test_grouped_postmix_falls_back(self) -> None:
        """grouped=True has no [nh,nh] GEMM to split, so it must not fuse."""
        config = _force_fuse_flag(
            _make_config(vha_postmix_grouped=True, vha_postmix_rank=1)
        )
        attn = _build(config)
        self.assertFalse(attn._can_fuse_inv_rope_postmix(False))
        attn.train()
        paddle.seed(0)
        x = paddle.randn([1, 32, HIDDEN], "bfloat16")
        out, _ = attn(x, attention_mask=None)
        self.assertEqual(out.shape, [1, 32, HIDDEN])

    def test_high_precision_rope_falls_back(self) -> None:
        config = _force_fuse_flag(_make_config(high_precision_rope=True))
        attn = _build(config)
        self.assertFalse(attn._can_fuse_inv_rope_postmix(False))

    def test_the_config_rejects_a_flag_that_can_never_fire(self) -> None:
        """The fallback is bitwise identical, so nothing else would report it."""
        for overrides, expected in (
            ({"vha_postmix_grouped": True, "vha_postmix_rank": 1}, "grouped"),
            ({"high_precision_rope": True}, "high_precision_rope"),
            ({"apply_rope_fusion": False}, "apply_rope_fusion"),
            ({"use_vha_attention": False}, "use_vha_attention"),
            ({"qk_pos_emb_head_dim": 0}, "qk_pos_emb_head_dim"),
            # No DSv4HybridAttention is built at all in these two, so there is
            # no postmix to fuse into rather than a postmix of the wrong shape.
            (
                {"experimental_attention_variant": None},
                "experimental_attention_variant",
            ),
            ({"csa_compress_ratios": [-2] * 4}, "csa_compress_ratios"),
        ):
            with (
                self.subTest(**overrides),
                self.assertRaisesRegex(ValueError, expected),
            ):
                _make_config(fuse_inv_rope_into_vha_postmix=True, **overrides)

    def test_selective_postmix_recompute_only_warns(self) -> None:
        """It disables the fusion per layer, so it is a choice, not a mistake."""
        with self.assertLogs(
            "paddlefleet.transformer.transformer_config", level="WARNING"
        ) as caught:
            config = _make_config(
                fuse_inv_rope_into_vha_postmix=True,
                recompute_granularity="selective",
                recompute_modules=["vha_postmix"],
            )
        self.assertTrue(config.fuse_inv_rope_into_vha_postmix)
        joined = "\n".join(caught.output)
        self.assertIn("fuse_inv_rope_into_vha_postmix", joined)
        self.assertIn("every layer", joined)

    def test_a_bounded_recompute_window_is_named_in_the_warning(self) -> None:
        """Only the covered layers lose the fusion; the message must say which."""
        with self.assertLogs(
            "paddlefleet.transformer.transformer_config", level="WARNING"
        ) as caught:
            _make_config(
                fuse_inv_rope_into_vha_postmix=True,
                recompute_granularity="selective",
                recompute_modules=["vha_postmix"],
                recompute_num_layers=2,
                recompute_method="block",
            )
        joined = "\n".join(caught.output)
        self.assertIn("block", joined)
        self.assertIn("2 layers", joined)


if __name__ == "__main__":
    unittest.main()
