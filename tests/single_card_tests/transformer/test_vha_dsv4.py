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

"""Unit tests for VHA (Virtual Head Attention) in DSv4 hybrid attention.

Covers the postmix (grouped / ungrouped low-rank cross-head mixing) and premix
(structured Q up-projection) additions to ``DSv4HybridSelfAttention`` /
``DSv4HybridAttention``, the config-validation ValueErrors, the muon_slice_specs
branching, and forward / backward (gradient) coverage of the postmix and premix
parameters.
"""

import unittest

import paddle
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.muon_utils import ortho_per_head, ortho_stacked
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

initialize_fleet(strategy=paddle.distributed.fleet.DistributedStrategy())

_SEED = 42


def _make_config(
    num_layers=4,
    hidden_size=256,
    num_attention_heads=8,
    v_head_dim=32,
    qk_nope_head_dim=16,
    qk_rope_head_dim=16,
    q_lora_rank=64,
    o_groups=4,
    o_lora_rank=32,
    params_dtype=paddle.float32,
    bf16=False,
    csa_compress_ratios=None,
    **overrides,
):
    """Minimal dsv4-hybrid config; VHA knobs passed via **overrides."""
    if csa_compress_ratios is None:
        csa_compress_ratios = [0, 4, 128, 4]
    kwargs = {
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "params_dtype": params_dtype,
        "bf16": bf16,
        "use_bias": False,
        "multi_latent_attention": True,
        "experimental_attention_variant": "dsv4_hybrid",
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": v_head_dim - qk_rope_head_dim,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "qk_pos_emb_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "hybrid_mla_q_lora_rank": 1536,
        "hybrid_mla_kv_lora_rank": 512,
        "hybrid_mla_qk_nope_head_dim": 192,
        "hybrid_mla_qk_rope_head_dim": 64,
        "hybrid_mla_v_head_dim": 256,
        "hybrid_mla_num_attention_heads": 64,
        "hybrid_mla_num_key_value_heads": 64,
        "o_groups": o_groups,
        "o_lora_rank": o_lora_rank,
        "rope_type": "rope",
        "rotary_base": 10000.0,
        "rotary_percent": 1.0,
        "normalization": "RMSNorm",
        "use_qk_norm": True,
        "csa_compress_ratios": csa_compress_ratios,
        "csa_window_size": 16,
        "dsa_index_n_heads": 4,
        "dsa_index_head_dim": 32,
        "dsa_index_topk": 8,
        "dsa_indexer_loss_coeff": 1.0,
        "dsa_indexer_use_sparse_loss": False,
        "dsa_indexer_rotary_interleaved": False,
        "apply_rope_fusion": False,
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
    }
    kwargs.update(overrides)
    config = TransformerConfig(**kwargs)
    # Force float32 weights (config.dtype drives create_parameter for the
    # grouped o-proj and premix). Real training uses bf16 via amp, but these
    # are numerical unit tests; float32 keeps postmix / fuse / forward dtypes
    # consistent (bf16 has no GPU allclose kernel and breaks the grouped matmul
    # when mixed with the float32 postmix params).
    config.dtype = "float32"
    return config


def _build(config, layer_number=0):
    model_parallel_cuda_manual_seed(_SEED)
    spec = get_attention_spec(
        config=config,
        attention_layer_type="dsv4_hybrid_attention",
        attn_mask_type=AttnMaskType.causal,
    )
    return build_spec_layer(spec, config=config, layer_number=layer_number)


class TestPostmixInit(unittest.TestCase):
    def test_ungrouped_shapes(self):
        attn = _build(_make_config(use_vha_attention=True, vha_postmix_rank=4))
        self.assertTrue(attn.use_vha_postmix)
        self.assertFalse(attn.vha_postmix_grouped)
        self.assertEqual(attn.vha_postmix_U.shape, [8, 4])
        self.assertEqual(attn.vha_postmix_V.shape, [8, 4])

    def test_grouped_shapes(self):
        attn = _build(
            _make_config(
                use_vha_attention=True,
                vha_postmix_grouped=True,
                vha_postmix_rank=2,
            )
        )
        # o_groups=4, group_heads = nh(8)//4 = 2
        self.assertEqual(attn.vha_postmix_U.shape, [4, 2, 2])
        self.assertEqual(attn.vha_postmix_V.shape, [4, 2, 2])

    def test_rank_default_ungrouped(self):
        # rank None -> mix_heads // 4 = nh(8)//4 = 2
        attn = _build(
            _make_config(use_vha_attention=True, vha_postmix_rank=None)
        )
        self.assertEqual(attn.vha_postmix_rank, 2)

    def test_rank_default_grouped_clamped_min1(self):
        # grouped: mix_heads = group_heads = 2; 2//4 = 0 -> clamped to 1
        attn = _build(
            _make_config(
                use_vha_attention=True,
                vha_postmix_grouped=True,
                vha_postmix_rank=None,
            )
        )
        self.assertEqual(attn.vha_postmix_rank, 1)

    def test_rank_clamped_to_mix_heads(self):
        # rank far above nh is clamped down to nh (=8)
        attn = _build(
            _make_config(use_vha_attention=True, vha_postmix_rank=999)
        )
        self.assertEqual(attn.vha_postmix_rank, 8)

    def test_grouped_head_divisibility_error(self):
        # nh=6, v_head_dim=32, o_groups=4: 6*32 % 4 == 0 (base check passes) but
        # 6 % 4 != 0, so grouped postmix must raise.
        with self.assertRaises(ValueError):
            _build(
                _make_config(
                    num_attention_heads=6,
                    v_head_dim=32,
                    o_groups=4,
                    use_vha_attention=True,
                    vha_postmix_grouped=True,
                    vha_postmix_rank=1,
                )
            )

    def test_recompute_flag_on(self):
        attn = _build(
            _make_config(
                use_vha_attention=True,
                recompute_granularity="selective",
                recompute_modules=["vha_postmix"],
            )
        )
        self.assertTrue(attn.recompute_vha_postmix)

    def test_recompute_flag_off(self):
        attn = _build(_make_config(use_vha_attention=True))
        self.assertFalse(attn.recompute_vha_postmix)

    def test_recompute_first_n_hit(self):
        # num_layers=4, first_n with n=2 -> layers 0,1 recompute.
        attn = _build(
            _make_config(
                use_vha_attention=True,
                recompute_granularity="selective",
                recompute_modules=["vha_postmix"],
                recompute_method="first_n",
                recompute_num_layers=2,
            ),
            layer_number=1,
        )
        self.assertTrue(attn.recompute_vha_postmix)

    def test_recompute_first_n_miss(self):
        # first_n with n=2 -> layer 3 is beyond the window -> no recompute.
        attn = _build(
            _make_config(
                use_vha_attention=True,
                recompute_granularity="selective",
                recompute_modules=["vha_postmix"],
                recompute_method="first_n",
                recompute_num_layers=2,
            ),
            layer_number=3,
        )
        self.assertFalse(attn.recompute_vha_postmix)

    def test_recompute_block_hit(self):
        # single chunk (pp=vpp=1), block n=2 -> layers 0,1 recompute.
        attn = _build(
            _make_config(
                use_vha_attention=True,
                recompute_granularity="selective",
                recompute_modules=["vha_postmix"],
                recompute_method="block",
                recompute_num_layers=2,
            ),
            layer_number=0,
        )
        self.assertTrue(attn.recompute_vha_postmix)

    def test_recompute_block_miss(self):
        # block n=2 -> layer 2 falls outside the per-chunk window.
        attn = _build(
            _make_config(
                use_vha_attention=True,
                recompute_granularity="selective",
                recompute_modules=["vha_postmix"],
                recompute_method="block",
                recompute_num_layers=2,
            ),
            layer_number=2,
        )
        self.assertFalse(attn.recompute_vha_postmix)

    def test_recompute_dict_layer_count(self):
        # dict recompute_modules now selects layers per submodule: first_n with
        # n=1 covers layer 0 only.
        attn = _build(
            _make_config(
                use_vha_attention=True,
                recompute_granularity="selective",
                recompute_modules={"vha_postmix": 1},
                recompute_method="first_n",
            ),
            layer_number=0,
        )
        self.assertTrue(attn.recompute_vha_postmix)
        attn = _build(
            _make_config(
                use_vha_attention=True,
                recompute_granularity="selective",
                recompute_modules={"vha_postmix": 1},
                recompute_method="first_n",
            ),
            layer_number=1,
        )
        self.assertFalse(attn.recompute_vha_postmix)

    def test_recompute_dict_layer_list(self):
        # An explicit layer list needs no recompute_method.
        cfg_kwargs = {
            "use_vha_attention": True,
            "recompute_granularity": "selective",
            "recompute_modules": {"vha_postmix": [2]},
        }
        self.assertTrue(
            _build(
                _make_config(**cfg_kwargs), layer_number=2
            ).recompute_vha_postmix
        )
        self.assertFalse(
            _build(
                _make_config(**cfg_kwargs), layer_number=1
            ).recompute_vha_postmix
        )


class TestPostmixApply(unittest.TestCase):
    def test_ungrouped_identity_at_init(self):
        # V is zero-initialised -> postmix is identity at construction.
        attn = _build(_make_config(use_vha_attention=True, vha_postmix_rank=4))
        x = paddle.randn([2, 16, 8 * 32])
        out = attn._apply_vha_postmix(x)
        self.assertEqual(out.shape, [2, 16, 8 * 32])
        self.assertTrue(paddle.allclose(out, x, atol=1e-6).item())

    def test_grouped_identity_at_init(self):
        attn = _build(
            _make_config(
                use_vha_attention=True,
                vha_postmix_grouped=True,
                vha_postmix_rank=2,
            )
        )
        x = paddle.randn([2, 16, 8 * 32])
        out = attn._apply_vha_postmix(x)
        self.assertEqual(out.shape, [2, 16, 8 * 32])
        self.assertTrue(paddle.allclose(out, x, atol=1e-6).item())

    def test_ungrouped_matches_manual_einsum(self):
        attn = _build(_make_config(use_vha_attention=True, vha_postmix_rank=4))
        # Randomise V so the correction is non-trivial.
        attn.vha_postmix_V.set_value(paddle.randn(attn.vha_postmix_V.shape))
        x = paddle.randn([2, 16, 8 * 32])
        out = attn._apply_vha_postmix(x)
        nh, vd = 8, 32
        xm = x.reshape([2, 16, nh, vd])
        z = paddle.einsum("bthd,hr->btrd", xm, attn.vha_postmix_U)
        delta = paddle.einsum("btrd,hr->bthd", z, attn.vha_postmix_V)
        ref = (xm + delta).reshape([2, 16, nh * vd])
        self.assertTrue(paddle.allclose(out, ref, atol=1e-5).item())


class TestPremixInit(unittest.TestCase):
    def test_shapes_and_replaces_up_proj(self):
        # nh=8, q_lora_rank=64, g_q=2 -> k=4, d_q=32, q_head_dim=32.
        attn = _build(
            _make_config(
                use_vha_attention=True, use_vha_premix=True, vha_premix_groups=2
            )
        )
        self.assertTrue(attn.use_vha_premix)
        self.assertIsNone(attn.linear_q_up_proj)
        self.assertEqual(attn.vha_premix_groups, 2)
        self.assertEqual(attn.vha_premix_expand, 4)  # k = nh // g_q
        self.assertEqual(attn.vha_premix_dq, 32)  # d_q = q_lora_rank // g_q
        self.assertEqual(attn.vha_premix_weight.shape, [4, 32, 32])

    def test_premix_requires_vha_attention(self):
        # use_vha_premix is gated by use_vha_attention: off unless both set.
        attn = _build(
            _make_config(
                use_vha_attention=False,
                use_vha_premix=True,
                vha_premix_groups=2,
            )
        )
        self.assertFalse(attn.use_vha_premix)
        self.assertIsNotNone(attn.linear_q_up_proj)

    def test_groups_none_raises(self):
        with self.assertRaises(ValueError):
            _build(
                _make_config(
                    use_vha_attention=True,
                    use_vha_premix=True,
                    vha_premix_groups=None,
                )
            )

    def test_groups_nonpositive_raises(self):
        with self.assertRaises(ValueError):
            _build(
                _make_config(
                    use_vha_attention=True,
                    use_vha_premix=True,
                    vha_premix_groups=0,
                )
            )

    def test_groups_nh_indivisible_raises(self):
        # nh=8, g_q=3 -> 8 % 3 != 0.
        with self.assertRaises(ValueError):
            _build(
                _make_config(
                    use_vha_attention=True,
                    use_vha_premix=True,
                    vha_premix_groups=3,
                )
            )

    def test_groups_qlora_indivisible_raises(self):
        # nh=8, q_lora_rank=20, g_q=8: 8 % 8 == 0 but 20 % 8 != 0.
        with self.assertRaises(ValueError):
            _build(
                _make_config(
                    q_lora_rank=20,
                    use_vha_attention=True,
                    use_vha_premix=True,
                    vha_premix_groups=8,
                )
            )


class TestMuonSliceSpecs(unittest.TestCase):
    def test_non_split_head_returns_empty(self):
        attn = _build(
            _make_config(
                use_vha_attention=True, use_vha_premix=True, vha_premix_groups=2
            )
        )
        self.assertEqual(
            attn.muon_slice_specs({"muon_qkv_update_mode": "tp"}), {}
        )

    def test_premix_on_marks_premix_weight(self):
        attn = _build(
            _make_config(
                use_vha_attention=True, use_vha_premix=True, vha_premix_groups=2
            )
        )
        specs = attn.muon_slice_specs({"muon_qkv_update_mode": "split_head"})
        self.assertIn("vha_premix_weight", specs)
        self.assertIs(specs["vha_premix_weight"][0], ortho_stacked)
        # premix replaces linear_q_up_proj (None) -> not marked.
        self.assertNotIn("linear_q_up_proj.weight", specs)
        self.assertIn("linear_o_group_proj", specs)

    def test_premix_off_marks_up_proj(self):
        attn = _build(
            _make_config(use_vha_attention=True, use_vha_premix=False)
        )
        specs = attn.muon_slice_specs({"muon_qkv_update_mode": "split_head"})
        self.assertIn("linear_q_up_proj.weight", specs)
        self.assertIs(specs["linear_q_up_proj.weight"][0], ortho_per_head)
        self.assertEqual(
            specs["linear_q_up_proj.weight"][1]["heads"],
            attn.num_attention_heads_per_partition,
        )
        self.assertNotIn("vha_premix_weight", specs)

    def test_default_mode_is_split_head(self):
        # muon_slice_specs defaults muon_qkv_update_mode to "split_head".
        attn = _build(
            _make_config(use_vha_attention=True, use_vha_premix=False)
        )
        specs = attn.muon_slice_specs({})
        self.assertIn("linear_o_group_proj", specs)


class TestPostmixBackward(unittest.TestCase):
    def _check(self, attn):
        # V is zero-initialised, so dL/dU is zero at init; randomise V (small)
        # to make both U and V receive a non-trivial gradient.
        attn.vha_postmix_V.set_value(
            paddle.randn(attn.vha_postmix_V.shape) * 0.1
        )
        x = paddle.randn([2, 16, 8 * 32])
        x.stop_gradient = False
        out = attn._apply_vha_postmix(x)
        out.sum().backward()
        self.assertIsNotNone(attn.vha_postmix_U.grad)
        self.assertIsNotNone(attn.vha_postmix_V.grad)
        self.assertIsNotNone(x.grad)
        self.assertGreater(attn.vha_postmix_U.grad.abs().sum().item(), 0.0)
        self.assertGreater(attn.vha_postmix_V.grad.abs().sum().item(), 0.0)

    def test_ungrouped_param_grads(self):
        self._check(
            _build(_make_config(use_vha_attention=True, vha_postmix_rank=4))
        )

    def test_grouped_param_grads(self):
        self._check(
            _build(
                _make_config(
                    use_vha_attention=True,
                    vha_postmix_grouped=True,
                    vha_postmix_rank=2,
                )
            )
        )


class TestPremixBackward(unittest.TestCase):
    def test_premix_weight_grad_via_einsum(self):
        attn = _build(
            _make_config(
                use_vha_attention=True, use_vha_premix=True, vha_premix_groups=2
            )
        )
        # Mirror the premix einsum (btgr,krd->btkgd) on the built weight and
        # check the gradient flows to vha_premix_weight and the compressed Q.
        b, sq = 2, 16
        g = attn.vha_premix_groups
        d_q = attn.vha_premix_dq
        q_comp = paddle.randn([b, sq, g, d_q])
        q_comp.stop_gradient = False
        w = attn.vha_premix_weight
        out = paddle.einsum("btgr,krd->btkgd", q_comp, w)
        out.sum().backward()
        self.assertIsNotNone(w.grad)
        self.assertIsNotNone(q_comp.grad)
        self.assertGreater(w.grad.abs().sum().item(), 0.0)


class TestForward(unittest.TestCase):
    # batch_size must be 1: DSv4HybridAttention rejects b>1 in indexer mode.
    def _run(self, attn, b=1, sq=16):
        attn.eval()
        x = paddle.randn([b, sq, attn.config.hidden_size])
        out, bias = attn(x, attention_mask=None)
        return out

    def test_forward_ungrouped_postmix(self):
        attn = _build(_make_config(use_vha_attention=True, vha_postmix_rank=4))
        out = self._run(attn)
        self.assertEqual(out.shape, [1, 16, 256])

    def test_forward_grouped_postmix(self):
        attn = _build(
            _make_config(
                use_vha_attention=True,
                vha_postmix_grouped=True,
                vha_postmix_rank=2,
            )
        )
        out = self._run(attn)
        self.assertEqual(out.shape, [1, 16, 256])

    def test_forward_premix(self):
        # Exercises the premix branch in get_query_key_value_tensors.
        attn = _build(
            _make_config(
                use_vha_attention=True, use_vha_premix=True, vha_premix_groups=2
            )
        )
        out = self._run(attn)
        self.assertEqual(out.shape, [1, 16, 256])

    def test_forward_premix_and_postmix(self):
        attn = _build(
            _make_config(
                use_vha_attention=True,
                use_vha_premix=True,
                vha_premix_groups=2,
                vha_postmix_grouped=True,
                vha_postmix_rank=2,
            )
        )
        out = self._run(attn)
        self.assertEqual(out.shape, [1, 16, 256])

    def test_forward_postmix_recompute_branch(self):
        # recompute_vha_postmix True + training -> selective-recompute path.
        attn = _build(
            _make_config(
                use_vha_attention=True,
                vha_postmix_rank=4,
                recompute_granularity="selective",
                recompute_modules=["vha_postmix"],
            )
        )
        self.assertTrue(attn.recompute_vha_postmix)
        attn.train()
        x = paddle.randn([1, 16, 256])
        out, bias = attn(x, attention_mask=None)
        self.assertEqual(out.shape, [1, 16, 256])

    def test_forward_postmix_identity_at_init(self):
        # postmix V=0 at init -> _apply_vha_postmix is the identity, so a forward
        # with postmix active must equal the same layer with postmix bypassed.
        attn = _build(_make_config(use_vha_attention=True, vha_postmix_rank=4))
        attn.eval()
        x = paddle.randn([1, 16, 256])
        out_with, _ = attn(x, attention_mask=None)
        attn.use_vha_postmix = False  # bypass the (identity) postmix
        out_without, _ = attn(x, attention_mask=None)
        self.assertTrue(
            paddle.allclose(out_with, out_without, atol=1e-6).item()
        )

    def test_backward_premix_and_postmix_params(self):
        # Full forward + backward with premix + (grouped) postmix active: every
        # VHA parameter and the input must receive a gradient.
        attn = _build(
            _make_config(
                use_vha_attention=True,
                use_vha_premix=True,
                vha_premix_groups=2,
                vha_postmix_grouped=True,
                vha_postmix_rank=2,
            )
        )
        attn.vha_postmix_V.set_value(
            paddle.randn(attn.vha_postmix_V.shape) * 0.1
        )
        attn.train()
        x = paddle.randn([1, 16, 256])
        x.stop_gradient = False
        out, _ = attn(x, attention_mask=None)
        out.sum().backward()
        self.assertIsNotNone(attn.vha_premix_weight.grad)
        self.assertIsNotNone(attn.vha_postmix_U.grad)
        self.assertIsNotNone(attn.vha_postmix_V.grad)
        self.assertIsNotNone(x.grad)

    def test_backward_postmix_recompute_branch(self):
        # Selective recompute + train + backward: gradients still reach the
        # input and the postmix parameters through the recompute wrapper.
        attn = _build(
            _make_config(
                use_vha_attention=True,
                vha_postmix_rank=4,
                recompute_granularity="selective",
                recompute_modules=["vha_postmix"],
            )
        )
        self.assertTrue(attn.recompute_vha_postmix)
        attn.vha_postmix_V.set_value(
            paddle.randn(attn.vha_postmix_V.shape) * 0.1
        )
        attn.train()
        x = paddle.randn([1, 16, 256])
        x.stop_gradient = False
        out, _ = attn(x, attention_mask=None)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(attn.vha_postmix_U.grad)
        self.assertIsNotNone(attn.vha_postmix_V.grad)


if __name__ == "__main__":
    unittest.main()
