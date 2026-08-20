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

"""Stage-level selective recompute for the HCA/CSA attention block.

``recompute_modules`` gains three entries that partition exactly what the coarse
``full_attn`` entry covers:

- ``hca_pre_csa``  : q/kv projections, qk norm, RoPE
- ``hca_csa``      : the CSA core attention (Compressor + Indexer + sparse attn)
- ``hca_post_csa`` : inverse RoPE, VHA postmix, grouped o_group_proj, gated_attn

Recompute must not change the math, so every combination is compared **bitwise**
(``paddle.equal_all``, not ``allclose``) against the no-recompute reference:
forward output, input gradient and every parameter gradient.

Determinism
-----------
A bitwise comparison is only meaningful if the kernels are run-to-run
deterministic, otherwise the reference itself moves. Three things are pinned:

- ``FLAGS_cudnn_deterministic`` / ``FLAGS_embedding_deterministic`` are set
  before paddle is imported.
- ``attention_dropout=0`` and ``preserve_rng_state=False`` on every segment: no
  RNG is consumed inside a segment, so a replayed forward cannot diverge.
- the attention backends are pinned per test. ``unfused`` is pure paddle;
  ``tilelang`` has a deterministic backward. The cuDNN DSA backward's dKV
  scatter-add is *not* deterministic (see ``test_block_sparse_dsa_gradcheck.py``)
  and is therefore not covered here.

Scope
-----
The bitwise tests use HCA layers (``csa_compress_ratios`` entry 128, no Indexer),
which is what the layer43 configs run. CSA layers with an active Indexer (ratio
2..127 with ``csa_dense_mode=false``) are deliberately not compared bitwise:
``csa_attention.py`` gates the indexer-loss path on ``need_indexer_loss =
self.training and paddle.is_grad_enabled()``, so whether the CSA core sits inside
a recomputed segment decides which branch its first, no-grad pass takes, and the
top-k handed to the sparse attention comes out different (~20% relative on the
layer output in this harness). That predates the stage split -- plain
``full_attn`` shows the same offset against no-recompute -- so recompute is not
value-preserving there and asserting equality would encode a property the code
does not have. What is checked for that shape is that the Indexer still receives
a gradient through the recomputed segment.

Also not covered: the VHA postmix inside the post-CSA stage. Its parameters are
created without an explicit dtype, so in this harness they stay fp32 while the
activations are bf16; production only lines up because of the O2 amp cast. The
``vha_postmix`` entry is still exercised as a no-op in the combinations below.
"""

import os
import unittest

os.environ.setdefault("FLAGS_cudnn_deterministic", "True")
os.environ.setdefault("FLAGS_embedding_deterministic", "1")

import paddle
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig

_SEED = 42
_HIDDEN_SIZE = 256

# Every stage combination that must stay bitwise identical to no-recompute.
_STAGE_COMBOS = [
    ["hca_pre_csa"],
    ["hca_csa"],
    ["hca_post_csa"],
    ["hca_pre_csa", "hca_csa"],
    ["hca_csa", "hca_post_csa"],
    ["hca_pre_csa", "hca_post_csa"],
    ["hca_pre_csa", "hca_csa", "hca_post_csa"],
    # full_attn covers the same span; kept here as a regression anchor.
    ["full_attn"],
    # Nested submodule recompute must be suppressed inside an outer segment
    # without changing the result.
    ["hca_pre_csa", "hca_csa", "hca_post_csa", "gated_attn", "vha_postmix"],
    ["hca_csa", "gated_attn", "vha_postmix"],
]


def _make_config(
    *,
    recompute_modules=None,
    compress_ratio=128,
    csa_dense_mode=True,
    indexer_backend="unfused",
    sparse_attn_backend="unfused",
    gated_attention=True,
    recompute_num_layers=None,
    recompute_method=None,
    v_head_dim=32,
):
    """DSv4 hybrid attention config, single layer, HCA (ratio 128) by default."""
    return TransformerConfig(
        num_hidden_layers=1,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=8,
        params_dtype=paddle.bfloat16,
        bf16=True,
        use_bias=False,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        q_lora_rank=64,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=v_head_dim,
        o_groups=4,
        o_lora_rank=32,
        rope_type="rope",
        rotary_base=10000.0,
        rotary_percent=1.0,
        normalization="RMSNorm",
        use_qk_norm=True,
        csa_compress_ratios=[compress_ratio],
        csa_window_size=16,
        dsa_index_n_heads=4,
        dsa_index_head_dim=32,
        dsa_index_topk=8,
        dsa_indexer_loss_coeff=1.0,
        dsa_indexer_use_sparse_loss=False,
        dsa_indexer_rotary_interleaved=False,
        apply_rope_fusion=False,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        softmax_type="vanilla",
        csa_indexer_backend=indexer_backend,
        csa_sparse_attn_backend=sparse_attn_backend,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        csa_dense_mode=csa_dense_mode,
        gated_attention=gated_attention,
        gated_attn_use_q_lora=False,
        recompute_granularity=(
            "selective" if recompute_modules is not None else None
        ),
        recompute_modules=recompute_modules,
        recompute_num_layers=recompute_num_layers,
        recompute_method=recompute_method,
    )


def _build_attention(config, layer_number=0):
    spec = get_attention_spec(
        config=config,
        attention_layer_type="dsv4_hybrid_attention",
        attn_mask_type=AttnMaskType.causal,
    )
    return build_spec_layer(spec, config=config, layer_number=layer_number)


def _startend(batch_size, seq_len):
    return paddle.full([batch_size, 1, seq_len, 1], seq_len, dtype="int32")


def _grad_source(hidden):
    """Return (leaf, layer_input) where layer_input is a NON-leaf tensor.

    ``paddle.distributed.fleet.utils.recompute`` rejects a leaf tensor that
    requires grad as a segment input ("Leaf Var ... can't use inplace
    strategy"). In a real model the attention input is always the output of a
    preceding op, so multiply by one to reproduce that. The multiplication is
    exact, so it does not perturb the bitwise comparison.
    """
    leaf = hidden.clone()
    leaf.stop_gradient = False
    return leaf, leaf * 1.0


def _forward_backward(attn, hidden, startend):
    leaf, x = _grad_source(hidden)
    output, _ = attn(
        hidden_states=x,
        attention_mask=None,
        attn_mask_startend_row_indices=startend,
    )
    output.cast("float32").sum().backward()
    grads = {
        name: (None if p.grad is None else p.grad.clone())
        for name, p in attn.named_parameters()
    }
    return output.detach().clone(), leaf.grad.clone(), grads


class TestHCAStagedRecompute(unittest.TestCase):
    """Bitwise forward/backward alignment of the three HCA recompute stages."""

    @classmethod
    def setUpClass(cls):
        if not paddle.device.is_compiled_with_cuda():
            raise unittest.SkipTest("CUDA build of Paddle is required")
        if paddle.device.cuda.device_count() == 0:
            raise unittest.SkipTest("No CUDA device available")
        paddle.set_flags({"FLAGS_cudnn_deterministic": True})

    def setUp(self):
        self.batch_size, self.seq_len = 1, 64
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

    def _assert_bitwise(self, ref, got, what):
        # equal_all has no bf16 GPU kernel; the bf16 -> fp32 cast is lossless,
        # so equality after the cast is still an exact bit comparison.
        ref32, got32 = ref.cast("float32"), got.cast("float32")
        self.assertTrue(
            paddle.equal_all(ref32, got32).item(),
            f"{what} is not bitwise identical; max abs diff = "
            f"{(ref32 - got32).abs().max().item()}",
        )

    def _run_combo(self, combo, **config_kwargs):
        """Compare one recompute_modules combo against no-recompute, bitwise."""
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        attn_ref = _build_attention(_make_config(**config_kwargs))
        attn_rc = _build_attention(
            _make_config(recompute_modules=combo, **config_kwargs)
        )
        attn_rc.set_state_dict(attn_ref.state_dict())
        attn_ref.train()
        attn_rc.train()

        hidden = paddle.randn(
            [self.batch_size, self.seq_len, _HIDDEN_SIZE],
            dtype=paddle.bfloat16,
        )
        startend = _startend(self.batch_size, self.seq_len)

        out_ref, xgrad_ref, grads_ref = _forward_backward(
            attn_ref, hidden, startend
        )
        out_rc, xgrad_rc, grads_rc = _forward_backward(
            attn_rc, hidden, startend
        )

        self._assert_bitwise(out_ref, out_rc, "forward output")
        self._assert_bitwise(xgrad_ref, xgrad_rc, "input gradient")

        self.assertEqual(set(grads_ref), set(grads_rc))
        compared = 0
        for name, g_ref in grads_ref.items():
            g_rc = grads_rc[name]
            self.assertEqual(
                g_ref is None,
                g_rc is None,
                f"gradient presence differs for {name}",
            )
            if g_ref is None:
                continue
            self._assert_bitwise(g_ref, g_rc, f"gradient of {name}")
            compared += 1
        self.assertGreater(compared, 0, "no parameter gradient was compared")

    def test_hca_unfused_backend(self):
        for combo in _STAGE_COMBOS:
            with self.subTest(modules=combo):
                self._run_combo(combo)

    def test_hca_tilelang_backend(self):
        try:
            import paddlefleet.tilelang_ops  # noqa: F401
        except ImportError:
            self.skipTest("TileLang CSA ops not available")
        for combo in _STAGE_COMBOS:
            with self.subTest(modules=combo):
                self._run_combo(
                    combo,
                    sparse_attn_backend="tilelang",
                    indexer_backend="tilelang",
                    # The TileLang sparse-attention kernel rejects a head dim of
                    # 32 ("warp_col_tiles must be greater than 8").
                    v_head_dim=64,
                )

    def test_csa_indexer_receives_gradient_under_staged_recompute(self):
        """The Indexer still trains through the recomputed CSA segment.

        The indexer loss is created *inside* ``core_attention`` and only on the
        grad-enabled pass, so a broken segment boundary shows up as a silently
        missing gradient rather than as an error.
        """
        config = _make_config(
            recompute_modules=["hca_pre_csa", "hca_csa", "hca_post_csa"],
            compress_ratio=4,
            csa_dense_mode=False,
        )
        attn = _build_attention(config)
        attn.train()
        hidden = paddle.randn(
            [self.batch_size, self.seq_len, _HIDDEN_SIZE],
            dtype=paddle.bfloat16,
        )
        leaf, x = _grad_source(hidden)
        output, _ = attn(
            hidden_states=x,
            attention_mask=None,
            attn_mask_startend_row_indices=_startend(
                self.batch_size, self.seq_len
            ),
        )
        output.cast("float32").sum().backward()
        self.assertIsNotNone(leaf.grad)

        indexer_params = [
            (name, p)
            for name, p in attn.named_parameters()
            if "indexer" in name and not p.stop_gradient
        ]
        self.assertGreater(len(indexer_params), 0, "no indexer parameter found")
        for name, p in indexer_params:
            self.assertIsNotNone(p.grad, f"indexer param {name} has no grad")
            self.assertTrue(
                paddle.isfinite(p.grad.cast("float32")).all().item(),
                f"indexer param {name} has non-finite grad",
            )


class TestHCAStagedRecomputeShapesAndModes(unittest.TestCase):
    """Bitwise alignment in configurations the combination sweep does not cover.

    Representative stage combinations only -- the point is to catch a change in
    behaviour for a *shape* (batch > 1, gate disabled) or a *mode* (eval), not to
    re-run the full sweep.
    """

    # One combination per wrapping kind exercised by _attn_forward:
    # plain recompute (csa), RecomputeWithoutOutput on the tail (csa+post),
    # and pre as its own segment plus a tail segment (all three).
    REPRESENTATIVE = [
        ["hca_csa"],
        ["hca_csa", "hca_post_csa"],
        ["hca_pre_csa", "hca_csa", "hca_post_csa"],
    ]

    @classmethod
    def setUpClass(cls):
        if not paddle.device.is_compiled_with_cuda():
            raise unittest.SkipTest("CUDA build of Paddle is required")
        if paddle.device.cuda.device_count() == 0:
            raise unittest.SkipTest("No CUDA device available")
        paddle.set_flags({"FLAGS_cudnn_deterministic": True})

    def setUp(self):
        self.batch_size, self.seq_len = 1, 64
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

    _assert_bitwise = TestHCAStagedRecompute._assert_bitwise
    _run_combo = TestHCAStagedRecompute._run_combo

    def test_batch_size_two(self):
        """Covers the logical-batch pack/unpack around the staged segments."""
        self.batch_size = 2
        for combo in self.REPRESENTATIVE:
            with self.subTest(modules=combo):
                self._run_combo(combo)

    def test_gated_attention_disabled(self):
        """post-CSA without the gate: o_group_proj is then the last op."""
        for combo in self.REPRESENTATIVE:
            with self.subTest(modules=combo):
                self._run_combo(combo, gated_attention=False)

    def test_eval_mode_ignores_recompute(self):
        """No segment is created outside training, and the result is unchanged."""
        combo = ["hca_pre_csa", "hca_csa", "hca_post_csa"]
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        attn_ref = _build_attention(_make_config())
        attn_rc = _build_attention(_make_config(recompute_modules=combo))
        attn_rc.set_state_dict(attn_ref.state_dict())
        attn_ref.eval()
        attn_rc.eval()

        hidden = paddle.randn(
            [self.batch_size, self.seq_len, _HIDDEN_SIZE],
            dtype=paddle.bfloat16,
        )
        startend = _startend(self.batch_size, self.seq_len)
        with paddle.no_grad():
            out_ref, _ = attn_ref(
                hidden_states=hidden,
                attention_mask=None,
                attn_mask_startend_row_indices=startend,
            )
            out_rc, _ = attn_rc(
                hidden_states=hidden,
                attention_mask=None,
                attn_mask_startend_row_indices=startend,
            )
        self._assert_bitwise(out_ref, out_rc, "eval-mode forward output")
        self.assertIsNone(attn_rc._stage_recompute)

    def test_recompute_holders_cleared_after_forward(self):
        """The holders are per-forward state and must not leak between steps."""
        for combo in (["hca_csa", "hca_post_csa"], ["full_attn"]):
            with self.subTest(modules=combo):
                paddle.seed(_SEED)
                model_parallel_cuda_manual_seed(_SEED)
                attn = _build_attention(_make_config(recompute_modules=combo))
                attn.train()
                hidden = paddle.randn(
                    [self.batch_size, self.seq_len, _HIDDEN_SIZE],
                    dtype=paddle.bfloat16,
                )
                _forward_backward(
                    attn, hidden, _startend(self.batch_size, self.seq_len)
                )
                self.assertIsNone(attn._stage_recompute)
                self.assertIsNone(attn._gate_recompute)

    def test_all_stages_forward_matches_stage_by_stage(self):
        """The span delegates must stay pure composition of the three stages."""
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        attn = _build_attention(_make_config())
        attn.train()
        hidden = paddle.randn(
            [self.batch_size, self.seq_len, _HIDDEN_SIZE],
            dtype=paddle.bfloat16,
        )
        with paddle.no_grad():
            fused = attn._all_stages_forward(hidden, None, 0, None, None)
            query, key, value, q_compressed, _ = attn._pre_csa_forward(
                hidden, 0, None
            )
            core = attn._csa_forward(
                query, key, value, None, hidden, q_compressed, None, None
            )
            staged = attn._post_csa_forward(core, hidden, q_compressed, 0, None)
        self._assert_bitwise(fused, staged, "_all_stages_forward output")

    def test_pre_csa_dedupe_drops_key_value_alias(self):
        """key and value are one tensor; a segment may not return it twice.

        Returning the same object twice makes paddle.autograd.backward reject the
        segment with "contains duplicate paddle.Tensor object".
        """
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        attn = _build_attention(_make_config())
        attn.train()
        hidden = paddle.randn(
            [self.batch_size, self.seq_len, _HIDDEN_SIZE],
            dtype=paddle.bfloat16,
        )
        with paddle.no_grad():
            query, key, value, q_compressed, kv_compressed = (
                attn._pre_csa_forward(hidden, 0, None)
            )
            deduped = attn._pre_csa_deduped_forward(hidden, 0, None)
        self.assertIs(value, key, "expected the single-head KV alias")
        self.assertEqual(len(deduped), 4)
        self.assertEqual(len({id(t) for t in deduped}), 4)


class TestHCAStagedRecomputeConfig(unittest.TestCase):
    """Flag resolution for the three stage entries."""

    def setUp(self):
        # RowParallelLinear initialisation forks the model-parallel RNG.
        paddle.seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

    def _flags(self, **kwargs):
        attn = _build_attention(_make_config(**kwargs), layer_number=0)
        return (
            attn.recompute_pre_csa,
            attn.recompute_csa,
            attn.recompute_post_csa,
        )

    def test_individual_entries(self):
        self.assertEqual(
            self._flags(recompute_modules=["hca_pre_csa"]),
            (True, False, False),
        )
        self.assertEqual(
            self._flags(recompute_modules=["hca_csa"]), (False, True, False)
        )
        self.assertEqual(
            self._flags(recompute_modules=["hca_post_csa"]),
            (False, False, True),
        )

    def test_ignored_without_selective_granularity(self):
        config = _make_config(recompute_modules=["hca_csa"])
        config.recompute_granularity = "full"
        config.recompute_num_layers = 1
        config.recompute_method = "uniform"
        attn = _build_attention(config, layer_number=0)
        self.assertFalse(attn.recompute_csa)

    def test_honours_recompute_num_layers(self):
        # first_n with n=0 selects no layer
        self.assertEqual(
            self._flags(
                recompute_modules=["hca_csa"],
                recompute_num_layers=0,
                recompute_method="first_n",
            ),
            (False, False, False),
        )
        self.assertEqual(
            self._flags(
                recompute_modules=["hca_csa"],
                recompute_num_layers=1,
                recompute_method="first_n",
            ),
            (False, True, False),
        )

    def test_honours_block_method(self):
        # block with n=1 on a 1-layer model selects the layer
        self.assertEqual(
            self._flags(
                recompute_modules=["hca_post_csa"],
                recompute_num_layers=1,
                recompute_method="block",
            ),
            (False, False, True),
        )

    def test_rejects_unknown_recompute_method(self):
        # Rejected by TransformerConfig itself, so the flag resolution can
        # assume block/first_n.
        with self.assertRaises(AssertionError):
            _build_attention(
                _make_config(
                    recompute_modules=["hca_csa"],
                    recompute_num_layers=1,
                    recompute_method="uniform",
                )
            )

    def test_unrelated_entries_leave_stage_flags_off(self):
        self.assertEqual(
            self._flags(recompute_modules=["norm", "mlp", "gated_attn"]),
            (False, False, False),
        )

    def test_rejects_combination_with_full_attn(self):
        with self.assertRaises(ValueError):
            _build_attention(
                _make_config(recompute_modules=["full_attn", "hca_csa"])
            )

    def test_rejects_dict_configuration(self):
        # Rejected either by the stage flag resolution (ValueError) or already
        # upstream while building the layer (AssertionError); never silently
        # ignored.
        with self.assertRaises((ValueError, AssertionError)):
            _build_attention(_make_config(recompute_modules={"hca_csa": 1}))


if __name__ == "__main__":
    unittest.main()
