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

"""Phase 2 (``train_indexer_only``) regression guards.

Freezing every non-Indexer parameter breaks two things that have nothing to do
with the attention math, and both were found with the probes under
``.agents/probes/``:

1. The indexer-loss attach PyLayers return their first input unchanged. Once the
   backbone is frozen that input is a leaf with ``stop_gradient=True``, which
   Paddle rejects as an inplace alias on a grad-enabled node. The fix returns a
   fresh tensor and reports ``None`` for that position in backward.
2. ``recompute`` is a PyLayer, so its output is differentiable only if its input
   is. With a frozen backbone the segment input is detached, the segment output
   inherits that, and the indexer loss attached inside the segment silently
   never runs backward - no error, normal loss curve, zero indexer updates.
   ``keep_indexer_grad_path`` re-enters the graph to keep that path alive.

The tests below run on CPU and do not need the TileLang/cuDNN kernels.
"""

import unittest

import paddle
from paddle import nn
from paddle.distributed.fleet.utils import recompute

from paddlefleet.recompute_utils import keep_indexer_grad_path
from paddlefleet.tensor_parallel import RecomputeWithoutOutput
from paddlefleet.transformer.csa_attention import (
    TileLangCSAIndexerLossAutoScaler,
)
from paddlefleet.transformer.dsa_attention import DSAIndexerLossAutoScaler
from paddlefleet.transformer.transformer_config import TransformerConfig


def _make_dsv4_config(**overrides):
    kwargs = {
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 0,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "params_dtype": paddle.float32,
        "use_bias": False,
        "multi_latent_attention": True,
        "experimental_attention_variant": "dsv4_hybrid",
        "q_lora_rank": 64,
        "kv_lora_rank": 64,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 64,
        "qk_pos_emb_head_dim": 64,
        "v_head_dim": 128,
        "normalization": "RMSNorm",
        "csa_compress_ratios": [0, 4],
        "dsa_index_n_heads": 4,
        "dsa_index_head_dim": 64,
        "dsa_index_topk": 16,
        "dsa_indexer_loss_coeff": 0.01,
        "dsa_indexer_use_sparse_loss": False,
        "csa_dense_mode": False,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _make_mqa_config(**overrides):
    """dsv4-hybrid config whose ``-2`` layers run non-absorbed MQA.

    A ``-2`` layer is a hybrid MLA layer, so ``__post_init__`` demands the whole
    ``hybrid_mla_*`` block before it ever gets to the ``train_indexer_only``
    checks. ``csa_dense_mode=True`` matches the real new-attention configs: the
    128/HCA layers have no CSAIndexer, so the DSAIndexer of the ``-2`` layers is
    the only Indexer in the model.
    """
    kwargs = {
        "csa_dense_mode": True,
        "csa_compress_ratios": [128, -2],
        "non_absorbed_mqa": True,
        # The DSA indexer of a -2 layer runs the cuDNN kernel, which pins
        # index_head_dim=128 and index_topk to a multiple of 128 (<= 2048).
        "dsa_index_head_dim": 128,
        "dsa_index_topk": 128,
        "hybrid_mla_q_lora_rank": 64,
        "hybrid_mla_kv_lora_rank": 64,
        "hybrid_mla_qk_nope_head_dim": 64,
        "hybrid_mla_qk_rope_head_dim": 64,
        "hybrid_mla_v_head_dim": 128,
        "hybrid_mla_num_attention_heads": 8,
        "hybrid_mla_num_key_value_heads": 8,
    }
    kwargs.update(overrides)
    return _make_dsv4_config(**kwargs)


class TestPhase2ConfigValidation(unittest.TestCase):
    """``train_indexer_only`` must reject configs that cannot train."""

    def test_valid_phase2_config(self):
        config = _make_dsv4_config(train_indexer_only=True)
        self.assertTrue(config.train_indexer_only)
        self.assertFalse(config.csa_dense_mode)

    def test_default_is_off_for_phase1_and_phase3(self):
        self.assertFalse(
            _make_dsv4_config(csa_dense_mode=True).train_indexer_only
        )
        self.assertFalse(
            _make_dsv4_config(
                dsa_indexer_use_sparse_loss=True
            ).train_indexer_only
        )

    def test_dense_mode_without_mqa_indexer_rejected(self):
        # csa_dense_mode drops the CSAIndexer. On its own that leaves nothing to
        # train -- but it is *not* illegal per se: the non-absorbed MQA layers
        # carry a DSAIndexer of their own (see the tests below).
        with self.assertRaisesRegex(ValueError, "build at least one Indexer"):
            _make_dsv4_config(train_indexer_only=True, csa_dense_mode=True)

    def test_non_positive_loss_coeff_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "positive\n?.*dsa_indexer_loss_coeff"
        ):
            _make_dsv4_config(
                train_indexer_only=True, dsa_indexer_loss_coeff=0.0
            )

    def test_no_indexer_layer_rejected(self):
        with self.assertRaisesRegex(ValueError, "build at least one Indexer"):
            _make_dsv4_config(
                train_indexer_only=True, csa_compress_ratios=[0, 128]
            )

    def test_mqa_dsa_indexer_satisfies_the_check_without_any_csa_layer(self):
        # The new-attention phase 2: every CSA-ratio layer is HCA (128) and
        # csa_dense_mode is on, so there is no CSAIndexer anywhere. The only
        # Indexer is the DSAIndexer of the ``-2`` non-absorbed MQA layers.
        config = _make_mqa_config(train_indexer_only=True)
        self.assertTrue(config.train_indexer_only)
        self.assertTrue(config.csa_dense_mode)

    def test_mqa_dense_rejected(self):
        # non_absorbed_mqa_dense drops the DSAIndexer (gpt_layer_specs.py passes
        # indexer=None), which is exactly the phase-1 shape: nothing to train.
        with self.assertRaisesRegex(ValueError, "build at least one Indexer"):
            _make_mqa_config(
                train_indexer_only=True, non_absorbed_mqa_dense=True
            )

    def test_mqa_without_hybrid_layer_rejected(self):
        # non_absorbed_mqa only does anything to ``-2`` layers; without one there
        # is no MQALatentAttention and therefore no DSAIndexer.
        with self.assertRaisesRegex(ValueError, "build at least one Indexer"):
            _make_mqa_config(
                train_indexer_only=True, csa_compress_ratios=[0, 128]
            )

    def test_non_dsv4_variant_rejected(self):
        with self.assertRaisesRegex(ValueError, "dsv4_hybrid"):
            TransformerConfig(
                num_hidden_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                params_dtype=paddle.float32,
                train_indexer_only=True,
            )

    def test_hy_sparse_attention_rejected(self):
        # HySparseTransformerLayer does not call keep_indexer_grad_path(), so with a
        # frozen backbone the Indexer would silently get no gradient at all.
        with self.assertRaisesRegex(ValueError, "enable_hy_sparse_attention"):
            _make_dsv4_config(
                train_indexer_only=True, enable_hy_sparse_attention=True
            )


class TestKeepIndexerGradPath(unittest.TestCase):
    """The recompute segment input must stay differentiable in phase 2."""

    def setUp(self):
        self.hidden = paddle.randn([2, 4, 8])
        self.hidden.stop_gradient = True

    def test_noop_when_flag_disabled(self):
        config = _make_dsv4_config()
        out = keep_indexer_grad_path(self.hidden, config)
        self.assertIs(out, self.hidden)
        self.assertTrue(out.stop_gradient)

    def test_reenters_graph_when_flag_enabled(self):
        config = _make_dsv4_config(train_indexer_only=True)
        out = keep_indexer_grad_path(self.hidden, config)
        self.assertIsNot(out, self.hidden)
        self.assertFalse(out.stop_gradient)
        self.assertFalse(out.is_leaf)
        # Value must be untouched: the anchor adds a zero.
        self.assertTrue(paddle.allclose(out, self.hidden).item())

    def test_short_circuits_when_already_differentiable(self):
        config = _make_dsv4_config(train_indexer_only=True)
        hidden = paddle.randn([2, 4, 8])
        hidden.stop_gradient = False
        hidden = hidden * 1.0
        out = keep_indexer_grad_path(hidden, config)
        self.assertIs(out, hidden)

    def test_noop_under_no_grad(self):
        config = _make_dsv4_config(train_indexer_only=True)
        with paddle.no_grad():
            out = keep_indexer_grad_path(self.hidden, config)
        self.assertIs(out, self.hidden)

    def test_non_tensor_passthrough(self):
        config = _make_dsv4_config(train_indexer_only=True)
        self.assertIsNone(keep_indexer_grad_path(None, config))


class _IndexerBranch(nn.Layer):
    """Minimal stand-in for one CSA layer: frozen main path + trainable indexer.

    Mirrors ``csa_attention.py`` structurally: the indexer consumes a detached
    copy of the layer input, and its loss is attached to the attention output
    through ``DSAIndexerLossAutoScaler``.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.main = nn.Linear(hidden_size, hidden_size)
        self.indexer = nn.Linear(hidden_size, hidden_size)

    def freeze_backbone(self):
        self.main.weight.stop_gradient = True
        self.main.bias.stop_gradient = True
        self.indexer.weight.stop_gradient = False
        self.indexer.bias.stop_gradient = False

    def forward(self, hidden_states):
        output = self.main(hidden_states)
        detached = hidden_states.detach()
        detached.stop_gradient = False
        indexer_loss = self.indexer(detached).square().mean()
        return DSAIndexerLossAutoScaler.apply(output, indexer_loss)


class TestAttachUnderFrozenBackbone(unittest.TestCase):
    """The attach PyLayers must survive a frozen backbone and still feed grads."""

    def setUp(self):
        paddle.seed(0)
        self.hidden_size = 8
        self.ids = paddle.randn([2, 4, self.hidden_size])
        self.ids.stop_gradient = True

    def test_dsa_autoscaler_produces_indexer_grad(self):
        layer = _IndexerBranch(self.hidden_size)
        layer.freeze_backbone()
        head = nn.Linear(self.hidden_size, self.hidden_size)
        head.weight.stop_gradient = True
        head.bias.stop_gradient = True

        out = layer(self.ids)
        self.assertFalse(out.stop_gradient)
        head(out).sum().backward()

        self.assertIsNotNone(layer.indexer.weight.grad)
        self.assertGreater(
            float(paddle.abs(layer.indexer.weight.grad).sum()), 0.0
        )
        self.assertIsNone(layer.main.weight.grad)
        self.assertIsNone(head.weight.grad)

    def test_dsa_autoscaler_still_passes_grad_when_backbone_trains(self):
        layer = _IndexerBranch(self.hidden_size)
        hidden = paddle.randn([2, 4, self.hidden_size])
        hidden.stop_gradient = False
        hidden = hidden * 1.0

        layer(hidden).sum().backward()

        self.assertIsNotNone(layer.main.weight.grad)
        self.assertIsNotNone(layer.indexer.weight.grad)

    def test_tilelang_autoscaler_forward_returns_fresh_tensor_when_frozen(self):
        # Only the forward contract is checked here: the backward needs the
        # TileLang/cuDNN indexer kernels, which single-card CPU tests skip.
        output = paddle.randn([2, 4, 8])
        output.stop_gradient = True
        index_q = paddle.randn([2, 4, 8])
        index_q.stop_gradient = False
        weights = paddle.randn([2, 4, 8])
        weights.stop_gradient = False
        index_k = paddle.randn([2, 4, 8])
        index_k.stop_gradient = False
        topk_indices = paddle.zeros([2, 4, 2], dtype="int32")
        topk_probs = paddle.rand([2, 4, 2])
        target = paddle.rand([2, 4, 2])

        attached = TileLangCSAIndexerLossAutoScaler.apply(
            output,
            index_q,
            weights,
            index_k,
            topk_indices,
            topk_probs,
            target,
            0.01,
        )
        self.assertIsNot(attached, output)
        self.assertFalse(attached.stop_gradient)
        # The frozen path returns a clone, so the value must be identical.
        self.assertTrue(paddle.allclose(attached, output).item())


class TestRecomputeSegmentKeepsIndexerGrad(unittest.TestCase):
    """Full-recompute regression guard for the silent zero-gradient failure."""

    def setUp(self):
        paddle.seed(0)
        self.hidden_size = 8
        self.hidden = paddle.randn([2, 4, self.hidden_size])
        self.hidden.stop_gradient = True

    def _run(self, config):
        layer = _IndexerBranch(self.hidden_size)
        layer.freeze_backbone()
        head = nn.Linear(self.hidden_size, self.hidden_size)
        head.weight.stop_gradient = True
        head.bias.stop_gradient = True

        segment_input = keep_indexer_grad_path(self.hidden, config)
        out = recompute(layer, segment_input)
        head(out).sum().backward()
        return layer, out

    def test_without_flag_indexer_gets_no_grad(self):
        layer, out = self._run(_make_dsv4_config())
        self.assertTrue(out.stop_gradient)
        self.assertIsNone(layer.indexer.weight.grad)

    def test_with_flag_indexer_gets_grad(self):
        layer, out = self._run(_make_dsv4_config(train_indexer_only=True))
        self.assertFalse(out.stop_gradient)
        self.assertIsNotNone(layer.indexer.weight.grad)
        self.assertGreater(
            float(paddle.abs(layer.indexer.weight.grad).sum()), 0.0
        )
        self.assertIsNone(layer.main.weight.grad)


class TestFullAttnRecomputeKeepsIndexerGrad(unittest.TestCase):
    """``recompute_granularity=selective`` + ``recompute_modules=["full_attn"]``.

    ``DSv4HybridAttention.forward`` wraps qkv + ``core_attention`` (i.e. the CSA
    Indexer and its side-attached loss) in ``RecomputeWithoutOutput``. The
    layer-level guard in ``TransformerLayer.forward`` does not cover this: it only
    runs in the ``full_recompute`` branch, which is off under selective
    granularity.

    This wrapper fails even more quietly than ``recompute``:
    ``discard_output_and_register_recompute`` registers its recompute hook only
    when the hook tensor is differentiable (``tensor_parallel/random.py:590``). A
    frozen backbone makes the segment input detached, so the whole backward is
    skipped with no error and not even the WARNING that ``recompute`` logs.
    """

    def setUp(self):
        paddle.seed(0)
        self.hidden_size = 8
        self.hidden = paddle.randn([2, 4, self.hidden_size])
        self.hidden.stop_gradient = True

    def _run(self, config):
        layer = _IndexerBranch(self.hidden_size)
        layer.freeze_backbone()
        # Stands in for DSv4HybridAttention.o_proj, which is frozen in phase 2.
        o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        o_proj.weight.stop_gradient = True
        o_proj.bias.stop_gradient = True

        wrapper = RecomputeWithoutOutput()
        core_attn_out = wrapper.recompute(
            layer,
            keep_indexer_grad_path(self.hidden, config),
            preserve_rng_state=False,
        )
        output = o_proj(core_attn_out)
        wrapper.discard_output_and_register_recompute(output)
        output.sum().backward()
        return layer, core_attn_out

    def test_without_flag_indexer_gets_no_grad(self):
        layer, core_attn_out = self._run(_make_dsv4_config())
        self.assertTrue(core_attn_out.stop_gradient)
        self.assertIsNone(layer.indexer.weight.grad)

    def test_with_flag_indexer_gets_grad(self):
        layer, core_attn_out = self._run(
            _make_dsv4_config(train_indexer_only=True)
        )
        self.assertFalse(core_attn_out.stop_gradient)
        self.assertIsNotNone(layer.indexer.weight.grad)
        self.assertGreater(
            float(paddle.abs(layer.indexer.weight.grad).sum()), 0.0
        )
        self.assertIsNone(layer.main.weight.grad)


class TestIndexerParameterOwnership(unittest.TestCase):
    """The trainer resolves phase-2 trainable params by module ownership.

    ``CompressedSparseAttention`` owns ``attn_sink`` and its own main
    ``compressor`` next to the ``indexer``. Both are backbone parameters, and the
    main compressor's parameter names contain ``compressor`` just like the
    Indexer's nested one, so any name-substring rule would misclassify them.
    This test pins the module-tree rule used by
    ``PretrainingTrainer._collect_indexer_params``.
    """

    def _build_core_attention(self, config=None):
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        from paddlefleet.models.gpt.gpt_layer_specs import LocalSpecProvider
        from paddlefleet.tensor_parallel.random import (
            model_parallel_cuda_manual_seed,
        )
        from paddlefleet.transformer.csa_attention import (
            CompressedSparseAttention,
            CompressedSparseAttentionSublayersSpec,
            Compressor,
            CompressorSublayersSpec,
            CSAIndexer,
            CSAIndexerSublayersSpec,
        )
        from paddlefleet.transformer.enums import AttnMaskType

        paddle.seed(0)
        model_parallel_cuda_manual_seed(0)
        backend = LocalSpecProvider()
        compressor_spec = LayerSpec(
            layer=Compressor,
            sublayers_spec=CompressorSublayersSpec(
                linear_wkv=backend.linear(),
                linear_wgate=backend.linear(),
                norm=backend.layer_norm(rms_norm=True, for_qk=False),
            ),
        )
        indexer_spec = LayerSpec(
            layer=CSAIndexer,
            sublayers_spec=CSAIndexerSublayersSpec(
                linear_wq_b=backend.linear(),
                linear_weights_proj=backend.linear(),
                compressor=compressor_spec,
            ),
        )
        config = config or _make_dsv4_config(
            train_indexer_only=True, csa_compress_ratios=[0, 4]
        )
        core_attention = CompressedSparseAttention(
            config=config,
            sublayers_spec=CompressedSparseAttentionSublayersSpec(
                compressor=compressor_spec,
                indexer=indexer_spec,
            ),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            compress_ratio=4,
        )
        return core_attention, CSAIndexer

    def test_indexer_ownership_excludes_backbone_siblings(self):
        core_attention, csa_indexer_cls = self._build_core_attention()
        self.assertIsNotNone(core_attention.indexer)
        self.assertIsNotNone(core_attention.compressor)

        expected, seen = [], set()
        for _, sublayer in core_attention.named_sublayers():
            if not isinstance(sublayer, csa_indexer_cls):
                continue
            for param in sublayer.parameters():
                if id(param) not in seen:
                    seen.add(id(param))
                    expected.append(param)
        expected_ids = {id(p) for p in expected}

        self.assertGreater(len(expected_ids), 0)
        # Indexer Q / weights projections and its nested compressor are included.
        for param in core_attention.indexer.parameters():
            self.assertIn(id(param), expected_ids)
        # attn_sink and the main compressor stay frozen.
        self.assertNotIn(id(core_attention.attn_sink), expected_ids)
        for param in core_attention.compressor.parameters():
            self.assertNotIn(id(param), expected_ids)

        all_ids = {id(p) for p in core_attention.parameters()}
        self.assertTrue(expected_ids.issubset(all_ids))
        self.assertLess(len(expected_ids), len(all_ids))

    def test_no_indexer_when_dense_mode(self):
        core_attention, _ = self._build_core_attention(
            config=_make_dsv4_config(
                csa_dense_mode=True, csa_compress_ratios=[0, 4]
            )
        )
        self.assertIsNone(core_attention.indexer)
        self.assertIsNotNone(core_attention.compressor)


if __name__ == "__main__":
    unittest.main()
