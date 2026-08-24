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

"""Tests for the ``separate_mtp_input`` feature.

``separate_mtp_input`` hands the shifted MTP embeddings computed by
``GPTEmbedding`` to ``MultiTokenPredictionLayer`` through a dedicated
``mtp_decoder_inputs`` entry in ``dict_args`` instead of concatenating them into
``hidden_states``.  That removes the per-layer split/concat from every backbone
TransformerLayer.  It is the ``pipeline_model_parallel_size == 1`` counterpart of
``enable_mtp_magic_send``.

End-to-end bitwise equivalence (forward + backward) against the concat/split
baseline is covered on real devices by

* ``tests/multi_card_tests/test_separate_mtp_input_cp.py`` (CP > 1)
* ``tests/multi_card_tests/tensor_parallel/test_separate_mtp_input_tp_sp.py``
  (TP > 1 + sequence_parallel)

This file covers the branches those runs cannot reach on a single card.
"""

import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import paddle
from paddle import nn

from paddlefleet.models.gpt import GPTConfig
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig

# =====================================================================
# Helpers
# =====================================================================

_BASE_CFG = {
    "vocab_size": 512,
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "tensor_model_parallel_size": 1,
    "expert_model_parallel_size": 1,
}


def _cfg(**kw):
    """separate_mtp_input config (always PP == 1)."""
    return GPTConfig(
        **{
            **_BASE_CFG,
            "separate_mtp_input": True,
            "num_nextn_predict_layers": 1,
            "pipeline_model_parallel_size": 1,
            **kw,
        }
    )


def _cfg_baseline(**kw):
    """Concat/split baseline: neither separate_mtp_input nor magic send."""
    return _cfg(separate_mtp_input=False, **kw)


def _cfg_magic(**kw):
    """magic send config, used for contrast assertions."""
    return GPTConfig(
        **{
            **_BASE_CFG,
            "enable_mtp_magic_send": True,
            "num_nextn_predict_layers": 1,
            "pipeline_model_parallel_size": 2,
            **kw,
        }
    )


class _FakeNorm(nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()

    def forward(self, x):
        return x


class _FakeLinear(nn.Layer):
    def __init__(self, in_f=128, out_f=64, *a, **kw):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias_attr=False)

    def forward(self, x):
        return self.linear(x), None


class _FakeTransformerLayer(nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()

    def forward(self, d):
        return {"hidden_states": d["hidden_states"]}


class _FakeEmbed(nn.Layer):
    """Stand-in for the magic-send local re-embedding of input_ids."""

    def __init__(self, hidden_size):
        super().__init__()
        self.weight = self.create_parameter(
            [1], default_initializer=nn.initializer.Constant(0.0)
        )
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        return paddle.randn(
            [input_ids.shape[0], input_ids.shape[1], self.hidden_size]
        )


@dataclass
class _FakeMTPSpec:
    enorm: object = None
    hnorm: object = None
    eh_proj: object = None
    e_proj: object = None
    h_proj: object = None
    transformer_layer: object = None
    layer_norm: object = None

    def __post_init__(self):
        tl = MagicMock()
        tl.sublayers_spec = MagicMock()
        tl.sublayers_spec.self_attn = MagicMock()
        tl.sublayers_spec.self_attn.extra_kwargs = {
            "attn_mask_type": AttnMaskType.causal
        }
        self.transformer_layer = tl


def _build_mtp_layer(config, layer_number=0):
    from paddlefleet.transformer.multi_token_prediction import (
        MultiTokenPredictionLayer,
    )

    spec = _FakeMTPSpec()
    mock_pg = MagicMock(cp=None, tp=None)
    with (
        patch(
            "paddlefleet.transformer.multi_token_prediction.build_spec_layer",
            side_effect=lambda s, *a, **kw: _FakeTransformerLayer()
            if s is spec.transformer_layer
            else _FakeNorm(),
        ),
        patch(
            "paddlefleet.transformer.multi_token_prediction.ProcessGroupCollection.use_mpu_process_groups",
            return_value=mock_pg,
        ),
        patch(
            "paddlefleet.transformer.multi_token_prediction.paddle.distributed.get_world_size",
            return_value=1,
        ),
    ):
        layer = MultiTokenPredictionLayer(
            config=config,
            sublayers_spec=spec,
            layer_number=layer_number,
            pg_collection=mock_pg,
        )

    if layer.eh_proj is not None:
        layer.eh_proj = _FakeLinear(config.hidden_size * 2, config.hidden_size)
    layer.enorm = _FakeNorm()
    layer.hnorm = _FakeNorm()
    if hasattr(layer, "norm") and layer.norm is not None:
        layer.norm = _FakeNorm()
    layer.transformer_layer = _FakeTransformerLayer()
    return layer


class _Emb(nn.Layer):
    """Minimal stand-in for the language embedding sublayer."""

    def __init__(self, v, h):
        super().__init__()
        self.embed_tokens = nn.Embedding(v, h)
        self.reduce_scatter_embeddings = self.scatter_to_sequence_parallel = (
            self.sequence_parallel
        ) = False

    @property
    def embedding_weight(self):
        return self.embed_tokens.weight

    def forward(self, input_ids, position_ids=None):
        return self.embed_tokens(input_ids)


def _build_gpt_embedding(config, emb_layer=None):
    """Build GPTEmbedding; pass ``emb_layer`` to share weights across instances."""
    from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

    mock_spec = MagicMock(rope_embedding=None)
    if emb_layer is None:
        emb_layer = _Emb(config.vocab_size, config.hidden_size)
    with (
        patch(
            "paddlefleet.models.gpt.gpt_embedding.build_spec_layer",
            side_effect=lambda s, *a, **kw: emb_layer
            if s is mock_spec.language_embedding
            else None,
        ),
        patch(
            "paddlefleet.models.gpt.gpt_embedding.mark_context_parallel_parameter_disable_scale_grad"
        ),
    ):
        return GPTEmbedding(
            sublayers_spec=mock_spec,
            config=config,
            vocab_size=config.vocab_size,
            max_sequence_length=128,
            position_embedding_type="rope",
        )


def _run_emb(config, input_ids, cp_world_size=1, emb_layer=None):
    """Run GPTEmbedding.forward with CP/SP scatter replaced by identity."""
    emb = _build_gpt_embedding(config, emb_layer=emb_layer)
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "paddlefleet.models.gpt.gpt_embedding.get_context_parallel_world_size",
                return_value=cp_world_size,
            )
        )
        sc = stack.enter_context(
            patch("paddlefleet.models.gpt.gpt_embedding.ScatterOp")
        )
        sc.apply = lambda x: x
        cp = stack.enter_context(
            patch(
                "paddlefleet.models.gpt.gpt_embedding.ContextParallelScatterOp"
            )
        )
        cp.apply = lambda x, axis=0, **kwargs: x
        return emb.forward({"input_ids": input_ids})


def _mtp_forward_ctx(
    cp_world_size=1,
    scatter_fn=None,
    cp_scatter_fn=None,
    proj_override=None,
    layer=None,
):
    """Context manager wiring up the mocks MTP forward needs on a single card."""
    stack = ExitStack()

    def _enter(stack_ref):
        stack_ref.enter_context(
            patch(
                "paddlefleet.transformer.multi_token_prediction.get_context_parallel_world_size",
                return_value=cp_world_size,
            )
        )
        tp = stack_ref.enter_context(
            patch(
                "paddlefleet.transformer.multi_token_prediction.tensor_parallel"
            )
        )
        tp.get_cuda_rng_tracker.return_value.fork.return_value = nullcontext()

        if scatter_fn is not None:
            so = stack_ref.enter_context(
                patch(
                    "paddlefleet.transformer.multi_token_prediction.ScatterOp"
                )
            )
            so.apply = scatter_fn
        if cp_scatter_fn is not None:
            co = stack_ref.enter_context(
                patch(
                    "paddlefleet.transformer.multi_token_prediction.ContextParallelScatterOp"
                )
            )
            co.apply = cp_scatter_fn
        if proj_override is not None and layer is not None:
            stack_ref.enter_context(
                patch.object(
                    layer,
                    "_proj_and_transformer_layer",
                    side_effect=proj_override,
                )
            )

    class _Ctx:
        def __enter__(self):
            _enter(stack)
            return stack

        def __exit__(self, *a):
            stack.__exit__(*a)

    return _Ctx()


def _assert_bitwise_equal(testcase, a, b, msg):
    """Bit-for-bit tensor comparison (no tolerance)."""
    testcase.assertEqual(list(a.shape), list(b.shape), f"{msg}: shape differs")
    testcase.assertTrue(
        paddle.equal_all(a, b).item(),
        f"{msg}: values differ (max abs diff "
        f"{paddle.max(paddle.abs(a.astype('float32') - b.astype('float32'))).item()})",
    )


# =====================================================================
# Tests
# =====================================================================


class TestSeparateMTPInputConfig(unittest.TestCase):
    """TransformerConfig.__post_init__ validation for separate_mtp_input."""

    def test_default_is_off(self):
        self.assertFalse(TransformerConfig().separate_mtp_input)

    def test_valid_config_accepted(self):
        cfg = TransformerConfig(
            separate_mtp_input=True,
            num_nextn_predict_layers=1,
            pipeline_model_parallel_size=1,
        )
        self.assertTrue(cfg.separate_mtp_input)

    def test_rejects_multiple_mtp_layers(self):
        # ValueError, not assert: must survive ``python -O``.
        with self.assertRaisesRegex(ValueError, "num_nextn_predict_layers=2"):
            TransformerConfig(
                separate_mtp_input=True,
                num_nextn_predict_layers=2,
                pipeline_model_parallel_size=1,
            )

    def test_rejects_pipeline_parallel(self):
        with self.assertRaisesRegex(
            ValueError, "pipeline_model_parallel_size=2"
        ):
            TransformerConfig(
                separate_mtp_input=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
            )

    def test_mutually_exclusive_with_magic_send(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            TransformerConfig(
                separate_mtp_input=True,
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=1,
            )

    def test_rejects_mtp_load_weight_only(self):
        with self.assertRaisesRegex(ValueError, "mtp_load_weight_only=True"):
            TransformerConfig(
                separate_mtp_input=True,
                mtp_load_weight_only=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=1,
            )

    def test_validation_survives_python_O(self):
        """The checks must not be assert-based (stripped by ``python -O``)."""
        import inspect
        import textwrap

        source = textwrap.dedent(
            inspect.getsource(TransformerConfig.__post_init__)
        )
        block = source.split("if self.separate_mtp_input:", 1)[1]
        # stop at the next top-level statement of __post_init__ (4-space indent)
        block = block.split("\n    if ", 1)[0]
        self.assertNotIn("assert ", block)
        self.assertEqual(block.count("raise ValueError"), 4)


class TestGPTEmbeddingSeparateOutput(unittest.TestCase):
    """GPTEmbedding hands the shifted MTP embedding over a dedicated key."""

    B, S, H = 2, 10, 64  # main decoder length is S - num_nextn_predict_layers

    def _input_ids(self):
        paddle.seed(20260819)
        return paddle.randint(0, _BASE_CFG["vocab_size"], [self.B, self.S])

    def test_hidden_states_carries_backbone_only(self):
        main_len = self.S - 1
        out = _run_emb(_cfg(), self._input_ids())
        self.assertEqual(out["hidden_states"].shape, [self.B, main_len, self.H])
        self.assertIn("mtp_decoder_inputs", out)
        self.assertEqual(
            out["mtp_decoder_inputs"].shape, [1, self.B, main_len, self.H]
        )

    def test_baseline_concatenates_instead(self):
        main_len = self.S - 1
        out = _run_emb(_cfg_baseline(), self._input_ids())
        self.assertEqual(
            out["hidden_states"].shape, [self.B * 2, main_len, self.H]
        )
        self.assertNotIn("mtp_decoder_inputs", out)

    def _assert_matches_baseline(self, cfg_kw, cp_world_size=1):
        """Same weights + same input => bitwise identical tensors, only regrouped."""
        input_ids = self._input_ids()
        shared_emb = _Emb(_BASE_CFG["vocab_size"], self.H)

        base = _run_emb(
            _cfg_baseline(**cfg_kw),
            input_ids,
            cp_world_size=cp_world_size,
            emb_layer=shared_emb,
        )
        sep = _run_emb(
            _cfg(**cfg_kw),
            input_ids,
            cp_world_size=cp_world_size,
            emb_layer=shared_emb,
        )

        # baseline packs [backbone, mtp_0] along axis 0
        chunks = paddle.split(base["hidden_states"], 2)
        _assert_bitwise_equal(
            self, sep["hidden_states"], chunks[0], "backbone embedding"
        )
        _assert_bitwise_equal(
            self,
            sep["mtp_decoder_inputs"][0],
            chunks[1],
            "shifted MTP embedding",
        )

    def test_bitwise_equal_to_baseline(self):
        self._assert_matches_baseline({})

    def test_bitwise_equal_to_baseline_with_cp(self):
        self._assert_matches_baseline(
            {"experimental_dataflow": True}, cp_world_size=2
        )

    def test_bitwise_equal_to_baseline_with_sp(self):
        self._assert_matches_baseline(
            {"sequence_parallel": True, "tensor_model_parallel_size": 2}
        )


class TestMTPLayerSeparateForward(unittest.TestCase):
    """MultiTokenPredictionLayer.forward(): separate_mtp_input branch."""

    B, S, H = 2, 8, 64

    def _dict_args(self, **extra):
        args = {
            "hidden_states": paddle.randn([self.B, self.S, self.H]),
            "mtp_decoder_inputs": paddle.stack(
                [paddle.randn([self.B, self.S, self.H])]
            ),
            "mtp_startend_row_indices_all": paddle.randn(
                [self.B, 1, self.S, 4]
            ),
            "mtp_hidden_inputs_mask_all": paddle.ones([self.B, 1, self.S]),
            "mtp_input_ids_for_moe_mask": paddle.randint(
                0, 512, [self.B, 1, self.S]
            ),
            "input_ids": paddle.randint(0, 512, [self.B, self.S]),
            "rotary_pos_emb": paddle.randn([1, self.S + 2, 32]),
            "rotary_pos_cos": paddle.randn([1, self.S + 2, 32]),
            "rotary_pos_sin": paddle.randn([1, self.S + 2, 32]),
            "labels": paddle.randint(0, 100, [self.B, self.S]),
        }
        args.update(extra)
        return args

    def test_forward_output_shape_and_keys(self):
        layer = _build_mtp_layer(_cfg())
        with _mtp_forward_ctx():
            result = layer.forward(self._dict_args())
        # [backbone | mtp] concatenated on axis 0 for the downstream LM heads
        self.assertEqual(
            result["hidden_states"].shape, [2 * self.B, self.S, self.H]
        )
        # the MTP input was consumed here and must not travel any further
        self.assertNotIn("mtp_decoder_inputs", result)
        self.assertNotIn("decoder_input", result)
        self.assertIn("labels", result)

    def test_decoder_input_passed_through_untouched(self):
        """mtp_decoder_inputs[depth] must reach the MTP block unmodified."""
        layer = _build_mtp_layer(_cfg())
        args = self._dict_args()
        expected = args["mtp_decoder_inputs"][0]
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return kwargs["hidden_states"]

        with _mtp_forward_ctx(proj_override=_capture, layer=layer):
            layer.forward(args)

        self.assertIn("decoder_input", captured)
        _assert_bitwise_equal(
            self, captured["decoder_input"], expected, "decoder_input"
        )

    def test_backbone_hidden_states_preserved_in_output(self):
        layer = _build_mtp_layer(_cfg())
        args = self._dict_args()
        backbone = args["hidden_states"]
        marker = paddle.full([self.B, self.S, self.H], 7.0)

        with _mtp_forward_ctx(proj_override=lambda **kw: marker, layer=layer):
            result = layer.forward(args)

        halves = paddle.split(result["hidden_states"], 2)
        _assert_bitwise_equal(self, halves[0], backbone, "backbone half")
        _assert_bitwise_equal(self, halves[1], marker, "mtp half")

    def test_missing_mtp_decoder_inputs_raises(self):
        layer = _build_mtp_layer(_cfg())
        args = self._dict_args()
        args.pop("mtp_decoder_inputs")
        with (
            self.assertRaisesRegex(RuntimeError, "mtp_decoder_inputs"),
            _mtp_forward_ctx(),
        ):
            layer.forward(args)

    def test_per_depth_mask_slice_by_experimental_version(self):
        """The shared branch slices the depth mask differently per dataflow."""
        for exp_ver, last_dim in ((True, 4), (False, 1)):
            layer = _build_mtp_layer(
                _cfg(gpt_model_use_experimental_version=exp_ver)
            )
            captured = {}

            def _capture(**kwargs):
                captured.update(kwargs)
                return kwargs["hidden_states"]

            with _mtp_forward_ctx(proj_override=_capture, layer=layer):
                layer.forward(self._dict_args())

            self.assertEqual(
                captured["attn_mask_startend_row_indices"].shape,
                [self.B, 1, self.S, last_dim],
                f"gpt_model_use_experimental_version={exp_ver}",
            )


class TestMTPLayerNoDoubleScatter(unittest.TestCase):
    """The scatter already happened in GPTEmbedding: never scatter again here.

    ``magic_send`` re-embeds ``input_ids`` at full global length, so it has to
    slice and then apply the very same CP/SP scatter GPTEmbedding would have
    applied.  ``separate_mtp_input`` receives tensors that are already CP/SP
    local, so a second scatter would corrupt the layout.
    """

    B, S, H = 2, 8, 64

    def _scatter_spies(self):
        return (
            MagicMock(side_effect=lambda x, **kw: x),
            MagicMock(side_effect=lambda x, axis=0, **kw: x),
        )

    def test_separate_does_not_scatter(self):
        cfg = _cfg(
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            experimental_dataflow=True,
        )
        layer = _build_mtp_layer(cfg)
        sp_spy, cp_spy = self._scatter_spies()
        # SP layout: hidden_states / decoder_input are [S_local, B, H]
        args = {
            "hidden_states": paddle.randn([self.S, self.B, self.H]),
            "mtp_decoder_inputs": paddle.stack(
                [paddle.randn([self.S, self.B, self.H])]
            ),
            "labels": paddle.randint(0, 100, [self.B, self.S]),
        }
        with _mtp_forward_ctx(
            cp_world_size=4,
            scatter_fn=sp_spy,
            cp_scatter_fn=cp_spy,
            proj_override=lambda **kw: kw["hidden_states"],
            layer=layer,
        ):
            layer.forward(args)

        sp_spy.assert_not_called()
        cp_spy.assert_not_called()

    def test_magic_send_does_scatter(self):
        """Contrast case: guards against a refactor silently breaking magic send."""
        from paddlefleet.models.gpt.mtp_embedding_layer import (
            mtp_magic_instance,
        )

        cfg = _cfg_magic(experimental_dataflow=True)
        layer = _build_mtp_layer(cfg)
        sp_spy, cp_spy = self._scatter_spies()
        # magic send re-embeds input_ids itself, addressed through the module
        # level MagicInstance the data loader normally fills in.
        input_ids = paddle.randint(0, 512, [self.B, self.S * 4 + 1])
        mtp_magic_instance.set_data({"input_ids": [input_ids]})
        mtp_magic_instance.set_magic_count(layer.magic_key, -1)
        layer.mtp_embed = _FakeEmbed(self.H)

        args = {
            "hidden_states": paddle.randn([self.B, self.S, self.H]),
            "labels": paddle.randint(0, 100, [self.B, self.S]),
        }
        with _mtp_forward_ctx(
            cp_world_size=4,
            scatter_fn=sp_spy,
            cp_scatter_fn=cp_spy,
            proj_override=lambda **kw: kw["hidden_states"],
            layer=layer,
        ):
            layer.forward(args)

        cp_spy.assert_called_once()


class TestTransformerLayerSplitGuard(unittest.TestCase):
    """Backbone layers must not split hidden_states when separate_mtp_input is on.

    ``TransformerLayer.forward`` has the condition inline, so it is mirrored
    here; the real code path is covered end to end by
    ``tests/multi_card_tests/test_separate_mtp_input_cp.py`` (dropping the guard
    there makes ``paddle.split`` fail on the un-concatenated hidden_states).
    ``HySparseTransformerLayer`` exposes it as a method, so that one is called
    for real.
    """

    def test_split_guard_logic(self):
        def should_split(cfg, is_mtp=False):
            return (
                cfg.num_nextn_predict_layers is not None
                and cfg.num_nextn_predict_layers > 0
                and not is_mtp
                and not cfg.mtp_load_weight_only
                and not cfg.enable_mtp_magic_send
                and not cfg.separate_mtp_input
            )

        self.assertFalse(should_split(_cfg()))
        self.assertFalse(should_split(_cfg_magic()))
        self.assertTrue(should_split(_cfg_baseline()))
        self.assertFalse(should_split(_cfg_baseline(), is_mtp=True))

    def test_hysparse_mtp_enabled(self):
        from paddlefleet.transformer.transformer_layer import (
            HySparseTransformerLayer,
        )

        def mtp_enabled(cfg, is_mtp=False):
            return HySparseTransformerLayer._mtp_enabled(
                SimpleNamespace(config=cfg), is_mtp
            )

        self.assertFalse(mtp_enabled(_cfg()))
        self.assertFalse(mtp_enabled(_cfg_magic()))
        self.assertTrue(mtp_enabled(_cfg_baseline()))
        self.assertFalse(mtp_enabled(_cfg_baseline(), is_mtp=True))


class TestHySparseLayerMTPSplit(unittest.TestCase):
    """``HySparseTransformerLayer`` is the layer class the production DSA config
    uses, and it is the only one that implements the MTP split/restore as
    methods (``_mtp_split`` / ``_mtp_restore``) instead of inline in ``forward``.
    They only touch ``self.config``, so they can be exercised on a bare instance
    without the FA4/DSA attention backend.
    """

    B, S, H = 2, 8, 64

    def _layer(self, cfg):
        from paddlefleet.transformer.transformer_layer import (
            HySparseTransformerLayer,
        )

        layer = HySparseTransformerLayer.__new__(HySparseTransformerLayer)
        layer.config = cfg
        return layer

    def _dict_args(self, hidden_states):
        return {
            "hidden_states": hidden_states,
            "position_ids": paddle.arange(self.S + 1).reshape([1, -1]),
            "rotary_pos_emb": paddle.randn([1, self.S + 1, 32]),
        }

    def test_separate_skips_split(self):
        layer = self._layer(_cfg())
        hidden_states = paddle.randn([self.B, self.S, self.H])
        dict_args = self._dict_args(hidden_states)
        rotary_before = dict_args["rotary_pos_emb"]

        ctx = layer._mtp_split(dict_args, is_mtp=False)

        self.assertIsNone(ctx, "separate_mtp_input must not split in the layer")
        _assert_bitwise_equal(
            self, dict_args["hidden_states"], hidden_states, "hidden_states"
        )
        # untouched: no trimming of the auxiliary tensors either
        self.assertIs(dict_args["rotary_pos_emb"], rotary_before)
        self.assertEqual(dict_args["position_ids"].shape, [1, self.S + 1])

    def test_baseline_splits_and_restores(self):
        layer = self._layer(_cfg_baseline())
        chunks = [
            paddle.randn([self.B, self.S, self.H]),
            paddle.randn([self.B, self.S, self.H]),
        ]
        dict_args = self._dict_args(paddle.concat(chunks))

        ctx = layer._mtp_split(dict_args, is_mtp=False)

        self.assertIsNotNone(ctx)
        _assert_bitwise_equal(
            self, dict_args["hidden_states"], chunks[0], "main chunk"
        )
        self.assertEqual(len(ctx["mtp_input"]), 1)
        _assert_bitwise_equal(self, ctx["mtp_input"][0], chunks[1], "mtp chunk")
        self.assertEqual(dict_args["position_ids"].shape, [1, self.S])
        self.assertEqual(dict_args["rotary_pos_emb"].shape, [1, self.S, 32])

        restored = layer._mtp_restore(dict_args, chunks[0], ctx)
        self.assertEqual(restored.shape, [2 * self.B, self.S, self.H])
        _assert_bitwise_equal(
            self, restored, paddle.concat(chunks), "restored hidden_states"
        )
        self.assertEqual(dict_args["position_ids"].shape, [1, self.S + 1])
        self.assertEqual(dict_args["rotary_pos_emb"].shape, [1, self.S + 1, 32])

    def test_magic_send_also_skips_split(self):
        layer = self._layer(_cfg_magic())
        hidden_states = paddle.randn([self.B, self.S, self.H])
        dict_args = self._dict_args(hidden_states)
        self.assertIsNone(layer._mtp_split(dict_args, is_mtp=False))


class TestWrappedPaddleNormPipeSeparate(unittest.TestCase):
    """WrappedPaddleNormPipe treats separate_mtp_input like magic send."""

    def _norm_out(self, cfg, hidden_states):
        from paddlefleet.transformer.paddle_norm import WrappedPaddleNormPipe

        return WrappedPaddleNormPipe(cfg, hidden_size=64).forward(
            {"hidden_states": hidden_states}
        )["hidden_states"]

    def test_separate_normalizes_without_split(self):
        cfg = TransformerConfig(
            separate_mtp_input=True,
            num_nextn_predict_layers=1,
            pipeline_model_parallel_size=1,
            hidden_size=64,
            normalization="RMSNorm",
            tensor_model_parallel_size=1,
        )
        out = self._norm_out(cfg, paddle.ones([1, 4, 64]) * 2.0)
        self.assertTrue(
            paddle.allclose(out, paddle.ones([1, 4, 64]), atol=1e-5).item()
        )

    def test_matches_magic_send_behaviour(self):
        common = {
            "num_nextn_predict_layers": 1,
            "hidden_size": 64,
            "normalization": "RMSNorm",
            "tensor_model_parallel_size": 1,
        }
        hidden_states = paddle.randn([2, 4, 64])
        sep = self._norm_out(
            TransformerConfig(
                separate_mtp_input=True,
                pipeline_model_parallel_size=1,
                **common,
            ),
            hidden_states,
        )
        magic = self._norm_out(
            TransformerConfig(
                enable_mtp_magic_send=True,
                pipeline_model_parallel_size=2,
                **common,
            ),
            hidden_states,
        )
        _assert_bitwise_equal(self, sep, magic, "final norm output")

    def test_baseline_still_splits(self):
        cfg = TransformerConfig(
            num_nextn_predict_layers=1,
            hidden_size=64,
            normalization="RMSNorm",
            tensor_model_parallel_size=1,
        )
        out = self._norm_out(cfg, paddle.randn([4, 8, 64]))
        self.assertEqual(out.shape, [4, 8, 64])

    def test_experimental_version_splits_and_norms_every_chunk(self):
        """gpt_model_use_experimental_version flips the guard back to the split
        path: by then hidden_states is the [backbone | mtp] concat rebuilt by the
        MTP layer, and the MTP chunk gets normalized too."""
        from paddlefleet.transformer.paddle_norm import WrappedPaddleNormPipe

        cfg = TransformerConfig(
            separate_mtp_input=True,
            num_nextn_predict_layers=1,
            pipeline_model_parallel_size=1,
            hidden_size=64,
            normalization="RMSNorm",
            tensor_model_parallel_size=1,
            gpt_model_use_experimental_version=True,
        )
        pipe = WrappedPaddleNormPipe(cfg, hidden_size=64)
        x = paddle.randn([4, 8, 64])  # [backbone | mtp_0] stacked on axis 0
        expected = [pipe.norm(chunk) for chunk in paddle.split(x, 2)]

        out = pipe.forward({"hidden_states": x.clone()})["hidden_states"]
        self.assertEqual(out.shape, [4, 8, 64])
        got = paddle.split(out, 2)
        _assert_bitwise_equal(self, got[0], expected[0], "backbone chunk norm")
        _assert_bitwise_equal(self, got[1], expected[1], "mtp chunk norm")


class TestHyperConnectionContractSeparate(unittest.TestCase):
    """mHC contract layer: hidden_states is pure backbone, contract it whole."""

    def _build(self, **kw):
        from paddlefleet.transformer.hyper_connection import (
            HyperConnectionContractLayer,
        )

        return HyperConnectionContractLayer(
            TransformerConfig(
                hidden_size=64,
                num_residual_streams=4,
                num_nextn_predict_layers=1,
                tensor_model_parallel_size=1,
                **kw,
            )
        )

    def test_separate_contracts_entire_tensor(self):
        """The whole tensor is contracted, not just its first 1/(num_mtp+1)."""
        from paddlefleet.transformer.hyper_connection import (
            HyperConnectionModule,
        )

        layer = self._build(
            separate_mtp_input=True, pipeline_model_parallel_size=1
        )
        B, S, H, n = 2, 8, 64, 4
        x = paddle.randn([B, S, H * n])
        expected = HyperConnectionModule.learned_output_contract(
            x,
            layer.hc_head_fn,
            layer.hc_head_base,
            layer.hc_head_scale,
            n,
            layer.config.rms_norm_eps,
        )
        result = layer.forward({"hidden_states": x.clone()})
        self.assertEqual(result["hidden_states"].shape, [B, S, H])
        # mhc_multistream is expanded to num_mtp+1 slots (zeros for the MTP
        # depths, overwritten by the MTP layers); the first slot is the input.
        self.assertEqual(
            result["mhc_multistream"].shape,
            [B * (layer.num_mtp + 1), S, H * n],
        )
        _assert_bitwise_equal(
            self,
            paddle.split(result["mhc_multistream"], layer.num_mtp + 1)[0],
            x,
            "mhc_multistream first slot",
        )
        self.assertFalse(layer.magic_send)
        _assert_bitwise_equal(
            self, result["hidden_states"], expected, "mHC contract output"
        )

    def test_separate_matches_magic_send(self):
        B, S, H, n = 2, 8, 64, 4
        x = paddle.randn([B, S, H * n])
        sep = self._build(
            separate_mtp_input=True, pipeline_model_parallel_size=1
        )
        magic = self._build(
            enable_mtp_magic_send=True, pipeline_model_parallel_size=2
        )
        magic.hc_head_fn.set_value(sep.hc_head_fn)
        magic.hc_head_base.set_value(sep.hc_head_base)
        magic.hc_head_scale.set_value(sep.hc_head_scale)

        out_sep = sep.forward({"hidden_states": x.clone()})["hidden_states"]
        out_magic = magic.forward({"hidden_states": x.clone()})["hidden_states"]
        _assert_bitwise_equal(self, out_sep, out_magic, "mHC contract output")

    def test_baseline_splits_before_contracting(self):
        layer = self._build(pipeline_model_parallel_size=1)
        B, S, H, n = 2, 8, 64, 4
        result = layer.forward(
            {"hidden_states": paddle.randn([B * 2, S, H * n])}
        )
        self.assertEqual(result["hidden_states"].shape, [B * 2, S, H])


if __name__ == "__main__":
    unittest.main()
