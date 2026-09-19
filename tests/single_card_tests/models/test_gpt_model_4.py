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
"""Unit tests for paddlefleet.models.gpt.gpt_model helpers.

Environment: no-card (CPU). Every method exercises pure control/dispatch logic
of GPTModel and its module-level helpers; no GPU numerics are claimed here.
paddle is a hard dependency of the production module, so when it (or the
paddlefleet package) is not importable the whole file skips with an honest
reason instead of faking a pass.
"""

import os
import sys
import unittest
from unittest import mock

# Make the in-repo `src/paddlefleet` importable when the package is not installed.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

# Honest capability probe: only ImportError/ModuleNotFoundError count as
# "dependency missing". Any other import-time failure must surface, not be
# swallowed as a skip.
try:
    import paddle
    from paddle.distributed.fleet.meta_parallel import (
        LayerDesc,  # noqa: F401
        LayerSpec,
        ScheduleChunk,
        SharedLayerDesc,
    )

    import paddlefleet.models.gpt.gpt_model as gpt_model_mod
    from paddlefleet.models.gpt.gpt_model import (
        GPTModel,
        GPTSublayersSpec,
        build_overlapped_nodes,
        is_vision_merge_key,
    )

    HAVE_PADDLE = True
    _SKIP_REASON = ""
except ImportError as exc:  # paddle / paddlefleet not installed
    HAVE_PADDLE = False
    _SKIP_REASON = f"paddle/paddlefleet not importable: {exc!r}"


# --- Lightweight, paddle-free recording collaborators -----------------------
# These stand in for TransformerLayer / MultiTokenPredictionLayer purely so the
# dispatch logic (iterate -> isinstance -> forward kwargs) can be observed. We
# patch the module globals used by isinstance so the *real* dispatch code runs;
# only the leaf `fp8_quant_weight` / `use_fp8` implementations are recorded.
class _RecTL:
    """Records fp8_quant_weight kwargs; reports a fixed use_fp8 flag."""

    def __init__(self, fp8=False):
        self.calls = []
        self._fp8 = fp8

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        self.calls.append((batch_mode, quant_transpose))

    def use_fp8(self):
        return self._fp8


class _RecMTP:
    """MTP stand-in whose fp8 call must be forwarded via .transformer_layer."""

    def __init__(self):
        self.transformer_layer = _RecTL()


class _Cfg:
    """Explicit config object (no MagicMock: every flag is a real bool/value)."""

    def __init__(self, **kw):
        self.model_type = kw.get("model_type", "")
        self.enable_mtp_magic_send = kw.get("enable_mtp_magic_send", False)
        self.gpt_model_use_experimental_version = kw.get("experimental", False)
        self.num_nextn_predict_layers = kw.get("num_nextn", 0)
        self.mtp_shared_last_layer = kw.get("mtp_shared_last_layer", False)
        self.multimax_modules = kw.get("multimax_modules", None)
        self.separate_mtp_headloss = kw.get("separate_mtp_headloss", False)


if HAVE_PADDLE:

    class _DummyLayer(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

    def _spec(**overrides):
        defaults = {
            "embedding": LayerSpec(_DummyLayer),
            "head_empty_layers": [],
            "transformer_layers": [LayerSpec(_DummyLayer)],
            "tail_empty_layers": [],
            "layer_norm": LayerSpec(_DummyLayer),
            "lm_head": LayerSpec(_DummyLayer),
        }
        defaults.update(overrides)
        return GPTSublayersSpec(**defaults)

    def _model(cfg):
        m = GPTModel.__new__(GPTModel)
        m.config = cfg
        m._pipeline_name_mapping = None
        m._pp_to_single_mapping = None
        m._sequential_layers = []
        return m

    def _prefixes(layers):
        return [entry["name_prefix"] for entry in layers]

    def _is_shared(layers):
        return [isinstance(entry["layer"], SharedLayerDesc) for entry in layers]


@unittest.skipUnless(HAVE_PADDLE, _SKIP_REASON)
class TestIsVisionMergeKey(unittest.TestCase):
    """is_vision_merge_key: three disjoint branches, no swallowed errors."""

    def test_non_vision_merge_key_is_false(self):
        self.assertFalse(
            is_vision_merge_key("model.layers.0.self_attn.q_proj.weight")
        )
        self.assertFalse(is_vision_merge_key(""))

    def test_vision_model_key_is_true(self):
        self.assertTrue(
            is_vision_merge_key(
                "vision_merge.vision_model.blocks.0.attn.qkv.weight"
            )
        )

    def test_wrapper_only_key_raises_value_error(self):
        # Starts with "vision_merge." but not "vision_merge.vision_model." -> a
        # param that would be silently dropped, so the function must reject it.
        with self.assertRaises(ValueError):
            is_vision_merge_key("vision_merge.extra_head.weight")
        # Missing trailing dot after vision_model is likewise the reject branch.
        with self.assertRaises(ValueError):
            is_vision_merge_key("vision_merge.vision_modelX.weight")


@unittest.skipUnless(HAVE_PADDLE, _SKIP_REASON)
class TestBuildOverlappedNodes(unittest.TestCase):
    """build_overlapped_nodes: partition/pairing derived by hand."""

    def test_no_overlap_partitions_pre_and_post(self):
        # overlap = min(fwd_transformer=1, bwd_transformer=0) = 0, so the single
        # forward transformer node and everything after it fall into post.
        with mock.patch.object(gpt_model_mod, "TransformerLayerNode", _RecTL):
            n0, t1, n2 = object(), _RecTL(), object()
            m0, m1 = object(), object()
            fwd = ScheduleChunk([n0, t1, n2])
            bwd = ScheduleChunk([m0, m1])
            (
                fwd_pre,
                bwd_pre,
                overlap,
                fwd_post,
                bwd_post,
            ) = build_overlapped_nodes(fwd, bwd)
        self.assertEqual(fwd_pre.nodes, [n0])
        self.assertEqual(fwd_post.nodes, [t1, n2])
        self.assertEqual(bwd_pre.nodes, [m0, m1])
        self.assertEqual(bwd_post.nodes, [])
        self.assertEqual(overlap.nodes, [])

    def test_overlap_pairs_forward_with_reversed_backward(self):
        # fwd=[t0,t1,t2] bwd=[b0,b1]; overlap=min(3,2)=2. Forward keeps order,
        # backward is consumed in reverse -> pairs (t0,b1),(t1,b0); t2 -> post.
        class _Pair:
            def __init__(self, f, b):
                self.f, self.b = f, b

        with (
            mock.patch.object(gpt_model_mod, "TransformerLayerNode", _RecTL),
            mock.patch.object(
                gpt_model_mod, "TransformerLayerOverlappedScheduleNode", _Pair
            ),
        ):
            t0, t1, t2 = _RecTL(), _RecTL(), _RecTL()
            b0, b1 = _RecTL(), _RecTL()
            (
                fwd_pre,
                bwd_pre,
                overlap,
                fwd_post,
                bwd_post,
            ) = build_overlapped_nodes(
                ScheduleChunk([t0, t1, t2]), ScheduleChunk([b0, b1])
            )
        self.assertEqual(fwd_pre.nodes, [])
        self.assertEqual(fwd_post.nodes, [t2])
        self.assertEqual(bwd_pre.nodes, [])
        self.assertEqual(bwd_post.nodes, [])
        self.assertEqual(len(overlap.nodes), 2)
        self.assertIs(overlap.nodes[0].f, t0)
        self.assertIs(overlap.nodes[0].b, b1)
        self.assertIs(overlap.nodes[1].f, t1)
        self.assertIs(overlap.nodes[1].b, b0)


@unittest.skipUnless(HAVE_PADDLE, _SKIP_REASON)
class TestGPTSublayersSpec(unittest.TestCase):
    """GPTSublayersSpec dataclass defaults and field storage."""

    def test_all_fields_default_to_none(self):
        spec = GPTSublayersSpec()
        for field in (
            "embedding",
            "head_empty_layers",
            "mhc_expand",
            "transformer_layers",
            "mhc_contract",
            "tail_empty_layers",
            "mtp",
            "output_block_attn_res",
            "layer_norm",
            "lm_head",
            "mtp_lm_head",
            "mtp_loss",
        ):
            self.assertIsNone(getattr(spec, field), field)

    def test_fields_store_given_values(self):
        emb, ln, head = object(), object(), object()
        spec = GPTSublayersSpec(embedding=emb, layer_norm=ln, lm_head=head)
        self.assertIs(spec.embedding, emb)
        self.assertIs(spec.layer_norm, ln)
        self.assertIs(spec.lm_head, head)
        self.assertIsNone(spec.mtp)


@unittest.skipUnless(HAVE_PADDLE, _SKIP_REASON)
class TestGetLayerDescList(unittest.TestCase):
    """get_layer_desc_list: exact layer ordering, name_prefixes and which
    entries become SharedLayerDesc are all hand-derived from the source."""

    def test_default_prefix_and_plain_descs(self):
        layers = _model(_Cfg()).get_layer_desc_list(
            _spec(), tie_word_embeddings=False
        )
        self.assertEqual(
            _prefixes(layers),
            ["model", "model.layers.0", "model", "model.lm_head"],
        )
        self.assertEqual(_is_shared(layers), [False, False, False, False])

    def test_qwen3_5_uses_language_model_prefix(self):
        layers = _model(_Cfg(model_type="qwen3_5")).get_layer_desc_list(
            _spec(), tie_word_embeddings=False
        )
        self.assertEqual(
            _prefixes(layers),
            [
                "model.language_model",
                "model.language_model.layers.0",
                "model.language_model",
                "model.language_model.lm_head",
            ],
        )

    def test_tie_word_embeddings_shares_embed_and_head(self):
        layers = _model(_Cfg()).get_layer_desc_list(
            _spec(), tie_word_embeddings=True
        )
        self.assertEqual(
            _prefixes(layers),
            ["model", "model.layers.0", "model", "model.shared_head"],
        )
        # Embedding (idx 0) and head (idx 3) become SharedLayerDesc; the
        # transformer layer and the layer_norm stay plain LayerDesc.
        self.assertEqual(_is_shared(layers), [True, False, False, True])

    def test_experimental_mtp_defers_layer_norm_after_mtp(self):
        cfg = _Cfg(experimental=True, num_nextn=1)
        spec = _spec(
            mtp=[LayerSpec(_DummyLayer)],
            mtp_lm_head=LayerSpec(_DummyLayer),
            mtp_loss=LayerSpec(_DummyLayer),
        )
        layers = _model(cfg).get_layer_desc_list(
            spec, tie_word_embeddings=False
        )
        self.assertEqual(
            _prefixes(layers),
            [
                "model",
                "model.layers.0",
                "model.layers.1",
                "model.shared_mtp_lm_head",
                "model.mtp_loss",
                "model",
                "model.shared_head",
            ],
        )
        # layer_norm ("model" at idx 5) sits AFTER mtp/mtp_loss under the
        # experimental+MTP branch; shared_mtp_lm_head and shared_head are shared.
        self.assertEqual(
            _is_shared(layers),
            [False, False, False, True, False, False, True],
        )

    def test_mtp_shared_last_layer_shares_without_magic_send(self):
        # mtp_shared_last_layer is orthogonal to enable_mtp_magic_send: the last
        # backbone transformer and every mtp spec must become SharedLayerDesc
        # even though magic send is off.
        cfg = _Cfg(mtp_shared_last_layer=True)
        spec = _spec(mtp=[LayerSpec(_DummyLayer)])
        layers = _model(cfg).get_layer_desc_list(
            spec, tie_word_embeddings=False
        )
        self.assertEqual(
            _prefixes(layers),
            [
                "model",
                "model.layers.0",
                "model",
                "model.layers.1",
                "model.lm_head",
            ],
        )
        self.assertEqual(_is_shared(layers), [False, True, False, True, False])


@unittest.skipUnless(HAVE_PADDLE, _SKIP_REASON)
class TestSetPipelineNameMapping(unittest.TestCase):
    """_set_pipeline_name_mapping explicit-mapping branch."""

    def test_explicit_mapping_is_stored_and_returned(self):
        model = GPTModel.__new__(GPTModel)
        model._pipeline_name_mapping = None
        mapping = {"model.embed.weight": "0.embedding.weight"}
        result = model._set_pipeline_name_mapping(mapping)
        # Returns the very object passed in and installs it on the model.
        self.assertIs(result, mapping)
        self.assertIs(model._pipeline_name_mapping, mapping)


@unittest.skipUnless(HAVE_PADDLE, _SKIP_REASON)
class TestFp8QuantWeight(unittest.TestCase):
    """fp8_quant_weight: dispatch + exact kwarg forwarding for both the flat
    run_function path and the virtual-pipeline chunk path."""

    def test_dispatch_over_run_function(self):
        tl, mtp, other = _RecTL(), _RecMTP(), object()
        model = GPTModel.__new__(GPTModel)
        model._num_virtual_pipeline_stages = 1
        model.run_function = [tl, mtp, other]
        with (
            mock.patch.object(gpt_model_mod, "TransformerLayer", _RecTL),
            mock.patch.object(
                gpt_model_mod, "MultiTokenPredictionLayer", _RecMTP
            ),
        ):
            model.fp8_quant_weight(batch_mode=True, quant_transpose=False)
        # TransformerLayer gets the call directly; MTP forwards via its inner
        # transformer_layer; the unrelated object is left untouched.
        self.assertEqual(tl.calls, [(True, False)])
        self.assertEqual(mtp.transformer_layer.calls, [(True, False)])

    def test_dispatch_over_virtual_pipeline_chunks(self):
        tl, mtp = _RecTL(), _RecMTP()
        model = GPTModel.__new__(GPTModel)
        model._num_virtual_pipeline_stages = 2
        model._model_chunks = [[tl, mtp]]
        with (
            mock.patch.object(gpt_model_mod, "TransformerLayer", _RecTL),
            mock.patch.object(
                gpt_model_mod, "MultiTokenPredictionLayer", _RecMTP
            ),
        ):
            model.fp8_quant_weight()  # defaults: batch_mode=False, quant_transpose=True
        self.assertEqual(tl.calls, [(False, True)])
        self.assertEqual(mtp.transformer_layer.calls, [(False, True)])


@unittest.skipUnless(HAVE_PADDLE, _SKIP_REASON)
class TestUseFp8(unittest.TestCase):
    """use_fp8 return contract, including a real return-value bug."""

    def test_no_fp8_layer_returns_false_flat_path(self):
        model = GPTModel.__new__(GPTModel)
        model._num_virtual_pipeline_stages = 1
        model.run_function = [object(), object()]
        self.assertIs(model.use_fp8(), False)

    def test_virtual_stage_with_fp8_layer_returns_true(self):
        tl = _RecTL(fp8=True)
        model = GPTModel.__new__(GPTModel)
        model._num_virtual_pipeline_stages = 2
        model._model_chunks = [[tl]]
        with mock.patch.object(gpt_model_mod, "TransformerLayer", _RecTL):
            self.assertIs(model.use_fp8(), True)

    @unittest.expectedFailure
    def test_virtual_stage_without_fp8_should_return_false(self):
        # REAL BUG (gpt_model.py:986-996): the `_num_virtual_pipeline_stages > 1`
        # branch has no `return False` fallthrough -- the lone `return False`
        # lives inside the `else` block. So VPP>1 with no fp8 layer returns
        # None instead of False, making use_fp8()'s return type inconsistent.
        # We assert the CORRECT behavior and mark it expectedFailure rather than
        # editing production.
        model = GPTModel.__new__(GPTModel)
        model._num_virtual_pipeline_stages = 2
        model._model_chunks = []
        self.assertIs(model.use_fp8(), False)


if __name__ == "__main__":
    unittest.main()
