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
#
# Scope: the whole-model modular AOA generator -- the two whole-model entries,
# the live-unit walk they recurse through, the output head's resolution through
# the shipped mapping, and the pipeline-group gather. Exercised with lightweight
# fakes so no pipeline / fleet init is required.
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import types
import unittest
from unittest import mock

import paddle
from paddle.distributed.fleet.meta_parallel.parallel_layers.pp_layers import (
    PipelineLayerChunk,
)

from paddlefleet.models.gpt import aoa_generator as gen
from paddlefleet.models.gpt.gpt_model import GPTModel


class _Leaf(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[2])


class _Cfg:
    """Minimal stand-in for a Fleet config, carrying only AOA name fields."""

    def __init__(self, **kwargs):
        self.aoa_checkpoint_name_prefix = kwargs.get(
            "aoa_checkpoint_name_prefix", "hf"
        )
        self.aoa_checkpoint_name_mapping = kwargs.get(
            "aoa_checkpoint_name_mapping", {}
        )
        self.model_type = kwargs.get("model_type", "")


class _FakeWholeModel:
    """Duck-typed whole-model boundary for the pure generator functions.

    Exposes exactly what ``aoa_generator`` reads: the child sub-layers and the
    pipeline->single mapping.

    The model single-name root hook is borrowed from ``GPTModel`` verbatim
    rather than stubbed: the context takes the root from it, and a stub would
    put the test's own root under test instead of the shipped one.
    """

    _model_name_prefix = GPTModel._model_name_prefix

    def __init__(self, sub_layers, pp_mapping):
        self._sub_layers = sub_layers
        self._pp_to_single_mapping = pp_mapping
        self._pipeline_name_mapping = object()  # non-None: skip rebuild
        # ``_model_name_prefix`` is the shipped hook and reads ``model_type``
        # off the config, so the fake must carry one.
        self.config = _Cfg()

    def _set_pipeline_name_mapping(self):  # pragma: no cover - never hit
        raise AssertionError("mapping already set")


_ENTRY_METHODS = (
    "gen_aoa_statements",
    "gen_inv_aoa_statements",
)


def _two_leaf_model():
    """Embedding + output_layer, the smallest whole-model shape."""
    return _FakeWholeModel(
        {"embedding": _Leaf(), "output_layer": _Leaf()},
        {
            "embedding.weight": "model.embed_tokens.weight",
            "output_layer.weight": "model.lm_head.weight",
        },
    )


def _entry_model(*, config=None):
    """A fake boundary that borrows GPTModel's whole-model AOA entries verbatim.

    Avoids constructing a real pipeline model: the entries only touch what
    ``aoa_generator`` reads off the model. The modular path is therefore
    exercised for real, which is what keeps the component-dispatch contract
    under test.
    """
    model = _two_leaf_model()
    if config is not None:
        model.config = config
    for name in _ENTRY_METHODS:
        setattr(model, name, types.MethodType(getattr(GPTModel, name), model))
    return model


_EMBED = "hf.embed_tokens.weight -> model.embed_tokens.weight"
_HEAD = "hf.lm_head.weight -> model.lm_head.weight"


class TestWholeModelAOAEntry(unittest.TestCase):
    """The two whole-model entries: dispatch, config default, and boundary."""

    def test_entry_runs_the_modular_generator(self):
        model = _entry_model()
        cfg = _Cfg()
        self.assertEqual(
            model.gen_aoa_statements(cfg)["aoa_statements"], [_EMBED, _HEAD]
        )
        self.assertEqual(
            model.gen_inv_aoa_statements(cfg)["aoa_statements"],
            [
                "model.lm_head.weight -> hf.lm_head.weight",
                "model.embed_tokens.weight -> hf.embed_tokens.weight",
            ],
        )

    def test_config_falls_back_to_self_config(self):
        model = _entry_model(config=_Cfg(aoa_checkpoint_name_prefix="own"))
        stmts = model.gen_aoa_statements()["aoa_statements"]
        self.assertTrue(all(s.startswith("own.") for s in stmts))

    def test_recursion_protocol_call_is_rejected(self):
        """A container must call the entry, not recurse through it.

        The entry takes ``(config=None)`` only, so it cannot absorb the base
        ``Layer`` protocol's keyword-only ``structured_name_prefix`` /
        ``checkpoint_lookup_drop_segment``. Recursing into this boundary
        therefore raises instead of silently emitting statements that skip
        whole-model orchestration.
        """
        model = _entry_model()
        ctx = gen.build_aoa_context(_two_leaf_model(), _Cfg())
        with self.assertRaises(TypeError):
            model.gen_aoa_statements(ctx, structured_name_prefix="tower.")
        with self.assertRaises(TypeError):
            model.gen_inv_aoa_statements(ctx, structured_name_prefix="tower.")


class TestAOAContextBuild(unittest.TestCase):
    def test_context_snapshots_config_and_mapping(self):
        model = _two_leaf_model()
        cfg = _Cfg(aoa_checkpoint_name_prefix="hf")
        ctx = gen.build_aoa_context(model, cfg)
        self.assertEqual(ctx.checkpoint_name_prefix, "hf")
        self.assertEqual(ctx.pp_to_single_mapping, model._pp_to_single_mapping)
        self.assertEqual(ctx.model_name_prefix, "model")
        self.assertIs(ctx.config, cfg)

    def test_context_defaults_are_resolved_not_none(self):
        class _Bare:
            pass

        ctx = gen.build_aoa_context(_two_leaf_model(), _Bare())
        self.assertEqual(
            ctx.checkpoint_name_prefix, gen.DEFAULT_CHECKPOINT_NAME_PREFIX
        )
        # A config that declares no mapping gets an empty one, never ``None``:
        # the generic boundary carries no model-specific layout, so every name
        # resolves through the identity fallback.
        self.assertEqual(dict(ctx.checkpoint_name_mapping), {})


class TestWholeModelWalk(unittest.TestCase):
    """The live-unit walk both entries recurse through."""

    def setUp(self):
        self.model = _two_leaf_model()
        self.ctx = gen.build_aoa_context(self.model, _Cfg())

    def test_forward_targets_cover_distinct_single_names(self):
        targets = {
            s.split(" -> ")[1]
            for s in gen.gen_whole_model_aoa(self.model, self.ctx)[
                "aoa_statements"
            ]
        }
        # Every distinct model single-name is a forward target exactly once.
        self.assertEqual(
            targets, {"model.embed_tokens.weight", "model.lm_head.weight"}
        )

    def test_inverse_is_the_mirror_walked_in_reverse(self):
        fwd = gen.gen_whole_model_aoa(self.model, self.ctx)["aoa_statements"]
        inv = gen.gen_whole_model_inv_aoa(self.model, self.ctx)[
            "aoa_statements"
        ]
        swapped = [
            " -> ".join(reversed(statement.split(" -> ")))
            for statement in reversed(fwd)
        ]
        self.assertEqual(inv, swapped)

    def test_pipeline_chunks_surface_their_inner_layers(self):
        """VPP: a chunk's inner layer keeps its two-segment structured name."""
        chunk = PipelineLayerChunk()
        chunk.append(_Leaf())
        model = _FakeWholeModel(
            {"0": chunk}, {"0.0.weight": "model.layers.5.weight"}
        )
        ctx = gen.build_aoa_context(model, _Cfg())
        self.assertEqual(
            gen.gen_whole_model_aoa(model, ctx)["aoa_statements"],
            ["hf.layers.5.weight -> model.layers.5.weight"],
        )

    def test_shared_layer_dict_surfaces_the_layer_it_holds(self):
        """The container the pipeline holds shared layers in owns no params."""
        model = _FakeWholeModel(
            {"shared_layers": paddle.nn.LayerDict({"embed": _Leaf()})},
            {"shared_layers.embed.weight": "model.embed_tokens.weight"},
        )
        ctx = gen.build_aoa_context(model, _Cfg())
        self.assertEqual(
            gen.gen_whole_model_aoa(model, ctx)["aoa_statements"],
            ["hf.embed_tokens.weight -> model.embed_tokens.weight"],
        )

    def test_a_layer_registered_twice_is_generated_once(self):
        """A shared layer's pivot sits under its alias and in the layer list.

        Both paths resolve to the same single name, so generating the subtree
        twice would repeat every statement -- and repeats are not safe to
        remove after the fact, since statements are an ordered program.
        """
        pivot = _Leaf()
        chunk = PipelineLayerChunk()
        chunk.append(pivot)
        model = _FakeWholeModel(
            {
                "shared_layers": paddle.nn.LayerDict({"embed": pivot}),
                "0": chunk,
            },
            {
                "shared_layers.embed.weight": "model.embed_tokens.weight",
                "0.0.weight": "model.embed_tokens.weight",
            },
        )
        ctx = gen.build_aoa_context(model, _Cfg())
        self.assertEqual(
            gen.gen_whole_model_aoa(model, ctx)["aoa_statements"],
            ["hf.embed_tokens.weight -> model.embed_tokens.weight"],
        )
        self.assertEqual(
            gen.gen_whole_model_inv_aoa(model, ctx)["aoa_statements"],
            ["model.embed_tokens.weight -> hf.embed_tokens.weight"],
        )

    def test_statement_list_is_mutable(self):
        # The consumer contract hands the list on to callers that extend it.
        out = gen.gen_whole_model_aoa(self.model, self.ctx)
        out["aoa_statements"].append("hf.x -> model.x")


class _OverridingLeaf(paddle.nn.Layer):
    """Stands in for a component-level override (Linear / SelfAttention...)."""

    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[2])
        self.seen_ctx = None

    def gen_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        self.seen_ctx = ctx
        return [f"__component__:{structured_name_prefix}"]

    def gen_inv_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        self.seen_ctx = ctx
        return [f"__component_inv__:{structured_name_prefix}"]


class TestRecursionProtocol(unittest.TestCase):
    """The whole-model walk must go through the Layer protocol, not a fork.

    Regression guard: an earlier revision re-implemented the module walk inside
    the generator and recursed into itself, which silently bypassed every
    component override.
    """

    def setUp(self):
        self.comp = _OverridingLeaf()
        self.model = _FakeWholeModel(
            {"embedding": _Leaf(), "component": self.comp},
            {
                "embedding.weight": "model.embed_tokens.weight",
                "component.weight": "model.component.weight",
            },
        )
        self.ctx = gen.build_aoa_context(self.model, _Cfg())

    def test_component_override_is_dispatched_to(self):
        stmts = gen.gen_whole_model_aoa(self.model, self.ctx)["aoa_statements"]
        self.assertIn("__component__:component.", stmts)
        # The default per-parameter line must not also be emitted for it.
        self.assertNotIn("hf.component.weight -> model.component.weight", stmts)

    def test_component_override_is_dispatched_to_in_inverse(self):
        stmts = gen.gen_whole_model_inv_aoa(self.model, self.ctx)[
            "aoa_statements"
        ]
        self.assertIn("__component_inv__:component.", stmts)

    def test_context_reaches_the_component_unchanged(self):
        # The context is built once and forwarded as-is, so a component reads
        # the same checkpoint protocol the entry resolved.
        gen.gen_whole_model_aoa(self.model, self.ctx)
        self.assertIs(self.comp.seen_ctx, self.ctx)


class _HeadLeaf(paddle.nn.Layer):
    """Output head stand-in, carrying the multimax extras the real head owns."""

    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[2])
        self.multimax_ranges = self.create_parameter(shape=[4])


def _head_mapping_ctx(model):
    """Context whose only mapping entry is the output head.

    The head's off-root value (a top-level ``lm_head`` sibling of the backbone)
    is what these tests exercise, so a one-entry fixture stands in for a full
    model layout; every other name resolves through the identity fallback.
    """
    cfg = _Cfg(
        aoa_checkpoint_name_prefix="model",
        aoa_checkpoint_name_mapping={"model.lm_head.weight": "lm_head.weight"},
    )
    return gen.build_aoa_context(model, cfg)


class TestOutputHeadThroughTheMapping(unittest.TestCase):
    """The output head is an ordinary mapping entry, not a walk special case.

    The ``ForCausalLM`` layout keeps ``lm_head`` a top-level sibling of the
    backbone; the mapping value says exactly that, so the shared checkpoint root
    is not prepended to it.
    """

    def _statements(self, head_single_prefix):
        model = _FakeWholeModel(
            {"output_layer": _HeadLeaf()},
            {
                "output_layer.weight": f"{head_single_prefix}.weight",
                "output_layer.multimax_ranges": (
                    f"{head_single_prefix}.multimax_ranges"
                ),
            },
        )
        return gen.gen_whole_model_aoa(model, _head_mapping_ctx(model))[
            "aoa_statements"
        ]

    def test_head_weight_lands_at_the_checkpoint_top_level(self):
        self.assertIn(
            "lm_head.weight -> model.lm_head.weight",
            self._statements("model.lm_head"),
        )

    def test_multimax_extras_keep_the_backbone_root(self):
        # Unmapped, so identity resolution keeps them under the backbone root --
        # where the checkpoint holds them -- and an identical name emits nothing.
        for head in ("model.lm_head", "model.shared_head"):
            self.assertFalse(
                any("multimax" in s for s in self._statements(head))
            )

    def test_the_mtp_head_is_absent_from_the_mapping(self):
        # The MTP head's checkpoint names already match its single names.
        self.assertEqual(self._statements("model.shared_mtp_lm_head"), [])


class _FakeGroup:
    """A pipeline group stand-in; the gather only reads ``nranks``."""

    def __init__(self, nranks):
        self.nranks = nranks


def _no_gather(*args, **kwargs):  # pragma: no cover - must not be hit
    raise AssertionError("gathered without a multi-stage pipeline group")


class TestGatherGuardedOnLivePipelineGroup(unittest.TestCase):
    """The live pipeline group alone decides whether a gather happens.

    A group cannot exist before distributed init, so its absence covers both a
    single-process run and a launched-but-not-initialized one -- in either case
    the rank-local set already is the whole answer. The one case that must not
    degrade silently, a declared pipeline with no group, raises instead.
    """

    def _group(self, group):
        return mock.patch.object(
            gen, "get_pipeline_model_parallel_group", return_value=group
        )

    def _local_only(self, group):
        local = ["hf.a -> model.a", "hf.b -> model.b"]
        with (
            self._group(group),
            mock.patch.object(
                paddle.distributed, "all_gather_object", _no_gather
            ),
        ):
            self.assertEqual(gen._globalize_statements(_Cfg(), local), local)

    def test_no_group_stays_rank_local(self):
        self._local_only(None)

    def test_single_stage_group_stays_rank_local(self):
        self._local_only(_FakeGroup(1))

    def test_gather_concatenates_in_rank_order(self):
        # Statements are order-sensitive, so the stages' lists are concatenated
        # in rank order rather than merged.
        def _gather(out, obj, group):
            out.extend([["hf.a -> model.a"], ["hf.b -> model.b"]])

        with (
            self._group(_FakeGroup(2)),
            mock.patch.object(paddle.distributed, "all_gather_object", _gather),
        ):
            self.assertEqual(
                gen._globalize_statements(_Cfg(), ["hf.b -> model.b"]),
                ["hf.a -> model.a", "hf.b -> model.b"],
            )

    def test_declared_pipeline_without_a_group_is_an_error(self):
        """A stage-local config is wrong, not a degradation.

        Under pipeline parallelism the live module tree is one stage, so a
        rank-local statement set misses the other stages' keys while both
        consumers need the globally complete one.
        """
        model = _two_leaf_model()
        cfg = _Cfg()
        cfg.pipeline_model_parallel_size = 2
        ctx = gen.build_aoa_context(model, cfg)
        for entry in (gen.gen_whole_model_aoa, gen.gen_whole_model_inv_aoa):
            with self._group(None), self.assertRaises(RuntimeError):
                entry(model, ctx)

    def test_single_stage_pipeline_needs_no_gather(self):
        model = _two_leaf_model()
        cfg = _Cfg()
        cfg.pipeline_model_parallel_size = 1
        ctx = gen.build_aoa_context(model, cfg)
        with (
            self._group(None),
            mock.patch.object(
                paddle.distributed, "all_gather_object", _no_gather
            ),
        ):
            self.assertEqual(
                gen.gen_whole_model_aoa(model, ctx)["aoa_statements"],
                [_EMBED, _HEAD],
            )


if __name__ == "__main__":
    unittest.main()
