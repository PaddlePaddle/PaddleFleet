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
# Scope: the whole-model modular AOA generator (whole-model entry + gate +
# tied-alias fan-out), exercised with lightweight fakes so no pipeline / fleet
# init is required.
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

from paddlefleet.models.gpt import aoa_generator as gen
from paddlefleet.models.gpt.aoa_generator import FleetAOAContext
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
        self.aoa_dtype_cast_rules = kwargs.get("aoa_dtype_cast_rules", {})
        self.model_type = kwargs.get("model_type", "")


class _FakeWholeModel:
    """Duck-typed whole-model boundary for the pure generator functions.

    Exposes exactly what ``aoa_generator`` reads: the child sub-layers, the
    pipeline->single mapping, and the raw pipeline state_dict whose tensor
    identities drive tied-alias / VPP grouping.

    The shared-layer policy table, its lookup hook, the leaf-name resolver and
    the model single-name root hook are borrowed from ``GPTModel`` verbatim
    rather than stubbed: the alias plan and the whole-model entries read all
    four, and a stub would put the test's own policy under test instead of the
    shipped one.
    """

    _AOA_SHARED_LAYER_COLLAPSES = GPTModel._AOA_SHARED_LAYER_COLLAPSES
    _aoa_shared_layer_collapses = GPTModel._aoa_shared_layer_collapses
    _model_name_prefix = GPTModel._model_name_prefix
    _resolve_leaf_names = GPTModel._resolve_leaf_names

    def __init__(self, sub_layers, pp_mapping, raw_state):
        self._sub_layers = sub_layers
        self._pp_to_single_mapping = pp_mapping
        self._pipeline_name_mapping = object()  # non-None: skip rebuild
        self._raw_state = raw_state
        # ``_model_name_prefix`` is the shipped hook and reads ``model_type``
        # off the config, so the fake must carry one.
        self.config = _Cfg()

    def _set_pipeline_name_mapping(self):  # pragma: no cover - never hit
        raise AssertionError("mapping already set")

    def _raw_structured_state_dict(self):
        return self._raw_state


_ENTRY_METHODS = (
    "gen_aoa_statements",
    "gen_inv_aoa_statements",
)


def _entry_model(*, config=None):
    """A fake boundary that borrows GPTModel's whole-model AOA entries verbatim.

    Avoids constructing a real pipeline model: the entries only touch what
    ``aoa_generator`` reads off the model. The modular path is therefore
    exercised for real, which is what keeps the component-dispatch contract
    under test.
    """
    model = _tied_model()
    model.aoa_towers = None
    model.config = _Cfg() if config is None else config
    for name in _ENTRY_METHODS:
        setattr(model, name, types.MethodType(getattr(GPTModel, name), model))
    return model


def _tied_model():
    """Embedding + output_layer sharing one tensor (pp==1 tie)."""
    emb = _Leaf()
    out = _Leaf()
    shared = emb.weight
    raw = {"embedding.weight": shared, "output_layer.weight": shared}
    pp_mapping = {
        "embedding.weight": "model.embed_tokens.weight",
        "output_layer.weight": "model.lm_head.weight",
    }
    model = _FakeWholeModel(
        {"embedding": emb, "output_layer": out}, pp_mapping, raw
    )
    return model


def _vpp_dup_model():
    """One tensor under two structured names mapping to the *same* single."""
    a = _Leaf()
    shared = a.weight
    raw = {"layer.0.weight": shared, "layer.1.weight": shared}
    pp_mapping = {
        "layer.0.weight": "model.layers.0.weight",
        "layer.1.weight": "model.layers.0.weight",
    }
    # Recursion walks the live module tree; give it a single real leaf so the
    # structured name "leaf.weight" is unrelated to the raw-dup keys, letting us
    # isolate the grouping/exclusion helpers.
    model = _FakeWholeModel({"leaf": a}, pp_mapping, raw)
    return model


class TestWholeModelAOAEntry(unittest.TestCase):
    """The two whole-model entries: dispatch, config default, and boundary."""

    def test_entry_runs_the_modular_generator(self):
        model = _entry_model()
        cfg = _Cfg()
        fwd = model.gen_aoa_statements(cfg)["aoa_statements"]
        inv = model.gen_inv_aoa_statements(cfg)["aoa_statements"]
        self.assertIn("hf.embed_tokens.weight -> model.lm_head.weight", fwd)
        self.assertIn("model.lm_head.weight -> _", inv)

    def test_config_falls_back_to_self_config(self):
        model = _entry_model(config=_Cfg())
        self.assertTrue(model.gen_aoa_statements()["aoa_statements"])

    def test_recursion_protocol_call_is_rejected(self):
        """A container must call the entry, not recurse through it.

        The entry takes ``(config=None)`` only, so it cannot absorb the base
        ``Layer`` protocol's keyword-only ``structured_name_prefix`` /
        ``aoa_name_scope``. Recursing into this boundary therefore raises
        instead of silently emitting statements that skip whole-model
        orchestration.
        """
        model = _entry_model()
        ctx = gen.build_aoa_context(_tied_model(), _Cfg())
        with self.assertRaises(TypeError):
            model.gen_aoa_statements(ctx, structured_name_prefix="tower.")
        with self.assertRaises(TypeError):
            model.gen_inv_aoa_statements(ctx, structured_name_prefix="tower.")


class TestAOAContextBuild(unittest.TestCase):
    def test_context_snapshots_config_and_mapping(self):
        model = _tied_model()
        cfg = _Cfg(aoa_checkpoint_name_prefix="hf")
        ctx = gen.build_aoa_context(model, cfg)
        self.assertEqual(ctx.checkpoint_name_prefix, "hf")
        self.assertEqual(ctx.pp_to_single_mapping, model._pp_to_single_mapping)
        self.assertIs(ctx.config, cfg)

    def test_context_defaults_are_resolved_not_none(self):
        class _Bare:
            pass

        ctx = gen.build_aoa_context(_tied_model(), _Bare())
        self.assertEqual(
            ctx.checkpoint_name_prefix, gen.DEFAULT_CHECKPOINT_NAME_PREFIX
        )
        # A config that declares nothing gets the ERNIE-series checkpoint
        # layout, not an empty mapping: the defaults describe the in-house
        # checkpoint and an external model overrides only what diverges.
        self.assertEqual(
            dict(ctx.checkpoint_name_mapping),
            gen.DEFAULT_CHECKPOINT_NAME_MAPPING,
        )
        self.assertEqual(ctx.dtype_cast_rules, {})
        # The MTP checkpoint-prefix protocol lives off AOAContext now; its
        # resolver yields same-numbering defaults for an unmigrated model.
        spec = gen.resolve_mtp_checkpoint_prefix_spec(_Bare())
        self.assertEqual(spec.checkpoint_prefix, "layers.$LAYER_ID")
        self.assertEqual(spec.transformer_checkpoint_prefix, "layers.$LAYER_ID")
        self.assertFalse(spec.is_absolute)
        # The skip set is pinned by the whole-model function, not the builder.
        self.assertEqual(ctx.excluded_names, frozenset())


class TestTiedAliasFanOut(unittest.TestCase):
    def setUp(self):
        self.model = _tied_model()
        self.ctx = gen.build_aoa_context(
            self.model, _Cfg(aoa_checkpoint_name_prefix="hf")
        )

    def test_canonical_is_embedding_not_lm_head(self):
        _excluded, groups = gen.collect_alias_plan(self.model, self.ctx)
        self.assertEqual(len(groups), 1)
        (group,) = groups
        self.assertEqual(group.canonical_single, "model.embed_tokens.weight")
        self.assertEqual(group.alias_singles, ("model.lm_head.weight",))
        # The canonical checkpoint name is resolved once, at plan time, so the
        # emitter never has to re-resolve a name it may not own.
        self.assertEqual(group.canonical_checkpoint, "hf.embed_tokens.weight")

    def test_alias_structured_name_is_excluded(self):
        excluded, _groups = gen.collect_alias_plan(self.model, self.ctx)
        self.assertEqual(excluded, frozenset({"output_layer.weight"}))

    def test_forward_fans_out_canonical_source_to_alias_key(self):
        out = gen.gen_whole_model_aoa(self.model, self.ctx)
        stmts = out["aoa_statements"]
        canonical = next(
            s for s in stmts if s.endswith(" -> model.embed_tokens.weight")
        )
        fanout = next(
            s for s in stmts if s.endswith(" -> model.lm_head.weight")
        )
        canonical_src = canonical.split(" -> ")[0]
        fanout_src = fanout.split(" -> ")[0]
        # Both the canonical target and the aliased target are fed from the same
        # single checkpoint source (forward = one source fans out).
        self.assertEqual(canonical_src, fanout_src)
        # The excluded structured name never produced a normal recursion line.
        self.assertEqual(
            sum(s.endswith(" -> model.lm_head.weight") for s in stmts), 1
        )

    def test_inverse_deletes_alias_keeps_canonical_producer(self):
        out = gen.gen_whole_model_inv_aoa(self.model, self.ctx)
        stmts = out["aoa_statements"]
        # Canonical producer writes the checkpoint; alias key is a DELETE.
        self.assertTrue(
            any(s.startswith("model.embed_tokens.weight -> ") for s in stmts)
        )
        self.assertIn("model.lm_head.weight -> _", stmts)
        # No real inverse statement writes the checkpoint from the alias key.
        self.assertFalse(
            any(
                s.startswith("model.lm_head.weight -> ")
                and not s.endswith(" -> _")
                for s in stmts
            )
        )

    def test_forward_and_inverse_generated_independently(self):
        fwd = gen.gen_whole_model_aoa(self.model, self.ctx)["aoa_statements"]
        inv = gen.gen_whole_model_inv_aoa(self.model, self.ctx)[
            "aoa_statements"
        ]
        # The inverse is not the textual reverse of forward (alias handling
        # differs by direction: fan-out vs delete).
        reversed_fwd = [" -> ".join(reversed(s.split(" -> "))) for s in fwd]
        self.assertNotEqual(sorted(inv), sorted(reversed_fwd))


class TestVppDuplicateDedup(unittest.TestCase):
    def test_pure_duplicate_is_excluded_without_alias_group(self):
        model = _vpp_dup_model()
        ctx = gen.build_aoa_context(
            model, _Cfg(aoa_checkpoint_name_prefix="hf")
        )
        # Same single-name under two structured names => dedup, not a fan-out.
        excluded, groups = gen.collect_alias_plan(model, ctx)
        self.assertEqual(groups, [])
        self.assertEqual(excluded, frozenset({"layer.1.weight"}))


class TestWholeModelKeyCoverage(unittest.TestCase):
    def test_forward_targets_cover_distinct_single_names(self):
        model = _tied_model()
        ctx = gen.build_aoa_context(
            model, _Cfg(aoa_checkpoint_name_prefix="hf")
        )
        targets = {
            s.split(" -> ")[1]
            for s in gen.gen_whole_model_aoa(model, ctx)["aoa_statements"]
        }
        # Every distinct model single-name is a forward target exactly once.
        self.assertEqual(
            targets,
            {"model.embed_tokens.weight", "model.lm_head.weight"},
        )


class TestMultiTowerSkeleton(unittest.TestCase):
    def test_aoa_towers_declared_but_unimplemented_raises(self):
        model = _tied_model()
        model.aoa_towers = ["vision", "language"]
        ctx = gen.build_aoa_context(model, _Cfg())
        with self.assertRaises(NotImplementedError):
            gen.gen_whole_model_aoa(model, ctx)
        with self.assertRaises(NotImplementedError):
            gen.gen_whole_model_inv_aoa(model, ctx)


def _multi_alias_model():
    """Two independent alias groups: (embed/lm_head) and (adapter_a/adapter_b)."""
    emb = _Leaf()
    out = _Leaf()
    adp_a = _Leaf()
    adp_b = _Leaf()
    shared_emb = emb.weight
    shared_adp = adp_a.weight
    raw = {
        "embedding.weight": shared_emb,
        "output_layer.weight": shared_emb,
        "adapter_a.weight": shared_adp,
        "adapter_b.weight": shared_adp,
    }
    pp_mapping = {
        "embedding.weight": "model.embed_tokens.weight",
        "output_layer.weight": "model.lm_head.weight",
        "adapter_a.weight": "model.adapter_a.weight",
        "adapter_b.weight": "model.adapter_b.weight",
    }
    model = _FakeWholeModel(
        {
            "embedding": emb,
            "output_layer": out,
            "adapter_a": adp_a,
            "adapter_b": adp_b,
        },
        pp_mapping,
        raw,
    )
    return model


class TestMultipleAliasGroups(unittest.TestCase):
    def test_two_independent_alias_groups_detected(self):
        model = _multi_alias_model()
        ctx = gen.build_aoa_context(
            model, _Cfg(aoa_checkpoint_name_prefix="hf")
        )
        _excluded, groups = gen.collect_alias_plan(model, ctx)
        self.assertEqual(len(groups), 2)
        canonicals = sorted(g.canonical_single for g in groups)
        self.assertIn("model.adapter_a.weight", canonicals)
        self.assertIn("model.embed_tokens.weight", canonicals)

    def test_forward_fans_out_both_groups(self):
        model = _multi_alias_model()
        ctx = gen.build_aoa_context(
            model, _Cfg(aoa_checkpoint_name_prefix="hf")
        )
        stmts = gen.gen_whole_model_aoa(model, ctx)["aoa_statements"]
        targets = {s.split(" -> ")[1] for s in stmts}
        self.assertIn("model.lm_head.weight", targets)
        self.assertIn("model.adapter_b.weight", targets)
        self.assertIn("model.embed_tokens.weight", targets)
        self.assertIn("model.adapter_a.weight", targets)

    def test_inverse_deletes_both_alias_sets(self):
        model = _multi_alias_model()
        ctx = gen.build_aoa_context(
            model, _Cfg(aoa_checkpoint_name_prefix="hf")
        )
        stmts = gen.gen_whole_model_inv_aoa(model, ctx)["aoa_statements"]
        self.assertIn("model.lm_head.weight -> _", stmts)
        self.assertIn("model.adapter_b.weight -> _", stmts)


class _OverridingLeaf(paddle.nn.Layer):
    """Stands in for a component-level override (Linear / SelfAttention...)."""

    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[2])
        self.seen_excluded = None

    def gen_aoa_statements(
        self, ctx, *, structured_name_prefix="", aoa_name_scope=None
    ):
        self.seen_excluded = ctx.excluded_names
        return [f"__component__:{structured_name_prefix}"]

    def gen_inv_aoa_statements(
        self, ctx, *, structured_name_prefix="", aoa_name_scope=None
    ):
        self.seen_excluded = ctx.excluded_names
        return [f"__component_inv__:{structured_name_prefix}"]


def _override_model():
    """A tied pair (non-empty ``excluded``) plus a component-override child."""
    emb = _Leaf()
    out = _Leaf()
    comp = _OverridingLeaf()
    shared = emb.weight
    raw = {
        "embedding.weight": shared,
        "output_layer.weight": shared,
        "component.weight": comp.weight,
    }
    pp_mapping = {
        "embedding.weight": "model.embed_tokens.weight",
        "output_layer.weight": "model.lm_head.weight",
        "component.weight": "model.component.weight",
    }
    model = _FakeWholeModel(
        {"embedding": emb, "output_layer": out, "component": comp},
        pp_mapping,
        raw,
    )
    return model, comp


class TestRecursionProtocol(unittest.TestCase):
    """The whole-model walk must go through the Layer protocol, not a fork.

    Regression guard: an earlier revision re-implemented the module walk inside
    the generator and recursed into itself, which silently bypassed every
    component override and dropped dtype casts.
    """

    def setUp(self):
        self.model, self.comp = _override_model()
        self.ctx = gen.build_aoa_context(
            self.model, _Cfg(aoa_checkpoint_name_prefix="hf")
        )

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

    def test_excluded_names_reach_the_component_override(self):
        gen.gen_whole_model_aoa(self.model, self.ctx)
        self.assertEqual(
            self.comp.seen_excluded, frozenset({"output_layer.weight"})
        )

    def test_builder_context_is_not_mutated(self):
        gen.gen_whole_model_aoa(self.model, self.ctx)
        # ctx is frozen: the skip set is pinned on a copy, not in place.
        self.assertEqual(self.ctx.excluded_names, frozenset())


class TestDtypeCastOnWholeModelPath(unittest.TestCase):
    """``ctx.dtype_cast_rules`` is consumed by the base Layer recursion."""

    def setUp(self):
        self.model = _tied_model()
        self.cfg = _Cfg(
            aoa_checkpoint_name_prefix="hf",
            aoa_dtype_cast_rules={
                "embed_tokens.weight": {
                    "checkpoint_dtype": "float32",
                    "model_dtype": "bfloat16",
                }
            },
        )
        self.ctx = gen.build_aoa_context(self.model, self.cfg)

    def test_forward_statement_carries_the_cast(self):
        stmts = gen.gen_whole_model_aoa(self.model, self.ctx)["aoa_statements"]
        self.assertIn(
            "hf.embed_tokens.weight -> model.embed_tokens.weight"
            ", src_dtype='float32', dst_dtype='bfloat16'",
            stmts,
        )

    def test_inverse_statement_swaps_the_endpoints(self):
        stmts = gen.gen_whole_model_inv_aoa(self.model, self.ctx)[
            "aoa_statements"
        ]
        self.assertIn(
            "model.embed_tokens.weight -> hf.embed_tokens.weight"
            ", src_dtype='bfloat16', dst_dtype='float32'",
            stmts,
        )


class TestAliasFanOutDtypeCast(unittest.TestCase):
    """The alias fan-out resolves each alias single's OWN forward
    dtype-cast rule, so a fanned-out tensor lands in the alias's declared dtype
    even when the canonical producer carries no cast.

    Before the fix, ``emit_alias_aoa`` emitted a bare
    ``{checkpoint} -> {alias}`` with no dtype suffix, silently dropping a cast
    declared on the alias key. The rule below is keyed on ``lm_head.weight``
    (the alias), NOT ``embed_tokens.weight`` (the canonical), which isolates the
    fan-out path from the base-recursion path already covered above.
    """

    def _cfg(self):
        return _Cfg(
            aoa_checkpoint_name_prefix="hf",
            aoa_dtype_cast_rules={
                "lm_head.weight": {
                    "checkpoint_dtype": "float32",
                    "model_dtype": "bfloat16",
                }
            },
        )

    def test_fanout_carries_alias_keyed_cast(self):
        model = _tied_model()
        ctx = gen.build_aoa_context(model, self._cfg())
        stmts = gen.gen_whole_model_aoa(model, ctx)["aoa_statements"]
        self.assertIn(
            "hf.embed_tokens.weight -> model.lm_head.weight"
            ", src_dtype='float32', dst_dtype='bfloat16'",
            stmts,
        )

    def test_canonical_producer_is_uncast_when_only_alias_declares_it(self):
        model = _tied_model()
        ctx = gen.build_aoa_context(model, self._cfg())
        stmts = gen.gen_whole_model_aoa(model, ctx)["aoa_statements"]
        # The canonical target has no rule of its own -> plain identity line.
        self.assertIn(
            "hf.embed_tokens.weight -> model.embed_tokens.weight", stmts
        )

    def test_emit_alias_aoa_unit_applies_per_alias_cast(self):
        # Direct unit check on the emitter with a hand-built group, independent
        # of the whole-model walk.
        ctx = FleetAOAContext(
            config=None,
            checkpoint_name_prefix="hf",
            pp_to_single_mapping={
                "embedding.weight": "model.embed_tokens.weight"
            },
            checkpoint_name_mapping={},
            dtype_cast_rules={
                "lm_head.weight": {
                    "checkpoint_dtype": "float32",
                    "model_dtype": "bfloat16",
                }
            },
            model_name_prefix="model",
            excluded_names=frozenset(),
        )
        group = gen.AliasGroup(
            "model.embed_tokens.weight",
            ("model.lm_head.weight",),
            "hf.embed_tokens.weight",
        )
        self.assertEqual(
            gen.emit_alias_aoa(ctx, [group]),
            [
                "hf.embed_tokens.weight -> model.lm_head.weight"
                ", src_dtype='float32', dst_dtype='bfloat16'"
            ],
        )

    def test_inverse_alias_delete_has_no_cast(self):
        # Inverse deletes the alias key (``-> _``); a DELETE never carries a
        # dtype cast, regardless of any rule on the alias single.
        model = _tied_model()
        ctx = gen.build_aoa_context(model, self._cfg())
        stmts = gen.gen_whole_model_inv_aoa(model, ctx)["aoa_statements"]
        self.assertIn("model.lm_head.weight -> _", stmts)
        self.assertFalse(
            any(
                s.startswith("model.lm_head.weight -> ") and "dtype" in s
                for s in stmts
            )
        )


def _shared_layer_model(structured, single, *, sub_layers=None):
    """One PP stage's view of a ``SharedLayerDesc`` group.

    The peer half is built by ANOTHER process, so this rank holds exactly one
    member and object identity relates it to nothing.
    """
    leaf = _Leaf()
    return _FakeWholeModel(
        {"leaf": leaf} if sub_layers is None else sub_layers,
        {structured: single},
        {structured: leaf.weight},
    )


def _plan_with_peer_rank(model, peer_resolved):
    """Runs :func:`collect_alias_plan` with the world faked to two ranks.

    ``peer_resolved`` is what the other rank contributes to the gather:
    ``{layer_name: [(single_name, checkpoint_name)]}``, already resolved on its
    owner, which is the only form allowed to cross a rank boundary.

    ``is_initialized`` is faked alongside ``get_world_size`` because the gather
    is guarded on both: a size >1 alone does not mean there is a group to gather
    over (see ``_shared_layer_members``).
    """
    ctx = gen.build_aoa_context(model, _Cfg(aoa_checkpoint_name_prefix="hf"))

    def _fake_all_gather_object(out, obj, *args, **kwargs):
        out.extend([obj, peer_resolved])

    with (
        mock.patch.object(paddle.distributed, "get_world_size", return_value=2),
        mock.patch.object(
            paddle.distributed, "is_initialized", return_value=True
        ),
        mock.patch.object(
            paddle.distributed, "all_gather_object", _fake_all_gather_object
        ),
    ):
        return gen.collect_alias_plan(model, ctx)


class TestSharedLayerTieAcrossPPStages(unittest.TestCase):
    """A collapsing tie whose halves live on different PP stages.

    Every declaring stage builds its OWN instance in its OWN process
    (``pp_layers.py`` walks only its slice of ``_layers_desc``), so the two
    halves are unrelated objects and no ``id()`` groups them. The plan keys on
    ``SharedLayerDesc.layer_name`` -- globally identical, since it is what
    Paddle itself shares on -- and all-gathers resolved names.
    """

    EMBED_SIDE = ("model.embed_tokens.weight", "hf.embed_tokens.weight")
    HEAD_SIDE = ("model.lm_head.weight", "hf.lm_head.weight")

    def test_embedding_stage_keeps_producer_and_gains_the_fanout(self):
        model = _shared_layer_model(
            "shared_layers.embed.weight", self.EMBED_SIDE[0]
        )
        excluded, groups = _plan_with_peer_rank(
            model, {"embed": [self.HEAD_SIDE]}
        )
        self.assertEqual(excluded, frozenset())
        (group,) = groups
        self.assertEqual(group.canonical_single, self.EMBED_SIDE[0])
        self.assertEqual(group.alias_singles, (self.HEAD_SIDE[0],))
        self.assertEqual(group.canonical_checkpoint, self.EMBED_SIDE[1])

    def test_head_stage_drops_its_producer_and_sources_the_peers_key(self):
        # The regression this guards: without the gather this rank saw a lone
        # structured name, emitted a plain statement, and sourced a head
        # checkpoint key that a tied checkpoint does not contain.
        model = _shared_layer_model(
            "shared_layers.embed.weight", self.HEAD_SIDE[0]
        )
        excluded, groups = _plan_with_peer_rank(
            model, {"embed": [self.EMBED_SIDE]}
        )
        self.assertEqual(excluded, frozenset({"shared_layers.embed.weight"}))
        (group,) = groups
        # Canonical selection is rank-invariant, so both stages emit the same
        # fan-out and _globalize_statements dedups it.
        self.assertEqual(group.canonical_single, self.EMBED_SIDE[0])
        self.assertEqual(group.canonical_checkpoint, self.EMBED_SIDE[1])
        self.assertEqual(group.alias_singles, (self.HEAD_SIDE[0],))

    def test_canonical_is_the_embedding_even_under_the_shared_prefix(self):
        # pp>1 registers the head under ``shared_head``, not ``lm_head``; the
        # alias-segment table must cover that spelling too.
        model = _shared_layer_model(
            "shared_layers.embed.weight", "model.shared_head.weight"
        )
        _excluded, groups = _plan_with_peer_rank(
            model, {"embed": [self.EMBED_SIDE]}
        )
        (group,) = groups
        self.assertEqual(group.canonical_single, self.EMBED_SIDE[0])
        self.assertEqual(group.alias_singles, ("model.shared_head.weight",))


class TestNonCollapsingSharedLayer(unittest.TestCase):
    """``mtp_reuse_transformer``: one tensor, but one checkpoint key per member.

    Indistinguishable from the tie above model-side, which is why the policy is
    declared rather than inferred.
    """

    def test_every_member_stays_an_independent_producer(self):
        model = _shared_layer_model(
            "shared_layers.mtp_reuse_transformer.weight",
            "model.layers.12.weight",
        )
        excluded, groups = _plan_with_peer_rank(
            model,
            {
                "mtp_reuse_transformer": [
                    ("model.mtp.transformer_layer.weight", "hf.mtp.weight")
                ]
            },
        )
        self.assertEqual(groups, [])
        self.assertEqual(excluded, frozenset())


def _mixed_shared_and_plain_model():
    """One identity group mixing a plain path with a ``shared_layers.*`` name.

    Reachable with ``tie_word_embeddings=True``, ``pp == 1`` and
    ``spec.mtp_lm_head`` set: ``get_layer_desc_list`` gates its embedding tie
    branch on pp>1 so the embedding stays a plain ``LayerDesc``, while the head
    still becomes ``SharedLayerDesc("embed")``, and
    ``skip_weight_param_allocation`` ties the two by object.
    """
    embedding = _Leaf()
    head = _Leaf()
    shared = embedding.weight
    return _FakeWholeModel(
        {"embedding": embedding, "shared_head": head},
        {
            "embedding.weight": "model.embed_tokens.weight",
            "shared_layers.embed.weight": "model.lm_head.weight",
        },
        {
            "embedding.weight": shared,
            "shared_layers.embed.weight": shared,
        },
    )


class TestMixedSharedAndPlainGroup(unittest.TestCase):
    def test_planned_as_one_unit_under_the_shared_layer_name(self):
        model = _mixed_shared_and_plain_model()
        ctx = gen.build_aoa_context(
            model, _Cfg(aoa_checkpoint_name_prefix="hf")
        )
        excluded, groups = gen.collect_alias_plan(model, ctx)
        # One unit, not two: were the plain name left to form its own identity
        # group it would escape exclusion and a second producer would write the
        # same checkpoint key.
        self.assertEqual(len(groups), 1)
        (group,) = groups
        self.assertEqual(group.canonical_single, "model.embed_tokens.weight")
        self.assertEqual(group.alias_singles, ("model.lm_head.weight",))
        self.assertEqual(excluded, frozenset({"shared_layers.embed.weight"}))


class TestSharedLayerPolicyLookup(unittest.TestCase):
    def test_unregistered_layer_name_raises(self):
        model = _shared_layer_model(
            "shared_layers.mystery.weight", "model.mystery.weight"
        )
        ctx = gen.build_aoa_context(model, _Cfg())
        with self.assertRaisesRegex(ValueError, "unregistered SharedLayerDesc"):
            gen.collect_alias_plan(model, ctx)

    def test_one_tensor_under_two_shared_layer_names_raises(self):
        leaf = _Leaf()
        shared = leaf.weight
        model = _FakeWholeModel(
            {"leaf": leaf},
            {
                "shared_layers.embed.weight": "model.embed_tokens.weight",
                "shared_layers.mtp_embed.weight": "model.mtp_embedding.weight",
            },
            {
                "shared_layers.embed.weight": shared,
                "shared_layers.mtp_embed.weight": shared,
            },
        )
        ctx = gen.build_aoa_context(model, _Cfg())
        with self.assertRaisesRegex(ValueError, "several SharedLayerDesc"):
            gen.collect_alias_plan(model, ctx)


class TestGatherGuardedOnLiveProcessGroup(unittest.TestCase):
    """``get_world_size() > 1`` does not imply there is a group to gather over.

    With no group initialized it falls back to ``PADDLE_TRAINERS_NUM``, so a
    launched-but-not-initialized process (every single-process run on a cluster
    node, including this test suite) reports the launcher's rank count while
    ``all_gather_object`` would raise. Both gather sites must stay rank-local
    instead of crashing, and must say so.
    """

    def _no_group(self, world_size):
        def _explode(*args, **kwargs):  # pragma: no cover - must not be hit
            raise AssertionError("gathered without an initialized group")

        return (
            mock.patch.object(
                paddle.distributed, "get_world_size", return_value=world_size
            ),
            mock.patch.object(
                paddle.distributed, "is_initialized", return_value=False
            ),
            mock.patch.object(
                paddle.distributed, "all_gather_object", _explode
            ),
        )

    def test_alias_plan_stays_rank_local(self):
        model = _tied_model()
        ctx = gen.build_aoa_context(model, _Cfg())
        size, init, gather = self._no_group(4)
        with (
            size,
            init,
            gather,
            self.assertLogs(gen.__name__, level="WARNING") as logs,
        ):
            excluded, groups = gen.collect_alias_plan(model, ctx)
        self.assertTrue(any("rank-local" in line for line in logs.output))
        # The pp==1 tie is still planned from this rank's own registrations.
        self.assertEqual(len(groups), 1)
        self.assertTrue(excluded)

    def test_globalize_statements_stays_rank_local(self):
        local = ["hf.a -> model.a", "hf.b -> model.b"]
        size, init, gather = self._no_group(4)
        with (
            size,
            init,
            gather,
            self.assertLogs(gen.__name__, level="WARNING"),
        ):
            self.assertEqual(gen._globalize_statements(local), local)

    def test_single_process_world_is_silent(self):
        model = _tied_model()
        ctx = gen.build_aoa_context(model, _Cfg())
        with (
            mock.patch.object(
                paddle.distributed, "get_world_size", return_value=1
            ),
            mock.patch.object(
                paddle.distributed, "is_initialized", return_value=False
            ),
            mock.patch.object(gen.logger, "warning") as warn,
        ):
            gen.collect_alias_plan(model, ctx)
            gen._globalize_statements(["hf.a -> model.a"])
        warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
