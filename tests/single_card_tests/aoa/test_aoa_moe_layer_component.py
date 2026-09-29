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
# Scope: ``MoELayer`` owns no parameters of its own; it composes gate, optional
# latent projection, optional shared experts (MLP), and the routed experts. The
# routed-expert emission is dispatched on the *live* structure:
#   * no ``grouped_gemm_experts`` -> per-expert recursion, enumerated by global
#     expert id through a local template expert (see ``_global_expert_ctx``) so
#     every EP rank emits the same statement set;
#   * ``grouped_gemm_experts is not None`` -> the grouped-GEMM helper: per-expert
#     fuse (``^T`` / ``axis=1``) into transients, then concat on ``axis=0`` into
#     the two packed model weights.
# The grouped emission does NOT depend on the expert class -- SonicMoEExpert
# saves through the grouped layout -- which is pinned here too. Orthogonally, the
# grouped branch reads the *model-side* layout off the live expert
# (``intermediate_ep_sharded``: 'allgather'/'ringmoe' dispatcher with EP > 1
# gives every rank all experts but only ``I // EP`` of each), adding an
# un-interleaving stage on both directions. This test pins the composition
# contract, the per-expert recursion, the grouped per-expert concat, the
# expert-type independence, and the intermediate-sharded un-interleaving.
import os
import sys
import types

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

import paddle
from paddle.distributed.flex_checkpoint.aoa.generation import AOAContext

_PFX = "model.layers.0.mlp."
_HF = "hf.layers.0.mlp."
_MD = "model.layers.0.mlp."


def _ctx(pp_to_single_mapping=None):
    return AOAContext(
        config=None,
        checkpoint_name_prefix="hf",
        pp_to_single_mapping=pp_to_single_mapping or {},
        checkpoint_name_mapping={},
        model_name_prefix="model",
    )


def _identity_ctx(model):
    """A complete identity ``pp_to_single_mapping`` for a live stand-in.

    ``resolve_single_name`` treats a non-empty mapping as authoritative and
    raises on any missing key, so every real param the top-level generator
    touches (gate, every expert, shared experts) must be present. Build it from
    the model's own ``state_dict`` prefixed with the layer scope; the per-expert
    ``_global_expert_ctx`` then reads expert 0 as its template and leaves the
    rest untouched (already identity here).
    """
    return _ctx({f"{_PFX}{k}": f"{_PFX}{k}" for k in model.state_dict().keys()})


def _make_linear_leaf(bias=False):
    from paddlefleet.tensor_parallel.layers import Linear

    class _LinearLeaf(paddle.nn.Layer):
        gen_aoa_statements = Linear.gen_aoa_statements
        gen_inv_aoa_statements = Linear.gen_inv_aoa_statements

        def __init__(self):
            super().__init__()
            self.weight = self.create_parameter(shape=[2, 2])
            self.bias = self.create_parameter(shape=[2]) if bias else None

    return _LinearLeaf


def _make_mlp_leaf():
    from paddlefleet.transformer.mlp import MLP

    up_gate_cls = _make_linear_leaf(bias=False)
    down_cls = _make_linear_leaf(bias=False)

    class _MLPLeaf(paddle.nn.Layer):
        gen_aoa_statements = MLP.gen_aoa_statements
        gen_inv_aoa_statements = MLP.gen_inv_aoa_statements
        _gen_up_gate_fusion_aoa_statements = (
            MLP._gen_up_gate_fusion_aoa_statements
        )
        _gen_inv_up_gate_fusion_aoa_statements = (
            MLP._gen_inv_up_gate_fusion_aoa_statements
        )
        _gen_up_gate_bias_aoa_statements = MLP._gen_up_gate_bias_aoa_statements
        _gen_inv_up_gate_bias_aoa_statements = (
            MLP._gen_inv_up_gate_bias_aoa_statements
        )
        _gen_up_gate_bias_fusion_aoa_statements = (
            MLP._gen_up_gate_bias_fusion_aoa_statements
        )
        _gen_inv_up_gate_bias_fusion_aoa_statements = (
            MLP._gen_inv_up_gate_bias_fusion_aoa_statements
        )

        def __init__(self):
            super().__init__()
            self.up_gate_proj = up_gate_cls()
            self.down_proj = down_cls()

    return _MLPLeaf


def _make_grouped_marker(
    dispatcher="deepep", expert_parallel=False, ep_size=None
):
    """Stand-in for the live ``grouped_gemm_experts``.

    The generators read the *model-side* weight layout off the live expert
    (``intermediate_ep_sharded``, then ``ep_group.nranks``), the same source
    ``sharded_state_dict`` dispatches on, so the real property is borrowed here
    and fed its two real inputs (``config.moe_token_dispatcher_type`` and
    ``expert_parallel``) instead of a hard-coded answer.
    """
    from paddlefleet.transformer.moe.moe_expert import GroupedMLPExpert

    class _GroupedMarker:
        intermediate_ep_sharded = GroupedMLPExpert.intermediate_ep_sharded

        def __init__(self):
            self.config = types.SimpleNamespace(
                moe_token_dispatcher_type=dispatcher
            )
            self.expert_parallel = expert_parallel
            self.ep_group = (
                None
                if ep_size is None
                else types.SimpleNamespace(nranks=ep_size)
            )

    return _GroupedMarker


def _make_moe_leaf(
    num_experts=2, shared=False, grouped=False, grouped_marker_factory=None
):
    """Binds the MoELayer AOA overrides onto a controllable stand-in.

    ``gate`` is a plain leaf (router AOA is out of scope here); ``experts`` is a
    ``LayerList`` of MLP leaves; ``shared_experts`` is an optional MLP leaf. When
    ``grouped`` is set a ``grouped_gemm_experts`` stand-in forces the grouped
    branch. The generators consult the marker for the *model-side* layout only
    (``intermediate_ep_sharded`` / ``ep_group.nranks``), never for the checkpoint
    layout, so the default stand-in reports the ordinary (deepep) layout.
    """
    from paddlefleet.transformer.moe.moe_layer import MoELayer

    mlp_cls = _make_mlp_leaf()

    class _GateLeaf(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.weight = self.create_parameter(shape=[2, 2])

    class _MoELeaf(paddle.nn.Layer):
        gen_aoa_statements = MoELayer.gen_aoa_statements
        gen_inv_aoa_statements = MoELayer.gen_inv_aoa_statements
        _global_expert_ctx = MoELayer._global_expert_ctx
        _gen_grouped_expert_aoa_statements = (
            MoELayer._gen_grouped_expert_aoa_statements
        )
        _gen_inv_grouped_expert_aoa_statements = (
            MoELayer._gen_inv_grouped_expert_aoa_statements
        )
        _intermediate_ep_size = MoELayer._intermediate_ep_size
        _reject_nongated_grouped_experts = (
            MoELayer._reject_nongated_grouped_experts
        )
        _inv_intermediate_ep_unpack_statements = (
            MoELayer._inv_intermediate_ep_unpack_statements
        )

        def __init__(self):
            super().__init__()
            self.num_experts = num_experts
            # ``_reject_nongated_grouped_experts`` reads
            # ``self.config.gated_linear_unit``; the grouped path only supports
            # the gated layout.
            self.config = types.SimpleNamespace(gated_linear_unit=True)
            self.gate = _GateLeaf()
            self.experts = paddle.nn.LayerList(
                [mlp_cls() for _ in range(num_experts)]
            )
            if shared:
                self.shared_experts = mlp_cls()
            if grouped:
                factory = (
                    _make_grouped_marker()
                    if grouped_marker_factory is None
                    else grouped_marker_factory
                )
                self.grouped_gemm_experts = factory()

    return _MoELeaf


class TestMoELayerClassContract(unittest.TestCase):
    def test_override_present(self):
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        for name in (
            "gen_aoa_statements",
            "gen_inv_aoa_statements",
            "_global_expert_ctx",
            "_gen_grouped_expert_aoa_statements",
            "_gen_inv_grouped_expert_aoa_statements",
            "_intermediate_ep_size",
            "_reject_nongated_grouped_experts",
            "_inv_intermediate_ep_unpack_statements",
        ):
            self.assertIn(name, MoELayer.__dict__)


class TestMoELayerPerExpertForward(unittest.TestCase):
    def test_gate_then_per_expert_composition(self):
        model = _make_moe_leaf(num_experts=2, shared=False)()
        stmts = model.gen_aoa_statements(
            _identity_ctx(model), structured_name_prefix=_PFX
        )
        # each expert recursed as an MLP: fused_ffn + down_proj delegation
        for eid in (0, 1):
            self.assertIn(
                f"{_HF}experts.{eid}.gate_proj.weight^T, "
                f"{_HF}experts.{eid}.up_proj.weight^T "
                f"-> {_MD}experts.{eid}.up_gate_proj.weight, fused_ffn",
                stmts,
            )
            self.assertIn(
                f"{_HF}experts.{eid}.down_proj.weight^T "
                f"-> {_MD}experts.{eid}.down_proj.weight",
                stmts,
            )

    def test_shared_experts_included(self):
        model = _make_moe_leaf(num_experts=1, shared=True)()
        stmts = model.gen_aoa_statements(
            _identity_ctx(model), structured_name_prefix=_PFX
        )
        self.assertIn(
            f"{_HF}shared_experts.gate_proj.weight^T, "
            f"{_HF}shared_experts.up_proj.weight^T "
            f"-> {_MD}shared_experts.up_gate_proj.weight, fused_ffn",
            stmts,
        )


class TestMoELayerPerExpertInverse(unittest.TestCase):
    def test_gate_then_per_expert_inverse(self):
        model = _make_moe_leaf(num_experts=2, shared=False)()
        stmts = model.gen_inv_aoa_statements(
            _identity_ctx(model), structured_name_prefix=_PFX
        )
        for eid in (0, 1):
            # fused_ffn splits into model-side halves, each then transposed
            # into its checkpoint name.
            self.assertIn(
                f"{_MD}experts.{eid}.up_gate_proj.weight "
                f"-> {_MD}experts.{eid}.gate_proj.weight, "
                f"{_MD}experts.{eid}.up_proj.weight, fused_ffn",
                stmts,
            )
            for leaf in ("gate_proj", "up_proj", "down_proj"):
                self.assertIn(
                    f"{_MD}experts.{eid}.{leaf}.weight^T "
                    f"-> {_HF}experts.{eid}.{leaf}.weight",
                    stmts,
                )


class TestMoELayerGroupedPerExpertConcat(unittest.TestCase):
    """Grouped-GEMM live layer (ordinary deepep layout): per-expert fuse
    (``^T`` / ``axis=1``) into transients, then concat on ``axis=0`` into the
    two packed model weights (GroupedMLPExpert convention)."""

    def test_forward_concat(self):
        model = _make_moe_leaf(num_experts=2, grouped=True)()
        stmts = model._gen_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        for eid in (0, 1):
            self.assertIn(
                f"{_HF}experts.{eid}.gate_proj.weight^T, "
                f"{_HF}experts.{eid}.up_proj.weight^T "
                f"-> {_MD}experts.{eid}.up_gate_proj.weight, axis=1",
                stmts,
            )
            self.assertIn(
                f"{_HF}experts.{eid}.down_proj.weight^T "
                f"-> {_MD}experts.{eid}.down_proj.weight",
                stmts,
            )
        self.assertIn(
            f"{_MD}experts.0.up_gate_proj.weight,"
            f"{_MD}experts.1.up_gate_proj.weight "
            f"-> {_MD}grouped_gemm_experts.weight1, axis=0",
            stmts,
        )
        self.assertIn(
            f"{_MD}experts.0.down_proj.weight,"
            f"{_MD}experts.1.down_proj.weight "
            f"-> {_MD}grouped_gemm_experts.weight2, axis=0",
            stmts,
        )
        # 2 experts * 2 fuse/stage + 2 concat = 6
        self.assertEqual(len(stmts), 6)

    def test_inverse_split_then_defuse(self):
        model = _make_moe_leaf(num_experts=2, grouped=True)()
        stmts = model._gen_inv_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        # stage 1: split the two packed weights on the expert axis
        self.assertIn(
            f"{_MD}grouped_gemm_experts.weight1 "
            f"-> {_MD}experts.0.up_gate_proj.weight,"
            f"{_MD}experts.1.up_gate_proj.weight, axis=0",
            stmts,
        )
        self.assertIn(
            f"{_MD}grouped_gemm_experts.weight2 "
            f"-> {_MD}experts.0.down_proj.weight.intermediate,"
            f"{_MD}experts.1.down_proj.weight.intermediate, axis=0",
            stmts,
        )
        # stage 2: per-expert de-fuse on axis=1 into transposition
        # intermediates, then transpose each into its checkpoint name.
        for eid in (0, 1):
            self.assertIn(
                f"{_MD}experts.{eid}.up_gate_proj.weight "
                f"-> {_MD}experts.{eid}.gate_proj.weight.intermediate, "
                f"{_MD}experts.{eid}.up_proj.weight.intermediate, axis=1",
                stmts,
            )
            for leaf in ("gate_proj", "up_proj", "down_proj"):
                self.assertIn(
                    f"{_MD}experts.{eid}.{leaf}.weight.intermediate^T "
                    f"-> {_HF}experts.{eid}.{leaf}.weight",
                    stmts,
                )
        # 2 split + 2 experts * (1 de-fuse + 3 transposes) = 10
        self.assertEqual(len(stmts), 10)


class TestGroupedExpertLayoutIsExpertTypeAgnostic(unittest.TestCase):
    """The checkpoint-facing expert layout is the grouped-GEMM one regardless of
    the expert class. ``SonicMoEExpert`` keeps a different in-memory layout while
    computing but its ``sharded_state_dict()`` converts back to the grouped
    layout first, so branching the generators on the expert type would describe
    a layout that is never saved. These pin the type-independence."""

    @staticmethod
    def _sonic_marker():
        from paddlefleet.transformer.moe.moe_expert import SonicMoEExpert

        marker = SonicMoEExpert.__new__(SonicMoEExpert)
        marker.config = types.SimpleNamespace(
            moe_token_dispatcher_type="deepep"
        )
        marker.expert_parallel = False
        return marker

    def test_intermediate_ep_sharded_is_inherited(self):
        from paddlefleet.transformer.moe.moe_expert import (
            GroupedMLPExpert,
            SonicMoEExpert,
        )

        self.assertIs(
            SonicMoEExpert.intermediate_ep_sharded,
            GroupedMLPExpert.intermediate_ep_sharded,
        )

    def _pair(self):
        plain = _make_moe_leaf(num_experts=2, grouped=True)()
        sonic = _make_moe_leaf(
            num_experts=2,
            grouped=True,
            grouped_marker_factory=self._sonic_marker,
        )()
        return plain, sonic

    def test_forward_statements_match(self):
        plain, sonic = self._pair()
        self.assertEqual(
            sonic._gen_grouped_expert_aoa_statements(_ctx(), _PFX, None),
            plain._gen_grouped_expert_aoa_statements(_ctx(), _PFX, None),
        )

    def test_inverse_statements_match(self):
        plain, sonic = self._pair()
        self.assertEqual(
            sonic._gen_inv_grouped_expert_aoa_statements(_ctx(), _PFX, None),
            plain._gen_inv_grouped_expert_aoa_statements(_ctx(), _PFX, None),
        )


class TestGroupedExpertIntermediateEPSharded(unittest.TestCase):
    """Grouped-GEMM live layer under the 'allgather' dispatcher with EP > 1:
    every rank holds all ``E`` experts but only ``I // EP`` of each one's
    intermediate dim, so both directions gain an un-interleaving stage against
    the rank-major re-declared global layout."""

    EP = 2
    E = 2
    W1 = f"{_MD}grouped_gemm_experts.weight1"
    W2 = f"{_MD}grouped_gemm_experts.weight2"

    def _model(self, dispatcher="allgather", expert_parallel=True, ep_size=EP):
        return _make_moe_leaf(
            num_experts=self.E,
            grouped=True,
            grouped_marker_factory=_make_grouped_marker(
                dispatcher=dispatcher,
                expert_parallel=expert_parallel,
                ep_size=ep_size,
            ),
        )()

    def test_ep_size_read_off_the_live_expert(self):
        self.assertEqual(self._model()._intermediate_ep_size(), self.EP)
        # Both predicate inputs must hold, and neither implies the other.
        self.assertIsNone(
            self._model(expert_parallel=False)._intermediate_ep_size()
        )
        self.assertIsNone(
            self._model(dispatcher="deepep")._intermediate_ep_size()
        )

    def test_forward_slices_then_reinterleaves(self):
        model = self._model()
        stmts = model._gen_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        for e in range(self.E):
            ug = f"{_MD}experts.{e}.up_gate_proj.weight"
            dm = f"{_MD}experts.{e}.down_proj.weight"
            self.assertIn(
                f"{_HF}experts.{e}.gate_proj.weight^T "
                f"-> {ug}.gate_ep0,{ug}.gate_ep1, axis=1",
                stmts,
            )
            self.assertIn(
                f"{ug}.gate_ep0,{ug}.up_ep0,{ug}.gate_ep1,{ug}.up_ep1 "
                f"-> {ug}, axis=1",
                stmts,
            )
            self.assertIn(
                f"{_HF}experts.{e}.down_proj.weight^T "
                f"-> {dm}.ep0,{dm}.ep1, axis=0",
                stmts,
            )
        # weight2 sources are walked rank-outer, expert-inner.
        self.assertIn(
            f"{_MD}experts.0.down_proj.weight.ep0,"
            f"{_MD}experts.1.down_proj.weight.ep0,"
            f"{_MD}experts.0.down_proj.weight.ep1,"
            f"{_MD}experts.1.down_proj.weight.ep1 -> {self.W2}, axis=0",
            stmts,
        )
        # E * 4 slice/interleave + 2 concat = 10
        self.assertEqual(len(stmts), 10)

    def test_inverse_unpacks_into_the_ordinary_transients(self):
        model = self._model()
        stmts = model._gen_inv_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        w1 = self.W1
        # stage (1) un-interleaves weight1's rank-major columns back into the
        # full gate/up, then cuts expert-wise into the ordinary transients.
        self.assertIn(
            f"{w1} -> {w1}.ep_blk0,{w1}.ep_blk1,{w1}.ep_blk2,{w1}.ep_blk3, "
            f"axis=1",
            stmts,
        )
        self.assertIn(
            f"{w1}.ep_blk0,{w1}.ep_blk2 -> {w1}.gate_all, axis=1", stmts
        )
        # stage (2) is shared verbatim with the ordinary layout: compare tails.
        ordinary = _make_moe_leaf(
            num_experts=self.E, grouped=True
        )()._gen_inv_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        tail = 4 * self.E
        self.assertEqual(stmts[-tail:], ordinary[-tail:])
        # stage1 (10) + stage2 (8) = 18
        self.assertEqual(len(stmts), 18)

    def test_round_trip_transient_names_are_consistent(self):
        model = self._model()
        fwd = model._gen_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        inv = model._gen_inv_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        for e in range(self.E):
            ug = f"{_MD}experts.{e}.up_gate_proj.weight"
            self.assertTrue(any(s.endswith(f"-> {ug}, axis=1") for s in fwd))
            self.assertTrue(any(s.endswith(f"-> {ug}, axis=1") for s in inv))

    def test_ordinary_layout_statements_unchanged(self):
        marker_default = _make_moe_leaf(num_experts=self.E, grouped=True)()
        explicit_deepep = self._model(dispatcher="deepep")
        for gen in (
            "_gen_grouped_expert_aoa_statements",
            "_gen_inv_grouped_expert_aoa_statements",
        ):
            self.assertEqual(
                getattr(explicit_deepep, gen)(_ctx(), _PFX, None),
                getattr(marker_default, gen)(_ctx(), _PFX, None),
            )


class TestGroupedExpertRejectsNonGated(unittest.TestCase):
    """The grouped path only supports the gated expert layout (``weight1`` last
    dim = ``2 * intermediate``); a non-gated MoE must fail loudly rather than
    emit statements that reference a nonexistent gate half."""

    def test_forward_and_inverse_reject(self):
        model = _make_moe_leaf(num_experts=2, grouped=True)()
        model.config.gated_linear_unit = False
        with self.assertRaises(NotImplementedError):
            model._gen_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        with self.assertRaises(NotImplementedError):
            model._gen_inv_grouped_expert_aoa_statements(_ctx(), _PFX, None)


if __name__ == "__main__":
    unittest.main()
