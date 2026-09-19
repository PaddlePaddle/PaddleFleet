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
# Scope: ``MoELayer`` owns no parameters of its own; it composes gate
# (TopKRouter), optional latent projection, optional shared experts (MLP), and
# the routed experts. The routed-expert emission is dispatched on the *live*
# structure:
#   * no ``grouped_gemm_experts`` -> per-expert recursion (each expert is its
#     own MLP with an ``experts.{id}.`` prefix);
#   * ``grouped_gemm_experts is not None`` -> the grouped-GEMM helper, which
#     itself has two checkpoint-layout sub-paths selected by
#     ``ctx.moe_expert_checkpoint_layout``:
#       - ``"per_expert"`` (default): per-expert fuse+concat into the two packed
#         model weights;
#       - ``"packed"``: the checkpoint already packs every expert into two 3D
#         tensors, so the only transform is a per-tensor ``permute='[0,2,1]'``.
# The ``"packed"`` flag is consulted ONLY on the grouped branch; a per-expert
# live layer ignores it, so a single model may mix layouts across layers under
# one model-level flag (Qwen3.5: grouped+packed backbone, per-expert MTP). The
# grouped emission does NOT depend on the expert class -- SonicMoEExpert saves
# through the grouped layout -- which is pinned here too. Orthogonally, the
# grouped branch also reads the *model-side* layout off the live expert
# (``intermediate_ep_sharded``: 'allgather' dispatcher with EP > 1 gives every
# rank all experts but only ``I // EP`` of each), which adds an un-interleaving
# stage on both directions. This test pins the composition contract, both packed
# directions, the grouped per-expert concat, the expert-type independence, the
# per-expert-branch flag-independence, the mixed coexistence, and the
# intermediate-sharded un-interleaving plus its rejection of the packed
# combination.
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

from paddlefleet.models.gpt.aoa_generator import (
    FleetAOAContext,
)


def _ctx(moe_expert_checkpoint_layout="per_expert"):
    return FleetAOAContext(
        config=None,
        checkpoint_name_prefix="hf",
        pp_to_single_mapping={},
        checkpoint_name_mapping={},
        dtype_cast_rules={},
        model_name_prefix="model",
        excluded_names=frozenset(),
        moe_expert_checkpoint_layout=moe_expert_checkpoint_layout,
    )


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
    ``sharded_state_dict`` dispatches on, so a marker that merely hard-codes
    the answer would pin nothing. The real property is borrowed here and fed
    its two real inputs instead: ``config.moe_token_dispatcher_type`` and
    ``expert_parallel``. Everything else a real expert owns (parameters,
    device state, a full TransformerConfig) is irrelevant to plan generation.
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


def _make_router_leaf():
    from paddlefleet.transformer.moe.moe_router import TopKRouter

    class _RouterLeaf(paddle.nn.Layer):
        gen_aoa_statements = TopKRouter.gen_aoa_statements
        gen_inv_aoa_statements = TopKRouter.gen_inv_aoa_statements

        def __init__(self):
            super().__init__()
            self.weight = self.create_parameter(shape=[2, 2])

    return _RouterLeaf


def _make_moe_leaf(
    num_experts=2, shared=False, grouped=False, grouped_marker_factory=None
):
    """Binds the MoELayer AOA overrides onto a controllable stand-in.

    ``gate`` is a router leaf; ``experts`` is a ``LayerList`` of MLP leaves.
    ``shared_experts`` is an optional MLP leaf. When ``grouped`` is set a
    ``grouped_gemm_experts`` stand-in forces the grouped-fusion branch;
    ``num_experts`` is exposed for the per-expert concat sub-path. The
    generators consult the marker for the *model-side* layout only
    (``intermediate_ep_sharded`` / ``ep_group.nranks``) and never for the
    checkpoint layout -- the grouped emission is one layout for every expert
    type -- so the default stand-in reports the ordinary (deepep) layout.
    ``grouped_marker_factory`` overrides it to pin the expert-type
    independence or to select the intermediate-sharded layout.
    """
    from paddlefleet.transformer.moe.moe_layer import MoELayer

    router_cls = _make_router_leaf()
    mlp_cls = _make_mlp_leaf()

    class _MoELeaf(paddle.nn.Layer):
        gen_aoa_statements = MoELayer.gen_aoa_statements
        gen_inv_aoa_statements = MoELayer.gen_inv_aoa_statements
        _gen_grouped_expert_aoa_statements = (
            MoELayer._gen_grouped_expert_aoa_statements
        )
        _gen_inv_grouped_expert_aoa_statements = (
            MoELayer._gen_inv_grouped_expert_aoa_statements
        )
        _gen_packed_grouped_expert_aoa_statements = (
            MoELayer._gen_packed_grouped_expert_aoa_statements
        )
        _gen_inv_packed_grouped_expert_aoa_statements = (
            MoELayer._gen_inv_packed_grouped_expert_aoa_statements
        )
        _GROUPED_EXPERT_LOCAL_NAMES = MoELayer._GROUPED_EXPERT_LOCAL_NAMES
        _intermediate_ep_size = MoELayer._intermediate_ep_size
        _reject_packed_intermediate_ep = MoELayer._reject_packed_intermediate_ep
        _reject_nongated_grouped_experts = (
            MoELayer._reject_nongated_grouped_experts
        )
        _inv_intermediate_ep_unpack_statements = (
            MoELayer._inv_intermediate_ep_unpack_statements
        )

        def __init__(self):
            super().__init__()
            self.num_experts = num_experts
            self.gate = router_cls()
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


_PFX = "model.layers.0.mlp."
_HF = "hf.layers.0.mlp."
_MD = "model.layers.0.mlp."

# MTP-side prefix used by the mixed-layout coexistence test.
_MTP_PFX = "model.mtp.layers.0.mlp."
_MTP_HF = "hf.mtp.layers.0.mlp."
_MTP_MD = "model.mtp.layers.0.mlp."


class TestMoELayerClassContract(unittest.TestCase):
    def test_override_present(self):
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        for name in (
            "gen_aoa_statements",
            "gen_inv_aoa_statements",
            "_gen_grouped_expert_aoa_statements",
            "_gen_inv_grouped_expert_aoa_statements",
            "_gen_packed_grouped_expert_aoa_statements",
            "_gen_inv_packed_grouped_expert_aoa_statements",
            "_intermediate_ep_size",
            "_reject_packed_intermediate_ep",
            "_reject_nongated_grouped_experts",
            "_inv_intermediate_ep_unpack_statements",
        ):
            self.assertIn(name, MoELayer.__dict__)


class TestMoELayerForward(unittest.TestCase):
    def test_gate_then_per_expert_composition(self):
        model = _make_moe_leaf(num_experts=2, shared=False)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        # gate rename (router leaf)
        self.assertIn(f"{_HF}gate.weight -> {_MD}gate.weight", stmts)
        # each expert recursed as an MLP: fused_ffn + down_proj
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
        # gate(1) + 2 experts * 2 statements each = 5
        self.assertEqual(len(stmts), 5)

    def test_shared_experts_included(self):
        model = _make_moe_leaf(num_experts=1, shared=True)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        self.assertIn(
            f"{_HF}shared_experts.gate_proj.weight^T, "
            f"{_HF}shared_experts.up_proj.weight^T "
            f"-> {_MD}shared_experts.up_gate_proj.weight, fused_ffn",
            stmts,
        )


class TestMoELayerInverse(unittest.TestCase):
    def test_gate_then_per_expert_inverse(self):
        model = _make_moe_leaf(num_experts=2, shared=False)()
        stmts = model.gen_inv_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        self.assertIn(f"{_MD}gate.weight -> {_HF}gate.weight", stmts)
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
        # gate(1) + 2 experts * (1 fused_ffn split + 3 transposes) = 9
        self.assertEqual(len(stmts), 9)


class TestMoELayerGroupedPerExpertConcat(unittest.TestCase):
    """Grouped-GEMM live layer with the default ``"per_expert"`` checkpoint
    layout: per-expert fuse (``^T`` / ``axis=1``) into transients, then concat
    on ``axis=0`` into the two packed model weights (GroupedMLPExpert
    convention)."""

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
        # stage 1: split the two packed weights
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
    the expert class.

    ``SonicMoEExpert`` keeps a different in-memory layout while computing
    (``weight1`` ``[E, 2I, io]`` with gate/up interleaved row-wise), but its
    ``sharded_state_dict()`` calls ``convert_weights_to_grouped_layout()`` first,
    so that layout never reaches a checkpoint. Branching the generators on the
    expert type therefore produced statements for a layout that is never saved;
    these tests pin the type-independence so the special case cannot come back.
    A genuine checkpoint layout divergence belongs in
    ``ctx.moe_expert_checkpoint_layout``, which the model declares.
    """

    @staticmethod
    def _sonic_marker():
        from paddlefleet.transformer.moe.moe_expert import SonicMoEExpert

        # ``__new__`` only: the generators may inspect nothing but the
        # model-side layout predicate, and a real expert would need a full
        # TransformerConfig plus device state to construct. The two inputs
        # ``intermediate_ep_sharded`` reads are supplied so the inherited
        # property answers for real (ordinary layout here).
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

        # One property covers both expert types, so the layout branch in the
        # generators needs no expert-type dispatch of its own.
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

    def test_no_axis_zero_defuse_for_sonic(self):
        _, sonic = self._pair()
        stmts = sonic._gen_inv_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        # The de-fuse of the per-expert fused gate+up is the only place an
        # axis=0 split would be wrong; the cross-expert splits legitimately use
        # axis=0 (the expert axis).
        for eid in (0, 1):
            self.assertIn(
                f"{_MD}experts.{eid}.up_gate_proj.weight "
                f"-> {_MD}experts.{eid}.gate_proj.weight.intermediate, "
                f"{_MD}experts.{eid}.up_proj.weight.intermediate, axis=1",
                stmts,
            )


class TestGroupedExpertIntermediateEPSharded(unittest.TestCase):
    """Grouped-GEMM live layer under the 'allgather' dispatcher with EP > 1.

    Every rank then holds all ``E`` experts but only ``I // EP`` of each one's
    intermediate dim, and the HF directions run against a 2-D re-declaration of
    those shards (AOA cannot lower a tensor's rank on save). That re-declared
    *global* layout is rank-major interleaved, because each rank's local block
    has to stay one contiguous rectangle::

        weight1 [E*io, 2*I_full]  columns: gate_r0 | up_r0 | gate_r1 | up_r1 |..
        weight2 [E*I_full, io]    rows:    (r0: e0..eE-1) | (r1: e0..eE-1) |..

    So the ordinary statements are numerically wrong here even though every
    shape still checks out: their ``axis=1`` bisection of ``weight1`` takes
    ``gate_r0 + up_r0`` as "gate", and their E-way ``axis=0`` split of
    ``weight2`` mixes experts. These tests pin the un-interleaving on both
    directions, that it lands on exactly the per-expert transients the ordinary
    layout produces (so the shared stages are untouched), and that the ordinary
    layout is unaffected.
    """

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
        model = self._model()
        self.assertEqual(model._intermediate_ep_size(), self.EP)
        # Both inputs of the predicate must hold, and neither implies the
        # other: allgather without EP, and EP without allgather, are ordinary.
        self.assertIsNone(
            self._model(expert_parallel=False)._intermediate_ep_size()
        )
        self.assertIsNone(
            self._model(dispatcher="deepep")._intermediate_ep_size()
        )

    def test_forward_slices_then_reinterleaves(self):
        model = self._model()
        stmts = model._gen_grouped_expert_aoa_statements(_ctx(), _PFX, None)

        expected = []
        for e in range(self.E):
            ug = f"{_MD}experts.{e}.up_gate_proj.weight"
            dm = f"{_MD}experts.{e}.down_proj.weight"
            # Each checkpoint matrix is cut into the EP intermediate slices the
            # re-declared layout expects, then gate/up are re-interleaved
            # rank-major into the per-expert transient.
            expected += [
                f"{_HF}experts.{e}.gate_proj.weight^T "
                f"-> {ug}.gate_ep0,{ug}.gate_ep1, axis=1",
                f"{_HF}experts.{e}.up_proj.weight^T "
                f"-> {ug}.up_ep0,{ug}.up_ep1, axis=1",
                f"{ug}.gate_ep0,{ug}.up_ep0,{ug}.gate_ep1,{ug}.up_ep1 "
                f"-> {ug}, axis=1",
                f"{_HF}experts.{e}.down_proj.weight^T "
                f"-> {dm}.ep0,{dm}.ep1, axis=0",
            ]
        # weight1 rows stay expert-major; weight2 rows are rank-outer.
        expected.append(
            f"{_MD}experts.0.up_gate_proj.weight,"
            f"{_MD}experts.1.up_gate_proj.weight -> {self.W1}, axis=0"
        )
        expected.append(
            f"{_MD}experts.0.down_proj.weight.ep0,"
            f"{_MD}experts.1.down_proj.weight.ep0,"
            f"{_MD}experts.0.down_proj.weight.ep1,"
            f"{_MD}experts.1.down_proj.weight.ep1 -> {self.W2}, axis=0"
        )
        self.assertEqual(stmts, expected)

    def test_inverse_unpacks_into_the_ordinary_transients(self):
        model = self._model()
        stmts = model._gen_inv_grouped_expert_aoa_statements(_ctx(), _PFX, None)

        w1, w2 = self.W1, self.W2
        ug = [f"{_MD}experts.{e}.up_gate_proj.weight" for e in range(self.E)]
        di = [
            f"{_MD}experts.{e}.down_proj.weight.intermediate"
            for e in range(self.E)
        ]
        # Stage (1): un-interleave. weight1 -> 2*EP column blocks, even ones
        # regroup into the full gate and odd ones into the full up, each then
        # cut expert-wise; weight2 -> EP*E row parts gathered back per expert.
        stage1 = [
            f"{w1} -> {w1}.ep_blk0,{w1}.ep_blk1,{w1}.ep_blk2,{w1}.ep_blk3, "
            f"axis=1",
            f"{w1}.ep_blk0,{w1}.ep_blk2 -> {w1}.gate_all, axis=1",
            f"{w1}.ep_blk1,{w1}.ep_blk3 -> {w1}.up_all, axis=1",
            f"{w1}.gate_all -> {w1}.gate_e0,{w1}.gate_e1, axis=0",
            f"{w1}.up_all -> {w1}.up_e0,{w1}.up_e1, axis=0",
            f"{w1}.gate_e0,{w1}.up_e0 -> {ug[0]}, axis=1",
            f"{w1}.gate_e1,{w1}.up_e1 -> {ug[1]}, axis=1",
            f"{w2} -> {w2}.r0e0,{w2}.r0e1,{w2}.r1e0,{w2}.r1e1, axis=0",
            f"{w2}.r0e0,{w2}.r1e0 -> {di[0]}, axis=0",
            f"{w2}.r0e1,{w2}.r1e1 -> {di[1]}, axis=0",
        ]
        self.assertEqual(stmts[: len(stage1)], stage1)

        # Stage (2) is shared verbatim with the ordinary layout: the whole
        # point of landing on the same transients. Compared against the
        # ordinary generator's own tail rather than re-spelled here.
        ordinary = _make_moe_leaf(
            num_experts=self.E, grouped=True
        )()._gen_inv_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        tail = 4 * self.E
        self.assertEqual(stmts[-tail:], ordinary[-tail:])
        self.assertEqual(len(stmts), len(stage1) + tail)

    def test_round_trip_names_are_consistent(self):
        # The two directions are written independently; the per-expert
        # transients they meet on must still agree, otherwise the save plan
        # would target names the load plan never produces.
        model = self._model()
        fwd = model._gen_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        inv = model._gen_inv_grouped_expert_aoa_statements(_ctx(), _PFX, None)
        for e in range(self.E):
            ug = f"{_MD}experts.{e}.up_gate_proj.weight"
            self.assertTrue(any(s.endswith(f"-> {ug}, axis=1") for s in fwd))
            self.assertTrue(any(s.endswith(f"-> {ug}, axis=1") for s in inv))

    def test_packed_layout_combination_rejected(self):
        model = self._model()
        ctx = _ctx(moe_expert_checkpoint_layout="packed")
        # Orthogonal dimensions: "packed" describes the checkpoint side and
        # intermediate-sharded the model side. The packed branch maps whole 3D
        # checkpoint tensors through a permute, leaving nowhere to un-interleave
        # the rank-major ordering, so it must fail loudly instead of emitting
        # plausible statements that silently produce wrong data.
        with self.assertRaises(NotImplementedError):
            model._gen_grouped_expert_aoa_statements(ctx, _PFX, None)
        with self.assertRaises(NotImplementedError):
            model._gen_inv_grouped_expert_aoa_statements(ctx, _PFX, None)
        # The ordinary layout still reaches the packed branch.
        self.assertEqual(
            len(
                self._model(
                    dispatcher="deepep"
                )._gen_grouped_expert_aoa_statements(ctx, _PFX, None)
            ),
            2,
        )

    def test_ordinary_layout_statements_unchanged(self):
        # Regression guard: the deepep path must be byte-identical to the
        # pre-existing expectations pinned by
        # TestMoELayerGroupedPerExpertConcat, i.e. the new branch is inert
        # unless the live expert says otherwise.
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


class TestMoELayerPackedLayout(unittest.TestCase):
    """``"packed"`` checkpoint layout on the grouped branch: the only transform
    is a per-weight ``permute='[0,2,1]'`` (an involution reused verbatim by the
    inverse). Selected by ``ctx.moe_expert_checkpoint_layout == "packed"``."""

    def test_forward_permute(self):
        model = _make_moe_leaf(num_experts=2, grouped=True)()
        stmts = model._gen_grouped_expert_aoa_statements(
            _ctx(moe_expert_checkpoint_layout="packed"), _PFX, None
        )
        self.assertEqual(
            stmts,
            [
                f"{_HF}grouped_gemm_experts.weight1 "
                f"-> {_MD}grouped_gemm_experts.weight1, permute='[0,2,1]'",
                f"{_HF}grouped_gemm_experts.weight2 "
                f"-> {_MD}grouped_gemm_experts.weight2, permute='[0,2,1]'",
            ],
        )

    def test_inverse_permute(self):
        model = _make_moe_leaf(num_experts=2, grouped=True)()
        stmts = model._gen_inv_grouped_expert_aoa_statements(
            _ctx(moe_expert_checkpoint_layout="packed"), _PFX, None
        )
        self.assertEqual(
            stmts,
            [
                f"{_MD}grouped_gemm_experts.weight1 "
                f"-> {_HF}grouped_gemm_experts.weight1, permute='[0,2,1]'",
                f"{_MD}grouped_gemm_experts.weight2 "
                f"-> {_HF}grouped_gemm_experts.weight2, permute='[0,2,1]'",
            ],
        )

    def test_top_level_gen_routes_packed(self):
        # Going through the public generator (gate + grouped packed) confirms
        # the branch selection wiring, not just the inner helper.
        model = _make_moe_leaf(num_experts=2, grouped=True)()
        stmts = model.gen_aoa_statements(
            _ctx(moe_expert_checkpoint_layout="packed"),
            structured_name_prefix=_PFX,
        )
        self.assertIn(f"{_HF}gate.weight -> {_MD}gate.weight", stmts)
        self.assertIn(
            f"{_HF}grouped_gemm_experts.weight1 "
            f"-> {_MD}grouped_gemm_experts.weight1, permute='[0,2,1]'",
            stmts,
        )
        # gate(1) + weight1 + weight2 = 3, no per-expert concat noise
        self.assertEqual(len(stmts), 3)


class TestMoELayerPerExpertIgnoresPackedFlag(unittest.TestCase):
    """A per-expert *live* layer (no ``grouped_gemm_experts``) must produce the
    same statements whether or not the model-level ``"packed"`` flag is set --
    the flag only selects a grouped-branch sub-path. This is what lets Qwen3.5
    carry per-expert MTP layers under a model-level ``"packed"`` flag."""

    def test_forward_flag_independent(self):
        model = _make_moe_leaf(num_experts=2, shared=False)()
        default = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        packed = model.gen_aoa_statements(
            _ctx(moe_expert_checkpoint_layout="packed"),
            structured_name_prefix=_PFX,
        )
        self.assertEqual(default, packed)
        self.assertTrue(all("permute" not in s for s in packed))

    def test_inverse_flag_independent(self):
        model = _make_moe_leaf(num_experts=2, shared=False)()
        default = model.gen_inv_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        packed = model.gen_inv_aoa_statements(
            _ctx(moe_expert_checkpoint_layout="packed"),
            structured_name_prefix=_PFX,
        )
        self.assertEqual(default, packed)


class TestMoELayerMixedLayoutCoexistence(unittest.TestCase):
    """Qwen3.5-style mix under ONE model-level ``moe_expert_checkpoint_layout
    == "packed"`` flag: a grouped backbone layer emits the packed permute while
    a per-expert MTP layer emits per-expert recursion (flag ignored on the
    per-expert branch). Both are generated with the same ``ctx``."""

    def test_backbone_packed_mtp_per_expert(self):
        ctx = _ctx(moe_expert_checkpoint_layout="packed")

        backbone = _make_moe_leaf(num_experts=2, grouped=True)()
        backbone_stmts = backbone.gen_aoa_statements(
            ctx, structured_name_prefix=_PFX
        )
        # backbone -> packed permute, no per-expert fused_ffn
        self.assertIn(
            f"{_HF}grouped_gemm_experts.weight1 "
            f"-> {_MD}grouped_gemm_experts.weight1, permute='[0,2,1]'",
            backbone_stmts,
        )
        self.assertTrue(all("fused_ffn" not in s for s in backbone_stmts))

        mtp = _make_moe_leaf(num_experts=2, grouped=False)()
        mtp_stmts = mtp.gen_aoa_statements(ctx, structured_name_prefix=_MTP_PFX)
        # MTP -> per-expert recursion, no packed permute
        for eid in (0, 1):
            self.assertIn(
                f"{_MTP_HF}experts.{eid}.gate_proj.weight^T, "
                f"{_MTP_HF}experts.{eid}.up_proj.weight^T "
                f"-> {_MTP_MD}experts.{eid}.up_gate_proj.weight, fused_ffn",
                mtp_stmts,
            )
        self.assertTrue(all("permute" not in s for s in mtp_stmts))


if __name__ == "__main__":
    unittest.main()
