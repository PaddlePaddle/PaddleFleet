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
# Scope: an MTP block's AOA contract. The block declares one checkpoint lookup
# drop segment (``transformer_layer``) for its whole subtree and enumerates
# nothing itself, so this test pins: the inner transformer's tensors reach the
# ordinary-layer mapping entries while the model side keeps the segment; direct
# children are unaffected by the drop; ``eh_proj`` is transposed; and
# ``mtp_embed`` gets no rule of its own.
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

import unittest

import paddle
from paddle.distributed.flex_checkpoint.aoa.generation import AOAContext

# Two entries of the shape a real model declares: both are written for an
# ordinary layer, and the drop segment is what lets the second one also cover
# an MTP block's inner transformer.
_MAPPING = {
    "model.layers.$LAYER_ID.norm.weight": (
        "model.layers.$LAYER_ID.shared_head.norm.weight"
    ),
    "model.layers.$LAYER_ID.mlp.down_proj.weight": (
        "model.layers.$LAYER_ID.block_sparse_moe.experts.w2.weight"
    ),
}

# An MTP block is registered as a top-level pipeline layer at
# ``<model>.layers.<i>``, so its subtree starts one segment below that.
_PFX = "model.layers.3."


def _ctx(*, checkpoint_name_prefix="model", mapping=_MAPPING):
    return AOAContext(
        config=None,
        checkpoint_name_prefix=checkpoint_name_prefix,
        pp_to_single_mapping={},
        checkpoint_name_mapping=mapping,
        model_name_prefix="model",
    )


class _ParamLeaf(paddle.nn.Layer):
    """A plain param carrier, left to the base identity recursion."""

    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[2, 2])


def _fused_linear_leaf():
    """Binds ``FusedLinear``'s AOA overrides onto a controllable stand-in.

    The real class allocates a fused GEMM weight through its own constructor,
    so the two overrides (which only forward to the Linear family helpers) are
    bound onto a lightweight ``paddle.nn.Layer`` carrying a real param.
    """
    from paddlefleet.tensor_parallel.layers import FusedLinear

    class _FusedLinearLeaf(paddle.nn.Layer):
        gen_aoa_statements = FusedLinear.gen_aoa_statements
        gen_inv_aoa_statements = FusedLinear.gen_inv_aoa_statements

        def __init__(self):
            super().__init__()
            self.weight = self.create_parameter(shape=[2, 2])

    return _FusedLinearLeaf()


class _InnerMLP(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.down_proj = _ParamLeaf()


class _InnerTransformer(paddle.nn.Layer):
    """Stands in for the block's ``transformer_layer`` child subtree."""

    def __init__(self):
        super().__init__()
        self.mlp = _InnerMLP()


def _make_mtp_stand_in(*, mtp_embed=False):
    """A structural stand-in for an MTP block.

    Building the real layer needs a live config plus TP / PP process groups, so
    this subclass keeps the class (hence both AOA overrides and their
    ``super()`` chain) and only the child layout that decides resolved names.
    """
    from paddlefleet.transformer.multi_token_prediction import (
        MultiTokenPredictionLayer,
    )

    class _MTPStandIn(MultiTokenPredictionLayer):
        def __init__(self):
            paddle.nn.Layer.__init__(self)
            self.enorm = _ParamLeaf()
            self.hnorm = _ParamLeaf()
            self.eh_proj = _fused_linear_leaf()
            self.norm = _ParamLeaf()
            self.transformer_layer = _InnerTransformer()
            if mtp_embed:
                self.mtp_embed = _ParamLeaf()

    return _MTPStandIn()


_INNER_CK = "model.layers.3.block_sparse_moe.experts.w2.weight"
_INNER_MD = "model.layers.3.transformer_layer.mlp.down_proj.weight"
_NORM_CK = "model.layers.3.shared_head.norm.weight"
_NORM_MD = "model.layers.3.norm.weight"
_EH_MD = "model.layers.3.eh_proj.weight"


class TestMTPClassContract(unittest.TestCase):
    def test_mtp_layer_overrides_both_directions(self):
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )

        self.assertIn("gen_aoa_statements", MultiTokenPredictionLayer.__dict__)
        self.assertIn(
            "gen_inv_aoa_statements", MultiTokenPredictionLayer.__dict__
        )

    def test_weight_only_variant_inherits_the_same_overrides(self):
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
            WeightOnlyMTPLayer,
        )

        self.assertIs(
            WeightOnlyMTPLayer.gen_aoa_statements,
            MultiTokenPredictionLayer.gen_aoa_statements,
        )
        self.assertIs(
            WeightOnlyMTPLayer.gen_inv_aoa_statements,
            MultiTokenPredictionLayer.gen_inv_aoa_statements,
        )

    def test_fused_linear_takes_part_and_is_exported(self):
        import paddlefleet.tensor_parallel as tp
        from paddlefleet.tensor_parallel.layers import FusedLinear

        self.assertIn("gen_aoa_statements", FusedLinear.__dict__)
        self.assertIn("gen_inv_aoa_statements", FusedLinear.__dict__)
        self.assertIs(tp.FusedLinear, FusedLinear)
        self.assertIn("FusedLinear", tp.__all__)


class TestMTPForward(unittest.TestCase):
    def test_inner_transformer_reaches_ordinary_layer_entry(self):
        stmts = _make_mtp_stand_in().gen_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        # Checkpoint side went through the entry written for an ordinary
        # layer; model side kept ``transformer_layer``.
        self.assertIn(f"{_INNER_CK} -> {_INNER_MD}", stmts)

    def test_direct_children_are_unaffected_by_the_drop(self):
        stmts = _make_mtp_stand_in().gen_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        # ``norm`` carries no drop segment, so the lookup is unchanged and its
        # own mapping entry still applies.
        self.assertIn(f"{_NORM_CK} -> {_NORM_MD}", stmts)
        # ``enorm`` / ``hnorm`` have no entry: identity, hence omitted.
        self.assertFalse(any("enorm" in s or "hnorm" in s for s in stmts))

    def test_eh_proj_is_transposed(self):
        stmts = _make_mtp_stand_in().gen_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        self.assertIn(f"{_EH_MD}^T -> {_EH_MD}", stmts)

    def test_mtp_embed_gets_no_rule(self):
        stmts = _make_mtp_stand_in(mtp_embed=True).gen_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        self.assertFalse(any("mtp_embed" in s for s in stmts))

    def test_statement_set_is_exactly_the_transformed_tensors(self):
        stmts = _make_mtp_stand_in(mtp_embed=True).gen_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        self.assertEqual(
            sorted(stmts),
            sorted(
                [
                    f"{_INNER_CK} -> {_INNER_MD}",
                    f"{_NORM_CK} -> {_NORM_MD}",
                    f"{_EH_MD}^T -> {_EH_MD}",
                ]
            ),
        )

    def test_drop_is_a_no_op_for_children_without_the_segment(self):
        # With distinct roots the identity statements become visible, which is
        # what shows the drop only ever removes the segment where it occurs.
        stmts = _make_mtp_stand_in(mtp_embed=True).gen_aoa_statements(
            _ctx(checkpoint_name_prefix="hf"),
            structured_name_prefix=_PFX,
        )
        self.assertIn(
            "hf.layers.3.enorm.weight -> model.layers.3.enorm.weight", stmts
        )
        self.assertIn(
            "hf.layers.3.hnorm.weight -> model.layers.3.hnorm.weight", stmts
        )
        self.assertIn(
            "hf.layers.3.mtp_embed.weight -> model.layers.3.mtp_embed.weight",
            stmts,
        )
        # A mapping hit is a complete checkpoint name, so it ignores the root.
        self.assertIn(f"{_INNER_CK} -> {_INNER_MD}", stmts)


class TestMTPInverse(unittest.TestCase):
    def test_endpoints_swapped_and_drop_still_declared(self):
        stmts = _make_mtp_stand_in(mtp_embed=True).gen_inv_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        self.assertEqual(
            sorted(stmts),
            sorted(
                [
                    f"{_INNER_MD} -> {_INNER_CK}",
                    f"{_NORM_MD} -> {_NORM_CK}",
                    f"{_EH_MD}^T -> {_EH_MD}",
                ]
            ),
        )

    def test_mtp_embed_gets_no_rule(self):
        stmts = _make_mtp_stand_in(mtp_embed=True).gen_inv_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        self.assertFalse(any("mtp_embed" in s for s in stmts))


if __name__ == "__main__":
    unittest.main()
