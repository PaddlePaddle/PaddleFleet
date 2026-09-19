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
# Scope: the gated ``MLP`` fuses the checkpoint ``gate_proj`` and ``up_proj``
# (each transposed) into the model ``up_gate_proj`` via the ``fused_ffn``
# macro, adds an optional fused bias (no transpose), then delegates
# ``down_proj`` to the Linear family. MLP has no own parameters/buffers of its
# own; every statement comes from the fused pair or the delegated ``down_proj``
# leaf. This test pins that contract and the independent inverse split.
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

from paddlefleet.models.gpt.aoa_generator import (
    FleetAOAContext,
)


def _ctx(dtype_cast_rules=None):
    return FleetAOAContext(
        config=None,
        checkpoint_name_prefix="hf",
        pp_to_single_mapping={},
        checkpoint_name_mapping={},
        dtype_cast_rules=dtype_cast_rules or {},
        model_name_prefix="model",
        excluded_names=frozenset(),
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


def _make_mlp_leaf(up_gate_bias=False, down_bias=False):
    """Binds the gated MLP AOA overrides onto a controllable stand-in.

    ``up_gate_proj`` only needs a ``.bias`` attribute (it gates the fused-bias
    statement and is never recursed); ``down_proj`` is a real Linear leaf that
    the MLP delegates to.
    """
    from paddlefleet.transformer.mlp import MLP

    up_gate_cls = _make_linear_leaf(bias=up_gate_bias)
    down_cls = _make_linear_leaf(bias=down_bias)

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


_PFX = "model.layers.0.mlp."
_HF = "hf.layers.0.mlp."
_MD = "model.layers.0.mlp."


class TestMLPClassContract(unittest.TestCase):
    def test_override_present(self):
        from paddlefleet.transformer.mlp import MLP

        for name in (
            "gen_aoa_statements",
            "gen_inv_aoa_statements",
            "_gen_up_gate_fusion_aoa_statements",
            "_gen_inv_up_gate_fusion_aoa_statements",
            "_gen_up_gate_bias_aoa_statements",
            "_gen_inv_up_gate_bias_aoa_statements",
            "_gen_up_gate_bias_fusion_aoa_statements",
            "_gen_inv_up_gate_bias_fusion_aoa_statements",
        ):
            self.assertIn(name, MLP.__dict__)


class TestMLPForward(unittest.TestCase):
    def test_fused_ffn_and_down_proj_delegation(self):
        model = _make_mlp_leaf(up_gate_bias=False, down_bias=False)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        self.assertEqual(
            stmts,
            [
                f"{_HF}gate_proj.weight^T, {_HF}up_proj.weight^T "
                f"-> {_MD}up_gate_proj.weight, fused_ffn",
                f"{_HF}down_proj.weight^T -> {_MD}down_proj.weight",
            ],
        )

    def test_fused_bias_present_when_up_gate_has_bias(self):
        model = _make_mlp_leaf(up_gate_bias=True, down_bias=True)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        # fused bias (no transpose) sits between the fused weight and down_proj;
        # the single bias axis has to be named explicitly for fused_ffn
        self.assertIn(
            f"{_HF}gate_proj.bias, {_HF}up_proj.bias "
            f"-> {_MD}up_gate_proj.bias, fused_ffn, axis=0",
            stmts,
        )
        # down_proj leaf now carries its own bias identity too
        self.assertIn(f"{_HF}down_proj.bias -> {_MD}down_proj.bias", stmts)


class TestMLPInverse(unittest.TestCase):
    def test_fused_split_and_down_proj_delegation(self):
        model = _make_mlp_leaf(up_gate_bias=True, down_bias=False)()
        stmts = model.gen_inv_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        # The fused weight splits onto model-side half names (no ^T: fused_ffn
        # emits its halves on the target side), then each half is transposed
        # back onto its checkpoint name.
        self.assertIn(
            f"{_MD}up_gate_proj.weight "
            f"-> {_MD}gate_proj.weight, {_MD}up_proj.weight, fused_ffn",
            stmts,
        )
        self.assertIn(
            f"{_MD}gate_proj.weight^T -> {_HF}gate_proj.weight", stmts
        )
        self.assertIn(f"{_MD}up_proj.weight^T -> {_HF}up_proj.weight", stmts)
        # The 1-D bias needs no transpose, so it splits straight onto the
        # checkpoint names in one statement.
        self.assertIn(
            f"{_MD}up_gate_proj.bias "
            f"-> {_HF}gate_proj.bias, {_HF}up_proj.bias, fused_ffn, axis=0",
            stmts,
        )
        # down_proj inverse transpose (endpoints swapped)
        self.assertIn(
            f"{_MD}down_proj.weight^T -> {_HF}down_proj.weight", stmts
        )


class TestMLPDtypeCast(unittest.TestCase):
    def test_down_proj_cast_endpoints(self):
        rules = {
            "layers.0.mlp.down_proj.weight": {
                "checkpoint_dtype": "bfloat16",
                "model_dtype": "float32",
            }
        }
        model = _make_mlp_leaf(up_gate_bias=False, down_bias=False)()
        fwd = model.gen_aoa_statements(_ctx(rules), structured_name_prefix=_PFX)
        self.assertIn(
            f"{_HF}down_proj.weight^T -> {_MD}down_proj.weight, "
            "src_dtype='bfloat16', dst_dtype='float32'",
            fwd,
        )
        inv = model.gen_inv_aoa_statements(
            _ctx(rules), structured_name_prefix=_PFX
        )
        self.assertIn(
            f"{_MD}down_proj.weight^T -> {_HF}down_proj.weight, "
            "src_dtype='float32', dst_dtype='bfloat16'",
            inv,
        )


if __name__ == "__main__":
    unittest.main()
