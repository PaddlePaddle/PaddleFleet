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
# Scope: the base ``MLP`` NON-gated checkpoint layout, selected by the AOA flag
# ``ctx.mlp_gate_up_fused = False`` (declared per-model, e.g. the Qwen3-VL vision
# tower's ``aoa_mlp_gate_up_fused = False``). The default gated layout
# UNCONDITIONALLY fuses checkpoint ``gate_proj`` + ``up_proj`` into
# ``up_gate_proj`` via ``fused_ffn``. The non-gated layout instead treats
# ``up_gate_proj`` / ``down_proj`` as two plain Linears with no gate/up split,
# delegating both to the generic Linear family (plain ``^T`` weight renames +
# identity bias, NO ``fused_ffn``).
#
# We bind the base ``MLP`` non-gated head helpers onto a stand-in that owns real
# ``paddle.nn.Layer`` Linear children (whose generators route through the shared
# ``gen_linear_(inv_)aoa_statements`` helpers), which exercises the delegation
# contract without the heavy TP-linear ``MLP.__init__``. We also assert the
# flag-routing contract on the real ``MLP`` class.
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
from paddlefleet.tensor_parallel.layers import (
    gen_linear_aoa_statements,
    gen_linear_inv_aoa_statements,
)


class _FakeLinear(paddle.nn.Layer):
    """A minimal transposed-weight Linear leaf whose AOA generators route
    through the shared Linear-family helpers (the same helpers the real
    ColumnParallelLinear / RowParallelLinear delegate to)."""

    def __init__(self, bias=False):
        super().__init__()
        self.weight = self.create_parameter(shape=[2, 2])
        self.bias = self.create_parameter(shape=[2]) if bias else None

    def gen_aoa_statements(
        self, ctx, *, structured_name_prefix="", aoa_name_scope=None
    ):
        return gen_linear_aoa_statements(
            self,
            ctx,
            structured_name_prefix=structured_name_prefix,
            aoa_name_scope=aoa_name_scope,
        )

    def gen_inv_aoa_statements(
        self, ctx, *, structured_name_prefix="", aoa_name_scope=None
    ):
        return gen_linear_inv_aoa_statements(
            self,
            ctx,
            structured_name_prefix=structured_name_prefix,
            aoa_name_scope=aoa_name_scope,
        )


def _make_fake_cls():
    from paddlefleet.transformer.mlp import MLP

    class _FakeNonGatedMLP(paddle.nn.Layer):
        # Bind the base MLP non-gated head helpers (the branch selected by
        # ``ctx.mlp_gate_up_fused == False`` inside ``gen_aoa_statements``).
        _gen_nongated_mlp_aoa_statements = MLP._gen_nongated_mlp_aoa_statements
        _gen_inv_nongated_mlp_aoa_statements = (
            MLP._gen_inv_nongated_mlp_aoa_statements
        )

        def __init__(self, bias=False):
            super().__init__()
            self.up_gate_proj = _FakeLinear(bias=bias)
            self.down_proj = _FakeLinear(bias=bias)

        def gen_aoa_statements(
            self, ctx, *, structured_name_prefix="", aoa_name_scope=None
        ):
            return self._gen_nongated_mlp_aoa_statements(
                ctx, structured_name_prefix, aoa_name_scope
            )

        def gen_inv_aoa_statements(
            self, ctx, *, structured_name_prefix="", aoa_name_scope=None
        ):
            return self._gen_inv_nongated_mlp_aoa_statements(
                ctx, structured_name_prefix, aoa_name_scope
            )

    return _FakeNonGatedMLP


_PFX = "mlp."
_PP = {
    _PFX + "up_gate_proj.weight": "model.mlp.up_gate_proj.weight",
    _PFX + "up_gate_proj.bias": "model.mlp.up_gate_proj.bias",
    _PFX + "down_proj.weight": "model.mlp.down_proj.weight",
    _PFX + "down_proj.bias": "model.mlp.down_proj.bias",
}


def _ctx():
    return FleetAOAContext(
        config=None,
        checkpoint_name_prefix="hf",
        pp_to_single_mapping=_PP,
        checkpoint_name_mapping={},
        dtype_cast_rules={},
        model_name_prefix="model",
        excluded_names=frozenset(),
        mlp_gate_up_fused=False,
    )


class TestNonGatedMLPForward(unittest.TestCase):
    """Forward (checkpoint -> model): both projections are plain Linear ``^T``
    renames; no gate/up split, no ``fused_ffn``."""

    def setUp(self):
        self.ctx = _ctx()

    def test_weight_golden_no_fusion(self):
        m = _make_fake_cls()(bias=False)
        stmts = m.gen_aoa_statements(self.ctx, structured_name_prefix=_PFX)
        expected = [
            "hf.mlp.up_gate_proj.weight^T -> model.mlp.up_gate_proj.weight",
            "hf.mlp.down_proj.weight^T -> model.mlp.down_proj.weight",
        ]
        self.assertEqual(stmts, expected)

    def test_bias_is_identity_no_transpose(self):
        m = _make_fake_cls()(bias=True)
        stmts = m.gen_aoa_statements(self.ctx, structured_name_prefix=_PFX)
        expected = [
            "hf.mlp.up_gate_proj.weight^T -> model.mlp.up_gate_proj.weight",
            "hf.mlp.up_gate_proj.bias -> model.mlp.up_gate_proj.bias",
            "hf.mlp.down_proj.weight^T -> model.mlp.down_proj.weight",
            "hf.mlp.down_proj.bias -> model.mlp.down_proj.bias",
        ]
        self.assertEqual(stmts, expected)

    def test_no_fused_ffn_either_bias_mode(self):
        for bias in (False, True):
            m = _make_fake_cls()(bias=bias)
            stmts = m.gen_aoa_statements(self.ctx, structured_name_prefix=_PFX)
            self.assertFalse(any("fused_ffn" in s for s in stmts))


class TestNonGatedMLPInverse(unittest.TestCase):
    """Inverse (model -> checkpoint): plain Linear ``^T`` renames the other way;
    still no ``fused_ffn`` (independently generated)."""

    def setUp(self):
        self.ctx = _ctx()

    def test_weight_golden_no_fusion(self):
        m = _make_fake_cls()(bias=False)
        stmts = m.gen_inv_aoa_statements(self.ctx, structured_name_prefix=_PFX)
        expected = [
            "model.mlp.up_gate_proj.weight^T -> hf.mlp.up_gate_proj.weight",
            "model.mlp.down_proj.weight^T -> hf.mlp.down_proj.weight",
        ]
        self.assertEqual(stmts, expected)

    def test_bias_is_identity_no_transpose(self):
        m = _make_fake_cls()(bias=True)
        stmts = m.gen_inv_aoa_statements(self.ctx, structured_name_prefix=_PFX)
        expected = [
            "model.mlp.up_gate_proj.weight^T -> hf.mlp.up_gate_proj.weight",
            "model.mlp.up_gate_proj.bias -> hf.mlp.up_gate_proj.bias",
            "model.mlp.down_proj.weight^T -> hf.mlp.down_proj.weight",
            "model.mlp.down_proj.bias -> hf.mlp.down_proj.bias",
        ]
        self.assertEqual(stmts, expected)

    def test_no_fused_ffn_either_bias_mode(self):
        for bias in (False, True):
            m = _make_fake_cls()(bias=bias)
            stmts = m.gen_inv_aoa_statements(
                self.ctx, structured_name_prefix=_PFX
            )
            self.assertFalse(any("fused_ffn" in s for s in stmts))


class TestNonGatedMLPRoundTrip(unittest.TestCase):
    """Forward and inverse are endpoint-swaps over the same name pairs (the
    plain-Linear leaves have no head reorder, so each fwd ``a -> b`` has an
    inverse ``b -> a`` with matching ``^T`` placement)."""

    def setUp(self):
        self.ctx = _ctx()

    def test_endpoints_swap(self):
        m = _make_fake_cls()(bias=True)
        fwd = m.gen_aoa_statements(self.ctx, structured_name_prefix=_PFX)
        inv = m.gen_inv_aoa_statements(self.ctx, structured_name_prefix=_PFX)

        def _endpoints(stmt):
            lhs, rhs = stmt.split(" -> ")
            return lhs.replace("^T", ""), rhs.replace("^T", "")

        fwd_pairs = {_endpoints(s) for s in fwd}
        inv_pairs = {(rhs, lhs) for (lhs, rhs) in (_endpoints(s) for s in inv)}
        self.assertEqual(fwd_pairs, inv_pairs)


class TestFlagRoutingContract(unittest.TestCase):
    """The base ``MLP`` owns the non-gated head helpers and its public
    generators route to them only when ``ctx.mlp_gate_up_fused`` is False (no
    AOA-only subclass involved)."""

    def test_base_mlp_has_nongated_helpers(self):
        from paddlefleet.transformer.mlp import MLP

        self.assertIn("_gen_nongated_mlp_aoa_statements", MLP.__dict__)
        self.assertIn("_gen_inv_nongated_mlp_aoa_statements", MLP.__dict__)

    def test_generators_branch_on_flag(self):
        import inspect

        from paddlefleet.transformer.mlp import MLP

        fwd_src = inspect.getsource(MLP.gen_aoa_statements)
        inv_src = inspect.getsource(MLP.gen_inv_aoa_statements)
        self.assertIn("mlp_gate_up_fused", fwd_src)
        self.assertIn("_gen_nongated_mlp_aoa_statements", fwd_src)
        self.assertIn("mlp_gate_up_fused", inv_src)
        self.assertIn("_gen_inv_nongated_mlp_aoa_statements", inv_src)


if __name__ == "__main__":
    unittest.main()
