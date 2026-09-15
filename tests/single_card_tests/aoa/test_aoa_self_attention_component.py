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
# Scope: standard ``SelfAttention`` fuses the checkpoint q/k/v projections
# (each transposed) into the model ``qkv_proj`` via the ``fused_qkv`` macro,
# adds an optional fused ``axis=0`` bias, then recurses every other direct
# child generically (``o_proj`` -> Linear ``^T``) and emits an identity
# statement for each own persistable buffer. This test pins that coverage
# contract and the independent inverse split.
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


class _Config:
    num_attention_heads = 8
    num_key_value_heads = 2


def _ctx(dtype_cast_rules=None):
    return FleetAOAContext(
        config=_Config(),
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


def _make_attn_leaf(qkv_bias=False, buffer=False):
    """Binds the SelfAttention AOA overrides onto a controllable stand-in.

    The real ``SelfAttention`` needs a live TP process group, so we bind the
    two entry points, the qkv-head hooks they dispatch into, and the fused
    qkv-bias helpers those hooks delegate their bias tail to. The stand-in
    carries a ``qkv_proj`` leaf (whose ``.bias`` gates the fused-bias
    statement), an ``o_proj`` Linear leaf that is recursed generically, and an
    optional own persistable buffer. ``gated_attention = False`` selects the
    standard qkv branch (the gated layout has its own component test).
    """
    from paddlefleet.transformer.attention import SelfAttention

    linear_with_bias = _make_linear_leaf(bias=qkv_bias)
    linear_plain = _make_linear_leaf(bias=False)

    class _AttnLeaf(paddle.nn.Layer):
        gen_aoa_statements = SelfAttention.gen_aoa_statements
        gen_inv_aoa_statements = SelfAttention.gen_inv_aoa_statements
        _gen_qkv_head_aoa_statements = (
            SelfAttention._gen_qkv_head_aoa_statements
        )
        _gen_inv_qkv_head_aoa_statements = (
            SelfAttention._gen_inv_qkv_head_aoa_statements
        )
        _gen_qkv_bias_aoa_statements = (
            SelfAttention._gen_qkv_bias_aoa_statements
        )
        _gen_inv_qkv_bias_aoa_statements = (
            SelfAttention._gen_inv_qkv_bias_aoa_statements
        )
        gated_attention = False

        def __init__(self):
            super().__init__()
            self.qkv_proj = linear_with_bias()
            self.o_proj = linear_plain()
            if buffer:
                self.register_buffer(
                    "some_buf", paddle.zeros([2]), persistable=True
                )
                self.register_buffer(
                    "tmp_buf", paddle.zeros([2]), persistable=False
                )

    return _AttnLeaf


_PFX = "model.layers.0.self_attn."
_HF = "hf.layers.0.self_attn."
_MD = "model.layers.0.self_attn."


class TestSelfAttentionClassContract(unittest.TestCase):
    def test_override_present(self):
        from paddlefleet.transformer.attention import SelfAttention

        for name in (
            "gen_aoa_statements",
            "gen_inv_aoa_statements",
            "_gen_qkv_head_aoa_statements",
            "_gen_inv_qkv_head_aoa_statements",
            "_gen_qkv_bias_aoa_statements",
            "_gen_inv_qkv_bias_aoa_statements",
        ):
            self.assertIn(name, SelfAttention.__dict__)


class TestSelfAttentionForward(unittest.TestCase):
    def test_fused_qkv_weight_and_child_recursion(self):
        model = _make_attn_leaf(qkv_bias=False, buffer=True)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        # fused qkv weight: three transposed checkpoint inputs, one model output
        self.assertIn(
            f"{_HF}q_proj.weight^T, {_HF}k_proj.weight^T, "
            f"{_HF}v_proj.weight^T "
            f"-> {_MD}qkv_proj.weight, fused_qkv, "
            f"num_heads=8, num_key_value_groups=2",
            stmts,
        )
        # o_proj recursed generically -> Linear transpose
        self.assertIn(f"{_HF}o_proj.weight^T -> {_MD}o_proj.weight", stmts)
        # own persistable buffer -> identity; non-persistable absent
        self.assertIn(f"{_HF}some_buf -> {_MD}some_buf", stmts)
        self.assertFalse(any("tmp_buf" in s for s in stmts))

    def test_fused_qkv_bias_present_when_qkv_has_bias(self):
        model = _make_attn_leaf(qkv_bias=True, buffer=False)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        self.assertIn(
            f"{_HF}q_proj.bias, {_HF}k_proj.bias, {_HF}v_proj.bias "
            f"-> {_MD}qkv_proj.bias, fused_qkv, "
            f"num_heads=8, num_key_value_groups=2, axis=0",
            stmts,
        )

    def test_no_qkv_bias_statement_when_absent(self):
        model = _make_attn_leaf(qkv_bias=False, buffer=False)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        self.assertFalse(any("qkv_proj.bias" in s for s in stmts))
        # exactly two statements: fused qkv weight + o_proj weight
        self.assertEqual(len(stmts), 2)


class TestSelfAttentionInverse(unittest.TestCase):
    def test_fused_qkv_split_and_child_recursion(self):
        model = _make_attn_leaf(qkv_bias=True, buffer=True)()
        stmts = model.gen_inv_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        # inverse fused qkv split carries NO ^T
        self.assertIn(
            f"{_MD}qkv_proj.weight -> {_HF}q_proj.weight, "
            f"{_HF}k_proj.weight, {_HF}v_proj.weight, "
            f"fused_qkv, num_heads=8, num_key_value_groups=2",
            stmts,
        )
        # inverse fused bias split
        self.assertIn(
            f"{_MD}qkv_proj.bias -> {_HF}q_proj.bias, "
            f"{_HF}k_proj.bias, {_HF}v_proj.bias, "
            f"fused_qkv, num_heads=8, num_key_value_groups=2, axis=0",
            stmts,
        )
        # o_proj recursed -> Linear inverse transpose (endpoints swapped)
        self.assertIn(f"{_MD}o_proj.weight^T -> {_HF}o_proj.weight", stmts)
        # own persistable buffer identity (model -> checkpoint)
        self.assertIn(f"{_MD}some_buf -> {_HF}some_buf", stmts)


class TestSelfAttentionDtypeCast(unittest.TestCase):
    def test_own_buffer_cast_endpoints(self):
        rules = {
            "layers.0.self_attn.some_buf": {
                "checkpoint_dtype": "bfloat16",
                "model_dtype": "float32",
            }
        }
        model = _make_attn_leaf(qkv_bias=False, buffer=True)()
        fwd = model.gen_aoa_statements(_ctx(rules), structured_name_prefix=_PFX)
        self.assertIn(
            f"{_HF}some_buf -> {_MD}some_buf, "
            "src_dtype='bfloat16', dst_dtype='float32'",
            fwd,
        )
        inv = model.gen_inv_aoa_statements(
            _ctx(rules), structured_name_prefix=_PFX
        )
        self.assertIn(
            f"{_MD}some_buf -> {_HF}some_buf, "
            "src_dtype='float32', dst_dtype='bfloat16'",
            inv,
        )


if __name__ == "__main__":
    unittest.main()
