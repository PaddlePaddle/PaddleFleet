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
# Scope: the base ``SelfAttention`` PRE-FUSED qkv checkpoint layout, selected by
# the AOA flag ``ctx.qkv_checkpoint_prefused`` (declared per-model, e.g. the
# Qwen3-VL vision tower's ``aoa_qkv_checkpoint_prefused = True``).
# The default (``False``) layout fuses separate checkpoint q/k/v keys into the
# model ``qkv_proj`` via ``fused_qkv``. The pre-fused layout instead splits a
# single checkpoint ``qkv`` tensor into AOA-internal q/k/v temporaries and then
# re-fuses per head (vision has no GQA, so
# ``num_heads == num_key_value_groups == config.num_attention_heads``).
#
# The head helpers ``_gen_(inv_)prefused_qkv_head_aoa_statements`` read
# ``self.config.num_attention_heads`` and ``self.qkv_proj.bias``, so we bind them
# onto a light probe carrying just those two attributes and golden-test the
# emitted statement list. We also assert the flag-routing contract on the real
# ``SelfAttention`` class.
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

from paddlefleet.models.gpt.aoa_generator import (
    FleetAOAContext,
)


def _ctx(pp_to_single_mapping):
    return FleetAOAContext(
        config=None,
        checkpoint_name_prefix="hf",
        pp_to_single_mapping=pp_to_single_mapping,
        checkpoint_name_mapping={},
        dtype_cast_rules={},
        model_name_prefix="model",
        excluded_names=frozenset(),
        qkv_checkpoint_prefused=True,
    )


_PFX = "layers.0.self_attn."
_PP = {
    _PFX + "qkv_proj.weight": "model.layers.0.self_attn.qkv_proj.weight",
    _PFX + "qkv_proj.bias": "model.layers.0.self_attn.qkv_proj.bias",
}
_QKV_MODEL_W = "model.layers.0.self_attn.qkv_proj.weight"
_QKV_MODEL_B = "model.layers.0.self_attn.qkv_proj.bias"
_QKV_CKPT_W = "hf.layers.0.self_attn.qkv.weight"
_QKV_CKPT_B = "hf.layers.0.self_attn.qkv.bias"
_NH = 4


def _make_probe(bias):
    """A light stand-in carrying the two live attributes the base pre-fused head
    helpers read (``config.num_attention_heads`` and ``qkv_proj.bias``); the
    helpers are borrowed off the real ``SelfAttention`` class."""
    from paddlefleet.transformer.attention import SelfAttention

    class _Probe:
        _gen_prefused_qkv_head_aoa_statements = (
            SelfAttention._gen_prefused_qkv_head_aoa_statements
        )
        _gen_inv_prefused_qkv_head_aoa_statements = (
            SelfAttention._gen_inv_prefused_qkv_head_aoa_statements
        )

        def __init__(self):
            self.config = types.SimpleNamespace(num_attention_heads=_NH)
            self.qkv_proj = types.SimpleNamespace(
                bias=object() if bias else None
            )

    return _Probe()


class TestPrefusedForward(unittest.TestCase):
    """Forward (checkpoint -> model): equal-split the single fused ``qkv`` into
    q/k/v temporaries, then re-fuse (with transpose) into the model single. No
    GQA -> ``num_heads == num_key_value_groups``."""

    def setUp(self):
        self.ctx = _ctx(_PP)

    def test_weight_golden(self):
        stmts = _make_probe(bias=False)._gen_prefused_qkv_head_aoa_statements(
            self.ctx, _PFX, None
        )
        q_t = f"{_QKV_CKPT_W}._vis_q"
        k_t = f"{_QKV_CKPT_W}._vis_k"
        v_t = f"{_QKV_CKPT_W}._vis_v"
        expected = [
            f"{_QKV_CKPT_W} -> {q_t}, {k_t}, {v_t}, axis=0",
            f"{q_t}^T, {k_t}^T, {v_t}^T -> {_QKV_MODEL_W}, fused_qkv, "
            f"num_heads={_NH}, num_key_value_groups={_NH}",
        ]
        self.assertEqual(stmts, expected)

    def test_bias_appended_axis0_no_transpose(self):
        stmts = _make_probe(bias=True)._gen_prefused_qkv_head_aoa_statements(
            self.ctx, _PFX, None
        )
        qb = f"{_QKV_CKPT_B}._vis_q"
        kb = f"{_QKV_CKPT_B}._vis_k"
        vb = f"{_QKV_CKPT_B}._vis_v"
        self.assertEqual(len(stmts), 4)
        self.assertEqual(stmts[2], f"{_QKV_CKPT_B} -> {qb}, {kb}, {vb}, axis=0")
        self.assertEqual(
            stmts[3],
            f"{qb}, {kb}, {vb} -> {_QKV_MODEL_B}, fused_qkv, "
            f"num_heads={_NH}, num_key_value_groups={_NH}, axis=0",
        )
        self.assertNotIn("^T", stmts[3])


class TestPrefusedInverse(unittest.TestCase):
    """Inverse (model -> checkpoint): split the model single into q/k/v
    (fused_qkv) then merge (with transpose) back into the fused checkpoint.
    Independently generated."""

    def setUp(self):
        self.ctx = _ctx(_PP)

    def test_weight_golden(self):
        stmts = _make_probe(
            bias=False
        )._gen_inv_prefused_qkv_head_aoa_statements(self.ctx, _PFX, None)
        q_t = f"{_QKV_CKPT_W}._vis_q"
        k_t = f"{_QKV_CKPT_W}._vis_k"
        v_t = f"{_QKV_CKPT_W}._vis_v"
        expected = [
            f"{_QKV_MODEL_W} -> {q_t}, {k_t}, {v_t}, fused_qkv, "
            f"num_heads={_NH}, num_key_value_groups={_NH}",
            f"{q_t}^T, {k_t}^T, {v_t}^T -> {_QKV_CKPT_W}, axis=0",
        ]
        self.assertEqual(stmts, expected)


class TestFlagRoutingContract(unittest.TestCase):
    """The base ``SelfAttention`` owns the pre-fused head helpers and its qkv
    head dispatcher routes to them only when ``ctx.qkv_checkpoint_prefused`` is
    set (no AOA-only subclass involved)."""

    def test_base_has_prefused_helpers(self):
        from paddlefleet.transformer.attention import SelfAttention

        self.assertIn(
            "_gen_prefused_qkv_head_aoa_statements", SelfAttention.__dict__
        )
        self.assertIn(
            "_gen_inv_prefused_qkv_head_aoa_statements",
            SelfAttention.__dict__,
        )

    def test_dispatcher_branches_on_flag(self):
        import inspect

        from paddlefleet.transformer.attention import SelfAttention

        fwd = inspect.getsource(SelfAttention._gen_qkv_head_aoa_statements)
        inv = inspect.getsource(SelfAttention._gen_inv_qkv_head_aoa_statements)
        self.assertIn("ctx.qkv_checkpoint_prefused", fwd)
        self.assertIn("_gen_prefused_qkv_head_aoa_statements", fwd)
        self.assertIn("ctx.qkv_checkpoint_prefused", inv)
        self.assertIn("_gen_inv_prefused_qkv_head_aoa_statements", inv)


if __name__ == "__main__":
    unittest.main()
