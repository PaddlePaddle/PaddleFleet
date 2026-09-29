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
# Scope: the shared gated-attention qkv AOA helper ``gated_qkv_aoa``, which the
# base ``SelfAttention`` qkv head dispatcher routes to when
# ``self.gated_attention`` and the (non-experimental) gate is fused into
# ``qkv_proj``. The model side is a per-KV-group ``[Q, Gate, K, V]`` layout that
# ``fused_qkv`` cannot express (gate section between Q and K;
# ``head_dim != v_head_dim`` needs GCD sub-chunking). The checkpoint keeps the
# gate as its own ``gate_proj`` tensor (the ERNIE-Lite layout): Q/gate/k/v are
# four distinct keys.
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

import math
import unittest

import paddle  # noqa: F401  (kept for CI parity with sibling tests)
from paddle.distributed.flex_checkpoint.aoa.generation import AOAContext

from paddlefleet.transformer.gated_qkv_aoa import (
    _temp_names,
    gen_gated_qkv_aoa,
    gen_gated_qkv_inv_aoa,
)


def _ctx(pp_to_single_mapping):
    return AOAContext(
        config=None,
        checkpoint_name_prefix="hf",
        pp_to_single_mapping=pp_to_single_mapping,
        checkpoint_name_mapping={},
        model_name_prefix="model",
    )


_PFX = "layers.0.self_attn."
_PP = {
    _PFX + "qkv_proj.weight": "model.layers.0.self_attn.qkv_proj.weight",
    _PFX + "qkv_proj.bias": "model.layers.0.self_attn.qkv_proj.bias",
}
_W = "model.layers.0.self_attn.qkv_proj.weight"
_BIAS = "model.layers.0.self_attn.qkv_proj.bias"


def _checkpoint(local):
    return f"hf.layers.0.self_attn.{local}"


class _FakeQkv:
    def __init__(self, bias):
        self.bias = object() if bias else None


class _FakeAttn:
    """A light stand-in carrying the live dims the generators read off the
    attention module (head counts / head dims / ``gated_attention``) plus
    ``attn.qkv_proj.bias``."""

    def __init__(self, gated=True, hd=4, vhd=2, nh=4, nkv=2, bias=False):
        self.num_attention_heads = nh
        self.num_key_value_heads = nkv
        self.head_dim = hd
        self.v_head_dim = vhd
        self.gated_attention = gated
        self.qkv_proj = _FakeQkv(bias)


class TestTempNames(unittest.TestCase):
    """The pure name-builder: counts, no-duplication completeness, and the
    per-group ``[Q, Gate, K, V]`` fused ordering."""

    def _dims(self, hd, vhd, nh, nkv):
        gcd = math.gcd(hd, vhd)
        return hd // gcd, vhd // gcd, nh // nkv

    def test_counts_and_completeness_gated(self):
        q, g, k, v, fused = _temp_names(_W, 4, 2, 4, 2, True)
        hd_c, vhd_c, _ = self._dims(4, 2, 4, 2)
        self.assertEqual(len(q), 4 * hd_c)
        self.assertEqual(len(g), 4 * vhd_c)
        self.assertEqual(len(k), 2 * hd_c)
        self.assertEqual(len(v), 2 * vhd_c)
        allnames = q + g + k + v
        self.assertEqual(len(allnames), len(set(allnames)))
        self.assertEqual(sorted(fused), sorted(allnames))

    def test_non_gated_has_no_gate_section(self):
        q, g, k, v, fused = _temp_names(_W, 4, 2, 4, 2, False)
        self.assertEqual(g, [])
        self.assertEqual(sorted(fused), sorted(q + k + v))

    def test_fused_is_per_group_q_gate_k_v(self):
        q, g, k, v, fused = _temp_names(_W, 4, 2, 4, 2, True)
        hd_c, vhd_c, hpg = self._dims(4, 2, 4, 2)
        q_per, g_per, k_per, v_per = (
            hpg * hd_c,
            hpg * vhd_c,
            hd_c,
            vhd_c,
        )
        block = q_per + g_per + k_per + v_per
        for gi in range(2):
            seg = fused[gi * block : (gi + 1) * block]
            self.assertEqual(seg[:q_per], q[gi * q_per : (gi + 1) * q_per])
            self.assertEqual(
                seg[q_per : q_per + g_per], g[gi * g_per : (gi + 1) * g_per]
            )
            self.assertEqual(
                seg[q_per + g_per : q_per + g_per + k_per],
                k[gi * k_per : (gi + 1) * k_per],
            )
            self.assertEqual(
                seg[q_per + g_per + k_per :], v[gi * v_per : (gi + 1) * v_per]
            )

    def test_head_dim_neq_v_head_dim_chunks(self):
        q, g, k, v, fused = _temp_names(_W, 1, 1, 6, 4, True)
        # nh=1 nkv=1 hd=6 vhd=4 -> gcd 2 -> 3 vs 2 chunks per head.
        self.assertEqual(len(q), 1 * 3)
        self.assertEqual(len(g), 1 * 2)
        self.assertEqual(len(k), 1 * 3)
        self.assertEqual(len(v), 1 * 2)


class TestSeparateForward(unittest.TestCase):
    """Checkpoint q/gate/k/v are four distinct keys; forward splits each,
    regroups into the fused per-group layout, then transposes (weights) into
    the model single."""

    def setUp(self):
        self.ctx = _ctx(_PP)

    def test_weight_and_bias_golden(self):
        attn = _FakeAttn(gated=True, hd=4, vhd=2, nh=4, nkv=2, bias=True)
        q, g, k, v, fused = _temp_names(_W, 4, 2, 4, 2, True)
        qb, gb, kb, vb, fusedb = _temp_names(_BIAS, 4, 2, 4, 2, True)
        stmts = gen_gated_qkv_aoa(attn, self.ctx, _PFX)
        expected = [
            f"{_checkpoint('q_proj.weight')} -> {','.join(q)}, axis=0",
            f"{_checkpoint('gate_proj.weight')} -> {','.join(g)}, axis=0",
            f"{_checkpoint('k_proj.weight')} -> {','.join(k)}, axis=0",
            f"{_checkpoint('v_proj.weight')} -> {','.join(v)}, axis=0",
            f"{','.join(fused)} -> {_W}.qkv_fused_tmp, axis=0",
            f"{_W}.qkv_fused_tmp^T -> {_W}",
            f"{_checkpoint('q_proj.bias')} -> {','.join(qb)}, axis=0",
            f"{_checkpoint('gate_proj.bias')} -> {','.join(gb)}, axis=0",
            f"{_checkpoint('k_proj.bias')} -> {','.join(kb)}, axis=0",
            f"{_checkpoint('v_proj.bias')} -> {','.join(vb)}, axis=0",
            f"{','.join(fusedb)} -> {_BIAS}, axis=0",
        ]
        self.assertEqual(stmts, expected)

    def test_non_gated_omits_gate(self):
        attn = _FakeAttn(gated=False, hd=4, vhd=2, nh=4, nkv=2, bias=False)
        stmts = gen_gated_qkv_aoa(attn, self.ctx, _PFX)
        self.assertFalse(any("gate_proj" in s for s in stmts))


class TestInverse(unittest.TestCase):
    """Inverse (independently generated): transpose the fused single (weights),
    split along the per-group layout, then regroup each checkpoint tensor into
    the four separate keys."""

    def test_separate_weight_golden(self):
        ctx = _ctx(_PP)
        attn = _FakeAttn(gated=True, hd=4, vhd=2, nh=4, nkv=2, bias=False)
        q, g, k, v, fused = _temp_names(_W, 4, 2, 4, 2, True)
        stmts = gen_gated_qkv_inv_aoa(attn, ctx, _PFX)
        expected = [
            f"{_W}^T -> {_W}.qkv_fused_tmp",
            f"{_W}.qkv_fused_tmp -> {','.join(fused)}, axis=0",
            f"{','.join(q)} -> {_checkpoint('q_proj.weight')}, axis=0",
            f"{','.join(g)} -> {_checkpoint('gate_proj.weight')}, axis=0",
            f"{','.join(k)} -> {_checkpoint('k_proj.weight')}, axis=0",
            f"{','.join(v)} -> {_checkpoint('v_proj.weight')}, axis=0",
        ]
        self.assertEqual(stmts, expected)


class TestFlagRoutingContract(unittest.TestCase):
    """The base ``SelfAttention`` qkv head dispatcher routes gated attention to
    the shared helper (no AOA-only subclass involved)."""

    def test_dispatcher_uses_gated_helper(self):
        import inspect

        from paddlefleet.transformer.attention import SelfAttention

        fwd = inspect.getsource(SelfAttention._gen_qkv_head_aoa_statements)
        inv = inspect.getsource(SelfAttention._gen_inv_qkv_head_aoa_statements)
        self.assertIn("gated_attention", fwd)
        self.assertIn("gen_gated_qkv_aoa", fwd)
        self.assertIn("gated_attention", inv)
        self.assertIn("gen_gated_qkv_inv_aoa", inv)


if __name__ == "__main__":
    unittest.main()
