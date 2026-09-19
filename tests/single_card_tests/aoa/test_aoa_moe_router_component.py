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
# Scope: the MoE ``TopKRouter`` owns no fused layout. Its own
# ``local_state_dict`` (this layer only, no sub-layers) is the sole enumeration
# source, so every own Parameter / persistable buffer becomes a plain rename
# (``weight`` / ``e_score_correction_bias`` ...) with no ``^T``, and a dtype
# cast suffix appears only when the model rules match. This test pins that
# coverage contract and the independent inverse direction.
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


def _make_router_leaf():
    """Binds the router AOA overrides onto a controllable stand-in.

    The router enumerates only its own params/buffers, so the stand-in carries
    a ``weight`` Parameter, an ``e_score_correction_bias`` Parameter, one
    persistable buffer (must appear) and one non-persistable buffer (must be
    absent).
    """
    from paddlefleet.transformer.moe.moe_router import TopKRouter

    class _RouterLeaf(paddle.nn.Layer):
        gen_aoa_statements = TopKRouter.gen_aoa_statements
        gen_inv_aoa_statements = TopKRouter.gen_inv_aoa_statements

        def __init__(self):
            super().__init__()
            self.weight = self.create_parameter(shape=[2, 2])
            self.e_score_correction_bias = self.create_parameter(shape=[2])
            self.register_buffer(
                "some_buf", paddle.zeros([2]), persistable=True
            )
            self.register_buffer(
                "tmp_buf", paddle.zeros([2]), persistable=False
            )

    return _RouterLeaf


_PFX = "model.layers.0.mlp.gate."
_HF = "hf.layers.0.mlp.gate."
_MD = "model.layers.0.mlp.gate."


class TestRouterClassContract(unittest.TestCase):
    def test_override_present(self):
        from paddlefleet.transformer.moe.moe_router import TopKRouter

        self.assertIn("gen_aoa_statements", TopKRouter.__dict__)
        self.assertIn("gen_inv_aoa_statements", TopKRouter.__dict__)


class TestRouterForward(unittest.TestCase):
    def test_plain_rename_no_transpose(self):
        model = _make_router_leaf()()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        # every own key is a plain rename (no ^T)
        self.assertIn(f"{_HF}weight -> {_MD}weight", stmts)
        self.assertIn(
            f"{_HF}e_score_correction_bias -> {_MD}e_score_correction_bias",
            stmts,
        )
        self.assertIn(f"{_HF}some_buf -> {_MD}some_buf", stmts)
        self.assertFalse(any("^T" in s for s in stmts))
        # non-persistable buffer never appears
        self.assertFalse(any("tmp_buf" in s for s in stmts))
        # exactly three producing statements
        self.assertEqual(len(stmts), 3)


class TestRouterInverse(unittest.TestCase):
    def test_endpoints_swapped(self):
        model = _make_router_leaf()()
        stmts = model.gen_inv_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        self.assertIn(f"{_MD}weight -> {_HF}weight", stmts)
        self.assertIn(
            f"{_MD}e_score_correction_bias -> {_HF}e_score_correction_bias",
            stmts,
        )
        self.assertIn(f"{_MD}some_buf -> {_HF}some_buf", stmts)
        self.assertFalse(any("tmp_buf" in s for s in stmts))
        self.assertEqual(len(stmts), 3)


class TestRouterDtypeCast(unittest.TestCase):
    def test_weight_cast_endpoints(self):
        rules = {
            "layers.0.mlp.gate.weight": {
                "checkpoint_dtype": "bfloat16",
                "model_dtype": "float32",
            }
        }
        model = _make_router_leaf()()
        fwd = model.gen_aoa_statements(_ctx(rules), structured_name_prefix=_PFX)
        self.assertIn(
            f"{_HF}weight -> {_MD}weight, "
            "src_dtype='bfloat16', dst_dtype='float32'",
            fwd,
        )
        inv = model.gen_inv_aoa_statements(
            _ctx(rules), structured_name_prefix=_PFX
        )
        self.assertIn(
            f"{_MD}weight -> {_HF}weight, "
            "src_dtype='float32', dst_dtype='bfloat16'",
            inv,
        )


if __name__ == "__main__":
    unittest.main()
