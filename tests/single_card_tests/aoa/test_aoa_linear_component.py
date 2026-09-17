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
# Scope: the Linear family (``Linear`` / ``ColumnParallelLinear`` /
# ``RowParallelLinear``) share a single module-level helper
# ``gen_linear_aoa_statements`` / ``gen_linear_inv_aoa_statements``. This test
# pins the coverage contract: ``weight`` carries ``^T``; bias / buffers use
# identity only when their resolved names differ.
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


def _ctx(*, checkpoint_name_prefix="hf", model_name_prefix="model"):
    return AOAContext(
        config=None,
        checkpoint_name_prefix=checkpoint_name_prefix,
        pp_to_single_mapping={},
        checkpoint_name_mapping={},
        model_name_prefix=model_name_prefix,
    )


def _make_linear_leaf(bias=False, buffer=False):
    """Binds the Linear family AOA helpers onto a controllable stand-in.

    The real ``Linear`` needs a live TP process group to construct, so we bind
    the two override methods (which just forward to the module-level helpers)
    onto a lightweight ``paddle.nn.Layer`` carrying real params / buffers.
    """
    from paddlefleet.tensor_parallel.layers import Linear

    class _LinearLeaf(paddle.nn.Layer):
        gen_aoa_statements = Linear.gen_aoa_statements
        gen_inv_aoa_statements = Linear.gen_inv_aoa_statements

        def __init__(self):
            super().__init__()
            self.weight = self.create_parameter(shape=[2, 2])
            if bias:
                self.bias = self.create_parameter(shape=[2])
            if buffer:
                # A persistable buffer must appear in state_dict and therefore
                # get an identity statement (no orphan target).
                self.register_buffer(
                    "some_buf", paddle.zeros([2]), persistable=True
                )
                # A non-persistable buffer must NOT appear (no statement).
                self.register_buffer(
                    "tmp_buf", paddle.zeros([2]), persistable=False
                )

    return _LinearLeaf


_PFX = "model.layers.0.mlp.down_proj."
_W_CK = "hf.layers.0.mlp.down_proj.weight"
_W_MD = "model.layers.0.mlp.down_proj.weight"


class TestLinearClassContract(unittest.TestCase):
    def test_three_linear_classes_override(self):
        from paddlefleet.tensor_parallel.layers import (
            ColumnParallelLinear,
            Linear,
            RowParallelLinear,
        )

        for cls in (Linear, ColumnParallelLinear, RowParallelLinear):
            self.assertIn("gen_aoa_statements", cls.__dict__)
            self.assertIn("gen_inv_aoa_statements", cls.__dict__)


class TestLinearForward(unittest.TestCase):
    def test_weight_transposed_bias_and_buffer_identity(self):
        model = _make_linear_leaf(bias=True, buffer=True)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        # weight is transposed
        self.assertIn(f"{_W_CK}^T -> {_W_MD}", stmts)
        # bias is identity (no ^T)
        self.assertIn(
            "hf.layers.0.mlp.down_proj.bias -> "
            "model.layers.0.mlp.down_proj.bias",
            stmts,
        )
        # persistable buffer is identity, non-persistable buffer is absent
        self.assertIn(
            "hf.layers.0.mlp.down_proj.some_buf -> "
            "model.layers.0.mlp.down_proj.some_buf",
            stmts,
        )
        self.assertFalse(any("tmp_buf" in s for s in stmts))

    def test_no_orphan_targets(self):
        model = _make_linear_leaf(bias=True, buffer=True)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        # Every own state_dict key (weight/bias/some_buf) has exactly one
        # producing statement; tmp_buf (non-persistable) has none.
        self.assertEqual(len(stmts), 3)

    def test_bias_absent_when_not_present(self):
        model = _make_linear_leaf(bias=False, buffer=False)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        self.assertEqual(stmts, [f"{_W_CK}^T -> {_W_MD}"])

    def test_same_name_identity_is_omitted(self):
        model = _make_linear_leaf(bias=True, buffer=True)()
        ctx = _ctx(checkpoint_name_prefix="model", model_name_prefix="model")
        stmts = model.gen_aoa_statements(ctx, structured_name_prefix=_PFX)
        self.assertEqual(stmts, [f"{_W_MD}^T -> {_W_MD}"])


class TestLinearInverse(unittest.TestCase):
    def test_weight_transposed_back_endpoints_swapped(self):
        model = _make_linear_leaf(bias=True, buffer=True)()
        stmts = model.gen_inv_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        self.assertIn(f"{_W_MD}^T -> {_W_CK}", stmts)
        self.assertIn(
            "model.layers.0.mlp.down_proj.bias -> "
            "hf.layers.0.mlp.down_proj.bias",
            stmts,
        )
        self.assertIn(
            "model.layers.0.mlp.down_proj.some_buf -> "
            "hf.layers.0.mlp.down_proj.some_buf",
            stmts,
        )
        self.assertEqual(len(stmts), 3)


if __name__ == "__main__":
    unittest.main()
