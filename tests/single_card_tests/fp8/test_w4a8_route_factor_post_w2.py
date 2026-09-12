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

"""Tests for applying routed_scaling_factor after the W4A8 down projection.

Three things are worth locking down, in decreasing order of how badly they bite:

1. The router-side skip and the expert-side apply must be driven by the *same*
   predicate, and every fold site must honour it. ``StandardMoERouter._hash_routing``
   folds and then returns early, so it never reaches the fold in
   ``TopKRouter.forward``; missing it leaves the first ``moe_n_hash_layers`` MoE layers
   with outputs scaled down by ``routed_scaling_factor``.
2. The backward has to scale ``out_grad`` before it fans out to the dgrad and the w2
   wgrad. Scaling one but not the other leaves that gradient short by the factor --
   training still runs and the loss still looks sane.
3. Paths without a post-w2 apply point must reject the scale rather than silently
   drop it.
"""

import ast
import inspect
import unittest
from types import SimpleNamespace

from paddlefleet.transformer.moe import moe_router
from paddlefleet.transformer.moe.fp8_utils import ExpertsGroupGemmContiguousNode
from paddlefleet.transformer.transformer_config import (
    w4a8_route_factor_post_w2_scale,
)


def _cfg(**kwargs):
    base = dict(
        use_w4a8=True,
        use_w4a8_fused_quant=True,
        routed_scaling_factor=1.5,
        routed_scaling_factor_learnable=False,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestPostW2ScalePredicate(unittest.TestCase):
    def test_enabled_returns_scalar(self):
        self.assertEqual(w4a8_route_factor_post_w2_scale(_cfg()), 1.5)

    def test_requires_w4a8(self):
        self.assertIsNone(w4a8_route_factor_post_w2_scale(_cfg(use_w4a8=False)))

    def test_requires_fused_quant_switch(self):
        self.assertIsNone(
            w4a8_route_factor_post_w2_scale(_cfg(use_w4a8_fused_quant=False))
        )

    def test_learnable_factor_is_excluded(self):
        # A learnable factor is a per-(token, expert) gather, not a scalar that can
        # be hoisted past the grouped GEMM.
        self.assertIsNone(
            w4a8_route_factor_post_w2_scale(
                _cfg(routed_scaling_factor_learnable=True)
            )
        )

    def test_unit_factor_is_a_noop(self):
        self.assertIsNone(
            w4a8_route_factor_post_w2_scale(_cfg(routed_scaling_factor=1.0))
        )

    def test_tolerates_configs_without_the_fields(self):
        self.assertIsNone(w4a8_route_factor_post_w2_scale(SimpleNamespace()))


class TestEveryFoldSiteIsGated(unittest.TestCase):
    """Each `top_gate * self.routed_scaling_factor` must sit behind the predicate.

    Guards against the failure that actually happened: only the fold in
    ``TopKRouter.forward`` was gated, while ``_hash_routing`` kept folding, so on a
    model with ``moe_n_hash_layers > 0`` the first MoE layers double-counted the skip.
    """

    def test_all_fold_sites_check_the_predicate(self):
        tree = ast.parse(inspect.getsource(moe_router))
        folds = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)):
                continue
            if not isinstance(node.value.op, ast.Mult):
                continue
            right = node.value.right
            if isinstance(right, ast.Attribute) and right.attr == "routed_scaling_factor":
                folds.append(node.lineno)
        self.assertTrue(folds, "no routed_scaling_factor fold found; test is stale")

        gated = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            calls = [
                c
                for c in ast.walk(node.test)
                if isinstance(c, ast.Call)
                and getattr(c.func, "id", None) == "w4a8_route_factor_post_w2_scale"
            ]
            if not calls:
                continue
            body_lines = {
                x.lineno for stmt in node.body for x in ast.walk(stmt)
                if hasattr(x, "lineno")
            }
            gated.extend(body_lines)

        ungated = [ln for ln in folds if ln not in gated]
        self.assertEqual(
            ungated,
            [],
            f"routed_scaling_factor folded without checking the predicate at lines {ungated}",
        )


class TestBackwardScalesBeforeFanOut(unittest.TestCase):
    def test_out_grad_is_scaled_before_any_consumer(self):
        tree = ast.parse(inspect.getsource(ExpertsGroupGemmContiguousNode))
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "backward_impl_fp8"
        )
        scale_lines = [
            c.lineno
            for c in ast.walk(fn)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "scale_"
            and getattr(c.func.value, "id", None) == "out_grad"
        ]
        self.assertEqual(len(scale_lines), 1, "expected exactly one out_grad.scale_")
        reads = sorted(
            {
                c.lineno
                for c in ast.walk(fn)
                if isinstance(c, ast.Name)
                and c.id == "out_grad"
                and isinstance(c.ctx, ast.Load)
                and c.lineno != scale_lines[0]
            }
        )
        early = [ln for ln in reads if ln < scale_lines[0]]
        self.assertEqual(
            early,
            [],
            "out_grad is consumed before it is scaled, so that consumer's gradient "
            f"is short by routed_scaling_factor (lines {early})",
        )

    def test_scaling_is_in_place(self):
        # out_grad is reused as the dx output buffer and the subbatch path asserts
        # `tmp_dx is tmp_out_grad`; rebinding it would break that invariant. Note
        # paddle's `x *= s` also rebinds, so only scale_ is acceptable.
        src = inspect.getsource(ExpertsGroupGemmContiguousNode)
        self.assertNotIn("out_grad = out_grad *", src)


class TestUnsupportedPathIsRejected(unittest.TestCase):
    def test_non_w4a8_node_rejects_the_scale(self):
        with self.assertRaises(ValueError):
            ExpertsGroupGemmContiguousNode(
                SimpleNamespace(experts=[], grouped_gemm_experts=None),
                use_fp8_mlp=False,
                use_w4a8=False,
                w4a8_route_factor_post_w2=1.5,
            )


if __name__ == "__main__":
    unittest.main()
