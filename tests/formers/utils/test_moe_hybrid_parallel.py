# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Behaviour tests for ``MoEHybridParallelClipGrad``.

Scope (无卡 / world-size==1):
* ``__init__`` moe-group *selection* driven by the reported world sizes,
* the real attribute delegation used by ``_comm_and_clip`` (``self.clip_norm``),
* the HF-vs-paddle formula selection performed by ``unwrap_hf_bitexact_clip``
  over the *nested* wrapper graph that the optimizer actually builds.

The full ``_dygraph_clip`` numeric path and the ``MoEHybridParallelOptimizer``
construction depend on a real Fleet hybrid topology / process group and are
marked skipped or expectedFailure below with the reason -- a fabricated
``world_size`` plus a mocked collective is not multi-card proof (see the
distributed-training rules, antipattern #13).
"""

import unittest

import paddle
from paddle.nn import ClipGradByGlobalNorm

from paddlefleet.utils.hf_bitexact_clip import HFBitexactClipGradByGlobalNorm
from paddlefleet.utils.moe_hybrid_parallel_optimizer import (
    MoEHybridParallelClipGrad,
    unwrap_hf_bitexact_clip,
)


class _FakeHCG:
    """Minimal Fleet-topology stand-in.

    Only reports world sizes / group handles; it performs no communication, so
    it exercises the *selection* branches of ``MoEHybridParallelClipGrad`` and
    nothing that would require a real process group.
    """

    def __init__(
        self,
        *,
        moe_sharding_ws=0,
        sharding_ws=1,
        mp_ws=1,
        pp_ws=1,
        expert_group=None,
        moe_sharding_group=None,
    ):
        self._moe_sharding_ws = moe_sharding_ws
        self._sharding_ws = sharding_ws
        self._mp_ws = mp_ws
        self._pp_ws = pp_ws
        self._expert_group = expert_group
        self._moe_sharding_group = moe_sharding_group

    def get_moe_sharding_parallel_world_size(self):
        return self._moe_sharding_ws

    def get_sharding_parallel_world_size(self):
        return self._sharding_ws

    def get_model_parallel_world_size(self):
        return self._mp_ws

    def get_pipe_parallel_world_size(self):
        return self._pp_ws

    def get_expert_parallel_group(self):
        return self._expert_group

    def get_moe_sharding_parallel_group(self):
        return self._moe_sharding_group


class TestClipGradInitGroupSelection(unittest.TestCase):
    def test_no_moe_sharding_leaves_moe_groups_unset(self):
        clip = ClipGradByGlobalNorm(1.0)
        hcg = _FakeHCG(moe_sharding_ws=0)

        clip_grad = MoEHybridParallelClipGrad(clip, hcg)

        # vars() reads the instance __dict__ directly, so __getattr__ (which
        # would otherwise forward to the wrapped clip) cannot fabricate these.
        self.assertNotIn("moe_group", vars(clip_grad))
        self.assertNotIn("moe_sharding_group", vars(clip_grad))
        self.assertIs(clip_grad._clip, clip)
        self.assertIs(clip_grad._hcg, hcg)
        self.assertEqual(clip_grad.processed_steps, 0)
        self.assertEqual(clip_grad.stat, {})

    def test_moe_sharding_binds_expert_and_sharding_groups_unswapped(self):
        expert_group = object()
        moe_sharding_group = object()
        clip = ClipGradByGlobalNorm(1.0)
        hcg = _FakeHCG(
            moe_sharding_ws=2,
            expert_group=expert_group,
            moe_sharding_group=moe_sharding_group,
        )

        clip_grad = MoEHybridParallelClipGrad(clip, hcg)

        # Distinct sentinels so a swapped assignment is caught.
        self.assertIs(clip_grad.moe_group, expert_group)
        self.assertIs(clip_grad.moe_sharding_group, moe_sharding_group)


class TestClipGradAttributeDelegation(unittest.TestCase):
    def test_getattr_returns_real_clip_norm(self):
        # clip_norm is consumed by _comm_and_clip as self.clip_norm; delegation
        # must surface the wrapped clip's real value, not a placeholder.
        clip = ClipGradByGlobalNorm(2.5)
        clip_grad = MoEHybridParallelClipGrad(clip, _FakeHCG())
        self.assertEqual(clip_grad.clip_norm, 2.5)

    def test_getattr_missing_attribute_raises(self):
        clip = ClipGradByGlobalNorm(1.0)
        clip_grad = MoEHybridParallelClipGrad(clip, _FakeHCG())
        with self.assertRaises(AttributeError):
            clip_grad.definitely_missing_attr_xyz


class TestHFvsPaddleFormulaSelection(unittest.TestCase):
    """``unwrap_hf_bitexact_clip`` decides HF vs paddle global-norm recipe.

    ``MoEHybridParallelOptimizer`` wraps the inner grad-clip once for the
    optimizer and then wraps *that* again per param-group, so a group's clip is
    ``MoEHybridParallelClipGrad(MoEHybridParallelClipGrad(HFBitexact...))``.
    A one-level ``_clip`` check would miss it and silently revert to paddle's
    formula, so the nesting is reproduced here with the real objects.
    """

    def test_unwrap_finds_hf_clip_through_double_wrap(self):
        hf = HFBitexactClipGradByGlobalNorm(1.0)
        hcg = _FakeHCG()
        inner = MoEHybridParallelClipGrad(hf, hcg)
        outer = MoEHybridParallelClipGrad(inner, hcg)

        self.assertIs(unwrap_hf_bitexact_clip(outer), hf)
        self.assertIs(unwrap_hf_bitexact_clip(inner), hf)
        self.assertIs(unwrap_hf_bitexact_clip(hf), hf)

    def test_unwrap_returns_none_for_plain_paddle_clip_chain(self):
        plain = ClipGradByGlobalNorm(1.0)
        wrapped = MoEHybridParallelClipGrad(plain, _FakeHCG())
        # No HF clip anywhere in the chain -> paddle formula must be selected.
        self.assertIsNone(unwrap_hf_bitexact_clip(wrapped))
        self.assertIsNone(unwrap_hf_bitexact_clip(plain))


class TestDygraphClipWorldSizeOne(unittest.TestCase):
    @unittest.expectedFailure
    def test_world_size_one_scales_grads_by_global_norm_coef(self):
        """world-size==1, non-MoE clip should scale grads by the paddle coef.

        Independent reference (paddle recipe, no HF clip):
            global_norm = sqrt(3^2 + 4^2) = 5
            coef = clip_norm / (max(norm, clip_norm) + 1e-6) = 1 / (5 + 1e-6)
            g' = g * coef

        Currently this raises ``AttributeError`` before reaching the scaling:
        ``_global_norm`` unconditionally reads ``self.moe_sharding_group``, but
        ``__init__`` only sets it when ``get_moe_sharding_parallel_world_size()
        > 0``. With world size 0 the attribute is unset, ``__getattr__``
        forwards to the wrapped ``ClipGradByGlobalNorm`` (which has no such
        attribute) and the lookup fails. Marked expectedFailure: it asserts the
        CORRECT behaviour and flips to an unexpected success if the bug is fixed
        (see report -- production is NOT modified here).
        """
        clip = ClipGradByGlobalNorm(1.0)
        hcg = _FakeHCG(moe_sharding_ws=0, sharding_ws=1, mp_ws=1, pp_ws=1)
        clip_grad = MoEHybridParallelClipGrad(clip, hcg)

        p = paddle.create_parameter(
            [2],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )
        g = paddle.to_tensor([3.0, 4.0], dtype="float32")

        result = clip_grad([(p, g)])

        coef = 1.0 / (5.0 + 1e-6)
        expected = [3.0 * coef, 4.0 * coef]
        (_, out_g) = result[0]
        self.assertEqual(len(result), 1)
        for actual, want in zip(out_g.numpy().tolist(), expected):
            self.assertAlmostEqual(actual, want, places=5)


class TestMoEHybridParallelOptimizerConstruction(unittest.TestCase):
    @unittest.skip(
        "MoEHybridParallelOptimizer.__init__ reads a real Fleet HCG "
        "(get_parallel_mode / *_world_size) and strategy.hybrid_configs, and "
        "the sharding-optimizer / grad-clip-wrapping selection it performs "
        "(main_grad branch, ShardingOptimizer choice) only exercises a real "
        "hybrid topology. Belongs to a multi-card/fleet job; a fabricated HCG "
        "would only prove attribute plumbing, not the wrapping decision."
    )
    def test_wraps_grad_clip_in_moe_clip_when_not_dp_mode(self):
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
