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

"""Single-card coverage for MoEHybridParallelClipGrad.

The wrapper normally runs under a hybrid-parallel process group. Here the
communicate-groups object is stubbed with world size 1 everywhere and ``None``
groups, so every collective in ``_global_norm`` is skipped and the clipping
recipe can be exercised on one device.
"""

import unittest
from unittest.mock import patch

import paddle
from paddle.framework import core

from paddlefleet.utils import moe_hybrid_parallel_optimizer as mhpo
from paddlefleet.utils.moe_hybrid_parallel_optimizer import (
    MoEHybridParallelClipGrad,
)


class _StubHybridCommGroups:
    """Stand-in for ``HybridCommunicateGroup`` without a process group."""

    def __init__(
        self,
        moe_sharding_world_size=1,
        sharding_world_size=1,
        model_parallel_world_size=1,
        pipe_parallel_world_size=1,
        expert_parallel_group=None,
        moe_sharding_parallel_group=None,
    ):
        self._moe_sharding_world_size = moe_sharding_world_size
        self._sharding_world_size = sharding_world_size
        self._model_parallel_world_size = model_parallel_world_size
        self._pipe_parallel_world_size = pipe_parallel_world_size
        self._expert_parallel_group = expert_parallel_group
        self._moe_sharding_parallel_group = moe_sharding_parallel_group

    def get_moe_sharding_parallel_world_size(self):
        return self._moe_sharding_world_size

    def get_sharding_parallel_world_size(self):
        return self._sharding_world_size

    def get_model_parallel_world_size(self):
        return self._model_parallel_world_size

    def get_pipe_parallel_world_size(self):
        return self._pipe_parallel_world_size

    def get_expert_parallel_group(self):
        return self._expert_parallel_group

    def get_moe_sharding_parallel_group(self):
        return self._moe_sharding_parallel_group


class _StubClipGrad:
    """Stand-in for ``ClipGradByGlobalNorm`` with no nested ``_clip``."""

    def __init__(self, clip_norm=1.0):
        self.clip_norm = clip_norm


class _StubSelectedRowsGrad:
    """Gradient reporting the SELECTED_ROWS layout of a sparse update."""

    type = core.VarDesc.VarType.SELECTED_ROWS

    def __init__(self, dense):
        self.dense = dense
        self.dtype = dense.dtype

    def multiply_(self, coefficient):
        self.dense = self.dense * coefficient
        return self.dense


def _clip_grad(clip_norm=1.0, hcg=None):
    return MoEHybridParallelClipGrad(
        _StubClipGrad(clip_norm=clip_norm),
        _StubHybridCommGroups() if hcg is None else hcg,
    )


def _param_and_grad(values):
    grad = paddle.to_tensor(values, dtype="float32")
    param = paddle.create_parameter(shape=grad.shape, dtype="float32")
    return param, grad


class TestClipGradConstruction(unittest.TestCase):
    def test_moe_groups_come_from_hcg_for_hybrid_expert_parallel(self):
        expert_parallel_group = object()
        moe_sharding_parallel_group = object()

        clip_grad = _clip_grad(
            hcg=_StubHybridCommGroups(
                moe_sharding_world_size=2,
                expert_parallel_group=expert_parallel_group,
                moe_sharding_parallel_group=moe_sharding_parallel_group,
            )
        )

        self.assertIs(clip_grad.moe_group, expert_parallel_group)
        self.assertIs(clip_grad.moe_sharding_group, moe_sharding_parallel_group)

    def test_moe_groups_are_not_bound_without_moe_sharding(self):
        clip_grad = _clip_grad(
            hcg=_StubHybridCommGroups(moe_sharding_world_size=0)
        )

        self.assertFalse(hasattr(clip_grad, "moe_group"))
        self.assertFalse(hasattr(clip_grad, "moe_sharding_group"))


class TestGlobalNormLogging(unittest.TestCase):
    def _global_norm(self):
        clip_grad = _clip_grad()
        norms = [
            paddle.to_tensor([value], dtype="float32")
            for value in (1.0, 2.0, 3.0, 4.0)
        ]
        with patch.object(mhpo, "logger") as logger:
            clip_grad._global_norm(*norms)
        return logger, norms

    def test_norms_are_logged_before_and_after_the_reduction(self):
        logger, norms = self._global_norm()

        messages = [call.args[0] for call in logger.info.call_args_list]
        self.assertEqual(len(messages), 2)
        self.assertIn("before reduce", messages[0])
        self.assertIn("dist-moe-grad-norm=3.0", messages[0])
        self.assertIn("non-dist-moe-grad-norm=4.0", messages[0])
        self.assertIn("after reduce", messages[1])
        self.assertIn("dist-grad-norm=1.0", messages[1])
        self.assertIn("non-dist-grad-norm=2.0", messages[1])
        # Single card, no groups: nothing was reduced into the buckets.
        self.assertEqual(
            [float(norm.item()) for norm in norms], [1.0, 2.0, 3.0, 4.0]
        )


class TestDygraphClipCoefficient(unittest.TestCase):
    def test_stock_coefficient_shrinks_grads_below_threshold(self):
        values = [[0.5, 1.5], [2.0, 1.0]]
        param, grad = _param_and_grad(values)
        clip_grad = _clip_grad(clip_norm=10.0)

        clip_grad([(param, grad)])

        # clip_norm / (max(norm, clip_norm) + 1e-6) is just under one.
        scaled = grad.numpy().tolist()
        self.assertNotEqual(scaled, values)
        self.assertLess(scaled[0][0], 0.5)
        self.assertAlmostEqual(scaled[0][0], 0.5, places=6)


class TestSelectedRowsGradients(unittest.TestCase):
    def test_stock_mode_merges_selected_rows_and_clips(self):
        param, dense = _param_and_grad([[3.0, 4.0]])
        clip_grad = _clip_grad(clip_norm=1.0)
        grad = _StubSelectedRowsGrad(dense)

        with (
            patch.object(mhpo.clip, "merge_selected_rows", lambda g: g),
            patch.object(
                mhpo.clip,
                "get_tensor_from_selected_rows",
                lambda g: g.dense,
            ),
        ):
            clip_grad([(param, grad)])

        self.assertAlmostEqual(
            clip_grad.stat["global_grad_norm"], 5.0, places=5
        )
        scaled = grad.dense.numpy().tolist()[0]
        self.assertAlmostEqual(scaled[0], 0.6, places=5)
        self.assertAlmostEqual(scaled[1], 0.8, places=5)


if __name__ == "__main__":
    unittest.main()
