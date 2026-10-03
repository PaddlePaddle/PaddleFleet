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

import unittest

import paddle
from paddle.distributed import ShardedWeight

from paddlefleet.trainer.trainer import (
    _fused_expert_optimizer_save_views,
    restore_fused_expert_3d_layout,
)


class PipelineAliases(paddle.nn.Layer):
    checkpoint_key = "model.layers.3.mlp.grouped_gemm_experts.weight1"

    def __init__(self):
        super().__init__()
        expert = paddle.nn.Layer()
        expert.add_parameter(
            "weight1",
            self.create_parameter(
                [2, 4, 6],
                default_initializer=paddle.nn.initializer.Constant(0.5),
            ),
        )
        self.shared_layers = paddle.nn.LayerDict({"experts": expert})
        self.add_sublayer("4", expert)
        self._pipeline_name_mapping = {self.checkpoint_key: "4.weight1"}

    def shard(self):
        param = self.shared_layers["experts"].weight1
        flat = param.reshape([8, 6])
        flat.name = param.name
        return ShardedWeight(self.checkpoint_key, flat, (8, 6), (16, 6), (8, 0))


class FusedExpertAliasTests(unittest.TestCase):
    def test_shared_pipeline_alias_preserves_expert_values_and_ep_coordinates(
        self,
    ):
        model = PipelineAliases()
        param = model.shared_layers["experts"].weight1
        self.assertNotIn("4.weight1", dict(model.named_parameters()))
        shard = model.shard()

        restore_fused_expert_3d_layout(model, {model.checkpoint_key: shard})

        self.assertEqual(shard.local_shape, (2, 4, 6))
        self.assertEqual(shard.global_shape, (4, 4, 6))
        self.assertEqual(shard.global_offset, (2, 0, 0))
        self.assertTrue(paddle.equal_all(shard.local_tensor, param).item())
        self.assertEqual(tuple(param.shape), (2, 4, 6))

    def test_optimizer_alias_views_restore_live_moments_after_save_failure(
        self,
    ):
        model = PipelineAliases()
        param = model.shared_layers["experts"].weight1
        optimizer = paddle.optimizer.AdamW(
            learning_rate=0.01, parameters=model.parameters()
        )
        (param**2).sum().backward()
        optimizer.step()
        optimizer.clear_grad()
        moments = [
            (mapping, mapping[param.name])
            for mapping in optimizer._accumulators.values()
            if param.name in mapping and mapping[param.name].ndim == 3
        ]
        self.assertEqual(len(moments), 2)
        expected = [tensor.clone() for _, tensor in moments]
        shard = model.shard()

        with (
            self.assertRaisesRegex(RuntimeError, "save failed"),
            _fused_expert_optimizer_save_views(
                model, {model.checkpoint_key: shard}, optimizer
            ),
        ):
            for mapping, tensor in moments:
                self.assertEqual(tuple(mapping[param.name].shape), (8, 6))
                self.assertIsNot(mapping[param.name], tensor)
                self.assertEqual(tuple(tensor.shape), (2, 4, 6))
            raise RuntimeError("save failed")

        for (mapping, tensor), value in zip(moments, expected):
            self.assertIs(mapping[param.name], tensor)
            self.assertTrue(paddle.equal_all(tensor, value).item())


if __name__ == "__main__":
    unittest.main()
