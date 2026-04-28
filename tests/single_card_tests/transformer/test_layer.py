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

import unittest

import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import (
    TransformerLayerSublayersSpec,
    TransformerLayerWithOverlap,
)


class DummyLayer(FleetLayer):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.linear = paddle.nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear(x)


class DummyMoEGate:
    norm_topk_prob = False


class DummyOverlapMoELayer(MoELayer):
    def __init__(self, config: TransformerConfig):
        paddle.nn.Layer.__init__(self)
        self.gate = DummyMoEGate()
        self.expert_model_parallel_size = 2
        self.moe_token_dispatcher_type = config.moe_token_dispatcher_type

    def set_layer_number(self, layer_number):
        self.layer_number = layer_number


class TestFleetLayer(unittest.TestCase):
    def setUp(self):
        transformer_config = TransformerConfig(
            num_hidden_layers=2, hidden_size=12, num_attention_heads=4
        )
        self.fleet_layer = DummyLayer(config=transformer_config)

    def test_fleet_layer(self):
        fleet_layer = self.fleet_layer
        assert fleet_layer
        assert fleet_layer.config.hidden_size == 12
        assert fleet_layer.linear.weight.dtype == paddle.float32

        x = paddle.ones((2, 2)).cuda()
        assert fleet_layer(x).dtype == paddle.float32


class TestTransformerLayerWithOverlap(unittest.TestCase):
    def test_hybrid_ep_dispatcher_is_allowed_for_overlap_layer(self):
        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=4,
            num_attention_heads=1,
            moe_token_dispatcher_type="hybridep",
        )
        layer = TransformerLayerWithOverlap(
            config,
            TransformerLayerSublayersSpec(mlp=LayerSpec(DummyOverlapMoELayer)),
            pg_collection=ProcessGroupCollection(),
        )

        self.assertEqual(layer.mlp.moe_token_dispatcher_type, "hybridep")


if __name__ == "__main__":
    unittest.main()
