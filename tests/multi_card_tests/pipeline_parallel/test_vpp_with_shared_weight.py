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

import random
import unittest
from dataclasses import dataclass

import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.nn import Layer

from paddlefleet.pipeline_parallel import (
    LayerDesc,
    NoPipelineParallel,
    PipelineLayer,
    PipelineParallelWithInterleave,
    SharedLayerDesc,
)
from paddlefleet.spec_utils import LayerSpec, build_layer
from paddlefleet.transformer.identity_op import IdentityOp

batch_size = 16
micro_batch_size = 2


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples=80, shape=(64, 256)):
        self.num_samples = num_samples
        self.shape = shape

    def __getitem__(self, idx):
        img = np.random.rand(*self.shape).astype("float32")
        label = np.random.randint(0, 10, size=(64,), dtype="int64")
        return img, label

    def __len__(self):
        return self.num_samples


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class SharedLinear(Layer):
    def __init__(self):
        super().__init__()
        self.shared_net = nn.Linear(256, 256)

    @property
    def shared_weight(self):
        return self.shared_net.weight

    def forward(self, hidden_states):
        outputs = self.shared_net(hidden_states)
        return outputs


class ClassifyPipe(Layer):
    def __init__(self):
        super().__init__()
        self.classify_net = nn.Linear(256, 10)

    def forward(self, hidden_states):
        outputs = self.classify_net(hidden_states)
        return outputs


@dataclass
class SimpleNetLayerSpec:
    features: list[LayerSpec] | list[IdentityOp]
    shared: LayerSpec | type = IdentityOp
    classifier: LayerSpec | type = IdentityOp


class SimpleNet(PipelineLayer):
    def __init__(self, sublayers_spec: SimpleNetLayerSpec, **kwargs):
        self.layers = SimpleNet.get_layer_desc_list(sublayers_spec)

        super().__init__(layers=self.layers, **kwargs)

    @staticmethod
    def get_layer_desc_list(spec: SimpleNetLayerSpec):
        def _logits_helper(linear, output):
            return paddle.matmul(output, linear.shared_weight)

        layers = []
        layers.append(
            SharedLayerDesc(
                "shared",
                spec.shared,
                shared_weight_attr="shared_weight",
            )
        )
        for features_spec in spec.features:
            layers.append(LayerDesc(features_spec))
        layers.append(
            SharedLayerDesc(
                "shared",
                spec.shared,
                forward_func=_logits_helper,
                shared_weight_attr="shared_weight",
            )
        )
        layers.append(LayerDesc(spec.classifier))
        return layers


def get_simple_spec(num_classes=10):
    spec = LayerSpec(
        layer=SimpleNet,
        sublayers_spec=SimpleNetLayerSpec(
            shared=LayerSpec(layer=SharedLinear),
            features=[
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
            ],
            classifier=LayerSpec(
                layer=ClassifyPipe,
            ),
        ),
        extra_kwargs={
            "loss_fn": nn.CrossEntropyLoss(),
        },
    )
    return spec


class TestDistVppTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 4
        self.num_virtual_pipeline_stages = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        strategy.hybrid_configs["pp_configs"].sync_moment = True
        strategy.hybrid_configs["pp_configs"].sync_param = True
        self.strategy = strategy
        fleet.init(is_collective=True, strategy=strategy)

    def test_vpp_model(self):
        hcg = fleet.get_hybrid_communicate_group()

        set_random_seed(1024)
        simple_spec = get_simple_spec()

        nopp_model = build_layer(simple_spec, num_stages=1)
        nopp_model = NoPipelineParallel(nopp_model, self.strategy)
        nopp_scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        nopp_optimizer = paddle.optimizer.SGD(
            learning_rate=nopp_scheduler, parameters=nopp_model.parameters()
        )

        seg_method = "layer:Linear"
        vpp_model = build_layer(
            simple_spec,
            topology=hcg.topology(),
            seg_method=seg_method,
            num_stages=self.pipeline_parallel_size,
            num_virtual_pipeline_stages=self.num_virtual_pipeline_stages,
        )

        vpp_scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2, 3, 4], values=[0.01, 0.02, 0.03, 0.04], verbose=True
        )
        vpp_optimizer = paddle.optimizer.SGD(
            learning_rate=vpp_scheduler, parameters=vpp_model.parameters()
        )
        vpp_model = PipelineParallelWithInterleave(
            vpp_model, hcg, self.strategy
        )
        vpp_optimizer = fleet.distributed_optimizer(vpp_optimizer)

        layer_name_proj = {
            "_layers.shared_layers.shared.shared_net.weight": "_layers.shared_layers.shared.shared_net.weight",
            "_layers.shared_layers.shared.shared_net.bias": "_layers.shared_layers.shared.shared_net.bias",
            "_layers.1.0.weight": "_layers.1.weight",
            "_layers.1.0.bias": "_layers.1.bias",
            "_layers.2.0.weight": "_layers.2.weight",
            "_layers.2.0.bias": "_layers.2.bias",
            "_layers.3.0.weight": "_layers.3.weight",
            "_layers.3.0.bias": "_layers.3.bias",
            "_layers.4.0.weight": "_layers.4.weight",
            "_layers.4.0.bias": "_layers.4.bias",
            "_layers.5.0.weight": "_layers.5.weight",
            "_layers.5.0.bias": "_layers.5.bias",
            "_layers.6.0.weight": "_layers.6.weight",
            "_layers.6.0.bias": "_layers.6.bias",
            "_layers.7.2.classify_net.weight": "_layers.9.classify_net.weight",
            "_layers.7.2.classify_net.bias": "_layers.9.classify_net.bias",
        }

        nopp_model_param = {}
        for name, param in nopp_model.named_parameters():
            nopp_model_param[name] = param

        for name, param in vpp_model.named_parameters():
            param.set_value(nopp_model_param[layer_name_proj[name]])

        train_loader = paddle.io.DataLoader(
            RandomDataset(),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )

        for step_id, data in enumerate(train_loader()):
            img = paddle.to_tensor(data[0])
            label = paddle.to_tensor(data[1])
            img.stop_gradient = True
            label.stop_gradient = True

            nopp_loss = nopp_model.train_batch(
                [img, label], nopp_optimizer, nopp_scheduler
            )

            vpp_loss = vpp_model.train_batch(
                [img, label], vpp_optimizer, vpp_scheduler
            )

            print("loss:", nopp_loss.numpy(), vpp_loss.numpy())
            np.testing.assert_allclose(
                nopp_loss.numpy(), vpp_loss.numpy(), rtol=1e-6, atol=1e-8
            )


if __name__ == "__main__":
    unittest.main()
