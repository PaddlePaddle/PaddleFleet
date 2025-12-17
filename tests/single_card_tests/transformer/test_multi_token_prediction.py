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
from functools import partial

import paddle
from paddle.distributed import fleet

from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_layers_spec,
)
from paddlefleet.parallel_state import initialize_model_parallel
from paddlefleet.pipeline_parallel import NoPipelineParallel
from paddlefleet.spec_utils import build_layer
from paddlefleet.transformer.multi_token_prediction import MTPLossLoggingHelper

strategy = fleet.DistributedStrategy()
fleet.init(is_collective=True, strategy=strategy)
hcg = fleet.get_hybrid_communicate_group()
hcg._cp_sharding_comm_group = hcg._sharding_comm_group
initialize_model_parallel(hcg)

_SEED = 42


class TestMultiTokenPredictionLayer(unittest.TestCase):
    def _create_config_and_mtp_layer_spec(self):
        config = GPTConfig(
            num_nextn_predict_layers=2,
            num_hidden_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            use_cpu_initialization=True,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(config)
        mtp_layers_spec = get_gpt_mtp_layers_spec(
            config=config, spec=[transformer_layer_spec]
        )
        assert len(mtp_layers_spec) == config.num_nextn_predict_layers
        return config, mtp_layers_spec[0]

    def test_constructor_local(self):
        """Test basic construction of MTP layer."""

        paddle.seed(_SEED)
        config, mtp_layer_spec = self._create_config_and_mtp_layer_spec()
        mtp = build_layer(mtp_layer_spec)

        assert mtp.enorm.weight.shape[0] == config.hidden_size
        assert mtp.hnorm.weight.shape[0] == config.hidden_size
        assert mtp.eh_proj.weight.shape[0] == config.hidden_size * 2
        assert mtp.eh_proj.weight.shape[1] == config.hidden_size
        assert mtp.transformer_layer is not None
        num_weights = sum([p.size for p in mtp.parameters()])
        assert num_weights == 57664


class TestMultiTokenPrediction(unittest.TestCase):
    seq_length = 32
    micro_batch_size = 2

    def model_provider(self):
        config = GPTConfig(
            num_hidden_layers=2,
            num_nextn_predict_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            mtp_loss_scaling_factor=0.1,
            init_method=partial(paddle.nn.init.xavier_uniform_, gain=1.0),
            output_layer_init_method=partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
        )
        model = gpt_builder(config, num_stages=1)
        gpt_pipe_model = NoPipelineParallel(model, strategy=strategy)
        return gpt_pipe_model

    def get_batch(self, seq_length, micro_batch_size):
        data = list(range(seq_length))
        input_ids = paddle.tensor(data, dtype="int64").repeat(
            (micro_batch_size, 1)
        )
        labels = 1 + paddle.tensor(data, dtype="int64").repeat(
            (micro_batch_size, 1)
        )
        position_ids = paddle.tensor(data, dtype="int64").repeat(
            (micro_batch_size, 1)
        )
        attention_mask = paddle.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype="bool"
        )
        loss_mask = paddle.ones(seq_length).repeat((micro_batch_size, 1))
        batch = (
            {
                "input_ids": [input_ids],
                "attention_mask": [attention_mask],
                "position_ids": [position_ids],
                "loss_mask": [loss_mask],
            },
            [labels],
        )
        return batch

    def test_forward_backward(self):
        """Test MTP forward and backward with gptmodel."""
        paddle.seed(_SEED)
        gpt_pipe_model = self.model_provider()
        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        loss = gpt_pipe_model.forward_backward_pipeline(batch)

        tracker = MTPLossLoggingHelper.tracker
        assert "values" in tracker
        MTPLossLoggingHelper.clean_loss_in_tracker()

        # Check output logits shapes
        assert loss[1].shape[0] == self.micro_batch_size
        assert loss[1].shape[1] == self.seq_length

        for name, param in gpt_pipe_model.named_parameters():
            assert param.grad is not None


# class TestMTPLossLoggingHelper(unittest.TestCase):
#     num_hidden_layers = 4
#
#     def setUp(self):
#         # Reset the tracker before each test
#         MTPLossLoggingHelper.tracker = {}
#
#     def test_save_loss_to_tracker(self):
#         """Test saving loss to tracker."""
#         # Create a dummy loss tensor
#         loss = paddle.tensor(1.3)
#         layer_number = 2
#         num_hidden_layers = self.num_hidden_layers
#
#         # Test saving loss
#         MTPLossLoggingHelper.save_loss_to_tracker(
#             loss=loss,
#             layer_number=layer_number,
#             num_hidden_layers=num_hidden_layers,
#         )
#
#         # Verify tracker state
#         assert "values" in MTPLossLoggingHelper.tracker
#         assert MTPLossLoggingHelper.tracker["values"].shape == [
#             num_hidden_layers
#         ]
#         assert MTPLossLoggingHelper.tracker["values"][layer_number] == loss
#         assert MTPLossLoggingHelper.tracker["reduce_group"] is None
#         assert MTPLossLoggingHelper.tracker["avg_group"] is None
#
#     def test_track_mtp_metrics(self):
#         """Test tracking MTP metrics."""
#         # First save some losses
#         loss = paddle.tensor(2.3)
#         num_hidden_layers = self.num_hidden_layers
#         for i in range(num_hidden_layers):
#             MTPLossLoggingHelper.save_loss_to_tracker(
#                 loss=loss, layer_number=i, num_hidden_layers=num_hidden_layers
#             )
#
#         # Create dummy writer and loss dict
#         class DummyWriter:
#             def add_scalar(self, name, value, iteration):
#                 pass
#
#         class DummyWandBWriter:
#             def log(self, metrics, iteration):
#                 pass
#
#         loss_scale = 1.5
#         iteration = 2
#         writer = DummyWriter()
#         wandb_writer = DummyWandBWriter()
#         total_loss_dict = {}
#
#         # Test tracking metrics
#         MTPLossLoggingHelper.track_mtp_metrics(
#             loss_scale=loss_scale,
#             iteration=iteration,
#             writer=writer,
#             wandb_writer=wandb_writer,
#             total_loss_dict=total_loss_dict,
#         )
#
#         # Verify total_loss_dict is populated
#         for i in range(num_hidden_layers):
#             assert f"mtp_{i + 1} loss" in total_loss_dict
#             assert total_loss_dict[f"mtp_{i + 1} loss"] == loss * loss_scale
#
#         # Verify tracker is cleaned
#         assert paddle.all(MTPLossLoggingHelper.tracker["values"] == 0)
#         assert MTPLossLoggingHelper.tracker["reduce_group"] is None
#         assert MTPLossLoggingHelper.tracker["avg_group"] is None


if __name__ == "__main__":
    unittest.main()
