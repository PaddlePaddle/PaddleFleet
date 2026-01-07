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

import functools
import unittest

import paddle
from paddle.distributed import fleet

from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)
from paddlefleet.models.gpt.gpt_model import GPTModel
from paddlefleet.parallel_state import initialize_model_parallel
from paddlefleet.transformer.multi_token_prediction import (
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

strategy = fleet.DistributedStrategy()
fleet.init(is_collective=True, strategy=strategy)
hcg = fleet.get_hybrid_communicate_group()
hcg._cp_sharding_comm_group = hcg._sharding_comm_group
initialize_model_parallel(hcg)

_SEED = 42


class TestMultiTokenPredictionLayer(unittest.TestCase):
    def _create_config_and_mtp_block_spec(self):
        config = TransformerConfig(
            num_nextn_predict_layers=2,
            num_hidden_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            use_cpu_initialization=True,
        )
        transformer_layer_spec = get_gpt_layer_local_spec()
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=transformer_layer_spec
        )
        return config, mtp_block_spec

    def test_constructor_local(self):
        """Test basic construction of MTP layer."""

        paddle.seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec()
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)

        assert isinstance(mtp, MultiTokenPredictionBlock)
        assert len(mtp.layers) == config.num_nextn_predict_layers
        for i in range(config.num_nextn_predict_layers):
            assert mtp.layers[i].layer_number == i + 1
            assert mtp.layers[i].enorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].hnorm.weight.shape[0] == config.hidden_size
            assert (
                mtp.layers[i].eh_proj.weight.shape[0] == config.hidden_size * 2
            )
            assert mtp.layers[i].eh_proj.weight.shape[1] == config.hidden_size
            assert mtp.layers[i].transformer_layer is not None
        num_weights = sum([p.size for p in mtp.parameters()])
        assert num_weights == 57664 * config.num_nextn_predict_layers


class TestMultiTokenPrediction(unittest.TestCase):
    seq_length = 32
    micro_batch_size = 2

    def model_provider(self):
        config = TransformerConfig(
            num_hidden_layers=2,
            num_nextn_predict_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            mtp_loss_scaling_factor=0.1,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            use_qk_norm=True,
            multi_latent_attention=False,
            normalization="RMSNorm",
        )
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=transformer_layer_spec
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=100,
            max_sequence_length=64,
            tie_word_embeddings=True,
            position_embedding_type="rope",
            mtp_block_spec=mtp_block_spec,
        )
        return model

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
        batch = {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return batch

    def test_forward_backward(self):
        """Test MTP forward and backward with gptmodel."""
        paddle.seed(_SEED)

        gpt_model = self.model_provider()

        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        tokens, labels, loss_mask, attention_mask, position_ids = batch.values()

        output = gpt_model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )

        tracker = MTPLossLoggingHelper.tracker
        assert "values" in tracker
        MTPLossLoggingHelper.clean_loss_in_tracker()

        # Check output logits shapes
        assert output[1].shape[0] == self.micro_batch_size
        assert output[1].shape[1] == self.seq_length

        # Verify gradients
        output[0].backward()
        for name, param in gpt_model.named_parameters():
            assert param.grad is not None


class TestMTPLossLoggingHelper(unittest.TestCase):
    num_hidden_layers = 4

    def setUp(self):
        # Reset the tracker before each test
        MTPLossLoggingHelper.tracker = {}

    def test_save_loss_to_tracker(self):
        """Test saving loss to tracker."""
        # Create a dummy loss tensor
        loss = paddle.tensor(1.3)
        layer_number = 2
        num_hidden_layers = self.num_hidden_layers

        # Test saving loss
        MTPLossLoggingHelper.save_loss_to_tracker(
            loss=loss,
            layer_number=layer_number,
            num_hidden_layers=num_hidden_layers,
        )

        # Verify tracker state
        assert "values" in MTPLossLoggingHelper.tracker
        assert MTPLossLoggingHelper.tracker["values"].shape == [
            num_hidden_layers
        ]
        assert MTPLossLoggingHelper.tracker["values"][layer_number] == loss
        assert MTPLossLoggingHelper.tracker["reduce_group"] is None
        assert MTPLossLoggingHelper.tracker["avg_group"] is None

    def test_track_mtp_metrics(self):
        """Test tracking MTP metrics."""
        # First save some losses
        loss = paddle.tensor(2.3)
        num_hidden_layers = self.num_hidden_layers
        for i in range(num_hidden_layers):
            MTPLossLoggingHelper.save_loss_to_tracker(
                loss=loss, layer_number=i, num_hidden_layers=num_hidden_layers
            )

        # Create dummy writer and loss dict
        class DummyWriter:
            def add_scalar(self, name, value, iteration):
                pass

        class DummyWandBWriter:
            def log(self, metrics, iteration):
                pass

        loss_scale = 1.5
        iteration = 2
        writer = DummyWriter()
        wandb_writer = DummyWandBWriter()
        total_loss_dict = {}

        # Test tracking metrics
        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=loss_scale,
            iteration=iteration,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
        )

        # Verify total_loss_dict is populated
        for i in range(num_hidden_layers):
            assert f"mtp_{i + 1} loss" in total_loss_dict
            assert total_loss_dict[f"mtp_{i + 1} loss"] == loss * loss_scale

        # Verify tracker is cleaned
        assert paddle.all(MTPLossLoggingHelper.tracker["values"] == 0)
        assert MTPLossLoggingHelper.tracker["reduce_group"] is None
        assert MTPLossLoggingHelper.tracker["avg_group"] is None


if __name__ == "__main__":
    unittest.main()
