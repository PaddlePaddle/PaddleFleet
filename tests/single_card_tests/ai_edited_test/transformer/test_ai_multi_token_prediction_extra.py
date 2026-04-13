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
from unittest.mock import MagicMock, patch

import paddle

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.multi_token_prediction import (
    AttnMaskType,
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSublayersSpec,
    WeightOnlyMTPLayer,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "num_nextn_predict_layers": 1,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "perform_initialization": False,
        "recompute_granularity": None,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class SimpleLinear(paddle.nn.Layer):
    def __init__(self, in_f, out_f, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.linear(x), self.linear.bias


class MockNorm(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.w = paddle.create_parameter(
            shape=[kwargs.get("hidden_size", 64)],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        return x * self.w


class MockTransformerLayer(paddle.nn.Layer):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.linear = paddle.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, dict_args):
        h = dict_args["hidden_states"]
        dict_args["hidden_states"] = self.linear(h)
        return dict_args


class TestMultiTokenPredictionLayerSublayersSpec(unittest.TestCase):
    """Test MultiTokenPredictionLayerSublayersSpec dataclass."""

    def test_defaults(self):
        spec = MultiTokenPredictionLayerSublayersSpec()
        self.assertIsNone(spec.enorm)
        self.assertIsNone(spec.hnorm)
        self.assertIsNone(spec.eh_proj)


class TestMTPLossLoggingHelper(unittest.TestCase):
    """Test MTPLossLoggingHelper static methods."""

    def setUp(self):
        MTPLossLoggingHelper.tracker = {}

    def tearDown(self):
        MTPLossLoggingHelper.tracker = {}

    def test_save_loss_to_tracker(self):
        loss = paddle.to_tensor(0.5)
        MTPLossLoggingHelper.save_loss_to_tracker(loss, 0, 4)
        self.assertIn("values", MTPLossLoggingHelper.tracker)
        self.assertEqual(MTPLossLoggingHelper.tracker["values"][0].item(), 0.5)

    def test_save_loss_none_layer_skips(self):
        MTPLossLoggingHelper.save_loss_to_tracker(
            paddle.to_tensor(0.5), None, 4
        )
        self.assertNotIn("values", MTPLossLoggingHelper.tracker)

    def test_clean_loss_in_tracker(self):
        loss = paddle.to_tensor(0.5)
        MTPLossLoggingHelper.save_loss_to_tracker(loss, 0, 4)
        MTPLossLoggingHelper.clean_loss_in_tracker()
        self.assertEqual(
            MTPLossLoggingHelper.tracker["values"].sum().item(), 0.0
        )

    def test_reduce_loss_in_tracker_empty(self):
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        # Should not raise

    def test_track_mtp_metrics_no_writer(self):
        loss = paddle.to_tensor(0.5)
        MTPLossLoggingHelper.save_loss_to_tracker(loss, 0, 1)
        MTPLossLoggingHelper.track_mtp_metrics(1.0, 0, None, None, None)
        # Should not raise

    def test_track_mtp_metrics_with_total_loss_dict(self):
        loss = paddle.to_tensor(0.5)
        MTPLossLoggingHelper.save_loss_to_tracker(loss, 0, 1)
        total_dict = {}
        MTPLossLoggingHelper.track_mtp_metrics(1.0, 0, None, None, total_dict)
        self.assertIn("mtp_1 loss", total_dict)


class TestMTPLossAutoScaler(unittest.TestCase):
    """Test MTPLossAutoScaler."""

    def test_set_loss_scale(self):
        scale = paddle.to_tensor(2.0)
        MTPLossAutoScaler.set_loss_scale(scale)
        self.assertAlmostEqual(
            float(MTPLossAutoScaler.main_loss_backward_scale), 2.0
        )

    def test_forward_returns_output(self):
        output = paddle.randn([2, 4])
        mtp_loss = paddle.randn([2, 4])
        result = MTPLossAutoScaler.apply(output, mtp_loss)
        self.assertEqual(result.shape, output.shape)


class TestMultiTokenPredictionLayer(unittest.TestCase):
    """Test MultiTokenPredictionLayer."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.multi_token_prediction.build_layer")
    def test_construction(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg.return_value = mock_pg_obj

        mock_tl_spec = MagicMock()
        mock_tl_spec.sublayers_spec.self_attn.extra_kwargs = {
            "attn_mask_type": AttnMaskType.padding,
        }

        def build_side_effect(*a, **kw):
            if "transformer_layer" in str(kw):
                return MockTransformerLayer(_make_config())
            return MockNorm()

        mock_build.side_effect = build_side_effect

        config = _make_config()
        spec = MultiTokenPredictionLayerSublayersSpec(
            enorm=MagicMock(),
            hnorm=MagicMock(),
            eh_proj=MagicMock(),
            transformer_layer=mock_tl_spec,
            layer_norm=MagicMock(),
        )
        layer = MultiTokenPredictionLayer(
            config, spec, layer_number=0, pg_collection=mock_pg_obj
        )
        self.assertIsNotNone(layer.enorm)
        self.assertIsNotNone(layer.hnorm)
        self.assertIsNotNone(layer.eh_proj)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.multi_token_prediction.build_layer")
    def test_build_schedule_node(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg.return_value = mock_pg_obj

        mock_tl_spec = MagicMock()
        mock_tl_spec.sublayers_spec.self_attn.extra_kwargs = {
            "attn_mask_type": AttnMaskType.padding,
        }

        mock_build.side_effect = lambda *a, **kw: MockNorm()

        config = _make_config()
        spec = MultiTokenPredictionLayerSublayersSpec(
            enorm=MagicMock(),
            hnorm=MagicMock(),
            eh_proj=MagicMock(),
            transformer_layer=mock_tl_spec,
            layer_norm=MagicMock(),
        )
        layer = MultiTokenPredictionLayer(
            config, spec, layer_number=0, pg_collection=mock_pg_obj
        )
        node = layer.build_schedule_node()
        self.assertIsNotNone(node)


class TestWeightOnlyMTPLayer(unittest.TestCase):
    """Test WeightOnlyMTPLayer."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.multi_token_prediction.build_layer")
    def test_forward_returns_dict_unchanged(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg.return_value = mock_pg_obj

        mock_tl_spec = MagicMock()
        mock_tl_spec.sublayers_spec.self_attn.extra_kwargs = {
            "attn_mask_type": AttnMaskType.padding,
        }

        mock_build.side_effect = lambda *a, **kw: MockNorm()

        config = _make_config()
        spec = MultiTokenPredictionLayerSublayersSpec(
            enorm=MagicMock(),
            hnorm=MagicMock(),
            eh_proj=MagicMock(),
            transformer_layer=mock_tl_spec,
            layer_norm=MagicMock(),
        )
        layer = WeightOnlyMTPLayer(
            config, spec, layer_number=0, pg_collection=mock_pg_obj
        )
        dict_args = {"hidden_states": paddle.randn([2, 4, 64])}
        result = layer(dict_args)
        # WeightOnlyMTPLayer should pass through unchanged
        self.assertIs(result, dict_args)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.multi_token_prediction.build_layer")
    def test_build_schedule_node(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg.return_value = mock_pg_obj

        mock_tl_spec = MagicMock()
        mock_tl_spec.sublayers_spec.self_attn.extra_kwargs = {
            "attn_mask_type": AttnMaskType.padding,
        }

        mock_build.side_effect = lambda *a, **kw: MockNorm()

        config = _make_config()
        spec = MultiTokenPredictionLayerSublayersSpec(
            enorm=MagicMock(),
            hnorm=MagicMock(),
            eh_proj=MagicMock(),
            transformer_layer=mock_tl_spec,
            layer_norm=MagicMock(),
        )
        layer = WeightOnlyMTPLayer(
            config, spec, layer_number=0, pg_collection=mock_pg_obj
        )
        node = layer.build_schedule_node()
        self.assertIsNotNone(node)


class TestMultiTokenPredictionLayerForward(unittest.TestCase):
    """Test MultiTokenPredictionLayer forward."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.multi_token_prediction.build_layer")
    @patch("paddlefleet.transformer.multi_token_prediction.tensor_parallel")
    def test_forward_context_assertion(self, mock_tp, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg.return_value = mock_pg_obj
        mock_rng = MagicMock()
        mock_rng.__enter__ = MagicMock(return_value=None)
        mock_rng.__exit__ = MagicMock(return_value=None)
        mock_tp.get_cuda_rng_tracker.return_value.fork.return_value = mock_rng

        mock_tl = MockTransformerLayer(_make_config())
        mock_tl_spec = MagicMock()
        mock_tl_spec.sublayers_spec.self_attn.extra_kwargs = {
            "attn_mask_type": AttnMaskType.padding,
        }

        call_count = [0]

        def build_side_effect(*a, **kw):
            call_count[0] += 1
            if "transformer_layer" in str(kw):
                return mock_tl
            return MockNorm()

        mock_build.side_effect = build_side_effect

        config = _make_config()
        spec = MultiTokenPredictionLayerSublayersSpec(
            enorm=MagicMock(),
            hnorm=MagicMock(),
            eh_proj=MagicMock(),
            transformer_layer=mock_tl_spec,
            layer_norm=MagicMock(),
        )
        layer = MultiTokenPredictionLayer(
            config, spec, layer_number=0, pg_collection=mock_pg_obj
        )

        h = paddle.randn([4, 8, 64], dtype="float32")
        dict_args = {"hidden_states": h, "context": "fake"}
        with self.assertRaises(AssertionError):
            layer(dict_args)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.multi_token_prediction.build_layer")
    @patch("paddlefleet.transformer.multi_token_prediction.tensor_parallel")
    def test_forward_packed_seq_assertion(self, mock_tp, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg.return_value = mock_pg_obj

        mock_tl_spec = MagicMock()
        mock_tl_spec.sublayers_spec.self_attn.extra_kwargs = {
            "attn_mask_type": AttnMaskType.padding,
        }

        mock_build.side_effect = lambda *a, **kw: MockNorm()

        config = _make_config()
        spec = MultiTokenPredictionLayerSublayersSpec(
            enorm=MagicMock(),
            hnorm=MagicMock(),
            eh_proj=MagicMock(),
            transformer_layer=mock_tl_spec,
            layer_norm=MagicMock(),
        )
        layer = MultiTokenPredictionLayer(
            config, spec, layer_number=0, pg_collection=mock_pg_obj
        )

        h = paddle.randn([4, 8, 64], dtype="float32")
        dict_args = {"hidden_states": h, "packed_seq_params": "fake"}
        with self.assertRaises(AssertionError):
            layer(dict_args)


if __name__ == "__main__":
    unittest.main()
