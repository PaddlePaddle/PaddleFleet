# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import paddle


def _make_mtp_config(**overrides):
    """Helper to create config for MTP testing."""
    from paddlefleet.transformer.transformer_config import TransformerConfig

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "num_nextn_predict_layers": 1,
        "train_mtp_only": False,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _mock_transformer_layer_spec():
    """Create a mock transformer layer spec with the needed attribute chain."""
    from paddlefleet.transformer.enums import AttnMaskType

    mock_self_attn = MagicMock()
    mock_self_attn.extra_kwargs = {"attn_mask_type": AttnMaskType.causal}

    mock_sublayers_spec = MagicMock()
    mock_sublayers_spec.self_attn = mock_self_attn

    mock_transformer_layer = MagicMock()
    mock_transformer_layer.sublayers_spec = mock_sublayers_spec
    return mock_transformer_layer


def _mock_build_layer_side_effect(*a, **kw):
    """Create a mock layer that is callable and returns tensors."""
    mock = MagicMock()
    # Make the mock itself callable to return (input, None) like ColumnParallelLinear
    mock.side_effect = lambda x, *args, **kwargs: (x, None)
    mock.backward_dw = MagicMock()
    return mock


def _make_pg_collection():
    """Helper to create a mock ProcessGroupCollection."""
    mock_pg = MagicMock()
    mock_pg.cp = MagicMock()
    mock_pg.cp.rank.return_value = 0
    mock_pg.cp.size.return_value = 1
    mock_pg.tp = MagicMock()
    return mock_pg


def _common_patches():
    """Return common patches needed for MultiTokenPredictionLayer tests."""
    return [
        patch(
            "paddlefleet.transformer.multi_token_prediction.ProcessGroupCollection.use_mpu_process_groups",
            return_value=_make_pg_collection(),
        ),
        patch(
            "paddlefleet.transformer.multi_token_prediction.build_spec_layer",
            side_effect=_mock_build_layer_side_effect,
        ),
        patch(
            "paddlefleet.transformer.multi_token_prediction.gather_from_tensor_model_parallel_region",
            side_effect=lambda x: x,
        ),
        patch(
            "paddlefleet.transformer.multi_token_prediction.scatter_to_sequence_parallel_region",
            side_effect=lambda x: x,
        ),
        patch(
            "paddlefleet.transformer.multi_token_prediction.tensor_parallel.get_cuda_rng_tracker",
            return_value=MagicMock(fork=nullcontext),
        ),
    ]


class TestMultiTokenPrediction(unittest.TestCase):
    """Unit tests for multi_token_prediction module."""

    def test_dense_mtp_mask_is_selected_by_prediction_depth(self):
        from paddlefleet.transformer.multi_token_prediction import (
            _apply_mtp_layer_masks,
        )

        dense_mask = paddle.arange(18, dtype="float32").reshape([1, 2, 3, 3])
        hidden_mask = paddle.arange(6, dtype="float32").reshape([1, 2, 3])
        dict_args = {
            "mtp_attn_mask": dense_mask,
            "mtp_hidden_inputs_mask_all": hidden_mask,
        }

        _apply_mtp_layer_masks(
            dict_args,
            depth=1,
            config=SimpleNamespace(gpt_model_use_experimental_version=False),
        )

        self.assertEqual(list(dict_args["attention_mask"].shape), [1, 1, 3, 3])
        self.assertTrue(
            paddle.equal_all(dict_args["attention_mask"], dense_mask[:, 1:2])
        )
        self.assertEqual(
            list(dict_args["mtp_hidden_inputs_mask"].shape), [1, 1, 3]
        )
        self.assertNotIn("mtp_attn_mask", dict_args)
        self.assertNotIn("mtp_hidden_inputs_mask_all", dict_args)

    def test_mtp_loss_logging_helper_save_loss(self):
        """Test MTPLossLoggingHelper.save_loss_to_tracker."""
        from paddlefleet.transformer.multi_token_prediction import (
            MTPLossLoggingHelper,
        )

        MTPLossLoggingHelper.tracker = {}
        loss = paddle.to_tensor(0.5, dtype=paddle.float32)
        MTPLossLoggingHelper.save_loss_to_tracker(
            loss, layer_number=0, num_hidden_layers=2
        )
        self.assertIn("values", MTPLossLoggingHelper.tracker)
        self.assertEqual(MTPLossLoggingHelper.tracker["values"].shape[0], 2)

    def test_mtp_loss_logging_helper_save_none_layer(self):
        """Test MTPLossLoggingHelper.save_loss_to_tracker with None layer."""
        from paddlefleet.transformer.multi_token_prediction import (
            MTPLossLoggingHelper,
        )

        MTPLossLoggingHelper.tracker = {}
        loss = paddle.to_tensor(0.5, dtype=paddle.float32)
        MTPLossLoggingHelper.save_loss_to_tracker(
            loss, layer_number=None, num_hidden_layers=2
        )
        self.assertNotIn("values", MTPLossLoggingHelper.tracker)

    def test_mtp_loss_logging_helper_clean(self):
        """Test MTPLossLoggingHelper.clean_loss_in_tracker."""
        from paddlefleet.transformer.multi_token_prediction import (
            MTPLossLoggingHelper,
        )

        MTPLossLoggingHelper.tracker = {
            "values": paddle.ones([2]),
            "reduce_group": None,
            "avg_group": None,
        }
        MTPLossLoggingHelper.clean_loss_in_tracker()
        self.assertTrue(
            paddle.allclose(
                MTPLossLoggingHelper.tracker["values"], paddle.zeros([2])
            )
        )

    def test_mtp_loss_logging_helper_reduce_no_group(self):
        """Test MTPLossLoggingHelper.reduce_loss_in_tracker without group."""
        from paddlefleet.transformer.multi_token_prediction import (
            MTPLossLoggingHelper,
        )

        MTPLossLoggingHelper.tracker = {
            "values": paddle.to_tensor([1.0, 2.0], dtype=paddle.float32),
            "reduce_group": None,
            "avg_group": None,
        }
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        # Should not raise

    def test_mtp_loss_logging_helper_track_metrics(self):
        """Test MTPLossLoggingHelper.track_mtp_metrics."""
        from paddlefleet.transformer.multi_token_prediction import (
            MTPLossLoggingHelper,
        )

        MTPLossLoggingHelper.tracker = {
            "values": paddle.to_tensor([1.0], dtype=paddle.float32),
            "reduce_group": None,
            "avg_group": None,
        }
        # Test without writer/wandb_writer
        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=1.0,
            iteration=0,
            writer=None,
            wandb_writer=None,
        )
        # Should not raise

    def test_mtp_loss_auto_scaler_forward(self):
        """Test MTPLossAutoScaler forward."""
        from paddlefleet.transformer.multi_token_prediction import (
            MTPLossAutoScaler,
        )

        output = paddle.randn([4, 64], dtype=paddle.float32)
        mtp_loss = paddle.to_tensor(0.5, dtype=paddle.float32)
        result = MTPLossAutoScaler.apply(output, mtp_loss)
        self.assertTrue(paddle.allclose(result, output))

    def test_mtp_loss_auto_scaler_set_loss_scale(self):
        """Test MTPLossAutoScaler.set_loss_scale."""
        from paddlefleet.transformer.multi_token_prediction import (
            MTPLossAutoScaler,
        )

        scale = paddle.to_tensor(2.0, dtype=paddle.float32)
        MTPLossAutoScaler.set_loss_scale(scale)
        self.assertAlmostEqual(
            MTPLossAutoScaler.main_loss_backward_scale.item(), 2.0
        )

    def test_mtp_sublayers_spec_defaults(self):
        """Test MultiTokenPredictionLayerSublayersSpec defaults."""
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayerSublayersSpec,
        )

        spec = MultiTokenPredictionLayerSublayersSpec()
        self.assertIsNone(spec.enorm)
        self.assertIsNone(spec.hnorm)
        self.assertIsNone(spec.eh_proj)
        self.assertIsNone(spec.transformer_layer)
        self.assertIsNone(spec.layer_norm)

    def test_supported_attn_mask(self):
        """Test SUPPORTED_ATTN_MASK contains expected values."""
        from paddlefleet.transformer.enums import AttnMaskType
        from paddlefleet.transformer.multi_token_prediction import (
            SUPPORTED_ATTN_MASK,
        )

        self.assertIn(AttnMaskType.padding, SUPPORTED_ATTN_MASK)
        self.assertIn(AttnMaskType.causal, SUPPORTED_ATTN_MASK)
        self.assertIn(AttnMaskType.no_mask, SUPPORTED_ATTN_MASK)
        self.assertIn(AttnMaskType.padding_causal, SUPPORTED_ATTN_MASK)

    def test_mtp_eh_projection_accuracy_gate(self):
        """Accuracy mode uses functional linear only for TP=1."""
        from paddlefleet.transformer import multi_token_prediction as mtp

        class Projection:
            def __init__(self):
                self.weight = paddle.randn([4, 3])
                self.bias = None
                self.skip_bias_add = False
                self.called = False

            def __call__(self, hidden_states):
                self.called = True
                return paddle.full([2, 3], -1.0), None

        hidden_states = paddle.randn([2, 4])
        projection = Projection()
        expected = paddle.nn.functional.linear(hidden_states, projection.weight)

        with patch.object(mtp, "_ACCURACY_COMPATIBLE_KERNEL", True):
            actual, bias = mtp._mtp_eh_projection(projection, hidden_states, 1)
            paddle.testing.assert_close(actual, expected)
            self.assertIsNone(bias)
            self.assertFalse(projection.called)

            native, bias = mtp._mtp_eh_projection(projection, hidden_states, 2)
            paddle.testing.assert_close(native, paddle.full([2, 3], -1.0))
            self.assertIsNone(bias)
            self.assertTrue(projection.called)

    def test_mtp_shift_position_ids_accuracy_gate(self):
        """Accuracy mode rolls MTP positions by one plus prediction depth."""
        from paddlefleet.transformer import multi_token_prediction as mtp

        position_ids = paddle.to_tensor([[0, 1, 2, 3]], dtype="int64")
        with patch.object(mtp, "_ACCURACY_COMPATIBLE_KERNEL", True):
            actual = mtp._mtp_shift_position_ids(
                position_ids,
                paddle.zeros([1, 4, 2]),
                layer_number=0,
                sequence_parallel=False,
            )
            self.assertTrue(
                paddle.equal_all(
                    actual, paddle.to_tensor([1, 2, 3, 0], dtype="int64")
                ).item()
            )

            raw_carrier = paddle.to_tensor(
                [[0, 1, 2, 3, 4, 5, 6, 0]], dtype="int64"
            )
            normalized = mtp._mtp_shift_position_ids(
                raw_carrier,
                paddle.zeros([1, 7, 2]),
                layer_number=0,
                sequence_parallel=False,
            )
            self.assertTrue(
                paddle.equal_all(
                    normalized,
                    paddle.to_tensor([1, 2, 3, 4, 5, 6, 0], dtype="int64"),
                ).item()
            )
            with self.assertRaisesRegex(ValueError, "shorter than"):
                mtp._mtp_shift_position_ids(
                    paddle.to_tensor([[0, 1, 2]], dtype="int64"),
                    paddle.zeros([1, 4, 2]),
                    layer_number=0,
                    sequence_parallel=False,
                )

            actual = mtp._mtp_shift_position_ids(
                position_ids,
                paddle.zeros([1, 4, 2]),
                layer_number=1,
                sequence_parallel=False,
            )
            self.assertTrue(
                paddle.equal_all(
                    actual, paddle.to_tensor([2, 3, 0, 1], dtype="int64")
                ).item()
            )
            inferred = mtp._mtp_shift_position_ids(
                None,
                paddle.zeros([1, 4, 2]),
                layer_number=0,
                sequence_parallel=False,
            )
            self.assertTrue(
                paddle.equal_all(
                    inferred, paddle.to_tensor([1, 2, 3, 0], dtype="int64")
                ).item()
            )

        with patch.object(mtp, "_ACCURACY_COMPATIBLE_KERNEL", False):
            self.assertIs(
                mtp._mtp_shift_position_ids(
                    position_ids,
                    paddle.zeros([1, 4, 2]),
                    layer_number=0,
                    sequence_parallel=False,
                ),
                position_ids,
            )

    def test_weight_only_mtp_layer_forward(self):
        """Test WeightOnlyMTPLayer.forward returns dict_args."""
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayerSublayersSpec,
            WeightOnlyMTPLayer,
        )

        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            config = _make_mtp_config(mtp_load_weight_only=True)
            sublayers = MultiTokenPredictionLayerSublayersSpec()
            sublayers.transformer_layer = _mock_transformer_layer_spec()

            layer = WeightOnlyMTPLayer(
                config=config,
                sublayers_spec=sublayers,
                layer_number=0,
            )
            # WeightOnlyMTPLayer forward just returns dict_args
            result = layer.forward({"hidden_states": paddle.randn([4, 64])})
            self.assertIn("hidden_states", result)

    def test_checkpointed_forward_invalid_method(self):
        """Test _checkpointed_forward raises for invalid recompute_method."""
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
            MultiTokenPredictionLayerSublayersSpec,
        )

        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            # Create config without recompute to pass validation,
            # then set recompute attributes directly
            config = _make_mtp_config()
            config.recompute_granularity = "full"
            config.recompute_method = "invalid"
            config.recompute_num_layers = 1
            sublayers = MultiTokenPredictionLayerSublayersSpec()
            sublayers.transformer_layer = _mock_transformer_layer_spec()

            layer = MultiTokenPredictionLayer(
                config=config,
                sublayers_spec=sublayers,
                layer_number=0,
            )
            layer.transformer_layer = MagicMock()
            with self.assertRaises(ValueError):
                layer._checkpointed_forward(
                    layer._proj_and_transformer_layer,
                    hidden_states=paddle.randn([4, 64]),
                    decoder_input=paddle.randn([4, 64]),
                )

    def test_forward_context_assertion(self):
        """Test forward raises when context is not None."""
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
            MultiTokenPredictionLayerSublayersSpec,
        )

        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            config = _make_mtp_config()
            sublayers = MultiTokenPredictionLayerSublayersSpec()
            sublayers.transformer_layer = _mock_transformer_layer_spec()

            layer = MultiTokenPredictionLayer(
                config=config,
                sublayers_spec=sublayers,
                layer_number=0,
            )
            layer.transformer_layer = MagicMock()
            with self.assertRaises(AssertionError):
                layer.forward(
                    {
                        "hidden_states": paddle.randn([4, 64]),
                        "context": MagicMock(),
                    }
                )

    def test_forward_packed_seq_assertion(self):
        """Test forward raises when packed_seq_params is not None."""
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
            MultiTokenPredictionLayerSublayersSpec,
        )

        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            config = _make_mtp_config()
            sublayers = MultiTokenPredictionLayerSublayersSpec()
            sublayers.transformer_layer = _mock_transformer_layer_spec()

            layer = MultiTokenPredictionLayer(
                config=config,
                sublayers_spec=sublayers,
                layer_number=0,
            )
            layer.transformer_layer = MagicMock()
            with self.assertRaises(AssertionError):
                layer.forward(
                    {
                        "hidden_states": paddle.randn([4, 64]),
                        "packed_seq_params": MagicMock(),
                    }
                )


if __name__ == "__main__":
    unittest.main()
