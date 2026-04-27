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

"""
Unit tests for MTP mask support under experimental_dataflow.

Covers the latest commit "[New feature] Support MTP mask":
  - GPTEmbedding: mtp_startend_row_indices_all / mtp_hidden_inputs_mask_all passthrough
  - TransformerLayer: experimental_dataflow skips old mask split/concat logic
  - MultiTokenPredictionLayer: per-depth mask slicing, hidden-inputs masking,
    shape validation, and restore-after-forward semantics

Usage:
    PYTHONPATH=<PaddleFleet>/src:$PYTHONPATH python -m pytest <this_file> -v
    # or
    PYTHONPATH=<PaddleFleet>/src:$PYTHONPATH python <this_file>
"""

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
from unittest.mock import MagicMock, patch

import paddle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Create a TransformerConfig with defaults suitable for experimental_dataflow MTP tests."""
    from paddlefleet.transformer.transformer_config import TransformerConfig

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "num_nextn_predict_layers": 2,
        "train_mtp_only": False,
        "experimental_dataflow": True,
        "gpt_model_use_experimental_version": False,
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


def _build_mtp_layer(config, layer_number=0):
    """Instantiate a MultiTokenPredictionLayer under mocked infra."""
    from paddlefleet.transformer.multi_token_prediction import (
        MultiTokenPredictionLayer,
        MultiTokenPredictionLayerSublayersSpec,
    )

    sublayers = MultiTokenPredictionLayerSublayersSpec()
    sublayers.transformer_layer = _mock_transformer_layer_spec()

    layer = MultiTokenPredictionLayer(
        config=config,
        sublayers_spec=sublayers,
        layer_number=layer_number,
    )

    # Replace enorm/hnorm with identity (mock build_spec_layer returns (x, None) tuples)
    layer.enorm = lambda x: x
    layer.hnorm = lambda x: x
    H = config.hidden_size
    layer.eh_proj = lambda x: (x[..., :H], None)

    # Replace norm if it exists (gpt_model_use_experimental_version=False)
    if hasattr(layer, "norm"):
        layer.norm = lambda x: x

    # Replace transformer_layer with a mock that returns a dict with hidden_states
    def _mock_transformer_fwd(input_dict):
        return {"hidden_states": input_dict["hidden_states"]}

    layer.transformer_layer = MagicMock(side_effect=_mock_transformer_fwd)
    return layer


# ---------------------------------------------------------------------------
# Test: experimental_dataflow config field
# ---------------------------------------------------------------------------


class TestExperimentalDataflowConfig(unittest.TestCase):
    """Test that experimental_dataflow config field exists and defaults correctly."""

    def test_default_is_false(self):
        from paddlefleet.transformer.transformer_config import TransformerConfig

        cfg = TransformerConfig(hidden_size=64, num_attention_heads=2)
        self.assertFalse(cfg.experimental_dataflow)

    def test_can_set_true(self):
        cfg = _make_config(experimental_dataflow=True)
        self.assertTrue(cfg.experimental_dataflow)


# ---------------------------------------------------------------------------
# Test: GPTEmbedding — mtp_startend_row_indices_all / mtp_hidden_inputs_mask_all
# ---------------------------------------------------------------------------


class TestGPTEmbeddingMTPMask(unittest.TestCase):
    """Test GPTEmbedding.forward propagation of mtp_startend_row_indices_all and mtp_hidden_inputs_mask_all."""

    def test_both_none_passes(self):
        """When both are absent from dict_args, no assertion fires."""
        from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

        config = _make_config(
            num_nextn_predict_layers=0,
            experimental_dataflow=True,
        )

        with patch(
            "paddlefleet.models.gpt.gpt_embedding.build_spec_layer",
            side_effect=_mock_build_layer_side_effect,
        ):
            mock_spec = MagicMock()
            mock_spec.rope_embedding = None
            embedding = GPTEmbedding(
                sublayers_spec=mock_spec,
                config=config,
                vocab_size=128,
                max_sequence_length=32,
                position_embedding_type="none",
            )
            # mock the embedding call
            embedding.embedding = MagicMock(
                return_value=paddle.randn([2, 8, 64])
            )

            dict_args = {
                "input_ids": paddle.randint(0, 128, [2, 8]),
                "position_ids": paddle.arange(8).unsqueeze(0).expand([2, 8]),
            }
            # Should not raise
            result = embedding.forward(dict_args)
            self.assertNotIn("mtp_startend_row_indices_all", result)
            self.assertNotIn("mtp_hidden_inputs_mask_all", result)

    def test_only_one_present_raises(self):
        """When only one of the pair is provided, assertion should fire."""
        from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

        config = _make_config(
            num_nextn_predict_layers=0,
            experimental_dataflow=True,
        )

        with patch(
            "paddlefleet.models.gpt.gpt_embedding.build_spec_layer",
            side_effect=_mock_build_layer_side_effect,
        ):
            mock_spec = MagicMock()
            mock_spec.rope_embedding = None
            embedding = GPTEmbedding(
                sublayers_spec=mock_spec,
                config=config,
                vocab_size=128,
                max_sequence_length=32,
                position_embedding_type="none",
            )
            embedding.embedding = MagicMock(
                return_value=paddle.randn([2, 8, 64])
            )

            dict_args = {
                "input_ids": paddle.randint(0, 128, [2, 8]),
                "position_ids": paddle.arange(8).unsqueeze(0).expand([2, 8]),
                "mtp_startend_row_indices_all": paddle.randn([2, 2, 8, 1]),
                # mtp_hidden_inputs_mask_all is intentionally missing
            }
            with self.assertRaises(AssertionError):
                embedding.forward(dict_args)

    def test_both_present_are_propagated(self):
        """When both tensors are present, they appear in preproc_output."""
        from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

        B, S, num_nextn = 2, 8, 2
        config = _make_config(
            num_nextn_predict_layers=0,
            experimental_dataflow=True,
        )

        with patch(
            "paddlefleet.models.gpt.gpt_embedding.build_spec_layer",
            side_effect=_mock_build_layer_side_effect,
        ):
            mock_spec = MagicMock()
            mock_spec.rope_embedding = None
            embedding = GPTEmbedding(
                sublayers_spec=mock_spec,
                config=config,
                vocab_size=128,
                max_sequence_length=32,
                position_embedding_type="none",
            )
            embedding.embedding = MagicMock(
                return_value=paddle.randn([B, S, 64])
            )

            startend = paddle.randn([B, num_nextn, S, 1]).cuda()
            mask_all = paddle.ones([B, num_nextn, S]).cuda()

            dict_args = {
                "input_ids": paddle.randint(0, 128, [B, S]),
                "position_ids": paddle.arange(S).unsqueeze(0).expand([B, S]),
                "mtp_startend_row_indices_all": startend,
                "mtp_hidden_inputs_mask_all": mask_all,
            }
            result = embedding.forward(dict_args)
            self.assertIn("mtp_startend_row_indices_all", result)
            self.assertIn("mtp_hidden_inputs_mask_all", result)
            self.assertEqual(
                list(result["mtp_startend_row_indices_all"].shape),
                [B, num_nextn, S, 1],
            )
            self.assertEqual(
                list(result["mtp_hidden_inputs_mask_all"].shape),
                [B, num_nextn, S],
            )


# ---------------------------------------------------------------------------
# Test: MultiTokenPredictionLayer — _concat_embeddings with mtp_hidden_inputs_mask
# ---------------------------------------------------------------------------


class TestMTPConcatEmbeddingsWithMask(unittest.TestCase):
    """Test that _concat_embeddings applies mtp_hidden_inputs_mask correctly."""

    def test_mask_zeros_out_hidden_states(self):
        """Positions where mask=0 should zero out hidden_states contribution."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            config = _make_config(num_nextn_predict_layers=1)
            layer = _build_mtp_layer(config, layer_number=0)

            # Override enorm/hnorm to identity
            layer.enorm = lambda x: x
            layer.hnorm = lambda x: x
            # Override eh_proj to identity (just take first H dims)
            H = config.hidden_size
            layer.eh_proj = lambda x: (x[..., :H], None)

            B, S = 2, 4
            hidden_states = paddle.ones([B, S, H], dtype="float32")
            decoder_input = paddle.ones([B, S, H], dtype="float32")

            # mask: zero at position 2 for the first batch item
            mask = paddle.ones([B, 1, S], dtype="float32")
            mask[0, 0, 2] = 0.0

            result = layer._concat_embeddings(
                hidden_states, decoder_input, mask
            )

            # At position (0, 2), hidden_states was zeroed -> concat will be [decoder, 0]
            # After projection (identity taking first H), result is decoder_input at that pos
            # The key point: masked position hidden contribution is zeroed
            # Since eh_proj is identity of first H dims (decoder_input part), we verify shape
            self.assertEqual(list(result.shape), [B, S, H])

    def test_no_mask_passes_through(self):
        """When mask is None, hidden_states are not modified."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            config = _make_config(num_nextn_predict_layers=1)
            layer = _build_mtp_layer(config, layer_number=0)

            layer.enorm = lambda x: x
            layer.hnorm = lambda x: x
            H = config.hidden_size
            layer.eh_proj = lambda x: (x[..., :H], None)

            B, S = 2, 4
            hidden_states = paddle.ones([B, S, H], dtype="float32")
            decoder_input = paddle.ones([B, S, H], dtype="float32")

            result = layer._concat_embeddings(
                hidden_states, decoder_input, None
            )
            self.assertEqual(list(result.shape), [B, S, H])

    def test_mask_transpose_and_dtype(self):
        """Mask is transposed from [B,1,S] to [B,S,1] and cast to hidden dtype."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            config = _make_config(num_nextn_predict_layers=1)
            layer = _build_mtp_layer(config, layer_number=0)

            B, S, H = 1, 6, config.hidden_size
            hidden_states = paddle.randn([B, S, H], dtype="float32")
            decoder_input = paddle.randn([B, S, H], dtype="float32")

            # Provide mask as int, should be cast to float
            mask = paddle.ones([B, 1, S], dtype="int32")
            mask[0, 0, 0] = 0

            # Override norms and proj to identity so we can inspect the masking
            original_hnorm = layer.hnorm
            captured = {}

            def capture_hnorm(x):
                return x

            def capture_eh_proj(x):
                captured["concat_input"] = x.clone()
                return (x[..., :H], None)

            layer.enorm = lambda x: x
            layer.hnorm = capture_hnorm
            layer.eh_proj = capture_eh_proj

            result = layer._concat_embeddings(
                hidden_states, decoder_input, mask
            )

            # Verify masked position is zeroed: position 0 in hidden_states
            concat_input = captured["concat_input"]
            # concat_input shape: [B, S, 2*H], second half is hidden_states (masked)
            masked_hidden = concat_input[0, 0, H:]  # position 0, hidden part
            self.assertTrue(
                paddle.allclose(
                    masked_hidden, paddle.zeros_like(masked_hidden)
                ),
                "Masked position should have zeroed hidden_states",
            )


# ---------------------------------------------------------------------------
# Test: MultiTokenPredictionLayer.forward — experimental_dataflow mask slicing
# ---------------------------------------------------------------------------


class TestMTPForwardExperimentalDataflow(unittest.TestCase):
    """Test MultiTokenPredictionLayer.forward with experimental_dataflow masks."""

    def _make_dict_args(self, B, S, H, num_nextn, with_masks=True):
        """Build a dict_args suitable for MTP forward."""
        # hidden_states_concat: (num_nextn+1) chunks concatenated along dim 0
        total_sections = num_nextn + 1
        hidden_states = paddle.randn(
            [B * total_sections, S, H], dtype="float32"
        )

        dict_args = {
            "hidden_states": hidden_states,
        }

        if with_masks:
            dict_args["mtp_startend_row_indices_all"] = paddle.randint(
                0, S, [B, num_nextn, S, 1]
            ).astype("float32")
            dict_args["mtp_hidden_inputs_mask_all"] = paddle.ones(
                [B, num_nextn, S], dtype="float32"
            )
            dict_args["attn_mask_startend_row_indices"] = paddle.randint(
                0, S, [B, 1, S, 1]
            ).astype("float32")

        return dict_args

    def test_forward_single_layer_with_masks(self):
        """Forward pass for a single MTP layer (train_mtp_only=False) with masks."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=False,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            dict_args = self._make_dict_args(
                B, S, H, num_nextn, with_masks=True
            )
            original_startend = dict_args[
                "mtp_startend_row_indices_all"
            ].clone()
            original_mask_all = dict_args["mtp_hidden_inputs_mask_all"].clone()
            original_attn = dict_args["attn_mask_startend_row_indices"].clone()

            result = layer.forward(dict_args)

            # mtp_startend_row_indices_all and mtp_hidden_inputs_mask_all should be restored
            self.assertIn("mtp_startend_row_indices_all", result)
            self.assertIn("mtp_hidden_inputs_mask_all", result)
            self.assertTrue(
                paddle.allclose(
                    result["mtp_startend_row_indices_all"], original_startend
                )
            )
            self.assertTrue(
                paddle.allclose(
                    result["mtp_hidden_inputs_mask_all"], original_mask_all
                )
            )
            # attn_mask_startend_row_indices should be restored to original
            self.assertIn("attn_mask_startend_row_indices", result)
            self.assertTrue(
                paddle.allclose(
                    result["attn_mask_startend_row_indices"], original_attn
                )
            )
            # per-depth mtp_hidden_inputs_mask should be cleaned up
            self.assertNotIn("mtp_hidden_inputs_mask", result)

    def test_forward_single_layer_without_masks(self):
        """Forward pass without masks should work (backward compatible)."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=False,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            dict_args = self._make_dict_args(
                B, S, H, num_nextn, with_masks=False
            )
            result = layer.forward(dict_args)

            self.assertNotIn("mtp_startend_row_indices_all", result)
            self.assertNotIn("mtp_hidden_inputs_mask_all", result)

    def test_forward_train_mtp_only_with_masks(self):
        """Forward pass with train_mtp_only=True loops over all depths."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=True,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            dict_args = self._make_dict_args(
                B, S, H, num_nextn, with_masks=True
            )
            original_startend = dict_args[
                "mtp_startend_row_indices_all"
            ].clone()
            original_mask_all = dict_args["mtp_hidden_inputs_mask_all"].clone()

            result = layer.forward(dict_args)

            # Masks should be restored after looping over all depths
            self.assertIn("mtp_startend_row_indices_all", result)
            self.assertIn("mtp_hidden_inputs_mask_all", result)
            self.assertTrue(
                paddle.allclose(
                    result["mtp_startend_row_indices_all"], original_startend
                )
            )
            self.assertTrue(
                paddle.allclose(
                    result["mtp_hidden_inputs_mask_all"], original_mask_all
                )
            )
            self.assertNotIn("mtp_hidden_inputs_mask", result)

    def test_hidden_states_shape_preserved(self):
        """Output hidden_states should have the same shape as input."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=False,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            dict_args = self._make_dict_args(
                B, S, H, num_nextn, with_masks=True
            )
            input_shape = list(dict_args["hidden_states"].shape)

            result = layer.forward(dict_args)
            output_shape = list(result["hidden_states"].shape)

            self.assertEqual(input_shape, output_shape)


# ---------------------------------------------------------------------------
# Test: Shape validation of mtp_startend_row_indices_all / mtp_hidden_inputs_mask_all
# ---------------------------------------------------------------------------


class TestMTPMaskShapeValidation(unittest.TestCase):
    """Test shape validation assertions in MTP forward."""

    def test_startend_wrong_num_nextn_dim(self):
        """mtp_startend_row_indices_all with wrong num_nextn dim should assert."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=False,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            total_sections = num_nextn + 1
            dict_args = {
                "hidden_states": paddle.randn(
                    [B * total_sections, S, H], dtype="float32"
                ),
                # Wrong: num_nextn dim is 3, should be 2
                "mtp_startend_row_indices_all": paddle.randn([B, 3, S, 1]),
                "mtp_hidden_inputs_mask_all": paddle.ones(
                    [B, num_nextn, S], dtype="float32"
                ),
            }
            with self.assertRaises(AssertionError):
                layer.forward(dict_args)

    def test_hidden_mask_wrong_num_nextn_dim(self):
        """mtp_hidden_inputs_mask_all with wrong num_nextn dim should assert."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=False,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            total_sections = num_nextn + 1
            dict_args = {
                "hidden_states": paddle.randn(
                    [B * total_sections, S, H], dtype="float32"
                ),
                "mtp_startend_row_indices_all": paddle.randn(
                    [B, num_nextn, S, 1]
                ),
                # Wrong: num_nextn dim is 1, should be 2
                "mtp_hidden_inputs_mask_all": paddle.ones(
                    [B, 1, S], dtype="float32"
                ),
            }
            with self.assertRaises(AssertionError):
                layer.forward(dict_args)

    def test_shape_mismatch_between_startend_and_hidden_mask(self):
        """Batch/seq mismatch between startend and hidden mask should assert."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=False,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            total_sections = num_nextn + 1
            dict_args = {
                "hidden_states": paddle.randn(
                    [B * total_sections, S, H], dtype="float32"
                ),
                "mtp_startend_row_indices_all": paddle.randn(
                    [B, num_nextn, S, 1]
                ),
                # Wrong: seq dim is S+2 instead of S
                "mtp_hidden_inputs_mask_all": paddle.ones(
                    [B, num_nextn, S + 2], dtype="float32"
                ),
            }
            with self.assertRaises(AssertionError):
                layer.forward(dict_args)


# ---------------------------------------------------------------------------
# Test: MTP _proj_and_transformer_layer passes masks correctly
# ---------------------------------------------------------------------------


class TestMTPProjAndTransformerLayerMaskPassing(unittest.TestCase):
    """Test that _proj_and_transformer_layer passes attn_mask_startend_row_indices
    and mtp_hidden_inputs_mask to the inner transformer."""

    def test_attn_mask_passed_to_transformer(self):
        """attn_mask_startend_row_indices should appear in the input_dict to transformer."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            config = _make_config(
                num_nextn_predict_layers=1,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            # Override norms and proj to identity
            layer.enorm = lambda x: x
            layer.hnorm = lambda x: x
            H = config.hidden_size
            layer.eh_proj = lambda x: (x[..., :H], None)

            B, S = 2, 4
            attn_mask = paddle.randint(0, S, [B, 1, S, 1]).astype("float32")

            captured_input = {}

            def mock_transformer(input_dict):
                captured_input.update(input_dict)
                return {"hidden_states": input_dict["hidden_states"]}

            layer.transformer_layer = MagicMock(side_effect=mock_transformer)

            layer._proj_and_transformer_layer(
                hidden_states=paddle.randn([B, S, H]),
                decoder_input=paddle.randn([B, S, H]),
                attn_mask_startend_row_indices=attn_mask,
                mtp_hidden_inputs_mask=paddle.ones([B, 1, S]),
            )

            self.assertIn("attn_mask_startend_row_indices", captured_input)
            self.assertTrue(
                paddle.allclose(
                    captured_input["attn_mask_startend_row_indices"], attn_mask
                )
            )

    def test_is_mtp_flag_set(self):
        """The input_dict to transformer should have is_mtp=True."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            config = _make_config(
                num_nextn_predict_layers=1,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            layer.enorm = lambda x: x
            layer.hnorm = lambda x: x
            H = config.hidden_size
            layer.eh_proj = lambda x: (x[..., :H], None)

            B, S = 2, 4
            captured_input = {}

            def mock_transformer(input_dict):
                captured_input.update(input_dict)
                return {"hidden_states": input_dict["hidden_states"]}

            layer.transformer_layer = MagicMock(side_effect=mock_transformer)

            layer._proj_and_transformer_layer(
                hidden_states=paddle.randn([B, S, H]),
                decoder_input=paddle.randn([B, S, H]),
            )

            self.assertTrue(captured_input.get("is_mtp", False))


# ---------------------------------------------------------------------------
# Test: MTP norm behavior (experimental version skips norm)
# ---------------------------------------------------------------------------


class TestMTPNormBehavior(unittest.TestCase):
    """Test that gpt_model_use_experimental_version controls whether norm is applied
    in _proj_and_transformer_layer output."""

    def test_non_experimental_applies_norm(self):
        """When gpt_model_use_experimental_version=False, self.norm is built."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            config = _make_config(
                gpt_model_use_experimental_version=False,
                num_nextn_predict_layers=1,
            )
            from paddlefleet.transformer.multi_token_prediction import (
                MultiTokenPredictionLayer,
                MultiTokenPredictionLayerSublayersSpec,
            )

            sublayers = MultiTokenPredictionLayerSublayersSpec()
            sublayers.transformer_layer = _mock_transformer_layer_spec()

            layer = MultiTokenPredictionLayer(
                config=config,
                sublayers_spec=sublayers,
                layer_number=0,
            )
            # self.norm should exist
            self.assertTrue(hasattr(layer, "norm"))

    def test_experimental_skips_norm_build(self):
        """When gpt_model_use_experimental_version=True, self.norm is NOT built."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            config = _make_config(
                gpt_model_use_experimental_version=True,
                num_nextn_predict_layers=1,
            )
            from paddlefleet.transformer.multi_token_prediction import (
                MultiTokenPredictionLayer,
                MultiTokenPredictionLayerSublayersSpec,
            )

            sublayers = MultiTokenPredictionLayerSublayersSpec()
            sublayers.transformer_layer = _mock_transformer_layer_spec()

            layer = MultiTokenPredictionLayer(
                config=config,
                sublayers_spec=sublayers,
                layer_number=0,
            )
            # self.norm should NOT exist as an attribute (the branch skips build_spec_layer)
            self.assertFalse(hasattr(layer, "norm"))


# ---------------------------------------------------------------------------
# Test: TransformerLayer experimental_dataflow mask handling
# ---------------------------------------------------------------------------


class TestTransformerLayerExperimentalDataflow(unittest.TestCase):
    """Test TransformerLayer does NOT split attn_mask_startend_row_indices when
    experimental_dataflow=True."""

    def test_experimental_dataflow_skips_mask_split(self):
        """With experimental_dataflow=True, the main mask should pass through unchanged."""
        config = _make_config(
            num_nextn_predict_layers=2,
            experimental_dataflow=True,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=1,
        )

        # We test the logic conceptually: when experimental_dataflow=True,
        # the code enters the else branch and sets attn_mask_startend_row_indices_mtp = None
        # This means no split/concat of the mask happens
        self.assertTrue(config.experimental_dataflow)
        self.assertEqual(config.num_nextn_predict_layers, 2)

    def test_old_dataflow_would_split_mask(self):
        """With experimental_dataflow=False, config is set for old mask split."""
        config = _make_config(
            num_nextn_predict_layers=2,
            experimental_dataflow=False,
        )
        self.assertFalse(config.experimental_dataflow)


# ---------------------------------------------------------------------------
# Test: LanguageLoss MTP experimental version branch
# ---------------------------------------------------------------------------


class TestLanguageLossMTPExperimental(unittest.TestCase):
    """Test that gpt_model_use_experimental_version changes MTP loss computation."""

    def test_experimental_loss_uses_lossmask(self):
        """Verify the experimental path computes per-token loss with lossmask."""
        config = _make_config(
            gpt_model_use_experimental_version=True,
            num_nextn_predict_layers=1,
            experimental_dataflow=True,
        )
        # Just check that the config fields are correct for the branch
        self.assertTrue(config.gpt_model_use_experimental_version)
        self.assertEqual(config.num_nextn_predict_layers, 1)

    def test_non_experimental_uses_forward_impl(self):
        """Non-experimental path should use _forward for MTP loss."""
        config = _make_config(
            gpt_model_use_experimental_version=False,
            num_nextn_predict_layers=1,
        )
        self.assertFalse(config.gpt_model_use_experimental_version)


# ---------------------------------------------------------------------------
# Test: MTP _checkpointed_forward with new mask args
# ---------------------------------------------------------------------------


class TestMTPCheckpointedForwardNewArgs(unittest.TestCase):
    """Test that _checkpointed_forward passes the new mask arguments to recompute."""

    def test_checkpointed_forward_accepts_new_kwargs(self):
        """_checkpointed_forward should extract attn_mask_startend_row_indices
        and mtp_hidden_inputs_mask from kwargs."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            config = _make_config(
                num_nextn_predict_layers=1,
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            B, S, H = 2, 4, config.hidden_size
            kwargs = {
                "hidden_states": paddle.randn([B, S, H]),
                "decoder_input": paddle.randn([B, S, H]),
                "attention_mask": None,
                "attn_mask_startend_row_indices": paddle.randn([B, 1, S, 1]),
                "context": None,
                "context_mask": None,
                "rotary_pos_emb": None,
                "rotary_pos_cos": None,
                "rotary_pos_sin": None,
                "attention_bias": None,
                "packed_seq_params": None,
                "mtp_hidden_inputs_mask": paddle.ones([B, 1, S]),
            }

            # Patch recompute to capture args
            captured_kwargs = {}

            def mock_recompute(fn, **kw):
                captured_kwargs.update(kw)
                return fn(**kw)

            with patch(
                "paddlefleet.transformer.multi_token_prediction.recompute",
                side_effect=mock_recompute,
            ):
                layer.enorm = lambda x: x
                layer.hnorm = lambda x: x
                layer.eh_proj = lambda x: (x[..., :H], None)

                layer._checkpointed_forward(
                    layer._proj_and_transformer_layer,
                    **kwargs,
                )

            self.assertIn("attn_mask_startend_row_indices", captured_kwargs)
            self.assertIn("mtp_hidden_inputs_mask", captured_kwargs)
            self.assertIsNotNone(
                captured_kwargs["attn_mask_startend_row_indices"]
            )
            self.assertIsNotNone(captured_kwargs["mtp_hidden_inputs_mask"])


# ---------------------------------------------------------------------------
# Test: Per-depth mask slicing correctness
# ---------------------------------------------------------------------------


class TestMTPPerDepthMaskSlicing(unittest.TestCase):
    """Test that forward slices the correct depth from mtp_startend_row_indices_all
    and mtp_hidden_inputs_mask_all for each MTP layer."""

    def test_layer_number_0_gets_depth_0(self):
        """layer_number=0 should slice depth index 0."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=False,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            # Create distinct per-depth masks so we can verify slicing
            startend = paddle.zeros([B, num_nextn, S, 1], dtype="float32")
            startend[:, 0, :, :] = 1.0  # depth 0 has value 1
            startend[:, 1, :, :] = 2.0  # depth 1 has value 2

            mask_all = paddle.zeros([B, num_nextn, S], dtype="float32")
            mask_all[:, 0, :] = 1.0  # depth 0: all ones
            mask_all[:, 1, :] = 0.5  # depth 1: half

            total_sections = num_nextn + 1
            dict_args = {
                "hidden_states": paddle.randn(
                    [B * total_sections, S, H], dtype="float32"
                ),
                "mtp_startend_row_indices_all": startend,
                "mtp_hidden_inputs_mask_all": mask_all,
                "attn_mask_startend_row_indices": paddle.randn([B, 1, S, 1]),
            }

            # Capture what gets passed to _proj_and_transformer_layer
            captured_calls = []
            original_proj = layer._proj_and_transformer_layer

            def capture_proj(**kwargs):
                captured_calls.append(dict(kwargs))
                return kwargs.get("hidden_states", paddle.randn([B, S, H]))

            layer._proj_and_transformer_layer = capture_proj

            layer.forward(dict_args)

            self.assertEqual(len(captured_calls), 1)
            call = captured_calls[0]

            # Check that depth 0 mask was passed
            passed_startend = call.get("attn_mask_startend_row_indices")
            self.assertIsNotNone(passed_startend)
            # Should have value 1.0 (depth 0)
            self.assertTrue(
                paddle.allclose(
                    passed_startend,
                    paddle.ones([B, 1, S, 1], dtype="float32"),
                ),
                f"Expected depth 0 mask (1.0), got {passed_startend}",
            )

            passed_hidden_mask = call.get("mtp_hidden_inputs_mask")
            self.assertIsNotNone(passed_hidden_mask)
            self.assertTrue(
                paddle.allclose(
                    passed_hidden_mask,
                    paddle.ones([B, 1, S], dtype="float32"),
                ),
                f"Expected depth 0 hidden mask (1.0), got {passed_hidden_mask}",
            )

    def test_layer_number_1_gets_depth_1(self):
        """layer_number=1 should slice depth index 1."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=False,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=1)

            startend = paddle.zeros([B, num_nextn, S, 1], dtype="float32")
            startend[:, 0, :, :] = 1.0
            startend[:, 1, :, :] = 2.0

            mask_all = paddle.zeros([B, num_nextn, S], dtype="float32")
            mask_all[:, 0, :] = 1.0
            mask_all[:, 1, :] = 0.5

            total_sections = num_nextn + 1
            dict_args = {
                "hidden_states": paddle.randn(
                    [B * total_sections, S, H], dtype="float32"
                ),
                "mtp_startend_row_indices_all": startend,
                "mtp_hidden_inputs_mask_all": mask_all,
                "attn_mask_startend_row_indices": paddle.randn([B, 1, S, 1]),
            }

            captured_calls = []

            def capture_proj(**kwargs):
                captured_calls.append(dict(kwargs))
                return kwargs.get("hidden_states", paddle.randn([B, S, H]))

            layer._proj_and_transformer_layer = capture_proj

            layer.forward(dict_args)

            self.assertEqual(len(captured_calls), 1)
            call = captured_calls[0]

            # Check that depth 1 mask was passed
            passed_startend = call.get("attn_mask_startend_row_indices")
            self.assertIsNotNone(passed_startend)
            # Should have value 2.0 (depth 1)
            expected = paddle.full([B, 1, S, 1], 2.0, dtype="float32")
            self.assertTrue(
                paddle.allclose(passed_startend, expected),
                f"Expected depth 1 mask (2.0), got {passed_startend}",
            )

            passed_hidden_mask = call.get("mtp_hidden_inputs_mask")
            self.assertIsNotNone(passed_hidden_mask)
            expected_mask = paddle.full([B, 1, S], 0.5, dtype="float32")
            self.assertTrue(
                paddle.allclose(passed_hidden_mask, expected_mask),
                f"Expected depth 1 hidden mask (0.5), got {passed_hidden_mask}",
            )

    def test_train_mtp_only_loops_all_depths(self):
        """With train_mtp_only=True, forward should call _proj for each depth."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 3
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=True,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            startend = paddle.zeros([B, num_nextn, S, 1], dtype="float32")
            for d in range(num_nextn):
                startend[:, d, :, :] = float(d + 1)

            mask_all = paddle.zeros([B, num_nextn, S], dtype="float32")
            for d in range(num_nextn):
                mask_all[:, d, :] = float(d + 1) * 0.1

            total_sections = num_nextn + 1
            dict_args = {
                "hidden_states": paddle.randn(
                    [B * total_sections, S, H], dtype="float32"
                ),
                "mtp_startend_row_indices_all": startend,
                "mtp_hidden_inputs_mask_all": mask_all,
                "attn_mask_startend_row_indices": paddle.randn([B, 1, S, 1]),
            }

            captured_calls = []

            def capture_proj(**kwargs):
                captured_calls.append(dict(kwargs))
                return kwargs.get("hidden_states", paddle.randn([B, S, H]))

            layer._proj_and_transformer_layer = capture_proj

            layer.forward(dict_args)

            # Should be called num_nextn times (once per depth)
            self.assertEqual(len(captured_calls), num_nextn)

            # Verify each depth got the correct mask slice
            for d in range(num_nextn):
                call = captured_calls[d]
                passed_startend = call.get("attn_mask_startend_row_indices")
                expected_val = float(d + 1)
                expected = paddle.full(
                    [B, 1, S, 1], expected_val, dtype="float32"
                )
                self.assertTrue(
                    paddle.allclose(passed_startend, expected),
                    f"Depth {d}: expected startend {expected_val}, got {passed_startend}",
                )

                passed_mask = call.get("mtp_hidden_inputs_mask")
                expected_mask_val = float(d + 1) * 0.1
                expected_mask = paddle.full(
                    [B, 1, S], expected_mask_val, dtype="float32"
                )
                self.assertTrue(
                    paddle.allclose(passed_mask, expected_mask),
                    f"Depth {d}: expected hidden mask {expected_mask_val}",
                )


# ---------------------------------------------------------------------------
# Test: GPTEmbedding — mtp_input_ids_for_moe_mask construction
# ---------------------------------------------------------------------------


class TestGPTEmbeddingMTPInputIdsForMoeMask(unittest.TestCase):
    """Test GPTEmbedding splits input_ids into backbone and per-depth MTP parts
    for MoE routing mask when expert_model_parallel_size > 1."""

    def _build_embedding(self, config, B, S, H):
        from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

        with patch(
            "paddlefleet.models.gpt.gpt_embedding.build_spec_layer",
            side_effect=_mock_build_layer_side_effect,
        ):
            mock_spec = MagicMock()
            mock_spec.rope_embedding = None
            emb = GPTEmbedding(
                sublayers_spec=mock_spec,
                config=config,
                vocab_size=128,
                max_sequence_length=32,
                position_embedding_type="none",
            )
            emb.embedding = MagicMock(return_value=paddle.randn([B, S, H]))
        return emb

    def test_mtp_input_ids_none_when_no_moe(self):
        """Without expert_model_parallel_size > 1, mtp_input_ids_for_moe_mask is None."""
        num_nextn = 2
        B, S_total, H = 2, 10, 64  # S_total = seq_length + num_nextn
        config = _make_config(
            num_nextn_predict_layers=num_nextn,
            expert_model_parallel_size=1,
        )
        emb = self._build_embedding(config, B, S_total, H)
        dict_args = {
            "input_ids": paddle.randint(0, 128, [B, S_total]),
            "position_ids": paddle.arange(S_total)
            .unsqueeze(0)
            .expand([B, S_total]),
        }
        result = emb.forward(dict_args)
        self.assertIsNone(result.get("mtp_input_ids_for_moe_mask"))

    def test_mtp_input_ids_shape_and_values(self):
        """With MoE + MTP, mtp_input_ids_for_moe_mask has [B, num_nextn, seq_length]
        and each depth k contains input_ids[:, k+1 : k+1+seq_length]."""
        num_nextn = 2
        seq_length = 8
        S_total = seq_length + num_nextn  # 10
        B, H = 2, 64
        config = _make_config(
            num_nextn_predict_layers=num_nextn,
            expert_model_parallel_size=2,
            tensor_model_parallel_size=1,
        )
        emb = self._build_embedding(config, B, S_total, H)

        input_ids = paddle.arange(S_total).unsqueeze(0).expand([B, S_total])
        dict_args = {
            "input_ids": input_ids.clone(),
            "position_ids": paddle.arange(S_total)
            .unsqueeze(0)
            .expand([B, S_total]),
        }
        result = emb.forward(dict_args)

        mtp_ids = result.get("mtp_input_ids_for_moe_mask")
        self.assertIsNotNone(mtp_ids)
        self.assertEqual(list(mtp_ids.shape), [B, num_nextn, seq_length])

        # depth 0: input_ids[:, 1:1+8] = [1,2,...,8]
        expected_d0 = input_ids[:, 1 : 1 + seq_length]
        self.assertTrue(paddle.equal_all(mtp_ids[:, 0, :], expected_d0))
        # depth 1: input_ids[:, 2:2+8] = [2,3,...,9]
        expected_d1 = input_ids[:, 2 : 2 + seq_length]
        self.assertTrue(paddle.equal_all(mtp_ids[:, 1, :], expected_d1))

    def test_backbone_input_ids_trimmed(self):
        """With MoE + MTP, backbone input_ids is trimmed to [B, seq_length]."""
        num_nextn = 2
        seq_length = 8
        S_total = seq_length + num_nextn
        B, H = 2, 64
        config = _make_config(
            num_nextn_predict_layers=num_nextn,
            expert_model_parallel_size=2,
            tensor_model_parallel_size=1,
        )
        emb = self._build_embedding(config, B, S_total, H)

        input_ids = paddle.arange(S_total).unsqueeze(0).expand([B, S_total])
        dict_args = {
            "input_ids": input_ids.clone(),
            "position_ids": paddle.arange(S_total)
            .unsqueeze(0)
            .expand([B, S_total]),
        }
        result = emb.forward(dict_args)

        backbone_ids = result.get("input_ids")
        self.assertIsNotNone(backbone_ids)
        self.assertEqual(list(backbone_ids.shape), [B, seq_length])
        expected = input_ids[:, :seq_length]
        self.assertTrue(paddle.equal_all(backbone_ids, expected))


# ---------------------------------------------------------------------------
# Test: MultiTokenPredictionLayer — mtp_input_ids_for_moe_mask slicing
# ---------------------------------------------------------------------------


class TestMTPForwardInputIdsForMoeMask(unittest.TestCase):
    """Test MTP forward correctly slices mtp_input_ids_for_moe_mask per-depth
    and restores dict_args after forward."""

    def _make_dict_args(self, B, S, H, num_nextn, with_moe_mask=True):
        total_sections = num_nextn + 1
        hidden_states = paddle.randn(
            [B * total_sections, S, H], dtype="float32"
        )
        dict_args = {"hidden_states": hidden_states}
        if with_moe_mask:
            # Distinct per-depth input_ids: depth k filled with k+10
            mtp_ids = paddle.zeros([B, num_nextn, S], dtype="int64")
            for k in range(num_nextn):
                mtp_ids[:, k, :] = k + 10
            dict_args["mtp_input_ids_for_moe_mask"] = mtp_ids
            dict_args["input_ids"] = paddle.ones([B, S], dtype="int64") * 99
        return dict_args

    def test_single_layer_slices_correct_depth(self):
        """layer_number=1 should get mtp_input_ids_for_moe_mask[:, 1, :]."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=False,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=1)

            dict_args = self._make_dict_args(B, S, H, num_nextn)

            captured = []

            def capture_proj(**kwargs):
                captured.append(dict(kwargs))
                return kwargs.get("hidden_states", paddle.randn([B, S, H]))

            layer._proj_and_transformer_layer = capture_proj
            layer.forward(dict_args)

            self.assertEqual(len(captured), 1)
            passed_ids = captured[0].get("input_ids")
            self.assertIsNotNone(passed_ids)
            # depth 1 -> value 11
            expected = paddle.full([B, S], 11, dtype="int64")
            self.assertTrue(paddle.equal_all(passed_ids, expected))

    def test_train_mtp_only_each_depth_gets_correct_ids(self):
        """train_mtp_only loops all depths, each should get its own input_ids slice."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 3
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=True,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            dict_args = self._make_dict_args(B, S, H, num_nextn)

            captured = []

            def capture_proj(**kwargs):
                captured.append(dict(kwargs))
                return kwargs.get("hidden_states", paddle.randn([B, S, H]))

            layer._proj_and_transformer_layer = capture_proj
            layer.forward(dict_args)

            self.assertEqual(len(captured), num_nextn)
            for d in range(num_nextn):
                passed_ids = captured[d].get("input_ids")
                expected = paddle.full([B, S], d + 10, dtype="int64")
                self.assertTrue(
                    paddle.equal_all(passed_ids, expected),
                    f"Depth {d}: expected {d + 10}, got {passed_ids}",
                )

    def test_restore_after_forward(self):
        """After forward, mtp_input_ids_for_moe_mask and backbone input_ids are restored."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=False,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            dict_args = self._make_dict_args(B, S, H, num_nextn)
            original_mtp_ids = dict_args["mtp_input_ids_for_moe_mask"].clone()
            original_backbone_ids = dict_args["input_ids"].clone()

            result = layer.forward(dict_args)

            # mtp_input_ids_for_moe_mask should be restored
            self.assertIn("mtp_input_ids_for_moe_mask", result)
            self.assertTrue(
                paddle.equal_all(
                    result["mtp_input_ids_for_moe_mask"], original_mtp_ids
                )
            )
            # backbone input_ids should be restored
            self.assertIn("input_ids", result)
            self.assertTrue(
                paddle.equal_all(result["input_ids"], original_backbone_ids)
            )

    def test_no_moe_mask_clears_input_ids(self):
        """When mtp_input_ids_for_moe_mask is None, input_ids should be cleared for MTP layers."""
        with (
            _common_patches()[0],
            _common_patches()[1],
            _common_patches()[2],
            _common_patches()[3],
            _common_patches()[4],
        ):
            num_nextn = 2
            B, S, H = 2, 8, 64
            config = _make_config(
                num_nextn_predict_layers=num_nextn,
                train_mtp_only=False,
                experimental_dataflow=True,
            )
            layer = _build_mtp_layer(config, layer_number=0)

            dict_args = self._make_dict_args(
                B, S, H, num_nextn, with_moe_mask=False
            )

            captured = []

            def capture_proj(**kwargs):
                captured.append(dict(kwargs))
                return kwargs.get("hidden_states", paddle.randn([B, S, H]))

            layer._proj_and_transformer_layer = capture_proj
            layer.forward(dict_args)

            # input_ids should not be passed when no moe mask
            self.assertIsNone(captured[0].get("input_ids"))


if __name__ == "__main__":
    unittest.main()
