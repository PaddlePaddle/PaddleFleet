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


def _make_mock_config(**overrides):
    """Create a mock TransformerConfig with required attributes."""
    config = MagicMock()
    config.normalization = "LayerNorm"
    config.use_qk_norm = False
    config.qk_l2_norm = False
    config.hidden_dropout_prob = 0.0
    config.num_hidden_layers = 2
    config.n_routed_experts = 8
    config.moe_grouped_gemm = False
    config.use_qk_norm = False
    config.multi_latent_attention = False
    config.moe_layer_freq = 0
    config.num_empty_layers_add_in_head = 0
    config.block_attention_residuals = False
    config.specific_layer = None
    config.hidden_size = 128
    config.rms_norm_eps = 1e-5
    config.sequence_parallel = False
    config.tensor_model_parallel_size = 1
    config.tie_word_embeddings = False
    config.pipeline_model_parallel_size = 1
    config.mrope_section = None
    config.init_method = MagicMock()
    config.mtp_num_layers = 0
    config.num_nextn_predict_layers = None
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


class TestGetAttentionSpecSelfAttention(unittest.TestCase):
    """Test get_attention_spec with self_attention type."""

    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_self_attention_returns_layer_spec(self, mock_provider_cls):
        from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec

        mock_provider = MagicMock()
        mock_provider.column_parallel_linear.return_value = MagicMock()
        mock_provider.row_parallel_linear.return_value = MagicMock()
        mock_provider.core_attention.return_value = MagicMock()
        mock_provider.layer_norm.return_value = MagicMock()
        mock_provider_cls.return_value = mock_provider

        config = _make_mock_config()
        result = get_attention_spec(config, "self_attention")
        self.assertIsNotNone(result)

    def test_self_attention_asserts_config_not_none(self):
        from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec

        with self.assertRaises(AssertionError):
            get_attention_spec(None, "self_attention")


class TestGetAttentionSpecGatedDeltaNet(unittest.TestCase):
    """Test get_attention_spec with gated_delta_net type."""

    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_gated_delta_net_returns_layer_spec(self, mock_provider_cls):
        from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec

        mock_provider = MagicMock()
        mock_provider.column_parallel_linear.return_value = MagicMock()
        mock_provider.row_parallel_linear.return_value = MagicMock()
        mock_provider.layer_norm.return_value = MagicMock()
        mock_provider_cls.return_value = mock_provider

        config = _make_mock_config()
        result = get_attention_spec(config, "gated_delta_net")
        self.assertIsNotNone(result)


class TestGetAttentionSpecMultiLatentAttention(unittest.TestCase):
    """Test get_attention_spec with multi_latent_attention type."""

    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_mla_returns_layer_spec(self, mock_provider_cls):
        from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec

        mock_provider = MagicMock()
        mock_provider.column_parallel_linear.return_value = MagicMock()
        mock_provider.row_parallel_linear.return_value = MagicMock()
        mock_provider.core_attention.return_value = MagicMock()
        mock_provider.layer_norm.return_value = MagicMock()
        mock_provider_cls.return_value = mock_provider

        config = _make_mock_config()
        result = get_attention_spec(config, "multi_latent_attention")
        self.assertIsNotNone(result)

    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_mla_with_qk_l2_norm_raises(self, mock_provider_cls):
        from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec

        mock_provider = MagicMock()
        mock_provider_cls.return_value = mock_provider

        config = _make_mock_config(qk_l2_norm=True)
        with self.assertRaises(AssertionError):
            get_attention_spec(config, "multi_latent_attention")

    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_mla_with_dsa_when_index_n_heads_set(self, mock_provider_cls):
        from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec

        mock_provider = MagicMock()
        mock_provider.column_parallel_linear.return_value = MagicMock()
        mock_provider.row_parallel_linear.return_value = MagicMock()
        mock_provider.core_attention.return_value = MagicMock()
        mock_provider.layer_norm.return_value = MagicMock()
        mock_provider_cls.return_value = mock_provider

        config = _make_mock_config(index_n_heads=4)
        result = get_attention_spec(config, "multi_latent_attention")
        self.assertIsNotNone(result)


class TestGetAttentionSpecInvalidType(unittest.TestCase):
    """Test get_attention_spec with invalid attention type."""

    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_unknown_type_raises_value_error(self, mock_provider_cls):
        from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec

        mock_provider = MagicMock()
        mock_provider_cls.return_value = mock_provider

        config = _make_mock_config()
        with self.assertRaises(ValueError):
            get_attention_spec(config, "unknown_type")


class TestGetGPTDecoderLayersSpecDense(unittest.TestCase):
    """Test get_gpt_decoder_layers_spec with all dense layers."""

    @patch(
        "paddlefleet.models.gpt.gpt_layer_specs.get_mlp_layer_spec_for_backend"
    )
    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_dense_layers_creates_correct_count(
        self, mock_provider_cls, mock_mlp
    ):
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_layers_spec,
        )

        mock_provider = MagicMock()
        mock_provider.column_parallel_linear.return_value = MagicMock()
        mock_provider.row_parallel_linear.return_value = MagicMock()
        mock_provider.core_attention.return_value = MagicMock()
        mock_provider.layer_norm.return_value = MagicMock()
        mock_provider_cls.return_value = mock_provider
        mock_mlp.return_value = MagicMock()

        # Use a list with all zeros for dense layers (int 0 causes ZeroDivisionError)
        config = _make_mock_config(
            num_hidden_layers=4, moe_layer_freq=[0, 0, 0, 0]
        )
        result = get_gpt_decoder_layers_spec(config)
        self.assertEqual(len(result), 4)


class TestGetGPTDecoderLayersSpecMoE(unittest.TestCase):
    """Test get_gpt_decoder_layers_spec with MoE layers."""

    @patch(
        "paddlefleet.models.gpt.gpt_layer_specs.get_mlp_layer_spec_for_backend"
    )
    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_moe_pattern_with_int_freq(self, mock_provider_cls, mock_mlp):
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_layers_spec,
        )

        mock_provider = MagicMock()
        mock_provider.column_parallel_linear.return_value = MagicMock()
        mock_provider.row_parallel_linear.return_value = MagicMock()
        mock_provider.core_attention.return_value = MagicMock()
        mock_provider.layer_norm.return_value = MagicMock()
        mock_provider_cls.return_value = mock_provider
        mock_mlp.return_value = MagicMock()

        # Every 2 layers is an expert layer
        config = _make_mock_config(
            num_hidden_layers=6, moe_layer_freq=2, n_routed_experts=4
        )
        result = get_gpt_decoder_layers_spec(config)
        self.assertEqual(len(result), 6)

    @patch(
        "paddlefleet.models.gpt.gpt_layer_specs.get_mlp_layer_spec_for_backend"
    )
    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_moe_pattern_with_list_freq(self, mock_provider_cls, mock_mlp):
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_layers_spec,
        )

        mock_provider = MagicMock()
        mock_provider.column_parallel_linear.return_value = MagicMock()
        mock_provider.row_parallel_linear.return_value = MagicMock()
        mock_provider.core_attention.return_value = MagicMock()
        mock_provider.layer_norm.return_value = MagicMock()
        mock_provider_cls.return_value = mock_provider
        mock_mlp.return_value = MagicMock()

        config = _make_mock_config(
            num_hidden_layers=4,
            moe_layer_freq=[1, 0, 1, 0],
            n_routed_experts=4,
        )
        result = get_gpt_decoder_layers_spec(config)
        self.assertEqual(len(result), 4)

    @patch(
        "paddlefleet.models.gpt.gpt_layer_specs.get_mlp_layer_spec_for_backend"
    )
    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_moe_list_pattern_wrong_length_raises(
        self, mock_provider_cls, mock_mlp
    ):
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_layers_spec,
        )

        mock_provider = MagicMock()
        mock_provider.column_parallel_linear.return_value = MagicMock()
        mock_provider_cls.return_value = mock_provider
        mock_mlp.return_value = MagicMock()

        config = _make_mock_config(
            num_hidden_layers=4,
            moe_layer_freq=[1, 0],
            n_routed_experts=4,
        )
        with self.assertRaises(AssertionError):
            get_gpt_decoder_layers_spec(config)


class TestGetGPTDecoderLayersSpecInvalidPattern(unittest.TestCase):
    """Test get_gpt_decoder_layers_spec with invalid moe pattern."""

    @patch(
        "paddlefleet.models.gpt.gpt_layer_specs.get_mlp_layer_spec_for_backend"
    )
    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_invalid_moe_layer_freq_type_raises(
        self, mock_provider_cls, mock_mlp
    ):
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_layers_spec,
        )

        mock_provider = MagicMock()
        mock_provider_cls.return_value = mock_provider
        mock_mlp.return_value = MagicMock()

        config = _make_mock_config(moe_layer_freq="invalid")
        with self.assertRaises(ValueError):
            get_gpt_decoder_layers_spec(config)

    @patch(
        "paddlefleet.models.gpt.gpt_layer_specs.get_mlp_layer_spec_for_backend"
    )
    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_invalid_pattern_values_raise(self, mock_provider_cls, mock_mlp):
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_decoder_layers_spec,
        )

        mock_provider = MagicMock()
        mock_provider.column_parallel_linear.return_value = MagicMock()
        mock_provider.row_parallel_linear.return_value = MagicMock()
        mock_provider.core_attention.return_value = MagicMock()
        mock_provider.layer_norm.return_value = MagicMock()
        mock_provider_cls.return_value = mock_provider
        mock_mlp.return_value = MagicMock()

        config = _make_mock_config(num_hidden_layers=2, moe_layer_freq=[2, 0])
        with self.assertRaises(ValueError):
            get_gpt_decoder_layers_spec(config)


class TestGetGPTSpecBasic(unittest.TestCase):
    """Test get_gpt_spec function."""

    @patch(
        "paddlefleet.models.gpt.gpt_layer_specs.get_mtp_layer_spec_for_backend"
    )
    @patch(
        "paddlefleet.models.gpt.gpt_layer_specs.get_mlp_layer_spec_for_backend"
    )
    @patch("paddlefleet.models.gpt.gpt_layer_specs.LocalSpecProvider")
    def test_get_gpt_spec_returns_layer_spec(
        self, mock_provider_cls, mock_mlp, mock_mtp
    ):
        from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_spec

        mock_provider = MagicMock()
        mock_provider.column_parallel_linear.return_value = MagicMock()
        mock_provider.row_parallel_linear.return_value = MagicMock()
        mock_provider.core_attention.return_value = MagicMock()
        mock_provider.layer_norm.return_value = MagicMock()
        mock_provider.fuse_layernorm_and_linear.return_value = False
        mock_provider.column_parallel_layer_norm_linear.return_value = None
        mock_provider_cls.return_value = mock_provider
        mock_mlp.return_value = MagicMock()
        mock_mtp.return_value = []

        config = _make_mock_config(hidden_size=64, head_dim=16)
        config.params_dtype = "float32"
        config.add_bias_linear = False
        config.gather_output = False
        config.perform_initialization = False
        config.num_nextn_predict_layers = None

        result = get_gpt_spec(
            config=config,
            transformer_layers_spec=[MagicMock()],
            mtp_layers_spec=[],
            vocab_size=1024,
            max_sequence_length=64,
        )
        self.assertIsNotNone(result)


class TestGetGPTMTPSpecs(unittest.TestCase):
    """Test MTP layer spec generation."""

    @patch(
        "paddlefleet.models.gpt.gpt_layer_specs.get_mtp_layer_spec_for_backend"
    )
    def test_get_gpt_mtp_layers_spec_empty(self, mock_mtp):
        from unittest.mock import MagicMock

        from paddle.distributed.fleet.meta_parallel import LayerSpec

        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_mtp_layers_spec,
        )

        mock_mtp.return_value = []
        config = _make_mock_config(
            mtp_num_layers=0, num_nextn_predict_layers=None
        )

        # Create a real LayerSpec for the last element, since the source asserts isinstance(spec[-1], LayerSpec)
        mock_layer_spec = LayerSpec(MagicMock())
        result = get_gpt_mtp_layers_spec(config, spec=[mock_layer_spec])
        self.assertEqual(result, [])


class TestGetMLPLayerSpecForBackend(unittest.TestCase):
    """Test get_mlp_layer_spec_for_backend helper."""

    @patch(
        "paddlefleet.models.gpt.gpt_layer_specs.get_moe_layer_spec_for_backend"
    )
    def test_dense_mlp_no_experts(self, mock_moe):
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_mlp_layer_spec_for_backend,
        )

        mock_moe.return_value = MagicMock()
        backend = MagicMock()
        backend.fuse_layernorm_and_linear.return_value = False
        backend.column_parallel_linear.return_value = MagicMock()
        backend.row_parallel_linear.return_value = MagicMock()

        result = get_mlp_layer_spec_for_backend(
            backend=backend, num_experts=None
        )
        self.assertIsNotNone(result)

    @patch(
        "paddlefleet.models.gpt.gpt_layer_specs.get_moe_layer_spec_for_backend"
    )
    def test_moe_mlp_with_experts(self, mock_moe):
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_mlp_layer_spec_for_backend,
        )

        mock_moe.return_value = MagicMock()
        backend = MagicMock()

        result = get_mlp_layer_spec_for_backend(
            backend=backend, num_experts=8, moe_grouped_gemm=True
        )
        mock_moe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
