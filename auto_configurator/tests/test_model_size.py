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
Unit tests for AutoConfigurator core.model_size module.
"""

# ============================================================================
# Import Module Under Test
# ============================================================================
import sys
from pathlib import Path

import pytest

test_dir = Path(__file__).parent
src_dir = test_dir.parent
sys.path.insert(0, str(src_dir))

from auto_configurator.core.model_size import (
    GPT_BASED_MODELS,
    ModelSizeParams,
    calculate_model_size,
)

# ============================================================================
# calculate_model_size Tests
# ============================================================================


class TestCalculateModelSize:
    """Test cases for calculate_model_size function."""

    def test_calculate_gpt_model_size_small(self):
        """Test calculating small GPT model size."""
        size = calculate_model_size(
            vocab_size=32000,
            seq_length=2048,
            hidden_size=768,
            num_layers=12,
            ffn_size=3072,
            model_name="gpt",
        )
        # Approximate GPT-2 (117M parameters)
        assert 0.1 < size < 0.2

    def test_calculate_gpt_model_size_medium(self):
        """Test calculating medium GPT model size."""
        size = calculate_model_size(
            vocab_size=50000,
            seq_length=4096,
            hidden_size=2048,
            num_layers=24,
            ffn_size=8192,
            model_name="gpt",
        )
        # GPT-style medium model (~1.3B parameters with given dimensions)
        assert 1 < size < 2

    def test_calculate_gpt_model_size_large(self):
        """Test calculating large GPT model size."""
        size = calculate_model_size(
            vocab_size=92500,
            seq_length=8192,
            hidden_size=12288,
            num_layers=96,
            ffn_size=49152,
            model_name="gpt",
        )
        # Approximate GPT-175B parameters
        assert 150 < size < 200

    def test_calculate_bert_model_size(self):
        """Test calculating BERT model size raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="BERT models are currently not supported"
        ):
            calculate_model_size(
                vocab_size=30000,
                seq_length=512,
                hidden_size=768,
                num_layers=12,
                ffn_size=3072,
                model_name="bert",
            )

    def test_calculate_t5_model_size(self):
        """Test calculating T5 model size raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="T5/mT5 models are currently not supported",
        ):
            calculate_model_size(
                vocab_size=32100,
                seq_length=512,
                hidden_size=512,
                num_layers=6,
                ffn_size=3072,
                model_name="t5",
            )

    def test_calculate_with_default_ffn(self):
        """Test calculating model size with default FFN (4x hidden_size)."""
        size = calculate_model_size(
            vocab_size=32000,
            seq_length=2048,
            hidden_size=2048,
            num_layers=24,
            ffn_size=None,  # Should default to 4 * hidden_size
            model_name="gpt",
        )
        # FFN = 4 * 2048 = 8192
        # Medium GPT model (~1.3B parameters with given dimensions)
        assert 1 < size < 2

    def test_unsupported_model_type_raises_error(self):
        """Test that unsupported model type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            calculate_model_size(
                vocab_size=32000,
                seq_length=2048,
                hidden_size=2048,
                num_layers=24,
                ffn_size=8192,
                model_name="unsupported",
            )
        assert "Unsupported model" in str(exc_info.value)


# ============================================================================
# ModelSizeParams Tests
# ============================================================================


class TestModelSizeParams:
    """Test cases for ModelSizeParams class."""

    def test_init_gpt_small_model(self):
        """Test initializing GPT small model parameters."""
        params = ModelSizeParams(
            model_size_in_b=0.5,
            vocab_size=32000,
            seq_length=2048,
            model_name="gpt",
        )
        params.init_params()

        # For 0.5B target, the algorithm finds ~37 layers to match
        assert params.hidden_size == 1024
        assert params.num_attention_heads == 16
        assert params.ffn_size == 4096
        assert 0.0002 < params.learning_rate < 0.0005

    def test_init_gpt_medium_model(self):
        """Test initializing GPT medium model parameters."""
        params = ModelSizeParams(
            model_size_in_b=7.0,
            vocab_size=50000,
            seq_length=4096,
            model_name="gpt",
        )
        params.init_params()

        # For 7B target, the algorithm finds ~34 layers to match
        assert params.hidden_size == 4096
        assert params.num_attention_heads == 32
        assert params.ffn_size == 16384
        assert 0.0001 < params.learning_rate < 0.0002

    def test_init_gpt_large_model(self):
        """Test initializing GPT large model parameters."""
        params = ModelSizeParams(
            model_size_in_b=175.0,
            vocab_size=92500,
            seq_length=8192,
            model_name="gpt",
        )
        params.init_params()

        # For 175B target, algorithm should find appropriate layers
        assert params.hidden_size == 12288
        assert params.num_attention_heads == 96
        assert params.ffn_size == 49152
        assert 5e-05 < params.learning_rate < 7e-05

    def test_init_bert_small_model(self):
        """Test initializing BERT small model parameters raises NotImplementedError."""
        params = ModelSizeParams(
            model_size_in_b=0.5,
            vocab_size=30000,
            seq_length=512,
            model_name="bert",
        )
        with pytest.raises(
            NotImplementedError, match="BERT models are currently not supported"
        ):
            params.init_params()

    def test_init_bert_medium_model(self):
        """Test initializing BERT medium model parameters raises NotImplementedError."""
        params = ModelSizeParams(
            model_size_in_b=2.0,
            vocab_size=30000,
            seq_length=512,
            model_name="bert",
        )
        with pytest.raises(
            NotImplementedError, match="BERT models are currently not supported"
        ):
            params.init_params()

    def test_init_t5_small_model(self):
        """Test initializing T5 small model parameters raises NotImplementedError."""
        # Use a model size that works with algorithm's 1% margin
        params = ModelSizeParams(
            model_size_in_b=2.0,
            vocab_size=32100,
            seq_length=512,
            model_name="t5",
        )
        with pytest.raises(
            NotImplementedError,
            match="T5/mT5 models are currently not supported",
        ):
            params.init_params()

    def test_init_t5_large_model(self):
        """Test initializing T5 large model parameters raises NotImplementedError."""
        # Use a model size that works with algorithm's 1% margin
        params = ModelSizeParams(
            model_size_in_b=20.0,
            vocab_size=32100,
            seq_length=2048,
            model_name="t5",
        )
        with pytest.raises(
            NotImplementedError,
            match="T5/mT5 models are currently not supported",
        ):
            params.init_params()

    def test_ffn_size_default_to_4x_hidden(self):
        """Test that FFN size defaults to 4x hidden_size."""
        params = ModelSizeParams(
            model_size_in_b=5.0,
            vocab_size=32000,
            seq_length=2048,
            model_name="gpt",
        )
        params.ffn_size = None  # Set to None to test default
        params.init_params()

        # After init_params, ffn_size should be set
        assert params.ffn_size is not None
        assert params.ffn_size == 4 * params.hidden_size

    def test_unsupported_model_type_raises_error(self):
        """Test that unsupported model type raises ValueError."""
        params = ModelSizeParams(
            model_size_in_b=1.0,
            vocab_size=32000,
            seq_length=2048,
            model_name="unsupported",
        )
        with pytest.raises(ValueError):
            params.init_params()

    def test_gpt_based_models_in_gpt_list(self):
        """Test that GPT-based models use correct parameters."""
        for model in ["llama", "qwen", "mixtral", "mistral", "gemma"]:
            params = ModelSizeParams(
                model_size_in_b=7.0,
                vocab_size=32000,
                seq_length=4096,
                model_name=model,
            )
            params.init_params()

            # Should use same parameters as GPT
            # Should use same parameters as GPT
            assert params.hidden_size == 4096
            assert params.num_attention_heads == 32

    def test_too_large_gpt_model_raises_error(self):
        """Test that too large GPT model raises ValueError."""
        params = ModelSizeParams(
            model_size_in_b=1200,  # Too large
            vocab_size=32000,
            seq_length=4096,
            model_name="gpt",
        )
        with pytest.raises(ValueError, match="Model size for GPT must be"):
            params.init_params()

    def test_too_large_bert_model_raises_error(self):
        """Test that too large BERT model raises NotImplementedError."""
        params = ModelSizeParams(
            model_size_in_b=300,  # Too large
            vocab_size=32000,
            seq_length=512,
            model_name="bert",
        )
        with pytest.raises(
            NotImplementedError, match="BERT models are currently not supported"
        ):
            params.init_params()

    def test_too_large_t5_model_raises_error(self):
        """Test that too large T5 model raises NotImplementedError."""
        params = ModelSizeParams(
            model_size_in_b=300,  # Too large
            vocab_size=32000,
            seq_length=2048,
            model_name="t5",
        )
        with pytest.raises(
            NotImplementedError,
            match="T5/mT5 models are currently not supported",
        ):
            params.init_params()


# ============================================================================
# GPT_BASED_MODELS Tests
# ============================================================================


class TestGPTBasedModelsList:
    """Test cases for GPT_BASED_MODELS constant."""

    def test_gpt_based_models_list(self):
        """Test that GPT_BASED_MODELS contains expected values."""
        expected = ["gpt", "llama", "qwen", "mixtral", "mistral", "gemma"]
        for model in expected:
            assert model in GPT_BASED_MODELS

    def test_t5_not_in_gpt_based(self):
        """Test that T5 is not in GPT_BASED_MODELS."""
        assert "t5" not in GPT_BASED_MODELS
        assert "mt5" not in GPT_BASED_MODELS

    def test_bert_not_in_gpt_based(self):
        """Test that BERT is not in GPT_BASED_MODELS."""
        assert "bert" not in GPT_BASED_MODELS


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
