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
Unit tests for AutoConfigurator paddlefleet_adapters module.
"""

from dataclasses import dataclass

import pytest

# ============================================================================
# Mock PaddleFleet Config Classes
# ============================================================================


@dataclass
class MockTransformerConfig:
    """Mock TransformerConfig for testing."""

    num_hidden_layers: int = 24
    hidden_size: int = 2048
    num_attention_heads: int = 32
    intermediate_size: int = 8192
    max_sequence_length: int = 4096
    vocab_size: int = 32000
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1


@dataclass
class MockGPTConfig:
    """Mock GPTConfig for testing."""

    num_hidden_layers: int = 24
    hidden_size: int = 2048
    num_attention_heads: int = 32
    intermediate_size: int = 8192
    max_sequence_length: int = 4096
    vocab_size: int = 32000
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    share_embeddings_and_output_weights: bool = False
    moe_grouped_gemm: bool = False
    parallel_output: bool = True


@dataclass
class MockBertConfig:
    """Mock BERT-style config for testing."""

    num_hidden_layers: int = 24
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    vocab_size: int = 30000
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1


# ============================================================================
# Import Module Under Test
# ============================================================================

import sys
from pathlib import Path

# Add parent directory to path
test_dir = Path(__file__).parent
src_dir = test_dir.parent
sys.path.insert(0, str(src_dir))

from auto_configurator.paddlefleet_adapters import (
    PaddleFleetCombinedConfigAdapter,
    PaddleFleetDataConfigAdapter,
    PaddleFleetModelConfigAdapter,
    PaddleFleetParallelConfigAdapter,
    PaddleFleetRecipe,
    PaddleFleetTrainingConfigAdapter,
    create_paddlefleet_adapter,
)

# ============================================================================
# PaddleFleetModelConfigAdapter Tests
# ============================================================================


class TestPaddleFleetModelConfigAdapter:
    """Test cases for PaddleFleetModelConfigAdapter."""

    def test_get_num_layers(self):
        """Test getting number of layers."""
        config = MockTransformerConfig(num_hidden_layers=32)
        adapter = PaddleFleetModelConfigAdapter(config)
        assert adapter.get_num_layers() == 32

    def test_set_num_layers(self):
        """Test setting number of layers."""
        config = MockTransformerConfig(num_hidden_layers=24)
        adapter = PaddleFleetModelConfigAdapter(config)
        adapter.set_num_layers(48)
        assert adapter.get_num_layers() == 48
        assert config.num_hidden_layers == 48

    def test_get_hidden_size(self):
        """Test getting hidden size."""
        config = MockTransformerConfig(hidden_size=4096)
        adapter = PaddleFleetModelConfigAdapter(config)
        assert adapter.get_hidden_size() == 4096

    def test_set_hidden_size(self):
        """Test setting hidden size."""
        config = MockTransformerConfig(hidden_size=2048)
        adapter = PaddleFleetModelConfigAdapter(config)
        adapter.set_hidden_size(4096)
        assert adapter.get_hidden_size() == 4096
        assert config.hidden_size == 4096

    def test_get_num_attention_heads(self):
        """Test getting number of attention heads."""
        config = MockTransformerConfig(num_attention_heads=64)
        adapter = PaddleFleetModelConfigAdapter(config)
        assert adapter.get_num_attention_heads() == 64

    def test_set_num_attention_heads(self):
        """Test setting number of attention heads."""
        config = MockTransformerConfig(num_attention_heads=32)
        adapter = PaddleFleetModelConfigAdapter(config)
        adapter.set_num_attention_heads(64)
        assert adapter.get_num_attention_heads() == 64
        assert config.num_attention_heads == 64

    def test_get_ffn_hidden_size(self):
        """Test getting FFN hidden size."""
        config = MockTransformerConfig(intermediate_size=16384)
        adapter = PaddleFleetModelConfigAdapter(config)
        assert adapter.get_ffn_hidden_size() == 16384

    def test_set_ffn_hidden_size(self):
        """Test setting FFN hidden size."""
        config = MockTransformerConfig(intermediate_size=8192)
        adapter = PaddleFleetModelConfigAdapter(config)
        adapter.set_ffn_hidden_size(16384)
        assert adapter.get_ffn_hidden_size() == 16384
        assert config.intermediate_size == 16384

    def test_get_seq_length(self):
        """Test getting sequence length."""
        config = MockTransformerConfig(max_sequence_length=8192)
        adapter = PaddleFleetModelConfigAdapter(config)
        assert adapter.get_seq_length() == 8192

    def test_set_seq_length(self):
        """Test setting sequence length."""
        config = MockTransformerConfig(max_sequence_length=4096)
        adapter = PaddleFleetModelConfigAdapter(config)
        adapter.set_seq_length(8192)
        assert adapter.get_seq_length() == 8192
        assert config.max_sequence_length == 8192

    def test_get_vocab_size(self):
        """Test getting vocabulary size."""
        config = MockGPTConfig(vocab_size=50000)
        adapter = PaddleFleetModelConfigAdapter(config)
        assert adapter.get_vocab_size() == 50000

    def test_set_vocab_size(self):
        """Test setting vocabulary size."""
        config = MockGPTConfig(vocab_size=32000)
        adapter = PaddleFleetModelConfigAdapter(config)
        adapter.set_vocab_size(50000)
        assert adapter.get_vocab_size() == 50000
        assert config.vocab_size == 50000

    def test_set_vocab_size_on_non_gpt_config(self):
        """Test setting vocab size on non-GPT config (uses setattr)."""
        config = MockBertConfig()
        adapter = PaddleFleetModelConfigAdapter(config)
        adapter.set_vocab_size(45000)
        assert adapter.get_vocab_size() == 45000

    def test_get_model_type_gpt(self):
        """Test getting model type for GPT."""
        config = MockGPTConfig()
        adapter = PaddleFleetModelConfigAdapter(config)
        assert adapter.get_model_type() == "gpt"

    def test_get_model_type_bert(self):
        """Test getting model type for BERT."""
        config = MockBertConfig()
        adapter = PaddleFleetModelConfigAdapter(config)
        assert adapter.get_model_type() == "bert"

    def test_get_model_type_default(self):
        """Test default model type (GPT-based)."""
        config = MockTransformerConfig()
        adapter = PaddleFleetModelConfigAdapter(config)
        assert adapter.get_model_type() == "gpt"


# ============================================================================
# PaddleFleetParallelConfigAdapter Tests
# ============================================================================


class TestPaddleFleetParallelConfigAdapter:
    """Test cases for PaddleFleetParallelConfigAdapter."""

    def test_get_tensor_parallel_size(self):
        """Test getting tensor parallel size."""
        config = MockTransformerConfig(tensor_model_parallel_size=4)
        adapter = PaddleFleetParallelConfigAdapter(config)
        assert adapter.get_tensor_parallel_size() == 4

    def test_set_tensor_parallel_size(self):
        """Test setting tensor parallel size."""
        config = MockTransformerConfig(tensor_model_parallel_size=1)
        adapter = PaddleFleetParallelConfigAdapter(config)
        adapter.set_tensor_parallel_size(8)
        assert adapter.get_tensor_parallel_size() == 8
        assert config.tensor_model_parallel_size == 8

    def test_get_pipeline_parallel_size(self):
        """Test getting pipeline parallel size."""
        config = MockTransformerConfig(pipeline_model_parallel_size=4)
        adapter = PaddleFleetParallelConfigAdapter(config)
        assert adapter.get_pipeline_parallel_size() == 4

    def test_set_pipeline_parallel_size(self):
        """Test setting pipeline parallel size."""
        config = MockTransformerConfig(pipeline_model_parallel_size=1)
        adapter = PaddleFleetParallelConfigAdapter(config)
        adapter.set_pipeline_parallel_size(8)
        assert adapter.get_pipeline_parallel_size() == 8
        assert config.pipeline_model_parallel_size == 8

    def test_get_virtual_pipeline_size(self):
        """Test getting virtual pipeline size."""
        config = MockTransformerConfig(virtual_pipeline_model_parallel_size=4)
        adapter = PaddleFleetParallelConfigAdapter(config)
        assert adapter.get_virtual_pipeline_size() == 4

    def test_set_virtual_pipeline_size(self):
        """Test setting virtual pipeline size."""
        config = MockTransformerConfig(
            virtual_pipeline_model_parallel_size=None
        )
        adapter = PaddleFleetParallelConfigAdapter(config)
        adapter.set_virtual_pipeline_size(4)
        assert adapter.get_virtual_pipeline_size() == 4
        assert config.virtual_pipeline_model_parallel_size == 4

    def test_set_virtual_pipeline_size_to_none(self):
        """Test setting virtual pipeline size to None."""
        config = MockTransformerConfig(virtual_pipeline_model_parallel_size=4)
        adapter = PaddleFleetParallelConfigAdapter(config)
        adapter.set_virtual_pipeline_size(None)
        assert adapter.get_virtual_pipeline_size() is None
        assert config.virtual_pipeline_model_parallel_size is None

    def test_get_context_parallel_size(self):
        """Test getting context parallel size."""
        config = MockTransformerConfig(context_parallel_size=2)
        adapter = PaddleFleetParallelConfigAdapter(config)
        assert adapter.get_context_parallel_size() == 2

    def test_set_context_parallel_size(self):
        """Test setting context parallel size."""
        config = MockTransformerConfig(context_parallel_size=1)
        adapter = PaddleFleetParallelConfigAdapter(config)
        adapter.set_context_parallel_size(4)
        assert adapter.get_context_parallel_size() == 4
        assert config.context_parallel_size == 4

    def test_get_expert_parallel_size(self):
        """Test getting expert parallel size."""
        config = MockTransformerConfig(expert_model_parallel_size=4)
        adapter = PaddleFleetParallelConfigAdapter(config)
        assert adapter.get_expert_parallel_size() == 4

    def test_set_expert_parallel_size(self):
        """Test setting expert parallel size."""
        config = MockTransformerConfig(expert_model_parallel_size=1)
        adapter = PaddleFleetParallelConfigAdapter(config)
        adapter.set_expert_parallel_size(4)
        assert adapter.get_expert_parallel_size() == 4
        assert config.expert_model_parallel_size == 4


# ============================================================================
# PaddleFleetDataConfigAdapter Tests
# ============================================================================


class TestPaddleFleetDataConfigAdapter:
    """Test cases for PaddleFleetDataConfigAdapter."""

    def test_get_micro_batch_size(self):
        """Test getting micro batch size."""
        adapter = PaddleFleetDataConfigAdapter(micro_batch_size=4)
        assert adapter.get_micro_batch_size() == 4

    def test_set_micro_batch_size(self):
        """Test setting micro batch size."""
        adapter = PaddleFleetDataConfigAdapter()
        adapter.set_micro_batch_size(8)
        assert adapter.get_micro_batch_size() == 8

    def test_get_global_batch_size(self):
        """Test getting global batch size."""
        adapter = PaddleFleetDataConfigAdapter(global_batch_size=2048)
        assert adapter.get_global_batch_size() == 2048

    def test_set_global_batch_size(self):
        """Test setting global batch size."""
        adapter = PaddleFleetDataConfigAdapter()
        adapter.set_global_batch_size(4096)
        assert adapter.get_global_batch_size() == 4096

    def test_default_values(self):
        """Test default values."""
        adapter = PaddleFleetDataConfigAdapter()
        assert adapter.get_micro_batch_size() == 1
        assert adapter.get_global_batch_size() == 512


# ============================================================================
# PaddleFleetTrainingConfigAdapter Tests
# ============================================================================


class TestPaddleFleetTrainingConfigAdapter:
    """Test cases for PaddleFleetTrainingConfigAdapter."""

    def test_get_num_nodes(self):
        """Test getting number of nodes."""
        adapter = PaddleFleetTrainingConfigAdapter(num_nodes=8)
        assert adapter.get_num_nodes() == 8

    def test_get_num_gpus_per_node(self):
        """Test getting GPUs per node."""
        adapter = PaddleFleetTrainingConfigAdapter(num_gpus_per_node=16)
        assert adapter.get_num_gpus_per_node() == 16

    def test_get_max_steps(self):
        """Test getting max steps."""
        adapter = PaddleFleetTrainingConfigAdapter(max_steps=100000)
        assert adapter.get_max_steps() == 100000

    def test_set_max_steps(self):
        """Test setting max steps."""
        adapter = PaddleFleetTrainingConfigAdapter()
        adapter.set_max_steps(50000)
        assert adapter.get_max_steps() == 50000

    def test_get_max_steps_none(self):
        """Test getting max steps when None."""
        adapter = PaddleFleetTrainingConfigAdapter(max_steps=None)
        assert adapter.get_max_steps() is None

    def test_get_recompute_config(self):
        """Test getting recompute config."""
        adapter = PaddleFleetTrainingConfigAdapter()
        adapter.set_recompute_config(
            granularity="selective",
            method="block",
            num_layers=4,
            modules=["attention", "mlp"],
        )
        config = adapter.get_recompute_config()
        assert config["granularity"] == "selective"
        assert config["method"] == "block"
        assert config["num_layers"] == 4
        assert config["modules"] == ["attention", "mlp"]

    def test_default_values(self):
        """Test default values."""
        adapter = PaddleFleetTrainingConfigAdapter()
        assert adapter.get_num_nodes() == 1
        assert adapter.get_num_gpus_per_node() == 8
        assert adapter.get_max_steps() is None
        config = adapter.get_recompute_config()
        assert all(v is None for v in config.values())


# ============================================================================
# PaddleFleetRecipe Tests
# ============================================================================


class TestPaddleFleetRecipe:
    """Test cases for PaddleFleetRecipe."""

    def test_total_gpus_property(self):
        """Test total_gpus property."""
        recipe = PaddleFleetRecipe(
            model_config=MockTransformerConfig(),
            num_nodes=4,
            num_gpus_per_node=8,
        )
        assert recipe.total_gpus == 32

    def test_default_values(self):
        """Test default recipe values."""
        recipe = PaddleFleetRecipe(model_config=MockTransformerConfig())
        assert recipe.micro_batch_size == 1
        assert recipe.global_batch_size == 512
        assert recipe.num_nodes == 1
        assert recipe.num_gpus_per_node == 8
        assert recipe.max_steps is None
        assert recipe.log_dir is None


# ============================================================================
# PaddleFleetCombinedConfigAdapter Tests
# ============================================================================


class TestPaddleFleetCombinedConfigAdapter:
    """Test cases for PaddleFleetCombinedConfigAdapter."""

    def test_get_model_config(self):
        """Test getting model config adapter."""
        model_config = MockTransformerConfig()
        adapter = PaddleFleetCombinedConfigAdapter(model_config=model_config)
        assert isinstance(
            adapter.get_model_config(), PaddleFleetModelConfigAdapter
        )

    def test_get_parallel_config_from_model(self):
        """Test getting parallel config from model config."""
        model_config = MockTransformerConfig(
            tensor_model_parallel_size=4, pipeline_model_parallel_size=2
        )
        adapter = PaddleFleetCombinedConfigAdapter(model_config=model_config)
        parallel = adapter.get_parallel_strategy()
        assert parallel.get_tensor_parallel_size() == 4
        assert parallel.get_pipeline_parallel_size() == 2

    def test_get_parallel_config_from_separate(self):
        """Test getting parallel config from separate config."""
        model_config = MockTransformerConfig()
        parallel_config = MockTransformerConfig(
            tensor_model_parallel_size=4, pipeline_model_parallel_size=2
        )
        adapter = PaddleFleetCombinedConfigAdapter(
            model_config=model_config, parallel_config=parallel_config
        )
        parallel = adapter.get_parallel_strategy()
        assert parallel.get_tensor_parallel_size() == 4
        assert parallel.get_pipeline_parallel_size() == 2

    def test_get_data_config_default(self):
        """Test getting data config with default values."""
        adapter = PaddleFleetCombinedConfigAdapter(
            model_config=MockTransformerConfig()
        )
        data = adapter.get_data_config()
        assert data.get_micro_batch_size() == 1
        assert data.get_global_batch_size() == 512

    def test_get_data_config_custom(self):
        """Test getting data config with custom values."""
        data_adapter = PaddleFleetDataConfigAdapter(
            micro_batch_size=4, global_batch_size=2048
        )
        adapter = PaddleFleetCombinedConfigAdapter(
            model_config=MockTransformerConfig(), data_config=data_adapter
        )
        data = adapter.get_data_config()
        assert data.get_micro_batch_size() == 4
        assert data.get_global_batch_size() == 2048

    def test_get_training_config_default(self):
        """Test getting training config with default values."""
        adapter = PaddleFleetCombinedConfigAdapter(
            model_config=MockTransformerConfig()
        )
        training = adapter.get_training_config()
        assert training.get_num_nodes() == 1
        assert training.get_num_gpus_per_node() == 8

    def test_get_training_config_custom(self):
        """Test getting training config with custom values."""
        training_adapter = PaddleFleetTrainingConfigAdapter(
            num_nodes=4, num_gpus_per_node=8, max_steps=10000
        )
        adapter = PaddleFleetCombinedConfigAdapter(
            model_config=MockTransformerConfig(),
            training_config=training_adapter,
        )
        training = adapter.get_training_config()
        assert training.get_num_nodes() == 4
        assert training.get_num_gpus_per_node() == 8
        assert training.get_max_steps() == 10000


# ============================================================================
# create_paddlefleet_adapter Tests
# ============================================================================


class TestCreatePaddleFleetAdapter:
    """Test cases for create_paddlefleet_adapter factory function."""

    def test_create_with_minimal_recipe(self):
        """Test creating adapter with minimal recipe."""
        recipe = PaddleFleetRecipe(model_config=MockTransformerConfig())
        adapter = create_paddlefleet_adapter(recipe)
        assert adapter.get_model_config().get_num_layers() == 24
        assert adapter.get_data_config().get_micro_batch_size() == 1
        assert adapter.get_training_config().get_num_nodes() == 1

    def test_create_with_full_recipe(self):
        """Test creating adapter with full recipe."""
        recipe = PaddleFleetRecipe(
            model_config=MockTransformerConfig(),
            parallel_config=MockTransformerConfig(
                tensor_model_parallel_size=4, pipeline_model_parallel_size=2
            ),
            micro_batch_size=4,
            global_batch_size=2048,
            num_nodes=4,
            num_gpus_per_node=8,
            max_steps=10000,
        )
        adapter = create_paddlefleet_adapter(recipe)
        model = adapter.get_model_config()
        parallel = adapter.get_parallel_strategy()
        data = adapter.get_data_config()
        training = adapter.get_training_config()

        assert model.get_num_layers() == 24
        assert parallel.get_tensor_parallel_size() == 4
        assert parallel.get_pipeline_parallel_size() == 2
        assert data.get_micro_batch_size() == 4
        assert data.get_global_batch_size() == 2048
        assert training.get_num_nodes() == 4
        assert training.get_num_gpus_per_node() == 8
        assert training.get_max_steps() == 10000


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
