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
Integration tests for AutoConfigurator.

These tests verify the full workflow of AutoConfigurator,
including initialization, configuration generation, and adapter functionality.
"""

# ============================================================================
# Import Module Under Test
# ============================================================================
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

test_dir = Path(__file__).parent
src_dir = test_dir.parent
sys.path.insert(0, str(src_dir))

from auto_configurator import (
    AutoConfigurator,
    PaddleFleetRecipe,
    estimate_model_size,
)

# ============================================================================
# Mock PaddleFleet Config Classes
# ============================================================================


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


# ============================================================================
# AutoConfigurator Integration Tests
# ============================================================================


class TestAutoConfiguratorInitialization:
    """Test cases for AutoConfigurator initialization."""

    def test_initialization_with_valid_gpt_config(self):
        """Test initializing AutoConfigurator with valid GPT config."""
        recipe = PaddleFleetRecipe(
            model_config=MockGPTConfig(),
            micro_batch_size=4,
            global_batch_size=2048,
            num_nodes=4,
            num_gpus_per_node=8,
        )

        runner = AutoConfigurator(
            recipe=recipe,
            path_to_logs="/tmp/test_logs",
            mode="pretrain",
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes=[1, 2, 4, 8],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            num_tokens_in_b=1400,
            tflops_per_gpu=989,
            max_minutes_per_run=30,
            max_training_days=2,
            max_steps_per_run=50,
            vocab_size=32000,
            calculate_model_size=False,
        )

        # Should initialize without errors
        assert runner.model_type == "gpt"
        assert runner.seq_length == 4096
        assert runner.gpu_count == 32

    def test_initialization_with_valid_llama_config(self):
        """Test initializing AutoConfigurator with valid Llama config."""

        @dataclass
        class MockLlamaConfig:
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

        recipe = PaddleFleetRecipe(
            model_config=MockLlamaConfig(),
            micro_batch_size=4,
            global_batch_size=2048,
            num_nodes=4,
            num_gpus_per_node=8,
        )

        runner = AutoConfigurator(
            recipe=recipe,
            path_to_logs="/tmp/test_logs",
            mode="pretrain",
            gpu_memory_gb=80,
        )

        # Should identify as GPT-based model
        assert runner.model_type == "gpt"

    def test_initialization_invalid_gpu_memory(self):
        """Test that invalid GPU memory raises ValueError."""
        recipe = PaddleFleetRecipe(model_config=MockGPTConfig())

        with pytest.raises(ValueError, match="gpu_memory_gb must be 40 or 80"):
            AutoConfigurator(
                recipe=recipe,
                path_to_logs="/tmp/test_logs",
                gpu_memory_gb=64,  # Invalid
            )

    def test_initialization_invalid_sequence_length(self):
        """Test that invalid sequence length for GPT raises ValueError."""

        @dataclass
        class InvalidSeqConfig:
            num_hidden_layers: int = 24
            hidden_size: int = 2048
            num_attention_heads: int = 32
            intermediate_size: int = 8192
            max_sequence_length: int = 500  # Invalid: not a multiple of 1024
            vocab_size: int = 32000
            tensor_model_parallel_size: int = 1
            pipeline_model_parallel_size: int = 1
            virtual_pipeline_model_parallel_size: int | None = None
            context_parallel_size: int = 1
            expert_model_parallel_size: int = 1

        recipe = PaddleFleetRecipe(model_config=InvalidSeqConfig())

        with pytest.raises(ValueError, match="seq_length.*not supported"):
            AutoConfigurator(
                recipe=recipe, path_to_logs="/tmp/test_logs", gpu_memory_gb=80
            )

    def test_initialization_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        recipe = PaddleFleetRecipe(model_config=MockGPTConfig())

        with pytest.raises(
            ValueError, match="Mode must be 'pretrain' or 'finetune'"
        ):
            AutoConfigurator(
                recipe=recipe, path_to_logs="/tmp/test_logs", mode="invalid"
            )

    def test_initialization_finetune_without_explicit_tp_pp(self):
        """Test that finetune mode without explicit TP/PP raises ValueError."""
        recipe = PaddleFleetRecipe(model_config=MockGPTConfig())

        with pytest.raises(
            ValueError, match="tensor_parallel_sizes must be specified"
        ):
            AutoConfigurator(
                recipe=recipe,
                path_to_logs="/tmp/test_logs",
                mode="finetune",
                tensor_parallel_sizes="auto",  # Not allowed in finetune
            )

    def test_initialization_auto_context_parallel(self):
        """Test that 'auto' for context parallel raises ValueError."""
        recipe = PaddleFleetRecipe(model_config=MockGPTConfig())

        with pytest.raises(
            ValueError, match="'auto' not supported for context_parallel_sizes"
        ):
            AutoConfigurator(
                recipe=recipe,
                path_to_logs="/tmp/test_logs",
                context_parallel_sizes=[
                    "auto"
                ],  # List containing 'auto' should be caught
            )

    def test_initialization_auto_expert_parallel(self):
        """Test that 'auto' for expert parallel raises ValueError."""
        recipe = PaddleFleetRecipe(model_config=MockGPTConfig())

        with pytest.raises(
            ValueError, match="'auto' not supported for expert_parallel_sizes"
        ):
            AutoConfigurator(
                recipe=recipe,
                path_to_logs="/tmp/test_logs",
                expert_parallel_sizes=[
                    "auto"
                ],  # List containing 'auto' should be caught
            )

    def test_initialization_zero_gpu_count(self):
        """Test that zero GPU count raises ValueError."""
        recipe = PaddleFleetRecipe(
            model_config=MockGPTConfig(), num_nodes=0, num_gpus_per_node=8
        )

        with pytest.raises(
            ValueError, match="num_nodes.*gpus_per_node must be > 0"
        ):
            AutoConfigurator(recipe=recipe, path_to_logs="/tmp/test_logs")

    def test_initialization_invalid_training_days(self):
        """Test that negative training days raises ValueError."""
        recipe = PaddleFleetRecipe(model_config=MockGPTConfig())

        with pytest.raises(ValueError, match="num_tokens_in_b must be > 0"):
            AutoConfigurator(
                recipe=recipe,
                path_to_logs="/tmp/test_logs",
                num_tokens_in_b=-1,  # Invalid
            )


class TestAutoConfiguratorAdapters:
    """Test cases for AutoConfigurator adapter methods."""

    def test_get_model_config_adapter(self):
        """Test getting model config adapter."""
        recipe = PaddleFleetRecipe(model_config=MockGPTConfig())
        runner = AutoConfigurator(recipe=recipe, path_to_logs="/tmp/test_logs")

        model_adapter = runner.get_model_config()
        assert model_adapter is not None
        assert model_adapter.get_num_layers() == 24
        assert model_adapter.get_hidden_size() == 2048
        assert model_adapter.get_num_attention_heads() == 32
        assert model_adapter.get_ffn_hidden_size() == 8192

    def test_get_parallel_strategy_adapter(self):
        """Test getting parallel strategy adapter."""
        recipe = PaddleFleetRecipe(
            model_config=MockGPTConfig(
                tensor_model_parallel_size=4, pipeline_model_parallel_size=2
            )
        )
        runner = AutoConfigurator(recipe=recipe, path_to_logs="/tmp/test_logs")

        parallel_adapter = runner.get_parallel_config()
        assert parallel_adapter is not None
        assert parallel_adapter.get_tensor_parallel_size() == 4
        assert parallel_adapter.get_pipeline_parallel_size() == 2
        assert parallel_adapter.get_context_parallel_size() == 1
        assert parallel_adapter.get_expert_parallel_size() == 1

    def test_get_data_config_adapter(self):
        """Test getting data config adapter."""
        recipe = PaddleFleetRecipe(
            model_config=MockGPTConfig(),
            micro_batch_size=4,
            global_batch_size=2048,
        )
        runner = AutoConfigurator(recipe=recipe, path_to_logs="/tmp/test_logs")

        data_adapter = runner.get_data_config()
        assert data_adapter is not None
        assert data_adapter.get_micro_batch_size() == 4
        assert data_adapter.get_global_batch_size() == 2048

    def test_get_training_config_adapter(self):
        """Test getting training config adapter."""
        recipe = PaddleFleetRecipe(
            model_config=MockGPTConfig(),
            num_nodes=4,
            num_gpus_per_node=8,
            max_steps=10000,
        )
        runner = AutoConfigurator(recipe=recipe, path_to_logs="/tmp/test_logs")

        training_adapter = runner.get_training_config()
        assert training_adapter is not None
        assert training_adapter.get_num_nodes() == 4
        assert training_adapter.get_num_gpus_per_node() == 8
        assert training_adapter.get_max_steps() == 10000


class TestAutoConfiguratorGeneration:
    """Test cases for configuration generation."""

    def test_generate_configs_basic(self):
        """Test basic configuration generation."""
        recipe = PaddleFleetRecipe(
            model_config=MockGPTConfig(),
            micro_batch_size=4,
            global_batch_size=2048,
            num_nodes=4,
            num_gpus_per_node=8,
        )

        runner = AutoConfigurator(
            recipe=recipe,
            path_to_logs="/tmp/test_logs",
            tensor_parallel_sizes=[1, 2],
            pipeline_parallel_sizes=[1, 2],
            micro_batch_sizes=[1, 2, 4],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            max_steps_per_run=50,
            calculate_model_size=False,
        )

        # Mock the internal model_size_in_b
        runner.model_size_in_b = 7.0

        from auto_configurator.core.grid_search import (
            generate_grid_search_configs,
        )

        configs = generate_grid_search_configs(
            runner_config=runner, adapter=runner.get_adapter()
        )

        # Should generate configurations
        assert len(configs) > 0

        # Check that generated configs have expected format
        for name, config in configs.items():
            assert "gpt" in name
            assert "tp_" in name
            assert "pp_" in name
            assert "mbs_" in name
            assert config.tensor_parallel_size in [1, 2]
            assert config.pipeline_parallel_size in [1, 2]
            assert config.micro_batch_size in [1, 2, 4]

    def test_generate_configs_with_80gb(self):
        """Test configuration generation with 80GB GPU."""
        recipe = PaddleFleetRecipe(
            model_config=MockGPTConfig(), num_nodes=4, num_gpus_per_node=8
        )

        runner = AutoConfigurator(
            recipe=recipe,
            path_to_logs="/tmp/test_logs",
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            max_steps_per_run=50,
            calculate_model_size=False,
        )

        runner.model_size_in_b = 7.0

        from auto_configurator.core.grid_search import (
            generate_grid_search_configs,
        )

        configs = generate_grid_search_configs(
            runner_config=runner, adapter=runner.get_adapter()
        )

        # Should generate configurations for 80GB
        assert len(configs) > 0

    def test_generate_configs_with_40gb(self):
        """Test configuration generation with 40GB GPU."""
        recipe = PaddleFleetRecipe(
            model_config=MockGPTConfig(), num_nodes=2, num_gpus_per_node=8
        )

        runner = AutoConfigurator(
            recipe=recipe,
            path_to_logs="/tmp/test_logs",
            gpu_memory_gb=40,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            max_steps_per_run=50,
            calculate_model_size=False,
        )

        runner.model_size_in_b = 7.0

        from auto_configurator.core.grid_search import (
            generate_grid_search_configs,
        )

        configs = generate_grid_search_configs(
            runner_config=runner, adapter=runner.get_adapter()
        )

        # Should generate configurations for 40GB
        assert len(configs) > 0


class TestEstimateModelSize:
    """Test cases for estimate_model_size function."""

    def test_estimate_model_size_from_constraints(self):
        """Test estimating model size from training constraints."""
        size = estimate_model_size(
            gpu_count=32,
            max_training_days=7,
            model_size_in_b=None,
            tflops_per_gpu=989,
            num_tokens_in_b=1400,
            model_name="gpt",
        )

        # Should return a positive model size
        assert size > 0
        assert size < 20  # Should be reasonable

    def test_estimate_training_time_from_model_size(self):
        """Test estimating training time from model size."""
        # Note: This function modifies max_training_days in place
        from auto_configurator import estimate_model_size

        max_training_days = 7.0

        size = estimate_model_size(
            gpu_count=32,
            max_training_days=max_training_days,
            model_size_in_b=7.0,
            tflops_per_gpu=989,
            num_tokens_in_b=1400,
            model_name="gpt",
        )

        # Should return the model size we provided
        assert abs(size - 7.0) < 0.1

    def test_estimate_t5_model_size_with_penalty(self):
        """Test estimating T5 model size with efficiency penalty."""
        size = estimate_model_size(
            gpu_count=32,
            max_training_days=7,
            model_size_in_b=None,
            tflops_per_gpu=989,
            num_tokens_in_b=1400,
            model_name="t5",
        )

        # T5 should have slightly lower effective capacity
        assert size > 0


class TestPaddleFleetRecipe:
    """Test cases for PaddleFleetRecipe dataclass."""

    def test_total_gpus_property(self):
        """Test total_gpus property calculation."""
        recipe = PaddleFleetRecipe(
            model_config=MockGPTConfig(), num_nodes=8, num_gpus_per_node=8
        )

        assert recipe.total_gpus == 64

    def test_default_values(self):
        """Test default recipe values."""
        recipe = PaddleFleetRecipe(model_config=MockGPTConfig())

        assert recipe.micro_batch_size == 1
        assert recipe.global_batch_size == 512
        assert recipe.num_nodes == 1
        assert recipe.num_gpus_per_node == 8
        assert recipe.max_steps is None
        assert recipe.log_dir is None


# ============================================================================
# Test Utilities
# ============================================================================


class TestConfigGeneration:
    """Test configuration generation workflow."""

    def test_config_name_format(self):
        """Test that generated config names follow expected format."""
        recipe = PaddleFleetRecipe(
            model_config=MockGPTConfig(), num_nodes=4, num_gpus_per_node=8
        )

        runner = AutoConfigurator(
            recipe=recipe,
            path_to_logs="/tmp/test_logs",
            tensor_parallel_sizes=[2, 4],
            pipeline_parallel_sizes=[1, 2],
            micro_batch_sizes=[1, 2, 4],
            max_steps_per_run=50,
            calculate_model_size=False,
        )
        runner.model_size_in_b = 7.0

        from auto_configurator.core.grid_search import (
            generate_grid_search_configs,
        )

        configs = generate_grid_search_configs(
            runner_config=runner, adapter=runner.get_adapter()
        )

        # Check that config names follow expected pattern
        for name in configs.keys():
            assert "gpt" in name
            assert "4nodes" in name
            assert "tp_" in name
            assert "pp_" in name
            assert "mbs_" in name
            assert "cp_1" in name
            assert "ep_1" in name

    def test_log_directory_format(self):
        """Test that log directories are properly formatted."""
        recipe = PaddleFleetRecipe(
            model_config=MockGPTConfig(), num_nodes=2, num_gpus_per_node=8
        )

        runner = AutoConfigurator(
            recipe=recipe,
            path_to_logs="/tmp/test_logs",
            tensor_parallel_sizes=[1, 2],
            pipeline_parallel_sizes=[1, 2],
            micro_batch_sizes=[1, 2],
            max_steps_per_run=50,
            calculate_model_size=False,
        )
        runner.model_size_in_b = 7.0

        from auto_configurator.core.grid_search import (
            generate_grid_search_configs,
        )

        configs = generate_grid_search_configs(
            runner_config=runner, adapter=runner.get_adapter()
        )

        # Check that log directories are properly formatted
        for config in configs.values():
            assert config.log_dir.startswith("/tmp/test_logs/")
            assert "gpt" in config.log_dir
            assert "tp_" in config.log_dir
            assert "pp_" in config.log_dir


# ============================================================================
# Bug 3 Fix Verification: finetune with None parallel sizes
# ============================================================================


class TestFinetuneValidation:
    """Verify finetune mode validates None and 'auto' parallel sizes correctly."""

    def test_finetune_with_none_tp_raises_error(self):
        """tensor_parallel_sizes=None in finetune mode should raise ValueError."""
        recipe = PaddleFleetRecipe(model_config=MockGPTConfig())

        with pytest.raises(
            ValueError, match="tensor_parallel_sizes must be specified"
        ):
            AutoConfigurator(
                recipe=recipe,
                path_to_logs="/tmp/test_logs",
                mode="finetune",
                tensor_parallel_sizes=None,  # None should be caught
                pipeline_parallel_sizes=[1, 2],
            )

    def test_finetune_with_none_pp_raises_error(self):
        """pipeline_parallel_sizes=None in finetune mode should raise ValueError."""
        recipe = PaddleFleetRecipe(model_config=MockGPTConfig())

        with pytest.raises(
            ValueError, match="pipeline_parallel_sizes must be specified"
        ):
            AutoConfigurator(
                recipe=recipe,
                path_to_logs="/tmp/test_logs",
                mode="finetune",
                tensor_parallel_sizes=[1, 2],
                pipeline_parallel_sizes=None,  # None should be caught
            )

    def test_finetune_with_explicit_tp_pp_passes(self):
        """Explicit TP/PP in finetune mode should pass validation."""
        recipe = PaddleFleetRecipe(model_config=MockGPTConfig())

        # Should not raise
        runner = AutoConfigurator(
            recipe=recipe,
            path_to_logs="/tmp/test_logs",
            mode="finetune",
            tensor_parallel_sizes=[1, 2],
            pipeline_parallel_sizes=[1, 2],
        )
        assert runner.mode == "finetune"


# ============================================================================
# Sequence Length Boundary Tests (relaxed validation)
# ============================================================================


class TestSeqLengthValidation:
    """Verify relaxed sequence length validation accepts valid multiples of 1024."""

    def test_seq_length_1024_valid(self):
        """seq_length=1024 should now be valid."""

        @dataclass
        class Config1024:
            num_hidden_layers: int = 24
            hidden_size: int = 2048
            num_attention_heads: int = 32
            intermediate_size: int = 8192
            max_sequence_length: int = 1024
            vocab_size: int = 32000
            tensor_model_parallel_size: int = 1
            pipeline_model_parallel_size: int = 1
            virtual_pipeline_model_parallel_size: int | None = None
            context_parallel_size: int = 1
            expert_model_parallel_size: int = 1

        recipe = PaddleFleetRecipe(model_config=Config1024())
        runner = AutoConfigurator(recipe=recipe, path_to_logs="/tmp/test_logs")
        assert runner.seq_length == 1024

    def test_seq_length_65536_valid(self):
        """seq_length=65536 (64K) should now be valid."""

        @dataclass
        class Config65536:
            num_hidden_layers: int = 24
            hidden_size: int = 2048
            num_attention_heads: int = 32
            intermediate_size: int = 8192
            max_sequence_length: int = 65536
            vocab_size: int = 32000
            tensor_model_parallel_size: int = 1
            pipeline_model_parallel_size: int = 1
            virtual_pipeline_model_parallel_size: int | None = None
            context_parallel_size: int = 1
            expert_model_parallel_size: int = 1

        recipe = PaddleFleetRecipe(model_config=Config65536())
        runner = AutoConfigurator(recipe=recipe, path_to_logs="/tmp/test_logs")
        assert runner.seq_length == 65536

    def test_seq_length_131072_valid(self):
        """seq_length=131072 (128K) should be valid."""

        @dataclass
        class Config128K:
            num_hidden_layers: int = 24
            hidden_size: int = 2048
            num_attention_heads: int = 32
            intermediate_size: int = 8192
            max_sequence_length: int = 131072
            vocab_size: int = 32000
            tensor_model_parallel_size: int = 1
            pipeline_model_parallel_size: int = 1
            virtual_pipeline_model_parallel_size: int | None = None
            context_parallel_size: int = 1
            expert_model_parallel_size: int = 1

        recipe = PaddleFleetRecipe(model_config=Config128K())
        runner = AutoConfigurator(recipe=recipe, path_to_logs="/tmp/test_logs")
        assert runner.seq_length == 131072

    def test_seq_length_not_multiple_of_1024_invalid(self):
        """seq_length=3000 (not multiple of 1024) should be invalid."""

        @dataclass
        class Config3000:
            num_hidden_layers: int = 24
            hidden_size: int = 2048
            num_attention_heads: int = 32
            intermediate_size: int = 8192
            max_sequence_length: int = 3000
            vocab_size: int = 32000
            tensor_model_parallel_size: int = 1
            pipeline_model_parallel_size: int = 1
            virtual_pipeline_model_parallel_size: int | None = None
            context_parallel_size: int = 1
            expert_model_parallel_size: int = 1

        recipe = PaddleFleetRecipe(model_config=Config3000())
        with pytest.raises(ValueError, match="seq_length.*not supported"):
            AutoConfigurator(recipe=recipe, path_to_logs="/tmp/test_logs")

    def test_seq_length_too_small_invalid(self):
        """seq_length=512 (< 1024) should be invalid."""

        @dataclass
        class Config512:
            num_hidden_layers: int = 24
            hidden_size: int = 2048
            num_attention_heads: int = 32
            intermediate_size: int = 8192
            max_sequence_length: int = 512
            vocab_size: int = 32000
            tensor_model_parallel_size: int = 1
            pipeline_model_parallel_size: int = 1
            virtual_pipeline_model_parallel_size: int | None = None
            context_parallel_size: int = 1
            expert_model_parallel_size: int = 1

        recipe = PaddleFleetRecipe(model_config=Config512())
        with pytest.raises(ValueError, match="seq_length.*not supported"):
            AutoConfigurator(recipe=recipe, path_to_logs="/tmp/test_logs")


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
