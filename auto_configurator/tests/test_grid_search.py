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
Unit tests for AutoConfigurator core.grid_search module.

Tests grid search generation and GPTGridSearch rules.
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

from auto_configurator.core.grid_search import (
    GeneratedConfig,
    GPTGridSearch,
    GridSearchConfig,
    get_grid_search_params,
)

# ============================================================================
# Mock AutoConfigurator for testing
# ============================================================================


@dataclass
class MockParallelConfig:
    """Mock parallel configuration."""

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1


@dataclass
class MockAutoConfigurator:
    """Mock AutoConfigurator for testing."""

    model_type: str = "gpt"
    model_size_in_b: float = 7.0
    gpu_memory_gb: int = 80
    path_to_logs: str = "/tmp/test_logs"
    tensor_parallel_sizes: list | None = None
    pipeline_parallel_sizes: list | None = None
    micro_batch_sizes: list | None = None
    context_parallel_sizes: list | None = None
    expert_parallel_sizes: list | None = None
    min_model_parallel_size: int | str | None = "auto"
    max_model_parallel_size: int | str | None = "auto"
    max_steps_per_run: int = 50

    class Adapter:
        """Mock adapter."""

        @staticmethod
        def get_model_config():
            """Return mock model config."""

            @dataclass
            class MockModelConfig:
                num_hidden_layers: int = 32
                hidden_size: int = 4096
                num_attention_heads: int = 32
                intermediate_size: int = 16384

            return MockModelConfig()

        @staticmethod
        def get_parallel_strategy():
            """Return mock parallel config."""
            return MockParallelConfig()

        @staticmethod
        def get_data_config():
            """Return mock data config."""

            @dataclass
            class MockDataConfig:
                global_batch_size: int = 2048

            return MockDataConfig()

        @staticmethod
        def get_training_config():
            """Return mock training config."""

            @dataclass
            class MockTrainingConfig:
                num_nodes: int = 4
                num_gpus_per_node: int = 8

            return MockTrainingConfig()

        def get_adapter(self):
            """Return combined adapter."""

            class Combined:
                """Combined adapter."""

                @staticmethod
                def get_model_config():
                    return MockAutoConfigurator.Adapter.get_model_config()

                @staticmethod
                def get_parallel_strategy():
                    return MockAutoConfigurator.Adapter.get_parallel_strategy()

                @staticmethod
                def get_data_config():
                    return MockAutoConfigurator.Adapter.get_data_config()

                @staticmethod
                def get_training_config():
                    return MockAutoConfigurator.Adapter.get_training_config()

            return Combined()


# ============================================================================
# GPTGridSearch Tests
# ============================================================================


class TestGPTGridSearch:
    """Test cases for GPTGridSearch class."""

    def test_init_params_default_80gb(self):
        """Test initializing default params for 80GB GPU."""
        search = GPTGridSearch(
            model_size_in_b=7.0,
            valid_pp=[1, 2, 4, 8, 16, 32],
            seq_length=4096,
            gpu_memory_gb=80,
        )
        search.init_params()

        # Should set default search space
        assert search.tp is not None
        assert search.pp is not None
        assert search.mbs is not None
        assert search.gbs > 0
        assert search.min_model_parallel > 0
        assert search.max_model_parallel > 0

    def test_init_params_small_model_2048(self):
        """Test initializing for small model with 2048 seq length."""
        search = GPTGridSearch(
            model_size_in_b=1.0,
            valid_pp=[1, 2],
            seq_length=2048,
            gpu_memory_gb=80,
        )
        search.init_params()

        # Small model should have smaller TP values
        assert search.tp == [1, 2]
        assert search.gbs == 256

    def test_init_params_medium_model_2048(self):
        """Test initializing for medium model with 2048 seq length."""
        search = GPTGridSearch(
            model_size_in_b=4.0,
            valid_pp=[1, 2, 4, 8],
            seq_length=2048,
            gpu_memory_gb=80,
        )
        search.init_params()

        # Medium model
        assert search.tp == [1, 2, 4]
        assert search.gbs == 1024

    def test_init_params_large_model_2048(self):
        """Test initializing for large model with 2048 seq length."""
        search = GPTGridSearch(
            model_size_in_b=13.0,
            valid_pp=[1, 2, 4],
            seq_length=2048,
            gpu_memory_gb=80,
        )
        search.init_params()

        # Large model with PP enabled
        assert search.tp == [1, 2, 4, 8]
        assert search.gbs == 2048
        assert search.min_model_parallel == 4
        assert search.max_model_parallel == 8

    def test_init_params_small_model_4096(self):
        """Test initializing for small model with 4096 seq length."""
        search = GPTGridSearch(
            model_size_in_b=1.0,
            valid_pp=[1, 2],
            seq_length=4096,
            gpu_memory_gb=80,
        )
        search.init_params()

        # Small model with 4096 seq length
        assert search.tp == [1, 2, 4]
        assert search.mbs == [1, 2, 4, 8]
        assert search.gbs == 128

    def test_init_params_medium_model_4096(self):
        """Test initializing for medium model with 4096 seq length."""
        search = GPTGridSearch(
            model_size_in_b=4.0,
            valid_pp=[1, 2],
            seq_length=4096,
            gpu_memory_gb=80,
        )
        search.init_params()

        # Medium model
        assert search.tp == [1, 2, 4]
        assert search.mbs == [1, 2, 4, 8]
        assert search.gbs == 512

    def test_init_params_large_model_4096(self):
        """Test initializing for large model with 4096 seq length."""
        search = GPTGridSearch(
            model_size_in_b=13.0,
            valid_pp=[1, 2, 4],
            seq_length=4096,
            gpu_memory_gb=80,
        )
        search.init_params()

        # Large model with PP enabled
        assert search.tp == [1, 2, 4, 8]
        assert search.mbs == [1, 2, 4, 8]
        assert search.min_model_parallel == 4
        assert search.max_model_parallel == 8

    def test_init_params_40gb_gpu(self):
        """Test initializing for 40GB GPU."""
        search = GPTGridSearch(
            model_size_in_b=7.0,
            valid_pp=[1, 2, 4],
            seq_length=2048,
            gpu_memory_gb=40,
        )
        search.init_params()

        # 40GB GPU should have adjusted search space
        assert search.tp is not None
        assert search.mbs is not None
        assert search.gbs > 0

    def test_init_params_8192_seq_length(self):
        """Test initializing for 8192 sequence length."""
        search = GPTGridSearch(
            model_size_in_b=7.0,
            valid_pp=[1, 2],
            seq_length=8192,
            gpu_memory_gb=80,
        )
        search.init_params()

        # 8192 seq length requires larger TP
        assert search.tp is not None

    def test_init_params_16384_seq_length(self):
        """Test initializing for 16384 sequence length."""
        search = GPTGridSearch(
            model_size_in_b=7.0,
            valid_pp=[1, 2],
            seq_length=16384,
            gpu_memory_gb=80,
        )
        search.init_params()

        # 16384 seq length requires larger TP
        assert search.tp is not None

    def test_init_params_32768_seq_length(self):
        """Test initializing for 32768 sequence length."""
        search = GPTGridSearch(
            model_size_in_b=7.0,
            valid_pp=[1, 2],
            seq_length=32768,
            gpu_memory_gb=80,
        )
        search.init_params()

        # 32768 seq length requires larger TP
        assert search.tp is not None


# ============================================================================
# get_grid_search_params Tests
# ============================================================================


class TestGetGridSearchParams:
    """Test cases for get_grid_search_params function."""

    def test_gpt_model_type(self):
        """Test grid search params for GPT model."""
        params = get_grid_search_params(
            model_type="gpt",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=2048,
        )

        # Should return GridSearchConfig
        assert isinstance(params, GridSearchConfig)
        assert params.tp is not None
        assert params.pp is not None
        assert params.mbs is not None

    def test_llama_model_type(self):
        """Test grid search params for Llama model (GPT-based)."""
        params = get_grid_search_params(
            model_type="llama",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=2048,
        )

        # Llama should use same rules as GPT
        assert isinstance(params, GridSearchConfig)
        assert params.tp is not None
        assert params.pp is not None

    def test_qwen_model_type(self):
        """Test grid search params for Qwen model (GPT-based)."""
        params = get_grid_search_params(
            model_type="qwen",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=2048,
        )

        # Qwen should use same rules as GPT
        assert isinstance(params, GridSearchConfig)
        assert params.tp is not None

    def test_mixtral_model_type(self):
        """Test grid search params for Mixtral model (GPT-based)."""
        params = get_grid_search_params(
            model_type="mixtral",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=2048,
        )

        # Mixtral should use same rules as GPT
        assert isinstance(params, GridSearchConfig)
        assert params.tp is not None

    def test_gemma_model_type(self):
        """Test grid search params for Gemma model (GPT-based)."""
        params = get_grid_search_params(
            model_type="gemma",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=2048,
        )

        # Gemma should use same rules as GPT
        assert isinstance(params, GridSearchConfig)
        assert params.tp is not None

    def test_explicit_tensor_parallel_sizes(self):
        """Test grid search params with explicit TP sizes."""
        explicit_tp = [1, 2, 4, 8]
        params = get_grid_search_params(
            model_type="gpt",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes=explicit_tp,
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=2048,
        )

        # Should use explicit TP sizes
        assert params.tp == explicit_tp

    def test_explicit_pipeline_parallel_sizes(self):
        """Test grid search params with explicit PP sizes."""
        explicit_pp = [1, 2, 4]
        params = get_grid_search_params(
            model_type="gpt",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes=explicit_pp,
            micro_batch_sizes="auto",
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=2048,
        )

        # Should use explicit PP sizes
        assert params.pp == explicit_pp

    def test_explicit_micro_batch_sizes(self):
        """Test grid search params with explicit MBS sizes."""
        explicit_mbs = [1, 2, 4, 8]
        params = get_grid_search_params(
            model_type="gpt",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes=explicit_mbs,
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=2048,
        )

        # Should use explicit MBS sizes
        assert params.mbs == explicit_mbs

    def test_explicit_global_batch_size(self):
        """Test grid search params with explicit GBS."""
        explicit_gbs = 1024
        params = get_grid_search_params(
            model_type="gpt",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=explicit_gbs,
        )

        # Should use explicit GBS
        assert params.gbs == explicit_gbs

    def test_explicit_context_parallel_sizes(self):
        """Test grid search params with explicit CP sizes."""
        explicit_cp = [1, 2, 4]
        params = get_grid_search_params(
            model_type="gpt",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            context_parallel_sizes=explicit_cp,
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=2048,
        )

        # Should use explicit CP sizes
        assert params.cp == explicit_cp

    def test_explicit_expert_parallel_sizes(self):
        """Test grid search params with explicit EP sizes."""
        explicit_ep = [1, 2, 4]
        params = get_grid_search_params(
            model_type="gpt",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            context_parallel_sizes=[1],
            expert_parallel_sizes=explicit_ep,
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=2048,
        )

        # Should use explicit EP sizes
        assert params.ep == explicit_ep

    def test_explicit_min_max_model_parallel(self):
        """Test grid search params with explicit min/max model parallel."""
        params = get_grid_search_params(
            model_type="gpt",
            model_size_in_b=7.0,
            num_layers=32,
            seq_length=4096,
            gpu_memory_gb=80,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size=4,
            max_model_parallel_size=8,
            global_batch_size=2048,
        )

        # Should use explicit min/max values
        assert params.min_model_parallel == 4
        assert params.max_model_parallel == 8

    def test_40gb_memory_adjusts_search_space(self):
        """Test that 40GB GPU adjusts search space."""
        params = get_grid_search_params(
            model_type="gpt",
            model_size_in_b=1.0,
            num_layers=24,
            seq_length=2048,
            gpu_memory_gb=40,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes="auto",
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            global_batch_size=2048,
        )

        # 40GB should have adjusted search space
        assert isinstance(params, GridSearchConfig)
        assert params.mbs is not None


# ============================================================================
# GeneratedConfig Tests
# ============================================================================


class TestGeneratedConfig:
    """Test cases for GeneratedConfig dataclass."""

    def test_create_generated_config(self):
        """Test creating a GeneratedConfig."""
        config = GeneratedConfig(
            name="gpt_7b_4nodes_tp_4_pp_2_cp_1_ep_1_mbs_4",
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            virtual_pipeline_size=None,
            context_parallel_size=1,
            expert_parallel_size=1,
            micro_batch_size=4,
            global_batch_size=2048,
            max_steps=50,
            recompute_granularity=None,
            recompute_method=None,
            recompute_num_layers=None,
            recompute_modules=None,
            log_dir="/tmp/test_logs/gpt_7b_4nodes_tp_4_pp_2_cp_1_ep_1_mbs_4",
        )

        assert config.name == "gpt_7b_4nodes_tp_4_pp_2_cp_1_ep_1_mbs_4"
        assert config.tensor_parallel_size == 4
        assert config.pipeline_parallel_size == 2
        assert config.virtual_pipeline_size is None
        assert config.context_parallel_size == 1
        assert config.expert_parallel_size == 1
        assert config.micro_batch_size == 4
        assert config.global_batch_size == 2048
        assert config.max_steps == 50
        assert config.recompute_granularity is None
        assert config.recompute_method is None
        assert config.recompute_num_layers is None
        assert config.recompute_modules is None

    def test_create_generated_config_with_virtual_pipeline(self):
        """Test creating a GeneratedConfig with virtual pipeline."""
        config = GeneratedConfig(
            name="gpt_7b_4nodes_tp_4_pp_4_vp_8_cp_1_ep_1_mbs_4",
            tensor_parallel_size=4,
            pipeline_parallel_size=4,
            virtual_pipeline_size=8,
            context_parallel_size=1,
            expert_parallel_size=1,
            micro_batch_size=4,
            global_batch_size=2048,
            max_steps=50,
            recompute_granularity=None,
            recompute_method=None,
            recompute_num_layers=None,
            recompute_modules=None,
            log_dir="/tmp/test_logs/gpt_7b_4nodes_tp_4_pp_4_vp_8_cp_1_ep_1_mbs_4",
        )

        assert config.name == "gpt_7b_4nodes_tp_4_pp_4_vp_8_cp_1_ep_1_mbs_4"
        assert config.tensor_parallel_size == 4
        assert config.pipeline_parallel_size == 4
        assert config.virtual_pipeline_size == 8

    def test_create_generated_config_with_recompute(self):
        """Test creating a GeneratedConfig with activation recompute."""
        config = GeneratedConfig(
            name="gpt_7b_4nodes_tp_4_pp_2_cp_1_ep_1_mbs_4",
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            virtual_pipeline_size=None,
            context_parallel_size=1,
            expert_parallel_size=1,
            micro_batch_size=4,
            global_batch_size=2048,
            max_steps=50,
            recompute_granularity="selective",
            recompute_method="block",
            recompute_num_layers=4,
            recompute_modules=["attention", "mlp"],
            log_dir="/tmp/test_logs/gpt_7b_4nodes_tp_4_pp_2_cp_1_ep_1_mbs_4",
        )

        assert config.name == "gpt_7b_4nodes_tp_4_pp_2_cp_1_ep_1_mbs_4"
        assert config.recompute_granularity == "selective"
        assert config.recompute_method == "block"
        assert config.recompute_num_layers == 4
        assert config.recompute_modules == ["attention", "mlp"]


# ============================================================================
# GridSearchConfig Tests
# ============================================================================


class TestGridSearchConfig:
    """Test cases for GridSearchConfig dataclass."""

    def test_create_grid_search_config(self):
        """Test creating a GridSearchConfig."""
        config = GridSearchConfig(
            tp=[1, 2, 4],
            pp=[1, 2],
            cp=[1],
            ep=[1],
            mbs=[1, 2, 4, 8],
            gbs=2048,
            min_model_parallel=1,
            max_model_parallel=8,
        )

        assert config.tp == [1, 2, 4]
        assert config.pp == [1, 2]
        assert config.cp == [1]
        assert config.ep == [1]
        assert config.mbs == [1, 2, 4, 8]
        assert config.gbs == 2048
        assert config.min_model_parallel == 1
        assert config.max_model_parallel == 8


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
