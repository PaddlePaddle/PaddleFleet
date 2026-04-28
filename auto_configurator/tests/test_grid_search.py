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

from auto_configurator import _apply_scoring_and_dedup
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
            """Return mock model config with adapter interface."""

            class MockModelConfig:
                def get_num_layers(self):
                    return 32

                def get_hidden_size(self):
                    return 4096

                def get_num_attention_heads(self):
                    return 32

                def get_ffn_hidden_size(self):
                    return 16384

                def get_seq_length(self):
                    return 4096

                def get_vocab_size(self):
                    return 32000

                def get_model_type(self):
                    return "gpt"

            return MockModelConfig()

        @staticmethod
        def get_parallel_strategy():
            """Return mock parallel config."""
            return MockParallelConfig()

        @staticmethod
        def get_data_config():
            """Return mock data config."""

            class MockDataConfig:
                def get_micro_batch_size(self):
                    return 4

                def get_global_batch_size(self):
                    return 2048

            return MockDataConfig()

        @staticmethod
        def get_training_config():
            """Return mock training config."""

            class MockTrainingConfig:
                def get_num_nodes(self):
                    return 4

                def get_num_gpus_per_node(self):
                    return 8

                def get_max_steps(self):
                    return 50

                def get_recompute_config(self):
                    return {
                        "granularity": None,
                        "method": None,
                        "num_layers": None,
                        "modules": None,
                    }

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
# MBS Multi-Config Tests (Bug 1 fix verification)
# ============================================================================


class TestMBSMultiConfig:
    """Verify that each (TP,PP,CP,EP) combo generates configs for ALL valid MBS values."""

    def test_multiple_mbs_per_parallel_combo(self):
        """Same (tp,pp,cp,ep) should produce multiple configs with different MBS."""
        from auto_configurator.core.grid_search import (
            generate_grid_search_configs,
        )

        mock = MockAutoConfigurator(
            model_type="gpt",
            model_size_in_b=7.0,
            gpu_memory_gb=80,
            tensor_parallel_sizes=[2],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1, 2, 4],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            max_steps_per_run=50,
            path_to_logs="/tmp/test_logs",
        )
        mock.adapter = mock.Adapter()
        mock.seq_length = 4096
        mock.global_batch_size = 2048

        configs = generate_grid_search_configs(
            runner_config=mock, adapter=mock.Adapter()
        )

        # Collect MBS values for tp=2, pp=1
        mbs_values = set()
        for config in configs.values():
            if (
                config.tensor_parallel_size == 2
                and config.pipeline_parallel_size == 1
            ):
                mbs_values.add(config.micro_batch_size)

        # Should have multiple MBS values, not just the first valid one
        assert len(mbs_values) > 1, (
            f"Expected multiple MBS values for (tp=2,pp=1), got {mbs_values}"
        )

    def test_all_requested_mbs_present(self):
        """All valid MBS values should appear in generated configs."""
        from auto_configurator.core.grid_search import (
            generate_grid_search_configs,
        )

        mock = MockAutoConfigurator(
            model_type="gpt",
            model_size_in_b=7.0,
            gpu_memory_gb=80,
            tensor_parallel_sizes=[1],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1, 2, 4, 8],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            max_steps_per_run=50,
            path_to_logs="/tmp/test_logs",
        )
        mock.adapter = mock.Adapter()
        mock.seq_length = 4096
        mock.global_batch_size = 2048

        configs = generate_grid_search_configs(
            runner_config=mock, adapter=mock.Adapter()
        )

        mbs_values = {
            c.micro_batch_size
            for c in configs.values()
            if c.tensor_parallel_size == 1 and c.pipeline_parallel_size == 1
        }

        # GBS=2048, GPUs=32, DP=32 (tp=1,pp=1) => MBS must divide 2048/32=64
        # MBS 1,2,4,8 all divide 64, so all 4 should be present
        assert mbs_values == {1, 2, 4, 8}, (
            f"Expected {{1,2,4,8}} MBS values, got {mbs_values}"
        )


# ============================================================================
# CP/EP Compatibility Tests (Bug 2 fix verification)
# ============================================================================


class TestCPEPCompatibility:
    """Verify CP/EP divisibility filtering logic."""

    def test_cp_and_ep_both_one_passes(self):
        """CP=1, EP=1 should always pass."""
        from auto_configurator.core.grid_search import (
            generate_grid_search_configs,
        )

        mock = MockAutoConfigurator(
            model_type="gpt",
            model_size_in_b=7.0,
            gpu_memory_gb=80,
            tensor_parallel_sizes=[1],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            max_steps_per_run=50,
            path_to_logs="/tmp/test_logs",
        )
        mock.adapter = mock.Adapter()
        mock.seq_length = 4096
        mock.global_batch_size = 2048

        configs = generate_grid_search_configs(
            runner_config=mock, adapter=mock.Adapter()
        )
        assert len(configs) > 0

    def test_cp_divides_ep_passes(self):
        """CP=2, EP=4 should pass (EP % CP == 0)."""
        from auto_configurator.core.grid_search import (
            generate_grid_search_configs,
        )

        mock = MockAutoConfigurator(
            model_type="gpt",
            model_size_in_b=7.0,
            gpu_memory_gb=80,
            tensor_parallel_sizes=[1],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
            context_parallel_sizes=[2],
            expert_parallel_sizes=[4],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            max_steps_per_run=50,
            path_to_logs="/tmp/test_logs",
        )
        mock.adapter = mock.Adapter()
        mock.seq_length = 4096
        mock.global_batch_size = 2048

        configs = generate_grid_search_configs(
            runner_config=mock, adapter=mock.Adapter()
        )
        # tp=1, pp=1, cp=2, ep=4 => model_parallelism=8, within default bounds
        assert len(configs) > 0, "CP=2, EP=4 (divisible) should produce configs"

    def test_cp_ep_not_divisible_filtered(self):
        """CP=3, EP=5 should be filtered out (no divisibility)."""
        from auto_configurator.core.grid_search import (
            generate_grid_search_configs,
        )

        mock = MockAutoConfigurator(
            model_type="gpt",
            model_size_in_b=7.0,
            gpu_memory_gb=80,
            tensor_parallel_sizes=[1],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
            context_parallel_sizes=[3],
            expert_parallel_sizes=[5],
            min_model_parallel_size="auto",
            max_model_parallel_size="auto",
            max_steps_per_run=50,
            path_to_logs="/tmp/test_logs",
        )
        mock.adapter = mock.Adapter()
        mock.seq_length = 4096
        mock.global_batch_size = 2048

        configs = generate_grid_search_configs(
            runner_config=mock, adapter=mock.Adapter()
        )
        # CP=3, EP=5 are coprime, should be filtered
        assert len(configs) == 0, (
            "CP=3, EP=5 (coprime) should produce no configs"
        )


# ============================================================================
# Unsupported Model Type Test (T5/BERT cleanup verification)
# ============================================================================


class TestUnsupportedModelInGridSearch:
    """Verify non-GPT models raise ValueError after T5/BERT stub removal."""

    def test_t5_raises_value_error(self):
        """T5 model should raise ValueError in get_grid_search_params."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            get_grid_search_params(
                model_type="t5",
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

    def test_bert_raises_value_error(self):
        """BERT model should raise ValueError in get_grid_search_params."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            get_grid_search_params(
                model_type="bert",
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


# ============================================================================
# Scoring and Dedup Tests
# ============================================================================


class TestScoringAndDedup:
    """Tests for _apply_scoring_and_dedup internal function."""

    @staticmethod
    def _make_config(tp, pp, cp, ep, mbs, name=None):
        """Helper to create GeneratedConfig for testing."""
        if name is None:
            name = f"test_tp_{tp}_pp_{pp}_cp_{cp}_ep_{ep}_mbs_{mbs}"
        return GeneratedConfig(
            name=name,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            virtual_pipeline_size=None,
            context_parallel_size=cp,
            expert_parallel_size=ep,
            micro_batch_size=mbs,
            global_batch_size=2048,
            max_steps=50,
            recompute_granularity=None,
            recompute_method=None,
            recompute_num_layers=None,
            recompute_modules=None,
            log_dir="/tmp/test",
        )

    def test_scoring_fn_with_no_max(self):
        """scoring_fn without max_configs: dedup by (TP,PP,CP,EP), keep best MBS."""
        configs = {
            "a": self._make_config(2, 1, 1, 1, 1),
            "b": self._make_config(2, 1, 1, 1, 4),
            "c": self._make_config(4, 1, 1, 1, 1),
        }
        scoring_fn = lambda c: c.micro_batch_size
        result = _apply_scoring_and_dedup(configs, scoring_fn, max_configs=None)
        assert len(result) == 2
        assert "b" in result  # Group (2,1,1,1): MBS=4 beats MBS=1
        assert "c" in result  # Group (4,1,1,1): only entry
        assert "a" not in result

    def test_max_configs_truncation(self):
        """max_configs limits the number of returned configs."""
        configs = {
            "a": self._make_config(1, 1, 1, 1, 1),
            "b": self._make_config(2, 1, 1, 1, 1),
            "c": self._make_config(4, 1, 1, 1, 1),
        }
        scoring_fn = lambda c: c.tensor_parallel_size
        result = _apply_scoring_and_dedup(configs, scoring_fn, max_configs=2)
        assert len(result) == 2
        assert "c" in result  # TP=4, highest score
        assert "b" in result  # TP=2, second

    def test_max_configs_none_returns_all_deduped(self):
        """max_configs=None returns all groups after dedup."""
        configs = {
            f"cfg_{tp}_{mbs}": self._make_config(tp, 1, 1, 1, mbs)
            for tp in [1, 2, 4]
            for mbs in [1, 2, 4]
        }
        scoring_fn = lambda c: c.micro_batch_size
        result = _apply_scoring_and_dedup(configs, scoring_fn, max_configs=None)
        # 3 groups (tp=1, tp=2, tp=4), each keeps best MBS=4
        assert len(result) == 3
        for cfg in result.values():
            assert cfg.micro_batch_size == 4

    def test_empty_configs(self):
        """Empty configs dict returns empty dict."""
        scoring_fn = lambda c: 1.0
        result = _apply_scoring_and_dedup({}, scoring_fn, max_configs=5)
        assert len(result) == 0

    def test_single_config(self):
        """Single config is returned as-is."""
        configs = {"a": self._make_config(2, 1, 1, 1, 4)}
        scoring_fn = lambda c: 1.0
        result = _apply_scoring_and_dedup(configs, scoring_fn, max_configs=5)
        assert len(result) == 1
        assert "a" in result

    def test_diversity_across_all_dimensions(self):
        """Dedup groups by all 4 dimensions: TP, PP, CP, EP."""
        configs = {
            "a": self._make_config(2, 2, 1, 1, 1),  # group (2,2,1,1)
            "b": self._make_config(2, 2, 1, 1, 4),  # same group, higher MBS
            "c": self._make_config(2, 2, 2, 1, 1),  # group (2,2,2,1)
            "d": self._make_config(2, 2, 1, 2, 1),  # group (2,2,1,2)
        }
        scoring_fn = lambda c: c.micro_batch_size
        result = _apply_scoring_and_dedup(configs, scoring_fn, max_configs=None)
        assert len(result) == 3
        assert "b" in result  # (2,2,1,1) keeps MBS=4

    def test_scoring_fn_negative_scores(self):
        """Scoring function can return negative values."""
        configs = {
            "a": self._make_config(1, 1, 1, 1, 1),
            "b": self._make_config(2, 1, 1, 1, 1),
        }
        scoring_fn = lambda c: -c.tensor_parallel_size
        result = _apply_scoring_and_dedup(configs, scoring_fn, max_configs=1)
        assert len(result) == 1
        assert "a" in result  # score=-1 > score=-2

    def test_max_configs_larger_than_available(self):
        """max_configs > len(configs) returns all available."""
        configs = {
            "a": self._make_config(1, 1, 1, 1, 1),
            "b": self._make_config(2, 1, 1, 1, 1),
        }
        scoring_fn = lambda c: 1.0
        result = _apply_scoring_and_dedup(configs, scoring_fn, max_configs=100)
        assert len(result) == 2

    def test_max_configs_zero_returns_all(self):
        """max_configs=0 does not truncate (0 is not > 0, no truncation applied)."""
        configs = {"a": self._make_config(1, 1, 1, 1, 1)}
        scoring_fn = lambda c: 1.0
        result = _apply_scoring_and_dedup(configs, scoring_fn, max_configs=0)
        assert len(result) == 1

    def test_score_ordering_preserved(self):
        """Result dict preserves score-descending order."""
        configs = {
            "low": self._make_config(1, 1, 1, 1, 1),
            "mid": self._make_config(2, 1, 1, 1, 1),
            "high": self._make_config(4, 1, 1, 1, 1),
        }
        scoring_fn = lambda c: c.tensor_parallel_size
        result = _apply_scoring_and_dedup(configs, scoring_fn, max_configs=None)
        names = list(result.keys())
        assert names[0] == "high"
        assert names[1] == "mid"
        assert names[2] == "low"


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
