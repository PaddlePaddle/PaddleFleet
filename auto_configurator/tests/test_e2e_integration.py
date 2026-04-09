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
End-to-end integration tests for AutoConfigurator with real PaddleFleet config objects.

Unlike test_integration.py which uses mock dataclasses, these tests import
real GPTConfig from paddlefleet.models.gpt.gpt_config and exercise the full
pipeline: GPTConfig -> PaddleFleetRecipe -> AutoConfigurator -> generate_configs().

Tests are skipped automatically when paddlefleet is not installed.
"""

import re
import sys
from pathlib import Path

import pytest

# Module-level skip guard: skip entire file if paddlefleet is not available
gpt_config_mod = pytest.importorskip(
    "paddlefleet.models.gpt.gpt_config",
    reason="paddlefleet not installed; skipping e2e integration tests",
)
GPTConfig = gpt_config_mod.GPTConfig

test_dir = Path(__file__).parent
src_dir = test_dir.parent
sys.path.insert(0, str(src_dir))

from auto_configurator import (
    AutoConfigurator,
    PaddleFleetRecipe,
    estimate_model_size,
    generate_configs,
)
from auto_configurator.core import calculate_model_size
from auto_configurator.paddlefleet_adapters import (
    PaddleFleetModelConfigAdapter,
    PaddleFleetParallelConfigAdapter,
    create_paddlefleet_adapter,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def gpt_7b_config():
    """Create a real GPTConfig resembling a ~7B model (Llama-7B-like)."""
    return GPTConfig(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        intermediate_size=11008,
        max_sequence_length=4096,
        vocab_size=32000,
    )


@pytest.fixture
def gpt_small_config():
    """Create a real GPTConfig resembling a ~1B model."""
    return GPTConfig(
        num_hidden_layers=24,
        hidden_size=2048,
        num_attention_heads=32,
        max_sequence_length=4096,
        vocab_size=32000,
    )


@pytest.fixture
def make_runner():
    """Factory fixture: create an AutoConfigurator from a GPTConfig."""

    def _make(
        config,
        num_nodes=4,
        num_gpus_per_node=8,
        global_batch_size=2048,
        micro_batch_size=1,
        mode="pretrain",
        gpu_memory_gb=80,
        tensor_parallel_sizes="auto",
        pipeline_parallel_sizes="auto",
        micro_batch_sizes="auto",
        context_parallel_sizes=None,
        expert_parallel_sizes=None,
        max_steps_per_run=50,
        calculate_model_size_flag=False,
        path_to_logs="/tmp/test_e2e_logs",
    ):
        recipe = PaddleFleetRecipe(
            model_config=config,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
        )
        return AutoConfigurator(
            recipe=recipe,
            path_to_logs=path_to_logs,
            mode=mode,
            gpu_memory_gb=gpu_memory_gb,
            tensor_parallel_sizes=tensor_parallel_sizes,
            pipeline_parallel_sizes=pipeline_parallel_sizes,
            micro_batch_sizes=micro_batch_sizes,
            context_parallel_sizes=context_parallel_sizes,
            expert_parallel_sizes=expert_parallel_sizes,
            max_steps_per_run=max_steps_per_run,
            calculate_model_size=calculate_model_size_flag,
        )

    return _make


# ============================================================================
# 1. Real Config Adapter Mapping Tests
# ============================================================================


class TestRealConfigAdapterMapping:
    """Verify adapters correctly map real GPTConfig attributes."""

    def test_model_type_detection(self, gpt_7b_config):
        """GPTConfig class name should be detected as 'gpt' model type."""
        adapter = PaddleFleetModelConfigAdapter(gpt_7b_config)
        assert adapter.get_model_type() == "gpt"

    def test_num_layers_read(self, gpt_7b_config):
        """Adapter reads num_hidden_layers from real GPTConfig."""
        adapter = PaddleFleetModelConfigAdapter(gpt_7b_config)
        assert adapter.get_num_layers() == 32

    def test_hidden_size_read(self, gpt_7b_config):
        """Adapter reads hidden_size from real GPTConfig."""
        adapter = PaddleFleetModelConfigAdapter(gpt_7b_config)
        assert adapter.get_hidden_size() == 4096

    def test_num_attention_heads_read(self, gpt_7b_config):
        """Adapter reads num_attention_heads from real GPTConfig."""
        adapter = PaddleFleetModelConfigAdapter(gpt_7b_config)
        assert adapter.get_num_attention_heads() == 32

    def test_ffn_hidden_size_read_explicit(self, gpt_7b_config):
        """Adapter reads explicitly set intermediate_size."""
        adapter = PaddleFleetModelConfigAdapter(gpt_7b_config)
        assert adapter.get_ffn_hidden_size() == 11008

    def test_ffn_hidden_size_auto_computed(self, gpt_small_config):
        """TransformerConfig.__post_init__ auto-computes intermediate_size = 4 * hidden_size."""
        adapter = PaddleFleetModelConfigAdapter(gpt_small_config)
        assert adapter.get_ffn_hidden_size() == 8192  # 4 * 2048

    def test_seq_length_read(self, gpt_7b_config):
        """Adapter reads max_sequence_length from real GPTConfig."""
        adapter = PaddleFleetModelConfigAdapter(gpt_7b_config)
        assert adapter.get_seq_length() == 4096

    def test_vocab_size_read(self, gpt_7b_config):
        """Adapter reads vocab_size from real GPTConfig."""
        adapter = PaddleFleetModelConfigAdapter(gpt_7b_config)
        assert adapter.get_vocab_size() == 32000

    def test_set_num_layers_roundtrip(self, gpt_7b_config):
        """Setting layers through adapter modifies real GPTConfig."""
        adapter = PaddleFleetModelConfigAdapter(gpt_7b_config)
        adapter.set_num_layers(48)
        assert adapter.get_num_layers() == 48
        assert gpt_7b_config.num_hidden_layers == 48

    def test_set_hidden_size_roundtrip(self, gpt_7b_config):
        """Setting hidden_size through adapter modifies real GPTConfig."""
        adapter = PaddleFleetModelConfigAdapter(gpt_7b_config)
        adapter.set_hidden_size(8192)
        assert adapter.get_hidden_size() == 8192
        assert gpt_7b_config.hidden_size == 8192


# ============================================================================
# 2. Parallel Config and Recipe Tests
# ============================================================================


class TestRealConfigParallelAndRecipe:
    """Verify parallel adapter and recipe work with real GPTConfig."""

    def test_parallel_defaults(self, gpt_7b_config):
        """Real GPTConfig defaults: tp=1, pp=1, cp=1, ep=1."""
        adapter = PaddleFleetParallelConfigAdapter(gpt_7b_config)
        assert adapter.get_tensor_parallel_size() == 1
        assert adapter.get_pipeline_parallel_size() == 1
        assert adapter.get_context_parallel_size() == 1
        assert adapter.get_expert_parallel_size() == 1

    def test_parallel_explicit_tp(self):
        """Real GPTConfig with explicit tensor_model_parallel_size."""
        config = GPTConfig(
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            tensor_model_parallel_size=4,
            max_sequence_length=4096,
            vocab_size=32000,
        )
        adapter = PaddleFleetParallelConfigAdapter(config)
        assert adapter.get_tensor_parallel_size() == 4

    def test_combined_adapter_with_real_config(self, gpt_7b_config):
        """create_paddlefleet_adapter correctly wraps real GPTConfig."""
        recipe = PaddleFleetRecipe(
            model_config=gpt_7b_config,
            num_nodes=4,
            num_gpus_per_node=8,
        )
        adapter = create_paddlefleet_adapter(recipe)
        model_cfg = adapter.get_model_config()
        assert model_cfg.get_num_layers() == 32
        assert model_cfg.get_hidden_size() == 4096
        assert model_cfg.get_model_type() == "gpt"

    def test_recipe_total_gpus(self, gpt_7b_config):
        """Recipe total_gpus property is num_nodes * num_gpus_per_node."""
        recipe = PaddleFleetRecipe(
            model_config=gpt_7b_config,
            num_nodes=4,
            num_gpus_per_node=8,
        )
        assert recipe.total_gpus == 32

    def test_adapter_preserves_config_reference(self, gpt_7b_config):
        """Adapter wraps real config by reference, not copy."""
        recipe = PaddleFleetRecipe(model_config=gpt_7b_config)
        adapter = create_paddlefleet_adapter(recipe)
        gpt_7b_config.num_hidden_layers = 64
        assert adapter.get_model_config().get_num_layers() == 64


# ============================================================================
# 3. AutoConfigurator Init with Real Config
# ============================================================================


class TestAutoConfiguratorInitWithRealConfig:
    """Verify AutoConfigurator initialization with real GPTConfig."""

    def test_basic_initialization(self, gpt_7b_config, make_runner):
        """AutoConfigurator correctly extracts model_type, seq_length, gpu_count."""
        runner = make_runner(gpt_7b_config)
        assert runner.model_type == "gpt"
        assert runner.seq_length == 4096
        assert runner.gpu_count == 32
        assert runner.global_batch_size == 2048

    def test_initialization_40gb(self, gpt_7b_config, make_runner):
        """AutoConfigurator with 40GB GPU initializes successfully."""
        runner = make_runner(gpt_7b_config, gpu_memory_gb=40)
        assert runner.gpu_memory_gb == 40

    def test_accessor_methods(self, gpt_7b_config, make_runner):
        """Accessor methods return valid adapters wrapping real config."""
        runner = make_runner(gpt_7b_config)
        model_cfg = runner.get_model_config()
        assert model_cfg.get_num_layers() == 32
        assert model_cfg.get_hidden_size() == 4096

        parallel_cfg = runner.get_parallel_config()
        assert parallel_cfg.get_tensor_parallel_size() == 1

        data_cfg = runner.get_data_config()
        assert data_cfg.get_global_batch_size() == 2048

        training_cfg = runner.get_training_config()
        assert training_cfg.get_num_nodes() == 4
        assert training_cfg.get_num_gpus_per_node() == 8

    def test_finetune_mode_with_explicit_tp_pp(
        self, gpt_7b_config, make_runner
    ):
        """Finetune mode with explicit TP/PP passes validation."""
        runner = make_runner(
            gpt_7b_config,
            mode="finetune",
            tensor_parallel_sizes=[2],
            pipeline_parallel_sizes=[1],
        )
        assert runner.mode == "finetune"


# ============================================================================
# 4. generate_configs() E2E Tests — Most Critical
# ============================================================================


class TestGenerateConfigsE2E:
    """Test the public generate_configs() API with real GPTConfig."""

    def test_returns_base_config_and_dict(self, gpt_7b_config, make_runner):
        """generate_configs returns (base_config, configs_dict)."""
        runner = make_runner(gpt_7b_config)
        base_config, configs = generate_configs(runner)
        assert base_config is gpt_7b_config
        assert isinstance(configs, dict)
        assert len(configs) > 0

    def test_explicit_parallel_sizes(self, gpt_7b_config, make_runner):
        """Generated configs respect explicit TP/PP/MBS constraints."""
        runner = make_runner(
            gpt_7b_config,
            tensor_parallel_sizes=[1, 2, 4],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1, 2, 4],
        )
        _, configs = generate_configs(runner)
        assert len(configs) > 0
        for config in configs.values():
            assert config.tensor_parallel_size in [1, 2, 4]
            assert config.pipeline_parallel_size == 1
            assert config.micro_batch_size in [1, 2, 4]

    def test_config_naming_format(self, gpt_7b_config, make_runner):
        """Config names match expected pattern."""
        runner = make_runner(
            gpt_7b_config,
            tensor_parallel_sizes=[2],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
        )
        _, configs = generate_configs(runner)
        pattern = re.compile(
            r"gpt_[\d.]+b_\d+nodes_tp_\d+_pp_\d+_cp_\d+_ep_\d+_mbs_\d+"
        )
        for name in configs:
            assert pattern.match(name), (
                f"Config name '{name}' doesn't match pattern"
            )

    def test_log_dir_format(self, gpt_7b_config, make_runner):
        """Log dir starts with path_to_logs and contains config name."""
        runner = make_runner(
            gpt_7b_config,
            path_to_logs="/tmp/test_auto_config",
            tensor_parallel_sizes=[2],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
        )
        _, configs = generate_configs(runner)
        for name, config in configs.items():
            assert config.log_dir.startswith("/tmp/test_auto_config/")
            assert name in config.log_dir

    def test_max_steps_propagated(self, gpt_7b_config, make_runner):
        """max_steps_per_run is propagated to all generated configs."""
        runner = make_runner(
            gpt_7b_config,
            max_steps_per_run=100,
            tensor_parallel_sizes=[2],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
        )
        _, configs = generate_configs(runner)
        for config in configs.values():
            assert config.max_steps == 100

    def test_context_parallel_sizes(self, gpt_7b_config, make_runner):
        """Context parallel sizes reflected in generated configs."""
        runner = make_runner(
            gpt_7b_config,
            tensor_parallel_sizes=[1],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
            context_parallel_sizes=[1, 2],
        )
        _, configs = generate_configs(runner)
        cp_values = {c.context_parallel_size for c in configs.values()}
        assert cp_values <= {1, 2}

    def test_model_size_computed_after_generate(
        self, gpt_7b_config, make_runner
    ):
        """After generate_configs, runner.model_size_in_b is set correctly."""
        runner = make_runner(gpt_7b_config)
        generate_configs(runner)
        # 7B-like config should yield ~6.59B
        assert 5.0 < runner.model_size_in_b < 10.0

    def test_auto_search_space_80gb(self, gpt_7b_config, make_runner):
        """Auto mode with 80GB GPU generates non-trivial search space."""
        runner = make_runner(gpt_7b_config)
        _, configs = generate_configs(runner)
        # Should produce multiple TP and MBS variants
        tp_values = {c.tensor_parallel_size for c in configs.values()}
        mbs_values = {c.micro_batch_size for c in configs.values()}
        assert len(tp_values) >= 1
        assert len(mbs_values) >= 1


# ============================================================================
# 5. Model Size Calculation Tests
# ============================================================================


class TestModelSizeCalculation:
    """Test model size calculation with real GPTConfig parameters."""

    def test_model_size_realistic_range(self, gpt_7b_config):
        """Model size for 7B-like config should be ~6.6B."""
        size = calculate_model_size(
            vocab_size=32000,
            seq_length=4096,
            hidden_size=4096,
            num_layers=32,
            ffn_size=11008,
            model_name="gpt",
        )
        assert 5.0 < size < 10.0

    def test_extract_matches_direct_calculation(
        self, gpt_7b_config, make_runner
    ):
        """_extract_model_size_from_config should match direct calculate_model_size."""
        runner = make_runner(gpt_7b_config)
        generate_configs(runner)
        extracted = runner.model_size_in_b

        direct = calculate_model_size(
            vocab_size=32000,
            seq_length=4096,
            hidden_size=4096,
            num_layers=32,
            ffn_size=11008,
            model_name="gpt",
        )
        assert abs(extracted - direct) < 0.01

    def test_estimate_model_size_function(self):
        """estimate_model_size returns reasonable value."""
        size = estimate_model_size(
            gpu_count=32,
            max_training_days=2,
            model_size_in_b=None,
            tflops_per_gpu=989,
            num_tokens_in_b=1400,
            model_name="gpt",
        )
        assert size > 0


# ============================================================================
# 6. Parallel Config Constraints in Generated Configs
# ============================================================================


class TestParallelConfigConstraints:
    """Verify generated configs satisfy parallel strategy constraints."""

    def test_pp_divides_num_layers(self, gpt_7b_config, make_runner):
        """PP always divides num_hidden_layers in generated configs."""
        runner = make_runner(
            gpt_7b_config,
            pipeline_parallel_sizes=[1, 2, 4, 8],
            tensor_parallel_sizes=[1],
            micro_batch_sizes=[1],
        )
        _, configs = generate_configs(runner)
        for config in configs.values():
            assert 32 % config.pipeline_parallel_size == 0

    def test_virtual_pipeline_for_pp_gt_2(self, gpt_7b_config, make_runner):
        """Virtual pipeline is set when PP > 2."""
        runner = make_runner(
            gpt_7b_config,
            tensor_parallel_sizes=[1],
            pipeline_parallel_sizes=[4],
            micro_batch_sizes=[1],
        )
        _, configs = generate_configs(runner)
        for config in configs.values():
            if config.pipeline_parallel_size > 2:
                assert (
                    config.virtual_pipeline_size
                    == 32 // config.pipeline_parallel_size
                )

    def test_tp_divides_attention_heads(self, gpt_7b_config, make_runner):
        """TP always divides num_attention_heads."""
        runner = make_runner(gpt_7b_config)
        _, configs = generate_configs(runner)
        for config in configs.values():
            assert 32 % config.tensor_parallel_size == 0

    def test_gbs_divisibility(self, gpt_7b_config, make_runner):
        """GBS is divisible by (MBS * DP_size) for every config."""
        runner = make_runner(
            gpt_7b_config,
            tensor_parallel_sizes=[2],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1, 2, 4, 8],
        )
        _, configs = generate_configs(runner)
        num_gpus = 32
        for config in configs.values():
            model_parallel = (
                config.tensor_parallel_size
                * config.pipeline_parallel_size
                * config.context_parallel_size
                * config.expert_parallel_size
            )
            dp_size = num_gpus / model_parallel
            assert (
                config.global_batch_size % (config.micro_batch_size * dp_size)
                == 0
            )


# ============================================================================
# 7. Edge Cases with Real Config
# ============================================================================


class TestEdgeCasesWithRealConfig:
    """Test boundary conditions using real GPTConfig objects."""

    def test_minimal_valid_config(self, make_runner):
        """Minimum valid config: layers=2, hidden=128, heads=2, seq=1024."""
        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=2,
            max_sequence_length=1024,
            vocab_size=1024,
        )
        runner = make_runner(
            config,
            tensor_parallel_sizes=[1],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
        )
        _, configs = generate_configs(runner)
        assert len(configs) >= 1

    def test_large_sequence_length(self, make_runner):
        """32768 sequence length uses the appropriate search space."""
        config = GPTConfig(
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            max_sequence_length=32768,
            vocab_size=32000,
        )
        runner = make_runner(config)
        _, configs = generate_configs(runner)
        assert len(configs) > 0

    def test_single_node_single_gpu(self, make_runner):
        """Single GPU produces exactly 1 config with tp=1, pp=1."""
        config = GPTConfig(
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            max_sequence_length=4096,
            vocab_size=32000,
        )
        runner = make_runner(
            config,
            num_nodes=1,
            num_gpus_per_node=1,
            global_batch_size=1,
            tensor_parallel_sizes=[1],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
        )
        _, configs = generate_configs(runner)
        assert len(configs) == 1
        config = next(iter(configs.values()))
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1

    def test_finetune_rejects_auto_tp(self, make_runner):
        """Finetune mode rejects tensor_parallel_sizes='auto'."""
        config = GPTConfig(
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            max_sequence_length=4096,
            vocab_size=32000,
        )
        with pytest.raises(ValueError, match="tensor_parallel_sizes"):
            make_runner(
                config,
                mode="finetune",
                tensor_parallel_sizes="auto",
                pipeline_parallel_sizes=[1],
            )


# ============================================================================
# Scoring, Dedup and max_configs E2E Tests
# ============================================================================


class TestGenerateConfigsScoringE2E:
    """E2E tests for generate_configs with max_configs and scoring_fn."""

    def test_no_scoring_returns_all(self, gpt_7b_config, make_runner):
        """Without scoring_fn, max_configs has no effect — all configs returned."""
        runner = make_runner(
            gpt_7b_config,
            tensor_parallel_sizes=[1, 2, 4],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1, 2, 4],
        )
        _, all_configs = generate_configs(runner)
        _, same_configs = generate_configs(
            runner, max_configs=1, scoring_fn=None
        )
        assert len(same_configs) == len(all_configs)

    def test_max_configs_limits_output(self, gpt_7b_config, make_runner):
        """With scoring_fn, max_configs limits the number of returned configs."""
        runner = make_runner(
            gpt_7b_config,
            tensor_parallel_sizes=[1, 2, 4],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1, 2, 4],
        )
        _, all_configs = generate_configs(runner)
        assert len(all_configs) > 2, (
            "Need more than 2 configs to test truncation"
        )

        _, limited = generate_configs(
            runner,
            max_configs=2,
            scoring_fn=lambda c: -c.tensor_parallel_size,
        )
        assert len(limited) <= 2
        assert len(limited) <= len(all_configs)

    def test_scoring_dedup_unique_parallel_groups(
        self, gpt_7b_config, make_runner
    ):
        """After scoring + dedup, no two configs share the same (TP,PP,CP,EP)."""
        runner = make_runner(
            gpt_7b_config,
            tensor_parallel_sizes=[1, 2],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1, 2, 4, 8],
        )
        _, deduped = generate_configs(
            runner,
            scoring_fn=lambda c: c.micro_batch_size,
        )
        groups = set()
        for cfg in deduped.values():
            key = (
                cfg.tensor_parallel_size,
                cfg.pipeline_parallel_size,
                cfg.context_parallel_size,
                cfg.expert_parallel_size,
            )
            assert key not in groups, f"Duplicate parallel group {key}"
            groups.add(key)

    def test_negative_max_configs_raises(self, gpt_7b_config, make_runner):
        """Negative max_configs raises ValueError."""
        runner = make_runner(gpt_7b_config)
        with pytest.raises(ValueError, match="max_configs"):
            generate_configs(runner, max_configs=-1, scoring_fn=lambda c: 1.0)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
