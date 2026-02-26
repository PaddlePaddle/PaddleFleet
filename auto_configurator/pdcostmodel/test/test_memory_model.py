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
Unit tests for pdcost memory_model module.

Tests ShardingConfig, RecomputeConfig, MemoryBreakdown, and MemoryModel classes.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
test_dir = Path(__file__).parent
src_dir = test_dir.parent.parent
sys.path.insert(0, str(src_dir))

from pdcostmodel.config import (
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
    ShardingStage,
    RecomputeGranularity,
)
from pdcostmodel.memory_model import (
    ShardingConfig,
    RecomputeConfig,
    MemoryBreakdown,
    MemoryModel,
)


# ============================================================================
# ShardingConfig Tests
# ============================================================================


class TestShardingConfig:
    """Test cases for ShardingConfig class."""

    def test_default_values(self):
        """Test default ShardingConfig values."""
        config = ShardingConfig()
        assert config.stage == ShardingStage.STAGE1
        assert config.degree == 1
        assert config.split_param is True
        assert config.release_grads is False
        assert config.tensorwise_offload is False

    def test_get_param_sharding_factor_stage3(self):
        """Test get_param_sharding_factor for stage3."""
        config = ShardingConfig(stage=ShardingStage.STAGE3, degree=8)
        assert config.get_param_sharding_factor() == 8

    def test_get_param_sharding_factor_stage1_with_split_param(self):
        """Test get_param_sharding_factor for stage1 with split_param."""
        config = ShardingConfig(
            stage=ShardingStage.STAGE1, degree=8, split_param=True
        )
        assert config.get_param_sharding_factor() == 8

    def test_get_param_sharding_factor_stage1_without_split_param(self):
        """Test get_param_sharding_factor for stage1 without split_param."""
        config = ShardingConfig(
            stage=ShardingStage.STAGE1, degree=8, split_param=False
        )
        assert config.get_param_sharding_factor() == 1

    def test_get_grad_sharding_factor_stage2(self):
        """Test get_grad_sharding_factor for stage2."""
        config = ShardingConfig(stage=ShardingStage.STAGE2, degree=4)
        assert config.get_grad_sharding_factor() == 4

    def test_get_grad_sharding_factor_stage3(self):
        """Test get_grad_sharding_factor for stage3."""
        config = ShardingConfig(stage=ShardingStage.STAGE3, degree=8)
        assert config.get_grad_sharding_factor() == 8

    def test_get_grad_sharding_factor_stage1_with_split_param(self):
        """Test get_grad_sharding_factor for stage1 with split_param."""
        config = ShardingConfig(
            stage=ShardingStage.STAGE1, degree=4, split_param=True
        )
        assert config.get_grad_sharding_factor() == 4

    def test_get_optimizer_sharding_factor_stage1(self):
        """Test get_optimizer_sharding_factor for stage1."""
        config = ShardingConfig(stage=ShardingStage.STAGE1, degree=8)
        assert config.get_optimizer_sharding_factor() == 8

    def test_get_optimizer_sharding_factor_none(self):
        """Test get_optimizer_sharding_factor for none."""
        config = ShardingConfig(stage=ShardingStage.NONE, degree=8)
        assert config.get_optimizer_sharding_factor() == 1

    def test_get_optimizer_memory_factor_no_offload(self):
        """Test get_optimizer_memory_factor without offload."""
        config = ShardingConfig(tensorwise_offload=False)
        assert config.get_optimizer_memory_factor() == 1.0

    def test_get_optimizer_memory_factor_with_tensorwise_offload(self):
        """Test get_optimizer_memory_factor with tensorwise offload."""
        config = ShardingConfig(tensorwise_offload=True, degree=8)
        factor = config.get_optimizer_memory_factor()
        assert 0 < factor < 1.0  # Should reduce memory

    def test_get_optimizer_memory_factor_offload_dp1(self):
        """Test get_optimizer_memory_factor with offload but dp=1."""
        config = ShardingConfig(tensorwise_offload=True, degree=1)
        assert config.get_optimizer_memory_factor() == 1.0  # No effect

    def test_uses_release_grads_true(self):
        """Test uses_release_grads returns True."""
        config = ShardingConfig(release_grads=True)
        assert config.uses_release_grads() is True

    def test_uses_release_grads_false(self):
        """Test uses_release_grads returns False."""
        config = ShardingConfig(release_grads=False)
        assert config.uses_release_grads() is False


# ============================================================================
# RecomputeConfig Tests
# ============================================================================


class TestRecomputeConfig:
    """Test cases for RecomputeConfig class."""

    def test_default_values(self):
        """Test default RecomputeConfig values."""
        config = RecomputeConfig()
        assert config.granularity == RecomputeGranularity.FULL
        assert config.method == "uniform"
        assert config.num_layers == 1

    def test_get_memory_reduction_factor_none(self):
        """Test get_memory_reduction_factor for none."""
        config = RecomputeConfig(granularity=RecomputeGranularity.NONE)
        assert config.get_memory_reduction_factor() == 1.0

    def test_get_memory_reduction_factor_selective(self):
        """Test get_memory_reduction_factor for selective."""
        config = RecomputeConfig(granularity=RecomputeGranularity.SELECTIVE)
        assert config.get_memory_reduction_factor() == 0.6

    def test_get_memory_reduction_factor_full_uniform_1(self):
        """Test get_memory_reduction_factor for full uniform with num_layers=1."""
        config = RecomputeConfig(
            granularity=RecomputeGranularity.FULL,
            method="uniform",
            num_layers=1,
        )
        factor = config.get_memory_reduction_factor()
        assert factor < 1.0  # Should reduce memory

    def test_get_memory_reduction_factor_full_uniform_4(self):
        """Test get_memory_reduction_factor for full uniform with num_layers=4."""
        config = RecomputeConfig(
            granularity=RecomputeGranularity.FULL,
            method="uniform",
            num_layers=4,
        )
        factor = config.get_memory_reduction_factor()
        assert factor < 1.0

    def test_get_recompute_overhead_none(self):
        """Test get_recompute_overhead for none."""
        config = RecomputeConfig(granularity=RecomputeGranularity.NONE)
        assert config.get_recompute_overhead() == 1.0

    def test_get_recompute_overhead_selective(self):
        """Test get_recompute_overhead for selective."""
        config = RecomputeConfig(granularity=RecomputeGranularity.SELECTIVE)
        assert config.get_recompute_overhead() == 1.15

    def test_get_recompute_overhead_full(self):
        """Test get_recompute_overhead for full."""
        config = RecomputeConfig(
            granularity=RecomputeGranularity.FULL,
            method="uniform",
            num_layers=1,
        )
        assert config.get_recompute_overhead() == 1.33


# ============================================================================
# MemoryBreakdown Tests
# ============================================================================


class TestMemoryBreakdown:
    """Test cases for MemoryBreakdown class."""

    def test_default_values(self):
        """Test default MemoryBreakdown values."""
        breakdown = MemoryBreakdown()
        assert breakdown.parameter_memory_gb == 0.0
        assert breakdown.gradient_memory_gb == 0.0
        assert breakdown.optimizer_memory_gb == 0.0
        assert breakdown.activation_memory_gb == 0.0
        assert breakdown.framework_overhead_gb == 2.0

    def test_allocated_memory_gb_property(self):
        """Test allocated_memory_gb property calculation."""
        breakdown = MemoryBreakdown(
            parameter_memory_gb=10.0,
            gradient_memory_gb=10.0,
            optimizer_memory_gb=20.0,
            activation_memory_gb=5.0,
            communication_buffer_gb=1.0,
            temporary_buffer_gb=0.5,
            framework_overhead_gb=2.0,
        )
        expected = 10.0 + 10.0 + 20.0 + 5.0 + 1.0 + 0.5 + 2.0
        assert breakdown.allocated_memory_gb == expected

    def test_reserved_memory_gb_property(self):
        """Test reserved_memory_gb property calculation."""
        breakdown = MemoryBreakdown(
            parameter_memory_gb=10.0,
            gradient_memory_gb=10.0,
            optimizer_memory_gb=20.0,
            activation_memory_gb=5.0,
            activation_buffer_pool_gb=3.0,
        )
        allocated = breakdown.allocated_memory_gb
        assert breakdown.reserved_memory_gb == allocated + 3.0

    def test_total_memory_gb_equals_reserved(self):
        """Test total_memory_gb equals reserved_memory_gb."""
        breakdown = MemoryBreakdown(
            parameter_memory_gb=10.0,
            activation_buffer_pool_gb=5.0,
        )
        assert breakdown.total_memory_gb == breakdown.reserved_memory_gb

    def test_model_states_gb_property(self):
        """Test model_states_gb property calculation."""
        breakdown = MemoryBreakdown(
            parameter_memory_gb=10.0,
            gradient_memory_gb=10.0,
            optimizer_memory_gb=20.0,
        )
        assert breakdown.model_states_gb == 40.0

    def test_to_dict(self):
        """Test to_dict method."""
        breakdown = MemoryBreakdown(
            parameter_memory_gb=10.5,
            gradient_memory_gb=10.5,
            optimizer_memory_gb=21.0,
            activation_memory_gb=5.0,
        )
        d = breakdown.to_dict()
        assert "parameter_memory_gb" in d
        assert "gradient_memory_gb" in d
        assert "optimizer_memory_gb" in d
        assert "total_memory_gb" in d

    def test_str_representation(self):
        """Test __str__ method."""
        breakdown = MemoryBreakdown(
            parameter_memory_gb=10.0,
            gradient_memory_gb=10.0,
        )
        s = str(breakdown)
        assert "Parameters" in s
        assert "Gradients" in s


# ============================================================================
# MemoryModel Tests
# ============================================================================


class TestMemoryModel:
    """Test cases for MemoryModel class."""

    @pytest.fixture
    def model_config(self):
        """Create a test ModelConfig."""
        return ModelConfig(
            num_hidden_layers=24,
            hidden_size=2048,
            intermediate_size=8192,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=64,
            num_experts=1,
            vocab_size=32000,
        )

    @pytest.fixture
    def moe_model_config(self):
        """Create a MoE test ModelConfig."""
        return ModelConfig(
            num_hidden_layers=24,
            hidden_size=2048,
            intermediate_size=8192,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=64,
            num_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=768,
            decoder_sparse_step=1,
            vocab_size=32000,
        )

    @pytest.fixture
    def training_config(self):
        """Create a test TrainingConfig."""
        return TrainingConfig(
            micro_batch_size=1,
            sequence_length=2048,
            dtype="bfloat16",
        )

    def test_estimate_parameter_count_per_gpu_dense(
        self, model_config, training_config
    ):
        """Test estimate_parameter_count_per_gpu for dense model."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=1)
        
        params = memory_model.estimate_parameter_count_per_gpu(parallel)
        
        assert "total_params" in params
        assert "attention_params" in params
        assert params["total_params"] > 0

    def test_estimate_parameter_count_per_gpu_with_tp(
        self, model_config, training_config
    ):
        """Test estimate_parameter_count_per_gpu with TP."""
        memory_model = MemoryModel(model_config, training_config)
        
        parallel_tp1 = ParallelConfig(tp=1, pp=1, dp=8)
        parallel_tp4 = ParallelConfig(tp=4, pp=1, dp=2)
        
        params_tp1 = memory_model.estimate_parameter_count_per_gpu(parallel_tp1)
        params_tp4 = memory_model.estimate_parameter_count_per_gpu(parallel_tp4)
        
        # With TP=4, attention and mlp params should be ~1/4
        assert params_tp4["attention_params"] < params_tp1["attention_params"]

    def test_estimate_parameter_count_per_gpu_moe(
        self, moe_model_config, training_config
    ):
        """Test estimate_parameter_count_per_gpu for MoE model."""
        memory_model = MemoryModel(moe_model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=1, ep=8)
        
        params = memory_model.estimate_parameter_count_per_gpu(parallel)
        
        assert "expert_params" in params
        assert params["expert_params"] > 0

    def test_estimate_parameter_memory(self, model_config, training_config):
        """Test estimate_parameter_memory method."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        sharding = ShardingConfig(stage=ShardingStage.STAGE1, degree=8)
        
        memory_gb = memory_model.estimate_parameter_memory(parallel, sharding)
        
        assert memory_gb > 0

    def test_estimate_gradient_memory(self, model_config, training_config):
        """Test estimate_gradient_memory method."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        sharding = ShardingConfig(stage=ShardingStage.STAGE1, degree=8)
        
        memory_gb = memory_model.estimate_gradient_memory(parallel, sharding)
        
        assert memory_gb > 0

    def test_estimate_optimizer_memory(self, model_config, training_config):
        """Test estimate_optimizer_memory method."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        sharding = ShardingConfig(stage=ShardingStage.STAGE1, degree=8)
        
        memory_gb = memory_model.estimate_optimizer_memory(parallel, sharding)
        
        assert memory_gb > 0

    def test_estimate_optimizer_memory_with_offload(
        self, model_config, training_config
    ):
        """Test estimate_optimizer_memory with tensorwise offload."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        sharding_no_offload = ShardingConfig(
            stage=ShardingStage.STAGE1, degree=8, tensorwise_offload=False
        )
        sharding_with_offload = ShardingConfig(
            stage=ShardingStage.STAGE1, degree=8, tensorwise_offload=True
        )
        
        mem_no_offload = memory_model.estimate_optimizer_memory(
            parallel, sharding_no_offload
        )
        mem_with_offload = memory_model.estimate_optimizer_memory(
            parallel, sharding_with_offload
        )
        
        # With offload should be less memory
        assert mem_with_offload < mem_no_offload

    def test_estimate_activation_memory(self, model_config, training_config):
        """Test estimate_activation_memory method."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        recompute = RecomputeConfig(granularity=RecomputeGranularity.FULL)
        
        memory_gb = memory_model.estimate_activation_memory(parallel, recompute)
        
        assert memory_gb > 0

    def test_estimate_activation_memory_with_recompute(
        self, model_config, training_config
    ):
        """Test estimate_activation_memory with different recompute configs."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        recompute_none = RecomputeConfig(granularity=RecomputeGranularity.NONE)
        recompute_full = RecomputeConfig(granularity=RecomputeGranularity.FULL)
        
        mem_no_recompute = memory_model.estimate_activation_memory(
            parallel, recompute_none
        )
        mem_with_recompute = memory_model.estimate_activation_memory(
            parallel, recompute_full
        )
        
        # With recompute should use less activation memory
        assert mem_with_recompute < mem_no_recompute

    def test_estimate_communication_buffer(self, model_config, training_config):
        """Test estimate_communication_buffer method."""
        memory_model = MemoryModel(model_config, training_config)
        
        # With TP, should have communication buffer
        parallel_tp = ParallelConfig(tp=4, pp=1, dp=2)
        buffer_gb = memory_model.estimate_communication_buffer(parallel_tp)
        assert buffer_gb > 0
        
        # Without TP, less buffer
        parallel_no_tp = ParallelConfig(tp=1, pp=1, dp=8)
        buffer_gb_no_tp = memory_model.estimate_communication_buffer(parallel_no_tp)
        assert buffer_gb_no_tp < buffer_gb

    def test_estimate_activation_buffer_pool_small_seq(
        self, model_config, training_config
    ):
        """Test estimate_activation_buffer_pool for small seq_len."""
        memory_model = MemoryModel(model_config, training_config)
        
        buffer_gb = memory_model.estimate_activation_buffer_pool(2048)
        assert buffer_gb > 0

    def test_estimate_activation_buffer_pool_large_seq(
        self, model_config, training_config
    ):
        """Test estimate_activation_buffer_pool for large seq_len."""
        memory_model = MemoryModel(model_config, training_config)
        
        buffer_small = memory_model.estimate_activation_buffer_pool(4096)
        buffer_large = memory_model.estimate_activation_buffer_pool(8192)
        
        # Larger seq_len should have larger buffer
        assert buffer_large > buffer_small

    def test_estimate_memory_complete(self, model_config, training_config):
        """Test estimate_memory returns complete MemoryBreakdown."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        breakdown = memory_model.estimate_memory(parallel)
        
        assert isinstance(breakdown, MemoryBreakdown)
        assert breakdown.parameter_memory_gb > 0
        assert breakdown.gradient_memory_gb > 0
        assert breakdown.optimizer_memory_gb > 0
        assert breakdown.total_memory_gb > 0

    def test_estimate_memory_with_sharding_config(
        self, model_config, training_config
    ):
        """Test estimate_memory with explicit sharding config."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        sharding = ShardingConfig(
            stage=ShardingStage.STAGE2, degree=8, split_param=True
        )
        
        breakdown = memory_model.estimate_memory(parallel, sharding)
        
        assert breakdown.total_memory_gb > 0

    def test_fits_memory_true(self, model_config, training_config):
        """Test fits_memory returns True when memory fits."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        fits, breakdown = memory_model.fits_memory(parallel, 80.0)
        
        # For small model, should fit in 80GB
        assert fits is True
        assert isinstance(breakdown, MemoryBreakdown)

    def test_fits_memory_false(self, model_config, training_config):
        """Test fits_memory returns False when memory doesn't fit."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=1)  # No sharding
        
        fits, breakdown = memory_model.fits_memory(parallel, 1.0)  # Only 1GB
        
        assert fits is False


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])