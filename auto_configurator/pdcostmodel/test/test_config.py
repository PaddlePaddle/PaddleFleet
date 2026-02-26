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
Unit tests for pdcost config module.

Tests configuration classes including GPUSpec, NetworkSpec, HardwareConfig,
ModelConfig, ParallelConfig, and TrainingConfig.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
test_dir = Path(__file__).parent
src_dir = test_dir.parent.parent
sys.path.insert(0, str(src_dir))

from pdcostmodel.config import (
    GPUSpec,
    NetworkSpec,
    HardwareConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
    ShardingStage,
    RecomputeGranularity,
)


# ============================================================================
# GPUSpec Tests
# ============================================================================


class TestGPUSpec:
    """Test cases for GPUSpec class."""

    def test_default_values(self):
        """Test default GPUSpec values."""
        gpu = GPUSpec()
        assert gpu.name == "H100-80GB-HBM3"
        assert gpu.memory_gb == 80.0
        assert gpu.fp32_tflops == 67.0
        assert gpu.fp16_tflops == 989.0
        assert gpu.bf16_tflops == 989.0
        assert gpu.memory_bandwidth_gbps == 3350.0

    def test_custom_values(self):
        """Test GPUSpec with custom values."""
        gpu = GPUSpec(
            name="Custom-GPU",
            memory_gb=40.0,
            fp32_tflops=30.0,
            fp16_tflops=200.0,
            bf16_tflops=200.0,
            memory_bandwidth_gbps=1500.0,
        )
        assert gpu.name == "Custom-GPU"
        assert gpu.memory_gb == 40.0
        assert gpu.fp32_tflops == 30.0
        assert gpu.fp16_tflops == 200.0

    def test_get_tflops_fp32(self):
        """Test get_tflops for fp32."""
        gpu = GPUSpec(fp32_tflops=50.0)
        assert gpu.get_tflops("fp32") == 50.0

    def test_get_tflops_fp16(self):
        """Test get_tflops for fp16."""
        gpu = GPUSpec(fp16_tflops=400.0)
        assert gpu.get_tflops("fp16") == 400.0

    def test_get_tflops_bf16(self):
        """Test get_tflops for bf16."""
        gpu = GPUSpec(bf16_tflops=500.0)
        assert gpu.get_tflops("bf16") == 500.0

    def test_get_tflops_default(self):
        """Test get_tflops returns bf16 for unknown dtype."""
        gpu = GPUSpec(bf16_tflops=600.0)
        assert gpu.get_tflops("unknown") == 600.0

    def test_from_name_h100(self):
        """Test GPUSpec.from_name for H100."""
        gpu = GPUSpec.from_name("H100-80GB-HBM3")
        assert gpu.name == "H100-80GB-HBM3"
        assert gpu.memory_gb == 80.0
        assert gpu.bf16_tflops == 989.0

    def test_from_name_a100_80gb(self):
        """Test GPUSpec.from_name for A100-80GB."""
        gpu = GPUSpec.from_name("A100-80GB")
        assert gpu.name == "A100-80GB"
        assert gpu.memory_gb == 80.0
        assert gpu.bf16_tflops == 312.0

    def test_from_name_a100_40gb(self):
        """Test GPUSpec.from_name for A100-40GB."""
        gpu = GPUSpec.from_name("A100-40GB")
        assert gpu.name == "A100-40GB"
        assert gpu.memory_gb == 40.0

    def test_from_name_v100(self):
        """Test GPUSpec.from_name for V100."""
        gpu = GPUSpec.from_name("V100-32GB")
        assert gpu.name == "V100-32GB"
        assert gpu.memory_gb == 32.0
        assert gpu.bf16_tflops == 0.0  # V100 doesn't support BF16

    def test_from_name_unknown_returns_h100(self):
        """Test GPUSpec.from_name returns H100 for unknown GPU."""
        gpu = GPUSpec.from_name("Unknown-GPU")
        assert gpu.name == "H100-80GB-HBM3"


# ============================================================================
# NetworkSpec Tests
# ============================================================================


class TestNetworkSpec:
    """Test cases for NetworkSpec class."""

    def test_default_values(self):
        """Test default NetworkSpec values."""
        network = NetworkSpec()
        assert network.intra_node_bandwidth_gbps == 900.0
        assert network.intra_node_latency_us == 1.0
        assert network.inter_node_bandwidth_gbps == 200.0
        assert network.inter_node_latency_us == 5.0
        assert network.allreduce_efficiency == 0.85
        assert network.allgather_efficiency == 0.80
        assert network.alltoall_efficiency == 0.70
        assert network.p2p_efficiency == 0.90

    def test_custom_values(self):
        """Test NetworkSpec with custom values."""
        network = NetworkSpec(
            intra_node_bandwidth_gbps=600.0,
            inter_node_bandwidth_gbps=100.0,
            allreduce_efficiency=0.90,
        )
        assert network.intra_node_bandwidth_gbps == 600.0
        assert network.inter_node_bandwidth_gbps == 100.0
        assert network.allreduce_efficiency == 0.90


# ============================================================================
# HardwareConfig Tests
# ============================================================================


class TestHardwareConfig:
    """Test cases for HardwareConfig class."""

    def test_default_values(self):
        """Test default HardwareConfig values."""
        config = HardwareConfig()
        assert config.num_nodes == 1
        assert config.gpus_per_node == 8

    def test_total_gpus_property(self):
        """Test total_gpus property calculation."""
        config = HardwareConfig(num_nodes=4, gpus_per_node=8)
        assert config.total_gpus == 32

    def test_is_intra_node_true(self):
        """Test is_intra_node returns True for small degree."""
        config = HardwareConfig(gpus_per_node=8)
        assert config.is_intra_node(4) is True
        assert config.is_intra_node(8) is True

    def test_is_intra_node_false(self):
        """Test is_intra_node returns False for large degree."""
        config = HardwareConfig(gpus_per_node=8)
        assert config.is_intra_node(16) is False


# ============================================================================
# ModelConfig Tests
# ============================================================================


class TestModelConfig:
    """Test cases for ModelConfig class."""

    def test_default_values(self):
        """Test default ModelConfig values."""
        config = ModelConfig()
        assert config.num_hidden_layers == 48
        assert config.hidden_size == 6144
        assert config.num_attention_heads == 32
        assert config.num_experts == 128

    def test_num_moe_layers_property(self):
        """Test num_moe_layers property for MoE model."""
        config = ModelConfig(
            num_hidden_layers=48,
            num_experts=128,
            decoder_sparse_step=1,
        )
        assert config.num_moe_layers == 48

    def test_num_moe_layers_for_dense_model(self):
        """Test num_moe_layers is 0 for dense model."""
        config = ModelConfig(num_experts=1)
        assert config.num_moe_layers == 0

    def test_num_dense_layers_property(self):
        """Test num_dense_layers property."""
        config = ModelConfig(
            num_hidden_layers=48,
            num_experts=1,
        )
        assert config.num_dense_layers == 48

    def test_estimate_parameters(self):
        """Test estimate_parameters method."""
        config = ModelConfig(
            num_hidden_layers=24,
            hidden_size=2048,
            intermediate_size=8192,
            num_attention_heads=32,
            num_experts=1,
            vocab_size=32000,
        )
        params = config.estimate_parameters()
        assert "total" in params
        assert "embedding" in params
        assert "attention" in params
        assert params["total"] > 0
        assert params["total_billion"] > 0

    def test_from_name_qwen3_30b(self):
        """Test ModelConfig.from_name for Qwen3-30B."""
        config = ModelConfig.from_name("qwen3-30b-a3b")
        assert config.num_hidden_layers == 48
        assert config.num_experts == 128
        assert config.num_experts_per_tok == 8

    def test_from_name_llama3_70b(self):
        """Test ModelConfig.from_name for LLaMA3-70B."""
        config = ModelConfig.from_name("llama3-70b")
        assert config.num_hidden_layers == 80
        assert config.hidden_size == 8192
        assert config.num_experts == 1

    def test_from_name_llama3_8b(self):
        """Test ModelConfig.from_name for LLaMA3-8B."""
        config = ModelConfig.from_name("llama3-8b")
        assert config.num_hidden_layers == 32
        assert config.hidden_size == 4096

    def test_from_name_deepseek_v3(self):
        """Test ModelConfig.from_name for DeepSeek-V3."""
        config = ModelConfig.from_name("deepseek-v3")
        assert config.num_experts == 256
        assert len(config.mlp_only_layers) > 0

    def test_from_name_unknown_raises_error(self):
        """Test ModelConfig.from_name raises error for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelConfig.from_name("unknown-model")

    def test_from_dict(self):
        """Test ModelConfig.from_dict method."""
        data = {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "vocab_size": 128000,
        }
        config = ModelConfig.from_dict(data)
        assert config.num_hidden_layers == 32
        assert config.hidden_size == 4096
        assert config.vocab_size == 128000


# ============================================================================
# ParallelConfig Tests
# ============================================================================


class TestParallelConfig:
    """Test cases for ParallelConfig class."""

    def test_default_values(self):
        """Test default ParallelConfig values."""
        config = ParallelConfig()
        assert config.tp == 1
        assert config.pp == 1
        assert config.dp == 1
        assert config.ep == 1
        assert config.sharding == "stage1"

    def test_custom_values(self):
        """Test ParallelConfig with custom values."""
        config = ParallelConfig(tp=4, pp=2, dp=4, ep=8, sharding="stage2")
        assert config.tp == 4
        assert config.pp == 2
        assert config.dp == 4
        assert config.ep == 8
        assert config.sharding == "stage2"

    def test_sharding_stage_property_stage1(self):
        """Test sharding_stage property for stage1."""
        config = ParallelConfig(sharding="stage1")
        assert config.sharding_stage == ShardingStage.STAGE1

    def test_sharding_stage_property_stage2(self):
        """Test sharding_stage property for stage2."""
        config = ParallelConfig(sharding="stage2")
        assert config.sharding_stage == ShardingStage.STAGE2

    def test_sharding_stage_property_stage3(self):
        """Test sharding_stage property for stage3."""
        config = ParallelConfig(sharding="stage3")
        assert config.sharding_stage == ShardingStage.STAGE3

    def test_sharding_stage_property_none(self):
        """Test sharding_stage property for none."""
        config = ParallelConfig(sharding="none")
        assert config.sharding_stage == ShardingStage.NONE

    def test_effective_sharding_degree_with_degree(self):
        """Test effective_sharding_degree with explicit degree."""
        config = ParallelConfig(dp=8, sharding="stage1", sharding_degree=4)
        assert config.effective_sharding_degree == 4

    def test_effective_sharding_degree_auto(self):
        """Test effective_sharding_degree defaults to dp."""
        config = ParallelConfig(dp=8, sharding="stage1", sharding_degree=-1)
        assert config.effective_sharding_degree == 8

    def test_effective_sharding_degree_none_sharding(self):
        """Test effective_sharding_degree is 1 when sharding is none."""
        config = ParallelConfig(dp=8, sharding="none")
        assert config.effective_sharding_degree == 1

    def test_world_size_property(self):
        """Test world_size property calculation."""
        config = ParallelConfig(tp=4, pp=2, dp=4)
        assert config.world_size == 32

    def test_validate_true(self):
        """Test validate returns True for valid config."""
        config = ParallelConfig(tp=4, pp=2, dp=4)
        assert config.validate(32) is True

    def test_validate_false_mismatch(self):
        """Test validate returns False for mismatched config."""
        config = ParallelConfig(tp=4, pp=2, dp=4)
        assert config.validate(16) is False

    def test_validate_false_invalid_values(self):
        """Test validate returns False for invalid values."""
        config = ParallelConfig(tp=0, pp=1, dp=1)
        assert config.validate(8) is False

    def test_to_dict(self):
        """Test to_dict method."""
        config = ParallelConfig(tp=4, pp=2, dp=4, ep=8, sharding="stage2")
        d = config.to_dict()
        assert d["tp"] == 4
        assert d["pp"] == 2
        assert d["dp"] == 4
        assert d["ep"] == 8
        assert d["sharding"] == "stage2"

    def test_from_dict(self):
        """Test from_dict method."""
        data = {"tp": 8, "pp": 1, "dp": 4, "ep": 8, "sharding": "stage1"}
        config = ParallelConfig.from_dict(data)
        assert config.tp == 8
        assert config.pp == 1
        assert config.dp == 4
        assert config.ep == 8

    def test_str_representation(self):
        """Test __str__ method."""
        config = ParallelConfig(tp=4, pp=2, dp=4, ep=8, sharding="stage1")
        s = str(config)
        assert "TP4" in s
        assert "PP2" in s
        assert "DP4" in s
        assert "EP8" in s


# ============================================================================
# TrainingConfig Tests
# ============================================================================


class TestTrainingConfig:
    """Test cases for TrainingConfig class."""

    def test_default_values(self):
        """Test default TrainingConfig values."""
        config = TrainingConfig()
        assert config.micro_batch_size == 1
        assert config.global_batch_size == 512
        assert config.gradient_accumulation_steps == 64
        assert config.sequence_length == 8192
        assert config.dtype == "bfloat16"

    def test_dtype_bytes_float32(self):
        """Test dtype_bytes for float32."""
        config = TrainingConfig(dtype="float32")
        assert config.dtype_bytes == 4

    def test_dtype_bytes_float16(self):
        """Test dtype_bytes for float16."""
        config = TrainingConfig(dtype="float16")
        assert config.dtype_bytes == 2

    def test_dtype_bytes_bfloat16(self):
        """Test dtype_bytes for bfloat16."""
        config = TrainingConfig(dtype="bfloat16")
        assert config.dtype_bytes == 2

    def test_recompute_config_property_none(self):
        """Test recompute_config property for none."""
        config = TrainingConfig(recompute_granularity="none")
        assert config.recompute_config == RecomputeGranularity.NONE

    def test_recompute_config_property_selective(self):
        """Test recompute_config property for selective."""
        config = TrainingConfig(recompute_granularity="selective")
        assert config.recompute_config == RecomputeGranularity.SELECTIVE

    def test_recompute_config_property_full(self):
        """Test recompute_config property for full."""
        config = TrainingConfig(recompute_granularity="full")
        assert config.recompute_config == RecomputeGranularity.FULL

    def test_to_dict(self):
        """Test to_dict method."""
        config = TrainingConfig(
            micro_batch_size=4,
            sequence_length=4096,
            dtype="float16",
        )
        d = config.to_dict()
        assert d["micro_batch_size"] == 4
        assert d["sequence_length"] == 4096
        assert d["dtype"] == "float16"

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "micro_batch_size": 2,
            "global_batch_size": 1024,
            "sequence_length": 2048,
            "dtype": "bfloat16",
        }
        config = TrainingConfig.from_dict(data)
        assert config.micro_batch_size == 2
        assert config.global_batch_size == 1024
        assert config.sequence_length == 2048


# ============================================================================
# ShardingStage Enum Tests
# ============================================================================


class TestShardingStage:
    """Test cases for ShardingStage enum."""

    def test_none_value(self):
        """Test ShardingStage.NONE value."""
        assert ShardingStage.NONE.value == "none"

    def test_stage1_value(self):
        """Test ShardingStage.STAGE1 value."""
        assert ShardingStage.STAGE1.value == "stage1"

    def test_stage2_value(self):
        """Test ShardingStage.STAGE2 value."""
        assert ShardingStage.STAGE2.value == "stage2"

    def test_stage3_value(self):
        """Test ShardingStage.STAGE3 value."""
        assert ShardingStage.STAGE3.value == "stage3"


# ============================================================================
# RecomputeGranularity Enum Tests
# ============================================================================


class TestRecomputeGranularity:
    """Test cases for RecomputeGranularity enum."""

    def test_none_value(self):
        """Test RecomputeGranularity.NONE value."""
        assert RecomputeGranularity.NONE.value == "none"

    def test_selective_value(self):
        """Test RecomputeGranularity.SELECTIVE value."""
        assert RecomputeGranularity.SELECTIVE.value == "selective"

    def test_full_value(self):
        """Test RecomputeGranularity.FULL value."""
        assert RecomputeGranularity.FULL.value == "full"


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])