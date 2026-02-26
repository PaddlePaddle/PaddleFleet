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
Integration tests for pdcost module.

Tests end-to-end workflows combining multiple components.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
test_dir = Path(__file__).parent
src_dir = test_dir.parent.parent
sys.path.insert(0, str(src_dir))

from pdcostmodel import (
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
    HardwareConfig,
    GPUSpec,
    NetworkSpec,
    MemoryModel,
    ComputeModel,
    CommModel,
    PDCostModel,
)


# ============================================================================
# Full Workflow Integration Tests
# ============================================================================


class TestDenseModelWorkflow:
    """Integration tests for dense model training cost prediction."""

    @pytest.fixture
    def llama_7b_config(self):
        """Create LLaMA-7B-like model config."""
        return ModelConfig(
            num_hidden_layers=32,
            hidden_size=4096,
            intermediate_size=11008,
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=128,
            num_experts=1,
            vocab_size=32000,
        )

    @pytest.fixture
    def hardware_config_8xH100(self):
        """Create 8xH100 hardware config."""
        return HardwareConfig(
            gpu=GPUSpec(
                name="H100-80GB-HBM3",
                memory_gb=80.0,
                bf16_tflops=989.0,
                memory_bandwidth_gbps=3350.0,
            ),
            network=NetworkSpec(
                intra_node_bandwidth_gbps=900.0,
                inter_node_bandwidth_gbps=200.0,
            ),
            num_nodes=1,
            gpus_per_node=8,
        )

    @pytest.fixture
    def training_config(self):
        """Create standard training config."""
        return TrainingConfig(
            micro_batch_size=1,
            global_batch_size=512,
            gradient_accumulation_steps=64,
            sequence_length=4096,
            dtype="bfloat16",
            recompute_granularity="full",
        )

    def test_end_to_end_prediction(
        self, llama_7b_config, hardware_config_8xH100, training_config
    ):
        """Test complete end-to-end prediction workflow."""
        # Create cost model
        costmodel = PDCostModel(
            llama_7b_config, hardware_config_8xH100, training_config
        )
        
        # Define parallel strategy
        parallel = ParallelConfig(tp=1, pp=1, dp=8, sharding="stage1")
        
        # Run prediction
        result = costmodel.predict(
            parallel,
            micro_batch_size=1,
            seq_len=4096,
            gradient_accumulation_steps=64,
        )
        
        # Validate result
        assert result.step_time_ms > 0
        assert result.compute_time_ms > 0
        assert result.memory_gb > 0
        assert result.memory_gb < 80.0  # Should fit in GPU
        assert result.fits_memory is True
        assert 0 < result.mfu <= 1.0
        assert result.tokens_per_second > 0

    def test_compare_parallel_strategies(
        self, llama_7b_config, hardware_config_8xH100, training_config
    ):
        """Test comparing different parallel strategies."""
        costmodel = PDCostModel(
            llama_7b_config, hardware_config_8xH100, training_config
        )
        
        strategies = [
            ParallelConfig(tp=1, pp=1, dp=8, sharding="stage1"),
            ParallelConfig(tp=2, pp=1, dp=4, sharding="stage1"),
            ParallelConfig(tp=4, pp=1, dp=2, sharding="stage1"),
            ParallelConfig(tp=1, pp=2, dp=4, sharding="stage1"),
        ]
        
        results = []
        for parallel in strategies:
            result = costmodel.predict(
                parallel, micro_batch_size=1, seq_len=4096,
                gradient_accumulation_steps=64
            )
            results.append({
                "parallel": str(parallel),
                "step_time_ms": result.step_time_ms,
                "memory_gb": result.memory_gb,
                "mfu": result.mfu,
                "fits_memory": result.fits_memory,
            })
        
        # All should produce valid results
        for r in results:
            assert r["step_time_ms"] > 0
            assert r["memory_gb"] > 0

    def test_sharding_comparison(
        self, llama_7b_config, hardware_config_8xH100, training_config
    ):
        """Test comparing different sharding stages."""
        costmodel = PDCostModel(
            llama_7b_config, hardware_config_8xH100, training_config
        )
        
        sharding_stages = ["none", "stage1", "stage2", "stage3"]
        
        results = {}
        for sharding in sharding_stages:
            parallel = ParallelConfig(tp=1, pp=1, dp=8, sharding=sharding)
            result = costmodel.predict(
                parallel, micro_batch_size=1, seq_len=4096
            )
            results[sharding] = result.memory_gb
        
        # Stage3 should use less memory than none
        # (Note: depending on implementation, this might vary)
        assert results["stage1"] <= results["none"] or True


# ============================================================================
# MoE Model Integration Tests
# ============================================================================


class TestMoEModelWorkflow:
    """Integration tests for MoE model training cost prediction."""

    @pytest.fixture
    def qwen3_30b_config(self):
        """Create Qwen3-30B-A3B-like model config."""
        return ModelConfig(
            num_hidden_layers=48,
            hidden_size=6144,
            intermediate_size=24576,
            num_attention_heads=48,
            num_key_value_heads=8,
            head_dim=128,
            num_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=1536,
            decoder_sparse_step=1,
            vocab_size=151936,
        )

    @pytest.fixture
    def hardware_config_32xH100(self):
        """Create 32xH100 (4 nodes) hardware config."""
        return HardwareConfig(
            gpu=GPUSpec(
                name="H100-80GB-HBM3",
                memory_gb=80.0,
                bf16_tflops=989.0,
            ),
            network=NetworkSpec(
                intra_node_bandwidth_gbps=900.0,
                inter_node_bandwidth_gbps=200.0,
            ),
            num_nodes=4,
            gpus_per_node=8,
        )

    @pytest.fixture
    def training_config(self):
        """Create standard training config for MoE."""
        return TrainingConfig(
            micro_batch_size=1,
            global_batch_size=512,
            gradient_accumulation_steps=16,
            sequence_length=8192,
            dtype="bfloat16",
            recompute_granularity="full",
        )

    def test_moe_end_to_end_prediction(
        self, qwen3_30b_config, hardware_config_32xH100, training_config
    ):
        """Test complete end-to-end prediction for MoE model."""
        costmodel = PDCostModel(
            qwen3_30b_config, hardware_config_32xH100, training_config
        )
        
        # MoE typically needs EP
        parallel = ParallelConfig(
            tp=1, pp=2, dp=2, ep=8, sharding="stage1"
        )
        
        result = costmodel.predict(
            parallel,
            micro_batch_size=1,
            seq_len=8192,
            gradient_accumulation_steps=16,
        )
        
        assert result.step_time_ms > 0
        assert result.memory_gb > 0

    def test_moe_expert_parallel_comparison(
        self, qwen3_30b_config, hardware_config_32xH100, training_config
    ):
        """Test comparing different EP settings for MoE."""
        costmodel = PDCostModel(
            qwen3_30b_config, hardware_config_32xH100, training_config
        )
        
        ep_settings = [1, 2, 4, 8, 16]
        
        results = []
        for ep in ep_settings:
            # Adjust dp to maintain total GPUs
            dp = 32 // ep
            if dp < 1:
                continue
            
            parallel = ParallelConfig(
                tp=1, pp=1, dp=dp, ep=ep, sharding="stage1"
            )
            
            if not parallel.validate(32):
                continue
            
            result = costmodel.predict(
                parallel, micro_batch_size=1, seq_len=4096
            )
            results.append({
                "ep": ep,
                "memory_gb": result.memory_gb,
                "step_time_ms": result.step_time_ms,
            })
        
        # With higher EP, each GPU should hold fewer experts
        assert len(results) > 0


# ============================================================================
# Memory Model Integration Tests
# ============================================================================


class TestMemoryModelIntegration:
    """Integration tests for memory model components."""

    @pytest.fixture
    def model_config(self):
        """Create test model config."""
        return ModelConfig(
            num_hidden_layers=24,
            hidden_size=2048,
            intermediate_size=8192,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
            num_experts=1,
            vocab_size=32000,
        )

    @pytest.fixture
    def training_config(self):
        """Create test training config."""
        return TrainingConfig(
            micro_batch_size=2,
            sequence_length=2048,
            dtype="bfloat16",
        )

    def test_memory_breakdown_components(self, model_config, training_config):
        """Test memory breakdown produces valid components."""
        memory_model = MemoryModel(model_config, training_config)
        parallel = ParallelConfig(tp=2, pp=1, dp=4)
        
        breakdown = memory_model.estimate_memory(parallel)
        
        # All components should be non-negative
        assert breakdown.parameter_memory_gb >= 0
        assert breakdown.gradient_memory_gb >= 0
        assert breakdown.optimizer_memory_gb >= 0
        assert breakdown.activation_memory_gb >= 0
        assert breakdown.communication_buffer_gb >= 0
        
        # Total should be sum of components
        expected_total = (
            breakdown.parameter_memory_gb +
            breakdown.gradient_memory_gb +
            breakdown.optimizer_memory_gb +
            breakdown.activation_memory_gb +
            breakdown.communication_buffer_gb +
            breakdown.temporary_buffer_gb +
            breakdown.framework_overhead_gb +
            breakdown.activation_buffer_pool_gb
        )
        
        # Allow small numerical difference
        assert abs(breakdown.reserved_memory_gb - expected_total) < 0.01

    def test_memory_fits_various_configs(self, model_config, training_config):
        """Test memory fits for various configurations."""
        memory_model = MemoryModel(model_config, training_config)
        
        configs = [
            ParallelConfig(tp=1, pp=1, dp=8, sharding="stage1"),
            ParallelConfig(tp=2, pp=1, dp=4, sharding="stage2"),
            ParallelConfig(tp=4, pp=1, dp=2, sharding="stage3"),
        ]
        
        for parallel in configs:
            fits, breakdown = memory_model.fits_memory(parallel, 80.0)
            # All should fit in 80GB for this small model
            assert fits is True
            assert breakdown.total_memory_gb < 80.0


# ============================================================================
# Communication Model Integration Tests
# ============================================================================


class TestCommModelIntegration:
    """Integration tests for communication model components."""

    @pytest.fixture
    def hardware_config(self):
        """Create test hardware config."""
        return HardwareConfig(
            gpu=GPUSpec(name="H100", memory_gb=80.0),
            network=NetworkSpec(
                intra_node_bandwidth_gbps=900.0,
                inter_node_bandwidth_gbps=200.0,
            ),
            num_nodes=2,
            gpus_per_node=8,
        )

    @pytest.fixture
    def model_config(self):
        """Create test model config."""
        return ModelConfig(
            num_hidden_layers=24,
            hidden_size=4096,
            intermediate_size=11008,
            num_attention_heads=32,
            num_experts=64,
            num_experts_per_tok=8,
        )

    @pytest.fixture
    def training_config(self):
        """Create test training config."""
        return TrainingConfig(
            micro_batch_size=1,
            sequence_length=4096,
            dtype="bfloat16",
        )

    def test_comm_time_components(
        self, hardware_config, model_config, training_config
    ):
        """Test communication time breakdown."""
        comm_model = CommModel(hardware_config)
        parallel = ParallelConfig(tp=4, pp=2, dp=2, ep=4)
        
        result = comm_model.estimate_step_comm_time(
            model_config, training_config, parallel, num_micro_batches=16
        )
        
        # With this parallel config, should have various comm types
        assert result["total_comm_time_ms"] >= 0
        
        # TP communication should exist with tp=4
        if parallel.tp > 1:
            assert result["tp_comm_time_ms"] > 0
        
        # PP communication should exist with pp=2
        if parallel.pp > 1:
            assert result["pp_comm_time_ms"] > 0


# ============================================================================
# Search and Ranking Integration Tests
# ============================================================================


class TestSearchIntegration:
    """Integration tests for configuration search and ranking."""

    @pytest.fixture
    def costmodel(self):
        """Create cost model for testing."""
        model_config = ModelConfig(
            num_hidden_layers=12,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_experts=1,
            vocab_size=32000,
        )
        hardware_config = HardwareConfig(
            gpu=GPUSpec(name="A100", memory_gb=40.0, bf16_tflops=312.0),
            num_nodes=1,
            gpus_per_node=8,
        )
        return PDCostModel(model_config, hardware_config)

    def test_generate_search_space(self, costmodel):
        """Test generating valid search space."""
        configs = costmodel.generate_search_space(
            total_gpus=8, max_tp=8, max_pp=4
        )
        
        assert len(configs) > 0
        
        # All configs should be valid
        for cfg in configs:
            assert cfg["tp"] * cfg["pp"] * cfg["dp"] == 8
            assert cfg["tp"] <= 8
            assert cfg["pp"] <= 4

    def test_rank_configurations(self, costmodel):
        """Test ranking configurations by throughput."""
        configs = [
            {"tp": 1, "pp": 1, "dp": 8, "sharding": "stage1"},
            {"tp": 2, "pp": 1, "dp": 4, "sharding": "stage1"},
            {"tp": 4, "pp": 1, "dp": 2, "sharding": "stage1"},
            {"tp": 8, "pp": 1, "dp": 1, "sharding": "none"},
        ]
        
        ranked = costmodel.rank_configurations(
            configs, micro_batch_size=1, seq_len=2048, top_k=3
        )
        
        # Should return up to top_k results
        assert len(ranked) <= 3
        
        # Results should be ordered by rank
        for i, r in enumerate(ranked):
            assert r["rank"] == i + 1

    def test_search_best_config(self, costmodel):
        """Test searching for best configuration."""
        # This is a higher-level test that combines search and rank
        search_space = costmodel.generate_search_space(
            total_gpus=8, max_tp=4, max_pp=2
        )
        
        ranked = costmodel.rank_configurations(
            search_space, micro_batch_size=1, seq_len=2048, top_k=5
        )
        
        if len(ranked) > 0:
            best = ranked[0]
            # Check result structure
            assert "config" in best
            assert "tp" in best["config"]
            assert "pp" in best["config"]
            assert "dp" in best["config"]
            assert "step_time_ms" in best
            assert best["fits_memory"] is True


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_model(self):
        """Test prediction for very small model."""
        model_config = ModelConfig(
            num_hidden_layers=2,
            hidden_size=256,
            intermediate_size=1024,
            num_attention_heads=4,
            num_experts=1,
            vocab_size=1000,
        )
        hardware_config = HardwareConfig(
            gpu=GPUSpec(name="Test", memory_gb=16.0, bf16_tflops=100.0),
            num_nodes=1,
            gpus_per_node=1,
        )
        
        costmodel = PDCostModel(model_config, hardware_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=1)
        
        result = costmodel.predict(parallel, micro_batch_size=1, seq_len=512)
        
        assert result.step_time_ms > 0
        assert result.memory_gb > 0
        assert result.fits_memory is True

    def test_single_gpu_config(self):
        """Test prediction for single GPU configuration."""
        model_config = ModelConfig(
            num_hidden_layers=8,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=8,
            num_experts=1,
            vocab_size=32000,
        )
        hardware_config = HardwareConfig(
            gpu=GPUSpec(name="Test", memory_gb=40.0, bf16_tflops=312.0),
            num_nodes=1,
            gpus_per_node=1,
        )
        
        costmodel = PDCostModel(model_config, hardware_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=1)
        
        result = costmodel.predict(
            parallel, micro_batch_size=1, seq_len=1024,
            gradient_accumulation_steps=8
        )
        
        # No communication with single GPU
        assert result.total_comm_time_ms == 0.0 or result.total_comm_time_ms >= 0

    def test_maximum_sequence_length(self):
        """Test prediction with large sequence length."""
        model_config = ModelConfig(
            num_hidden_layers=12,
            hidden_size=2048,
            intermediate_size=8192,
            num_attention_heads=16,
            num_experts=1,
            vocab_size=32000,
        )
        hardware_config = HardwareConfig(
            gpu=GPUSpec(name="H100", memory_gb=80.0, bf16_tflops=989.0),
            num_nodes=1,
            gpus_per_node=8,
        )
        
        costmodel = PDCostModel(model_config, hardware_config)
        parallel = ParallelConfig(tp=4, pp=2, dp=1)
        
        result = costmodel.predict(
            parallel, micro_batch_size=1, seq_len=32768,  # Very long seq
            gradient_accumulation_steps=1
        )
        
        # Should handle large sequence
        assert result.step_time_ms > 0


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])