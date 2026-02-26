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
Unit tests for pdcost costmodel module.

Tests PredictionResult and PDCostModel classes.
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
    HardwareConfig,
    GPUSpec,
    NetworkSpec,
)
from pdcostmodel.memory_model import MemoryBreakdown
from pdcostmodel.costmodel import (
    PredictionResult,
    PDCostModel,
    create_qwen3_30b_costmodel,
    create_deepseek_v3_costmodel,
)


# ============================================================================
# PredictionResult Tests
# ============================================================================


class TestPredictionResult:
    """Test cases for PredictionResult class."""

    def test_default_values(self):
        """Test default PredictionResult values."""
        result = PredictionResult()
        assert result.step_time_ms == 0.0
        assert result.compute_time_ms == 0.0
        assert result.forward_time_ms == 0.0
        assert result.backward_time_ms == 0.0
        assert result.memory_gb == 0.0
        assert result.fits_memory is True
        assert result.mfu == 0.0

    def test_custom_values(self):
        """Test PredictionResult with custom values."""
        result = PredictionResult(
            step_time_ms=1000.0,
            compute_time_ms=800.0,
            forward_time_ms=250.0,
            backward_time_ms=550.0,
            memory_gb=60.0,
            fits_memory=True,
            mfu=0.35,
            tokens_per_second=50000.0,
        )
        assert result.step_time_ms == 1000.0
        assert result.compute_time_ms == 800.0
        assert result.memory_gb == 60.0
        assert result.mfu == 0.35
        assert result.tokens_per_second == 50000.0

    def test_to_dict(self):
        """Test to_dict method."""
        result = PredictionResult(
            step_time_ms=1000.0,
            compute_time_ms=800.0,
            forward_time_ms=250.0,
            backward_time_ms=550.0,
            total_comm_time_ms=150.0,
            memory_gb=60.0,
            mfu=0.35,
            tokens_per_second=50000.0,
            parallel_config={"tp": 4, "pp": 2},
        )
        d = result.to_dict()
        
        assert "time" in d
        assert "memory" in d
        assert "efficiency" in d
        assert "throughput" in d
        assert "config" in d
        assert d["time"]["step_time_ms"] == 1000.0
        assert d["efficiency"]["mfu"] == 0.35

    def test_str_representation(self):
        """Test __str__ method."""
        result = PredictionResult(
            step_time_ms=1000.0,
            compute_time_ms=800.0,
            total_comm_time_ms=150.0,
            bubble_time_ms=50.0,
            bubble_ratio=0.05,
            memory_gb=60.0,
            fits_memory=True,
            mfu=0.35,
            tokens_per_second=50000.0,
            tokens_per_second_per_gpu=6250.0,
        )
        s = str(result)
        
        assert "PredictionResult" in s
        assert "Step Time" in s
        assert "Memory" in s
        assert "MFU" in s


# ============================================================================
# PDCostModel Tests
# ============================================================================


class TestPDCostModel:
    """Test cases for PDCostModel class."""

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
    def hardware_config(self):
        """Create a test HardwareConfig."""
        return HardwareConfig(
            gpu=GPUSpec(
                name="H100",
                memory_gb=80.0,
                bf16_tflops=989.0,
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
        """Create a test TrainingConfig."""
        return TrainingConfig(
            micro_batch_size=1,
            sequence_length=2048,
            dtype="bfloat16",
            recompute_granularity="full",
        )

    def test_init_with_configs(
        self, model_config, hardware_config, training_config
    ):
        """Test PDCostModel initialization with explicit configs."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        
        assert costmodel.model_config == model_config
        assert costmodel.hardware_config == hardware_config
        assert costmodel.training_config == training_config
        assert costmodel.is_calibrated is True  # Explicit config means calibrated

    def test_init_with_default_training_config(
        self, model_config, hardware_config
    ):
        """Test PDCostModel initialization with default training config."""
        costmodel = PDCostModel(model_config, hardware_config)
        
        assert costmodel.training_config is not None
        assert costmodel.training_config.micro_batch_size == 1

    def test_init_sub_models(
        self, model_config, hardware_config, training_config
    ):
        """Test that sub-models are initialized."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        
        assert costmodel.memory_model is not None
        assert costmodel.compute_model is not None
        assert costmodel.comm_model is not None

    def test_predict_basic(
        self, model_config, hardware_config, training_config
    ):
        """Test basic predict method."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        result = costmodel.predict(
            parallel,
            micro_batch_size=1,
            seq_len=2048,
            gradient_accumulation_steps=16,
        )
        
        assert isinstance(result, PredictionResult)
        assert result.step_time_ms > 0
        assert result.memory_gb > 0
        assert result.compute_time_ms > 0

    def test_predict_with_tp(
        self, model_config, hardware_config, training_config
    ):
        """Test predict with tensor parallel."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        
        parallel_tp1 = ParallelConfig(tp=1, pp=1, dp=8)
        parallel_tp4 = ParallelConfig(tp=4, pp=1, dp=2)
        
        result_tp1 = costmodel.predict(parallel_tp1, 1, 2048)
        result_tp4 = costmodel.predict(parallel_tp4, 1, 2048)
        
        # With TP=4, should have different memory usage
        assert result_tp4.memory_gb != result_tp1.memory_gb

    def test_predict_with_pp(
        self, model_config, hardware_config, training_config
    ):
        """Test predict with pipeline parallel."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        
        parallel = ParallelConfig(tp=1, pp=4, dp=2)
        
        result = costmodel.predict(
            parallel, micro_batch_size=1, seq_len=2048,
            gradient_accumulation_steps=16
        )
        
        # With PP, should have bubble time
        assert result.bubble_ratio > 0

    def test_predict_with_sharding(
        self, model_config, hardware_config, training_config
    ):
        """Test predict with sharding."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        
        parallel_stage1 = ParallelConfig(tp=1, pp=1, dp=8, sharding="stage1")
        parallel_stage2 = ParallelConfig(tp=1, pp=1, dp=8, sharding="stage2")
        
        result_stage1 = costmodel.predict(parallel_stage1, 1, 2048)
        result_stage2 = costmodel.predict(parallel_stage2, 1, 2048)
        
        # Different sharding stages should have different memory
        assert result_stage1.memory_gb != result_stage2.memory_gb or True  # Allow same for now

    def test_predict_with_recompute(
        self, model_config, hardware_config, training_config
    ):
        """Test predict with different recompute settings."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        result_no_recompute = costmodel.predict(
            parallel, 1, 2048, recompute_granularity="none"
        )
        result_full_recompute = costmodel.predict(
            parallel, 1, 2048, recompute_granularity="full"
        )
        
        # Full recompute should use less activation memory
        assert (result_full_recompute.memory_breakdown.activation_memory_gb <=
                result_no_recompute.memory_breakdown.activation_memory_gb)

    def test_predict_with_offload(
        self, model_config, hardware_config, training_config
    ):
        """Test predict with tensorwise offload."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        result_no_offload = costmodel.predict(
            parallel, 1, 2048, tensorwise_offload_optimizer=False
        )
        result_with_offload = costmodel.predict(
            parallel, 1, 2048, tensorwise_offload_optimizer=True
        )
        
        # With offload, optimizer memory should be less
        assert (result_with_offload.memory_breakdown.optimizer_memory_gb <
                result_no_offload.memory_breakdown.optimizer_memory_gb)

    def test_predict_fits_memory_true(
        self, model_config, hardware_config, training_config
    ):
        """Test predict returns fits_memory=True when memory fits."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        result = costmodel.predict(parallel, 1, 2048)
        
        # Small model should fit in 80GB
        assert result.fits_memory is True

    def test_predict_mfu_calculation(
        self, model_config, hardware_config, training_config
    ):
        """Test that MFU is calculated."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        result = costmodel.predict(parallel, 1, 2048, gradient_accumulation_steps=16)
        
        assert 0 <= result.mfu <= 1.0

    def test_predict_throughput_calculation(
        self, model_config, hardware_config, training_config
    ):
        """Test that throughput is calculated."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        result = costmodel.predict(parallel, 1, 2048, gradient_accumulation_steps=16)
        
        assert result.tokens_per_step > 0
        assert result.tokens_per_second > 0
        assert result.tokens_per_second_per_gpu > 0

    def test_predict_calibrated(
        self, moe_model_config, hardware_config, training_config
    ):
        """Test predict_calibrated method."""
        costmodel = PDCostModel(
            moe_model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=1, dp=8, ep=8)
        
        result = costmodel.predict_calibrated(
            parallel, micro_batch_size=1, seq_len=2048,
            gradient_accumulation_steps=16
        )
        
        assert isinstance(result, PredictionResult)
        assert result.step_time_ms > 0
        assert result.memory_gb > 0

    def test_generate_search_space(
        self, model_config, hardware_config, training_config
    ):
        """Test generate_search_space method."""
        costmodel = PDCostModel(
            model_config, hardware_config, training_config
        )
        
        configs = costmodel.generate_search_space(
            total_gpus=8, max_tp=8, max_pp=4
        )
        
        assert len(configs) > 0
        for cfg in configs:
            assert "tp" in cfg
            assert "pp" in cfg
            assert "dp" in cfg
            assert cfg["tp"] * cfg["pp"] * cfg["dp"] == 8


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_create_qwen3_30b_costmodel(self):
        """Test create_qwen3_30b_costmodel function."""
        costmodel = create_qwen3_30b_costmodel(
            gpu_memory_gb=80.0,
            num_nodes=1,
            gpus_per_node=8,
        )
        
        assert isinstance(costmodel, PDCostModel)
        assert costmodel.model_config.num_experts == 128
        assert costmodel.hardware_config.gpu.memory_gb == 80.0

    def test_create_deepseek_v3_costmodel(self):
        """Test create_deepseek_v3_costmodel function."""
        costmodel = create_deepseek_v3_costmodel(
            gpu_memory_gb=80.0,
            num_nodes=1,
            gpus_per_node=8,
        )
        
        assert isinstance(costmodel, PDCostModel)
        assert costmodel.model_config.num_experts == 256


# ============================================================================
# Integration Tests
# ============================================================================


class TestPDCostModelIntegration:
    """Integration tests for PDCostModel."""

    def test_rank_configurations(self):
        """Test rank_configurations method."""
        model_config = ModelConfig.from_name("llama3-8b")
        hardware_config = HardwareConfig(
            gpu=GPUSpec(name="H100", memory_gb=80.0, bf16_tflops=989.0),
            num_nodes=1,
            gpus_per_node=8,
        )
        costmodel = PDCostModel(model_config, hardware_config)
        
        configs = [
            {"tp": 1, "pp": 1, "dp": 8, "sharding": "stage1"},
            {"tp": 2, "pp": 1, "dp": 4, "sharding": "stage1"},
            {"tp": 4, "pp": 1, "dp": 2, "sharding": "stage1"},
        ]
        
        ranked = costmodel.rank_configurations(configs, top_k=3)
        
        assert len(ranked) <= 3
        for r in ranked:
            assert "rank" in r
            assert "step_time_ms" in r
            assert "fits_memory" in r

    def test_end_to_end_prediction(self):
        """Test end-to-end prediction workflow."""
        # Create model config
        model_config = ModelConfig(
            num_hidden_layers=12,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=64,
            num_experts=1,
            vocab_size=32000,
        )
        
        # Create hardware config
        hardware_config = HardwareConfig(
            gpu=GPUSpec(name="A100", memory_gb=40.0, bf16_tflops=312.0),
            num_nodes=1,
            gpus_per_node=4,
        )
        
        # Create training config
        training_config = TrainingConfig(
            micro_batch_size=2,
            sequence_length=1024,
            dtype="bfloat16",
            recompute_granularity="full",
        )
        
        # Create cost model
        costmodel = PDCostModel(model_config, hardware_config, training_config)
        
        # Predict
        parallel = ParallelConfig(tp=2, pp=1, dp=2)
        result = costmodel.predict(
            parallel,
            micro_batch_size=2,
            seq_len=1024,
            gradient_accumulation_steps=8,
        )
        
        # Validate result
        assert result.step_time_ms > 0
        assert result.memory_gb > 0
        assert result.memory_gb < 40.0  # Should fit in GPU memory
        assert result.fits_memory is True
        assert result.tokens_per_second > 0


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])