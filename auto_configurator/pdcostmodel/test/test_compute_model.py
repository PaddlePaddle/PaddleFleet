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
Unit tests for pdcost compute_model module.

Tests LayerProfile, ComputeModel and computation time estimation.
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
)
from pdcostmodel.compute_model import (
    LayerType,
    LayerProfile,
    ComputeModel,
)


# ============================================================================
# LayerType Enum Tests
# ============================================================================


class TestLayerType:
    """Test cases for LayerType enum."""

    def test_attention_value(self):
        """Test LayerType.ATTENTION value."""
        assert LayerType.ATTENTION.value == "attention"

    def test_dense_mlp_value(self):
        """Test LayerType.DENSE_MLP value."""
        assert LayerType.DENSE_MLP.value == "dense_mlp"

    def test_moe_router_value(self):
        """Test LayerType.MOE_ROUTER value."""
        assert LayerType.MOE_ROUTER.value == "moe_router"

    def test_moe_expert_value(self):
        """Test LayerType.MOE_EXPERT value."""
        assert LayerType.MOE_EXPERT.value == "moe_expert"

    def test_layernorm_value(self):
        """Test LayerType.LAYERNORM value."""
        assert LayerType.LAYERNORM.value == "layernorm"

    def test_embedding_value(self):
        """Test LayerType.EMBEDDING value."""
        assert LayerType.EMBEDDING.value == "embedding"


# ============================================================================
# LayerProfile Tests
# ============================================================================


class TestLayerProfile:
    """Test cases for LayerProfile class."""

    def test_default_values(self):
        """Test default LayerProfile values."""
        profile = LayerProfile(layer_type=LayerType.ATTENTION)
        assert profile.layer_type == LayerType.ATTENTION
        assert profile.flops_per_token == 0
        assert profile.efficiency == 0.5

    def test_custom_values(self):
        """Test LayerProfile with custom values."""
        profile = LayerProfile(
            layer_type=LayerType.DENSE_MLP,
            flops_per_token=1000000,
            efficiency=0.7,
        )
        assert profile.layer_type == LayerType.DENSE_MLP
        assert profile.flops_per_token == 1000000
        assert profile.efficiency == 0.7

    def test_estimate_time_ms(self):
        """Test estimate_time_ms method."""
        profile = LayerProfile(
            layer_type=LayerType.ATTENTION,
            flops_per_token=1000000,
            efficiency=0.5,
        )
        
        time_ms = profile.estimate_time_ms(
            tokens=1024,
            peak_tflops=100.0,
            parallel_factor=1,
        )
        
        assert time_ms > 0

    def test_estimate_time_ms_with_parallel(self):
        """Test estimate_time_ms with parallel factor."""
        profile = LayerProfile(
            layer_type=LayerType.ATTENTION,
            flops_per_token=1000000,
            efficiency=0.5,
        )
        
        time_no_parallel = profile.estimate_time_ms(1024, 100.0, 1)
        time_with_parallel = profile.estimate_time_ms(1024, 100.0, 4)
        
        # With parallel_factor=4, time should be ~1/4
        assert time_with_parallel < time_no_parallel
        assert abs(time_with_parallel - time_no_parallel / 4) < 0.01


# ============================================================================
# ComputeModel Tests
# ============================================================================


class TestComputeModel:
    """Test cases for ComputeModel class."""

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
        )

    def test_init_layer_profiles(
        self, model_config, hardware_config, training_config
    ):
        """Test layer profiles are initialized."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        
        assert LayerType.ATTENTION in compute_model.layer_profiles
        assert LayerType.DENSE_MLP in compute_model.layer_profiles
        assert LayerType.MOE_ROUTER in compute_model.layer_profiles
        assert LayerType.MOE_EXPERT in compute_model.layer_profiles
        assert LayerType.LAYERNORM in compute_model.layer_profiles

    def test_estimate_layer_time_attention(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_layer_time for attention."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        
        time_ms = compute_model.estimate_layer_time(
            LayerType.ATTENTION, batch_size=1, seq_len=2048, tp_degree=1
        )
        
        assert time_ms > 0

    def test_estimate_layer_time_mlp(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_layer_time for MLP."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        
        time_ms = compute_model.estimate_layer_time(
            LayerType.DENSE_MLP, batch_size=1, seq_len=2048, tp_degree=1
        )
        
        assert time_ms > 0

    def test_estimate_layer_time_with_different_batch(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_layer_time with different batch sizes."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        
        time_bs1 = compute_model.estimate_layer_time(
            LayerType.ATTENTION, batch_size=1, seq_len=2048, tp_degree=1
        )
        time_bs4 = compute_model.estimate_layer_time(
            LayerType.ATTENTION, batch_size=4, seq_len=2048, tp_degree=1
        )
        
        # Larger batch should take more time
        assert time_bs4 > time_bs1

    def test_estimate_attention_time(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_attention_time method."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        
        time_ms = compute_model.estimate_attention_time(
            batch_size=1, seq_len=2048, tp_degree=1
        )
        
        assert time_ms > 0

    def test_estimate_attention_time_with_tp(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_attention_time with TP."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        
        time_tp1 = compute_model.estimate_attention_time(1, 2048, tp_degree=1)
        time_tp4 = compute_model.estimate_attention_time(1, 2048, tp_degree=4)
        
        # With TP=4, time should be less
        assert time_tp4 < time_tp1

    def test_estimate_mlp_time(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_mlp_time method."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        
        time_ms = compute_model.estimate_mlp_time(
            batch_size=1, seq_len=2048, tp_degree=1
        )
        
        assert time_ms > 0

    def test_estimate_moe_time(
        self, moe_model_config, hardware_config, training_config
    ):
        """Test estimate_moe_time method."""
        compute_model = ComputeModel(
            moe_model_config, hardware_config, training_config
        )
        
        time_ms = compute_model.estimate_moe_time(
            batch_size=1, seq_len=2048, tp_degree=1, ep_degree=8
        )
        
        assert time_ms > 0

    def test_estimate_moe_time_with_ep(
        self, moe_model_config, hardware_config, training_config
    ):
        """Test estimate_moe_time with different EP."""
        compute_model = ComputeModel(
            moe_model_config, hardware_config, training_config
        )
        
        time_ep1 = compute_model.estimate_moe_time(1, 2048, tp_degree=1, ep_degree=1)
        time_ep8 = compute_model.estimate_moe_time(1, 2048, tp_degree=1, ep_degree=8)
        
        # With EP=8, time should be different (typically less)
        assert time_ep1 != time_ep8

    def test_estimate_dense_layer_time(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_dense_layer_time method."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        
        time_ms = compute_model.estimate_dense_layer_time(
            batch_size=1, seq_len=2048, tp_degree=1
        )
        
        # Should include attention + mlp + layernorm
        assert time_ms > 0

    def test_estimate_moe_layer_time(
        self, moe_model_config, hardware_config, training_config
    ):
        """Test estimate_moe_layer_time method."""
        compute_model = ComputeModel(
            moe_model_config, hardware_config, training_config
        )
        
        time_ms = compute_model.estimate_moe_layer_time(
            batch_size=1, seq_len=2048, tp_degree=1, ep_degree=8
        )
        
        assert time_ms > 0

    def test_estimate_forward_time(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_forward_time method."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        time_ms = compute_model.estimate_forward_time(
            batch_size=1, seq_len=2048, parallel=parallel
        )
        
        assert time_ms > 0

    def test_estimate_forward_time_with_pp(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_forward_time with pipeline parallel."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        
        parallel_pp1 = ParallelConfig(tp=1, pp=1, dp=8)
        parallel_pp4 = ParallelConfig(tp=1, pp=4, dp=2)
        
        time_pp1 = compute_model.estimate_forward_time(1, 2048, parallel_pp1)
        time_pp4 = compute_model.estimate_forward_time(1, 2048, parallel_pp4)
        
        # With PP=4, each stage has 1/4 layers, so should be less
        assert time_pp4 < time_pp1

    def test_estimate_backward_time(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_backward_time method."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        forward_time = compute_model.estimate_forward_time(1, 2048, parallel)
        backward_time = compute_model.estimate_backward_time(1, 2048, parallel)
        
        # Backward is ~2x forward
        assert backward_time > forward_time
        assert abs(backward_time - 2 * forward_time) < 0.01

    def test_estimate_pipeline_bubble_no_pp(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_pipeline_bubble with pp=1."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        
        bubble = compute_model.estimate_pipeline_bubble(
            forward_time=10.0, backward_time=20.0,
            pp_degree=1, num_micro_batches=16
        )
        
        assert bubble == 0.0

    def test_estimate_pipeline_bubble_with_pp(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_pipeline_bubble with pp > 1."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        
        bubble = compute_model.estimate_pipeline_bubble(
            forward_time=10.0, backward_time=20.0,
            pp_degree=4, num_micro_batches=16
        )
        
        # bubble_ratio = (4-1)/16 = 0.1875
        expected_ratio = 3 / 16
        expected_bubble = (10.0 + 20.0) * expected_ratio
        assert abs(bubble - expected_bubble) < 0.01

    def test_estimate_step_compute_time(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_step_compute_time method."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        result = compute_model.estimate_step_compute_time(
            batch_size=1, seq_len=2048,
            parallel=parallel, num_micro_batches=16,
            recompute_overhead=1.0
        )
        
        assert "forward_time_ms" in result
        assert "backward_time_ms" in result
        assert "bubble_time_ms" in result
        assert "compute_time_ms" in result
        assert "bubble_ratio" in result
        assert result["compute_time_ms"] > 0

    def test_estimate_step_compute_time_with_recompute(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_step_compute_time with recompute overhead."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=1, dp=8)
        
        result_no_recompute = compute_model.estimate_step_compute_time(
            1, 2048, parallel, 16, recompute_overhead=1.0
        )
        result_with_recompute = compute_model.estimate_step_compute_time(
            1, 2048, parallel, 16, recompute_overhead=1.33
        )
        
        # With recompute overhead, backward time should be higher
        assert result_with_recompute["backward_time_ms"] > result_no_recompute["backward_time_ms"]

    def test_estimate_step_compute_time_with_pp(
        self, model_config, hardware_config, training_config
    ):
        """Test estimate_step_compute_time with pipeline parallel."""
        compute_model = ComputeModel(
            model_config, hardware_config, training_config
        )
        parallel = ParallelConfig(tp=1, pp=4, dp=2)
        
        result = compute_model.estimate_step_compute_time(
            batch_size=1, seq_len=2048,
            parallel=parallel, num_micro_batches=16
        )
        
        # With PP, there should be bubble time
        assert result["bubble_time_ms"] > 0
        assert result["bubble_ratio"] > 0


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])