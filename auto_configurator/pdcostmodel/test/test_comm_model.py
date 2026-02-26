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
Unit tests for pdcost comm_model module.

Tests CommType, CommResult, and CommModel classes.
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
from pdcostmodel.comm_model import (
    CommType,
    CommResult,
    CommModel,
)


# ============================================================================
# CommType Enum Tests
# ============================================================================


class TestCommType:
    """Test cases for CommType enum."""

    def test_allreduce_value(self):
        """Test CommType.ALLREDUCE value."""
        assert CommType.ALLREDUCE.value == "allreduce"

    def test_allgather_value(self):
        """Test CommType.ALLGATHER value."""
        assert CommType.ALLGATHER.value == "allgather"

    def test_reduce_scatter_value(self):
        """Test CommType.REDUCE_SCATTER value."""
        assert CommType.REDUCE_SCATTER.value == "reduce_scatter"

    def test_alltoall_value(self):
        """Test CommType.ALLTOALL value."""
        assert CommType.ALLTOALL.value == "alltoall"

    def test_p2p_value(self):
        """Test CommType.P2P value."""
        assert CommType.P2P.value == "p2p"

    def test_broadcast_value(self):
        """Test CommType.BROADCAST value."""
        assert CommType.BROADCAST.value == "broadcast"


# ============================================================================
# CommResult Tests
# ============================================================================


class TestCommResult:
    """Test cases for CommResult class."""

    def test_default_values(self):
        """Test default CommResult values."""
        result = CommResult()
        assert result.time_ms == 0.0
        assert result.bandwidth_gbps == 0.0
        assert result.volume_bytes == 0
        assert result.latency_ms == 0.0
        assert result.transfer_ms == 0.0

    def test_custom_values(self):
        """Test CommResult with custom values."""
        result = CommResult(
            time_ms=10.5,
            bandwidth_gbps=500.0,
            volume_bytes=1024 * 1024 * 1024,
            latency_ms=0.5,
            transfer_ms=10.0,
        )
        assert result.time_ms == 10.5
        assert result.bandwidth_gbps == 500.0
        assert result.volume_bytes == 1024 * 1024 * 1024
        assert result.latency_ms == 0.5
        assert result.transfer_ms == 10.0


# ============================================================================
# CommModel Tests
# ============================================================================


class TestCommModel:
    """Test cases for CommModel class."""

    @pytest.fixture
    def hardware_config(self):
        """Create a test HardwareConfig."""
        return HardwareConfig(
            gpu=GPUSpec(name="H100", memory_gb=80.0),
            network=NetworkSpec(
                intra_node_bandwidth_gbps=900.0,
                inter_node_bandwidth_gbps=200.0,
                intra_node_latency_us=1.0,
                inter_node_latency_us=5.0,
                allreduce_efficiency=0.85,
                allgather_efficiency=0.80,
                alltoall_efficiency=0.70,
                p2p_efficiency=0.90,
            ),
            num_nodes=1,
            gpus_per_node=8,
        )

    @pytest.fixture
    def model_config(self):
        """Create a test ModelConfig."""
        return ModelConfig(
            num_hidden_layers=24,
            hidden_size=2048,
            intermediate_size=8192,
            num_attention_heads=32,
            num_experts=128,
            num_experts_per_tok=8,
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

    def test_predict_allreduce_single_gpu(self, hardware_config):
        """Test predict_allreduce returns empty for single GPU."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_allreduce(
            data_size_bytes=1024 * 1024,
            num_gpus=1,
        )
        
        assert result.time_ms == 0.0
        assert result.volume_bytes == 0

    def test_predict_allreduce_intra_node(self, hardware_config):
        """Test predict_allreduce for intra-node communication."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_allreduce(
            data_size_bytes=1024 * 1024 * 1024,  # 1GB
            num_gpus=8,
            is_intra_node=True,
        )
        
        assert result.time_ms > 0
        assert result.volume_bytes > 0
        assert result.bandwidth_gbps > 0

    def test_predict_allreduce_inter_node(self, hardware_config):
        """Test predict_allreduce for inter-node communication."""
        comm_model = CommModel(hardware_config)
        
        result_intra = comm_model.predict_allreduce(
            data_size_bytes=1024 * 1024 * 1024,
            num_gpus=8,
            is_intra_node=True,
        )
        result_inter = comm_model.predict_allreduce(
            data_size_bytes=1024 * 1024 * 1024,
            num_gpus=8,
            is_intra_node=False,
        )
        
        # Inter-node should be slower
        assert result_inter.time_ms > result_intra.time_ms

    def test_predict_allgather_single_gpu(self, hardware_config):
        """Test predict_allgather returns empty for single GPU."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_allgather(
            data_size_bytes=1024 * 1024,
            num_gpus=1,
        )
        
        assert result.time_ms == 0.0

    def test_predict_allgather(self, hardware_config):
        """Test predict_allgather method."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_allgather(
            data_size_bytes=1024 * 1024 * 512,  # 512MB
            num_gpus=8,
            is_intra_node=True,
        )
        
        assert result.time_ms > 0
        assert result.volume_bytes > 0

    def test_predict_reduce_scatter_single_gpu(self, hardware_config):
        """Test predict_reduce_scatter returns empty for single GPU."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_reduce_scatter(
            data_size_bytes=1024 * 1024,
            num_gpus=1,
        )
        
        assert result.time_ms == 0.0

    def test_predict_reduce_scatter(self, hardware_config):
        """Test predict_reduce_scatter method."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_reduce_scatter(
            data_size_bytes=1024 * 1024 * 512,
            num_gpus=8,
            is_intra_node=True,
        )
        
        assert result.time_ms > 0
        assert result.volume_bytes > 0

    def test_predict_alltoall_single_gpu(self, hardware_config):
        """Test predict_alltoall returns empty for single GPU."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_alltoall(
            data_size_bytes=1024 * 1024,
            num_gpus=1,
        )
        
        assert result.time_ms == 0.0

    def test_predict_alltoall(self, hardware_config):
        """Test predict_alltoall method."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_alltoall(
            data_size_bytes=1024 * 1024 * 256,
            num_gpus=8,
            is_intra_node=True,
            topk=8,
            num_experts=128,
        )
        
        assert result.time_ms > 0
        assert result.volume_bytes > 0

    def test_predict_p2p(self, hardware_config):
        """Test predict_p2p method."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_p2p(
            data_size_bytes=1024 * 1024 * 128,  # 128MB
            is_intra_node=False,
        )
        
        assert result.time_ms > 0
        assert result.volume_bytes == 1024 * 1024 * 128

    def test_predict_p2p_intra_vs_inter(self, hardware_config):
        """Test predict_p2p intra vs inter node."""
        comm_model = CommModel(hardware_config)
        
        result_intra = comm_model.predict_p2p(1024 * 1024 * 128, is_intra_node=True)
        result_inter = comm_model.predict_p2p(1024 * 1024 * 128, is_intra_node=False)
        
        # Inter-node should be slower
        assert result_inter.time_ms > result_intra.time_ms

    def test_predict_tp_comm_single_gpu(self, hardware_config):
        """Test predict_tp_comm returns empty for tp=1."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_tp_comm(
            activation_size_bytes=1024 * 1024,
            tp_degree=1,
        )
        
        assert result.time_ms == 0.0

    def test_predict_tp_comm(self, hardware_config):
        """Test predict_tp_comm method."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_tp_comm(
            activation_size_bytes=1024 * 1024 * 64,
            tp_degree=4,
        )
        
        assert result.time_ms > 0

    def test_predict_dp_comm_single_gpu(self, hardware_config):
        """Test predict_dp_comm returns empty for dp=1."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_dp_comm(
            gradient_size_bytes=1024 * 1024,
            dp_degree=1,
        )
        
        assert result.time_ms == 0.0

    def test_predict_dp_comm_allreduce(self, hardware_config):
        """Test predict_dp_comm without sharding (AllReduce)."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_dp_comm(
            gradient_size_bytes=1024 * 1024 * 1024,
            dp_degree=8,
            use_sharding=False,
        )
        
        assert result.time_ms > 0

    def test_predict_dp_comm_reduce_scatter(self, hardware_config):
        """Test predict_dp_comm with sharding (ReduceScatter)."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_dp_comm(
            gradient_size_bytes=1024 * 1024 * 1024,
            dp_degree=8,
            use_sharding=True,
        )
        
        assert result.time_ms > 0

    def test_predict_ep_comm_single_gpu(self, hardware_config):
        """Test predict_ep_comm returns empty for ep=1."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_ep_comm(
            token_data_bytes=1024 * 1024,
            ep_degree=1,
        )
        
        assert result.time_ms == 0.0

    def test_predict_ep_comm(self, hardware_config):
        """Test predict_ep_comm method."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_ep_comm(
            token_data_bytes=1024 * 1024 * 64,
            ep_degree=8,
            topk=8,
            num_experts=128,
        )
        
        assert result.time_ms > 0

    def test_predict_pp_comm_single_stage(self, hardware_config):
        """Test predict_pp_comm returns empty for pp=1."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_pp_comm(
            activation_size_bytes=1024 * 1024,
            pp_degree=1,
            num_micro_batches=16,
        )
        
        assert result.time_ms == 0.0

    def test_predict_pp_comm(self, hardware_config):
        """Test predict_pp_comm method."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_pp_comm(
            activation_size_bytes=1024 * 1024 * 64,
            pp_degree=4,
            num_micro_batches=16,
        )
        
        assert result.time_ms > 0

    def test_predict_sp_comm_single_gpu(self, hardware_config):
        """Test predict_sp_comm returns empty for tp=1."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_sp_comm(
            activation_size_bytes=1024 * 1024,
            tp_degree=1,
        )
        
        assert result.time_ms == 0.0

    def test_predict_sp_comm(self, hardware_config):
        """Test predict_sp_comm method."""
        comm_model = CommModel(hardware_config)
        
        result = comm_model.predict_sp_comm(
            activation_size_bytes=1024 * 1024 * 64,
            tp_degree=4,
        )
        
        assert result.time_ms > 0

    def test_estimate_step_comm_time(
        self, hardware_config, model_config, training_config
    ):
        """Test estimate_step_comm_time method."""
        comm_model = CommModel(hardware_config)
        parallel = ParallelConfig(tp=4, pp=2, dp=2, ep=8)
        
        result = comm_model.estimate_step_comm_time(
            model_config, training_config, parallel, num_micro_batches=16
        )
        
        assert "tp_comm_time_ms" in result
        assert "dp_comm_time_ms" in result
        assert "ep_comm_time_ms" in result
        assert "pp_comm_time_ms" in result
        assert "total_comm_time_ms" in result
        
        # With TP=4, should have TP comm time
        assert result["tp_comm_time_ms"] > 0
        
        # Total should be sum of all
        expected_total = (
            result["tp_comm_time_ms"] +
            result["ep_comm_time_ms"] +
            result["pp_comm_time_ms"] +
            result["dp_comm_time_ms"] +
            result["sp_comm_time_ms"]
        )
        assert abs(result["total_comm_time_ms"] - expected_total) < 0.01

    def test_estimate_step_comm_time_no_parallel(
        self, hardware_config, model_config, training_config
    ):
        """Test estimate_step_comm_time with no parallelism."""
        comm_model = CommModel(hardware_config)
        parallel = ParallelConfig(tp=1, pp=1, dp=1, ep=1)
        
        result = comm_model.estimate_step_comm_time(
            model_config, training_config, parallel, num_micro_batches=16
        )
        
        # No TP/PP/EP, minimal communication
        assert result["tp_comm_time_ms"] == 0.0
        assert result["pp_comm_time_ms"] == 0.0


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])