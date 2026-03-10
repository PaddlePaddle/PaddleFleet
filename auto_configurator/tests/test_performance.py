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
Unit tests for AutoConfigurator core.performance module.
"""

# ============================================================================
# Import Module Under Test
# ============================================================================
import sys
from pathlib import Path

import pytest

test_dir = Path(__file__).parent
src_dir = test_dir.parent
sys.path.insert(0, str(src_dir))

from auto_configurator.core.performance import calculate_tflops

# ============================================================================
# calculate_tflops Tests
# ============================================================================


class TestCalculateTflops:
    """Test cases for calculate_tflops function."""

    def test_calculate_gpt_small_model_tflops(self):
        """Test calculating TFLOPS for small GPT model."""
        model_tflops, per_gpu_tflops = calculate_tflops(
            model_name="gpt",
            gbs=64,
            enc_seq_len=2048,
            dec_seq_len=2048,
            hidden_size=1024,
            ffn_size=4096,
            num_layers=12,
            vocab=32000,
            num_nodes=1,
            gpus_per_node=8,
            time_per_step=0.5,
        )

        # Should be positive values
        assert model_tflops > 0
        assert per_gpu_tflops > 0

        # Per GPU should be 1/8 of model TFLOPS
        assert abs(per_gpu_tflops - (model_tflops / 8)) < 0.01

    def test_calculate_gpt_medium_model_tflops(self):
        """Test calculating TFLOPS for medium GPT model."""
        model_tflops, per_gpu_tflops = calculate_tflops(
            model_name="gpt",
            gbs=512,
            enc_seq_len=4096,
            dec_seq_len=4096,
            hidden_size=2048,
            ffn_size=8192,
            num_layers=24,
            vocab=50000,
            num_nodes=4,
            gpus_per_node=8,
            time_per_step=1.0,
        )

        assert model_tflops > 0
        assert per_gpu_tflops > 0
        # 32 GPUs total
        assert abs(per_gpu_tflops - (model_tflops / 32)) < 0.01

    def test_calculate_gpt_with_llama(self):
        """Test calculating TFLOPS for Llama model (same formula as GPT)."""
        model_tflops, per_gpu_tflops = calculate_tflops(
            model_name="llama",
            gbs=256,
            enc_seq_len=4096,
            dec_seq_len=4096,
            hidden_size=4096,
            ffn_size=16384,
            num_layers=32,
            vocab=32000,
            num_nodes=2,
            gpus_per_node=8,
            time_per_step=2.0,
        )

        assert model_tflops > 0
        assert per_gpu_tflops > 0
        assert abs(per_gpu_tflops - (model_tflops / 16)) < 0.01

    def test_calculate_gpt_with_mixtral(self):
        """Test calculating TFLOPS for Mixtral model (same formula as GPT)."""
        model_tflops, per_gpu_tflops = calculate_tflops(
            model_name="mixtral",
            gbs=256,
            enc_seq_len=4096,
            dec_seq_len=4096,
            hidden_size=4096,
            ffn_size=16384,
            num_layers=32,
            vocab=32000,
            num_nodes=4,
            gpus_per_node=8,
            time_per_step=2.5,
        )

        assert model_tflops > 0
        assert per_gpu_tflops > 0

    def test_calculate_bert_model_tflops(self):
        """Test calculating TFLOPS for BERT model raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="BERT models are currently not supported"
        ):
            calculate_tflops(
                model_name="bert",
                gbs=64,
                enc_seq_len=512,
                dec_seq_len=512,
                hidden_size=768,
                ffn_size=3072,
                num_layers=12,
                vocab=30000,
                num_nodes=1,
                gpus_per_node=8,
                time_per_step=0.3,
            )

    def test_calculate_t5_model_tflops(self):
        """Test calculating TFLOPS for T5 model raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="T5/mT5 models are currently not supported",
        ):
            calculate_tflops(
                model_name="t5",
                gbs=64,
                enc_seq_len=512,
                dec_seq_len=512,
                hidden_size=768,
                ffn_size=3072,
                num_layers=12,
                vocab=32100,
                num_nodes=1,
                gpus_per_node=8,
                time_per_step=0.5,
            )

    def test_unsupported_model_type_raises_error(self):
        """Test that unsupported model type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model name"):
            calculate_tflops(
                model_name="unsupported",
                gbs=64,
                enc_seq_len=2048,
                dec_seq_len=2048,
                hidden_size=2048,
                ffn_size=8192,
                num_layers=24,
                vocab=32000,
                num_nodes=1,
                gpus_per_node=8,
                time_per_step=1.0,
            )

    def test_faster_time_higher_tflops(self):
        """Test that faster time per step gives higher TFLOPS."""
        # Slow time
        slow_model_tflops, slow_per_gpu = calculate_tflops(
            model_name="gpt",
            gbs=64,
            enc_seq_len=2048,
            dec_seq_len=2048,
            hidden_size=1024,
            ffn_size=4096,
            num_layers=12,
            vocab=32000,
            num_nodes=1,
            gpus_per_node=8,
            time_per_step=1.0,
        )

        # Fast time
        fast_model_tflops, fast_per_gpu = calculate_tflops(
            model_name="gpt",
            gbs=64,
            enc_seq_len=2048,
            dec_seq_len=2048,
            hidden_size=1024,
            ffn_size=4096,
            num_layers=12,
            vocab=32000,
            num_nodes=1,
            gpus_per_node=8,
            time_per_step=0.5,
        )

        # Faster time should give higher TFLOPS
        assert fast_model_tflops > slow_model_tflops
        assert fast_per_gpu > slow_per_gpu

    def test_more_gpus_higher_total_tflops(self):
        """Test that more GPUs gives higher total TFLOPS."""
        # 8 GPUs
        tflops_8gpus, _ = calculate_tflops(
            model_name="gpt",
            gbs=64,
            enc_seq_len=2048,
            dec_seq_len=2048,
            hidden_size=1024,
            ffn_size=4096,
            num_layers=12,
            vocab=32000,
            num_nodes=1,
            gpus_per_node=8,
            time_per_step=1.0,
        )

        # 32 GPUs
        tflops_32gpus, _ = calculate_tflops(
            model_name="gpt",
            gbs=256,
            enc_seq_len=2048,
            dec_seq_len=2048,
            hidden_size=1024,
            ffn_size=4096,
            num_layers=12,
            vocab=32000,
            num_nodes=4,
            gpus_per_node=8,
            time_per_step=1.0,
        )

        # More GPUs should give higher total TFLOPS (assuming same time per step)
        assert tflops_32gpus > tflops_8gpus

    def test_rounding_to_two_decimals(self):
        """Test that TFLOPS values are rounded to 2 decimal places."""
        _, per_gpu_tflops = calculate_tflops(
            model_name="gpt",
            gbs=64,
            enc_seq_len=2048,
            dec_seq_len=2048,
            hidden_size=1024,
            ffn_size=4096,
            num_layers=12,
            vocab=32000,
            num_nodes=1,
            gpus_per_node=8,
            time_per_step=1.0,
        )

        # Check that value is rounded
        # Convert to string and check decimal places
        tflops_str = str(per_gpu_tflops)
        if "." in tflops_str:
            decimals = len(tflops_str.split(".")[1])
            assert decimals <= 2


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
