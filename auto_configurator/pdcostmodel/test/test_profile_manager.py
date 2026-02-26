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
Unit tests for pdcost profile_manager module.

Tests ProfileManager class and related functions.
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add parent directory to path
test_dir = Path(__file__).parent
src_dir = test_dir.parent.parent
sys.path.insert(0, str(src_dir))

from pdcostmodel.config import HardwareConfig, GPUSpec
from pdcostmodel.calibration import (
    PerformancePoint,
    PerformanceCurve,
    CalibrationResult,
)
from pdcostmodel.profile_manager import (
    ProfileManager,
    get_profile_filename,
    get_profile_path,
    PROFILES_DIR,
)


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Test cases for module-level helper functions."""

    def test_get_profile_filename(self):
        """Test get_profile_filename function."""
        filename = get_profile_filename("NVIDIA H800", 8, 1)
        assert "H800" in filename
        assert "8gpu" in filename
        assert "1node" in filename
        assert filename.endswith(".json")

    def test_get_profile_filename_cleans_name(self):
        """Test that get_profile_filename cleans GPU name."""
        filename = get_profile_filename("NVIDIA  H100  SXM", 4, 2)
        # Should not have double underscores
        assert "__" not in filename

    def test_get_profile_path(self):
        """Test get_profile_path function."""
        path = get_profile_path("H100", 8, 1)
        assert isinstance(path, Path)
        assert path.suffix == ".json"


# ============================================================================
# ProfileManager Tests
# ============================================================================


class TestProfileManager:
    """Test cases for ProfileManager class."""

    @pytest.fixture
    def temp_profile_dir(self):
        """Create a temporary directory for profiles and patch PROFILES_DIR."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        yield temp_path
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_calibration_result(self):
        """Create a sample calibration result."""
        points = [
            PerformancePoint(size=1024, tflops=500.0, efficiency=0.5, time_ms=50.0),
            PerformancePoint(size=2048, tflops=700.0, efficiency=0.7, time_ms=70.0),
        ]
        curve = PerformanceCurve(
            dtype="bfloat16",
            points=points,
            peak_tflops=1000.0,
            fit_a=0.05,
            fit_b=0.3,
        )
        
        return CalibrationResult(
            gpu_name="TestGPU",
            gpu_memory_gb=80.0,
            gpu_count=8,
            fp32_tflops=67.0,
            fp16_tflops=989.0,
            bf16_tflops=989.0,
            memory_bandwidth_gbps=3350.0,
            bf16_curve=curve,
            calibrated=True,
        )

    def test_init_default_dir(self):
        """Test ProfileManager initialization with default directory."""
        manager = ProfileManager()
        assert manager.profiles_dir is not None
        assert isinstance(manager.profiles_dir, Path)

    def test_init_custom_dir(self, temp_profile_dir):
        """Test ProfileManager initialization with custom directory."""
        manager = ProfileManager(profiles_dir=temp_profile_dir)
        assert manager.profiles_dir == temp_profile_dir

    def test_list_profiles_empty(self, temp_profile_dir):
        """Test list_profiles returns empty list when no profiles."""
        manager = ProfileManager(profiles_dir=temp_profile_dir)
        
        profiles = manager.list_profiles()
        
        assert profiles == []

    @patch('pdcostmodel.profile_manager.PROFILES_DIR')
    @patch('pdcostmodel.profile_manager.get_profile_path')
    def test_has_profile_with_mock(self, mock_get_path, mock_profiles_dir, temp_profile_dir):
        """Test has_profile with mocked path."""
        # Setup mock
        mock_profiles_dir.__truediv__ = lambda self, x: temp_profile_dir / x
        mock_get_path.return_value = temp_profile_dir / "TestGPU_8gpu_1node.json"
        
        manager = ProfileManager(profiles_dir=temp_profile_dir)
        
        # No file exists
        result = manager.has_profile("TestGPU", 8, 1)
        assert result is False
        
        # Create file
        (temp_profile_dir / "TestGPU_8gpu_1node.json").write_text("{}")
        mock_get_path.return_value = temp_profile_dir / "TestGPU_8gpu_1node.json"
        
        result = manager.has_profile("TestGPU", 8, 1)
        assert result is True

    def test_list_profiles_with_files(self, temp_profile_dir, sample_calibration_result):
        """Test list_profiles with saved files."""
        manager = ProfileManager(profiles_dir=temp_profile_dir)
        
        # Save profile data directly to temp dir
        profile_data = sample_calibration_result.to_dict()
        profile_data["node_count"] = 1
        profile_data["calibrated_at"] = "2025-01-01T00:00:00"
        
        filepath = temp_profile_dir / "TestGPU_8gpu_1node.json"
        with open(filepath, 'w') as f:
            json.dump(profile_data, f)
        
        profiles = manager.list_profiles()
        
        assert len(profiles) == 1
        assert profiles[0]["gpu_name"] == "TestGPU"

    def test_find_matching_profile_from_list(self, temp_profile_dir, sample_calibration_result):
        """Test find_matching_profile searches in profiles_dir."""
        manager = ProfileManager(profiles_dir=temp_profile_dir)
        
        # Save profile data directly to temp dir
        profile_data = sample_calibration_result.to_dict()
        profile_data["node_count"] = 1
        profile_data["calibrated_at"] = "2025-01-01T00:00:00"
        
        filepath = temp_profile_dir / "TestGPU_8gpu_1node.json"
        with open(filepath, 'w') as f:
            json.dump(profile_data, f)
        
        # Find matching
        match = manager.find_matching_profile(gpu_name="TestGPU", gpu_count=8)
        
        assert match is not None
        assert match["gpu_name"] == "TestGPU"

    def test_find_matching_profile_not_found(self, temp_profile_dir):
        """Test find_matching_profile returns None when no match."""
        manager = ProfileManager(profiles_dir=temp_profile_dir)
        
        match = manager.find_matching_profile(gpu_name="NonexistentGPU")
        
        assert match is None

    def test_load_profile_by_path(self, temp_profile_dir, sample_calibration_result):
        """Test load_profile_by_path method."""
        manager = ProfileManager(profiles_dir=temp_profile_dir)
        
        # Save profile data directly
        profile_data = sample_calibration_result.to_dict()
        profile_data["node_count"] = 1
        
        filepath = temp_profile_dir / "test_profile.json"
        with open(filepath, 'w') as f:
            json.dump(profile_data, f)
        
        # Load by path
        loaded = manager.load_profile_by_path(str(filepath))
        
        assert loaded is not None
        assert loaded["gpu_name"] == "TestGPU"
        assert loaded["bf16_tflops"] == 989.0

    def test_create_hardware_config(self, temp_profile_dir):
        """Test create_hardware_config method."""
        manager = ProfileManager(profiles_dir=temp_profile_dir)
        
        profile_data = {
            "gpu_name": "H100",
            "gpu_memory_gb": 80.0,
            "gpu_count": 8,
            "fp32_tflops": 67.0,
            "bf16_tflops": 989.0,
            "memory_bandwidth_gbps": 3350.0,
            "node_count": 1,
            "intra_node_bandwidth_gbps": 900.0,
            "inter_node_bandwidth_gbps": 200.0,
        }
        
        hw_config = manager.create_hardware_config(profile_data)
        
        assert isinstance(hw_config, HardwareConfig)
        assert hw_config.gpu.name == "H100"
        assert hw_config.gpu.memory_gb == 80.0
        assert hw_config.gpus_per_node == 8

    def test_delete_profile_not_found(self, temp_profile_dir):
        """Test delete_profile returns False when profile not found."""
        manager = ProfileManager(profiles_dir=temp_profile_dir)
        
        # Mock get_profile_path to use temp dir
        with patch('pdcostmodel.profile_manager.get_profile_path') as mock_path:
            mock_path.return_value = temp_profile_dir / "nonexistent.json"
            result = manager.delete_profile("NonexistentGPU", 8, 1)
        
        assert result is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestProfileManagerIntegration:
    """Integration tests for ProfileManager."""

    @pytest.fixture
    def temp_profile_dir(self):
        """Create a temporary directory for profiles."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_save_load_roundtrip_direct(self, temp_profile_dir):
        """Test full save/load roundtrip using direct file operations."""
        manager = ProfileManager(profiles_dir=temp_profile_dir)
        
        # Create calibration result with data
        points = [
            PerformancePoint(size=512, tflops=250.0, efficiency=0.25, time_ms=25.0),
            PerformancePoint(size=1024, tflops=500.0, efficiency=0.50, time_ms=50.0),
        ]
        curve = PerformanceCurve(
            dtype="bfloat16",
            points=points,
            peak_tflops=1000.0,
            fit_a=0.04,
            fit_b=0.2,
        )
        
        original = CalibrationResult(
            gpu_name="TestGPU",
            gpu_memory_gb=40.0,
            gpu_count=4,
            fp32_tflops=30.0,
            bf16_tflops=400.0,
            memory_bandwidth_gbps=2000.0,
            bf16_curve=curve,
            calibrated=True,
        )
        
        # Save directly to temp dir (simulating save_calibration behavior)
        profile_data = original.to_dict()
        profile_data["node_count"] = 1
        profile_data["calibrated_at"] = "2025-01-01T00:00:00"
        
        filepath = temp_profile_dir / "TestGPU_4gpu_1node.json"
        with open(filepath, 'w') as f:
            json.dump(profile_data, f)
        
        # Load via load_profile_by_path
        loaded = manager.load_profile_by_path(str(filepath))
        
        # Verify
        assert loaded["gpu_name"] == original.gpu_name
        assert loaded["gpu_memory_gb"] == original.gpu_memory_gb
        assert loaded["bf16_tflops"] == original.bf16_tflops

    def test_multiple_profiles_workflow_direct(self, temp_profile_dir):
        """Test workflow with multiple GPU profiles using direct file ops."""
        manager = ProfileManager(profiles_dir=temp_profile_dir)
        
        # Create and save multiple profiles directly
        gpus = [
            ("H800", 8, 80.0, 989.0),
            ("A100", 8, 80.0, 312.0),
            ("A100_40GB", 4, 40.0, 312.0),
        ]
        
        for gpu_name, gpu_count, memory, tflops in gpus:
            points = [
                PerformancePoint(size=1024, tflops=tflops*0.5, efficiency=0.5, time_ms=50.0),
            ]
            curve = PerformanceCurve(
                dtype="bfloat16",
                points=points,
                peak_tflops=tflops,
            )
            
            result = CalibrationResult(
                gpu_name=gpu_name,
                gpu_memory_gb=memory,
                gpu_count=gpu_count,
                bf16_tflops=tflops,
                bf16_curve=curve,
                calibrated=True,
            )
            
            profile_data = result.to_dict()
            profile_data["node_count"] = 1
            profile_data["calibrated_at"] = "2025-01-01T00:00:00"
            
            filepath = temp_profile_dir / f"{gpu_name}_{gpu_count}gpu_1node.json"
            with open(filepath, 'w') as f:
                json.dump(profile_data, f)
        
        # Verify all profiles saved
        profiles = manager.list_profiles()
        assert len(profiles) == 3
        
        # Test finding by name
        gpu_names = [p["gpu_name"] for p in profiles]
        assert "H800" in gpu_names
        assert "A100" in gpu_names


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])