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
Unit tests for pdcost calibration module.

Tests PerformancePoint, PerformanceCurve, CalibrationResult, and HardwareCalibrator classes.
"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add parent directory to path
test_dir = Path(__file__).parent
src_dir = test_dir.parent.parent
sys.path.insert(0, str(src_dir))

from pdcostmodel.config import GPUSpec, NetworkSpec, HardwareConfig
from pdcostmodel.calibration import (
    PerformancePoint,
    PerformanceCurve,
    CalibrationResult,
    HardwareCalibrator,
)


# ============================================================================
# PerformancePoint Tests
# ============================================================================


class TestPerformancePoint:
    """Test cases for PerformancePoint class."""

    def test_basic_creation(self):
        """Test basic PerformancePoint creation."""
        point = PerformancePoint(
            size=1024, tflops=500.0, efficiency=0.5, time_ms=100.0
        )
        assert point.size == 1024
        assert point.tflops == 500.0
        assert point.efficiency == 0.5
        assert point.time_ms == 100.0

    def test_custom_values(self):
        """Test PerformancePoint with custom values."""
        point = PerformancePoint(
            size=4096, tflops=800.0, efficiency=0.8, time_ms=50.0
        )
        assert point.size == 4096
        assert point.tflops == 800.0
        assert point.efficiency == 0.8
        assert point.time_ms == 50.0


# ============================================================================
# PerformanceCurve Tests
# ============================================================================


class TestPerformanceCurve:
    """Test cases for PerformanceCurve class."""

    @pytest.fixture
    def sample_points(self):
        """Create sample performance points."""
        return [
            PerformancePoint(size=256, tflops=100.0, efficiency=0.1, time_ms=10.0),
            PerformancePoint(size=512, tflops=200.0, efficiency=0.2, time_ms=20.0),
            PerformancePoint(size=1024, tflops=400.0, efficiency=0.4, time_ms=40.0),
            PerformancePoint(size=2048, tflops=600.0, efficiency=0.6, time_ms=60.0),
            PerformancePoint(size=4096, tflops=750.0, efficiency=0.75, time_ms=80.0),
        ]

    def test_basic_creation(self, sample_points):
        """Test basic PerformanceCurve creation."""
        curve = PerformanceCurve(
            dtype="bfloat16",
            points=sample_points,
            peak_tflops=1000.0,
        )
        assert curve.dtype == "bfloat16"
        assert len(curve.points) == 5
        assert curve.peak_tflops == 1000.0

    def test_predict_efficiency(self, sample_points):
        """Test predict_efficiency method."""
        curve = PerformanceCurve(
            dtype="bfloat16",
            points=sample_points,
            peak_tflops=1000.0,
            fit_a=0.05,
            fit_b=0.1,
            fit_max=0.85,
        )
        
        # Test prediction
        efficiency = curve.predict_efficiency(1024)
        assert 0 <= efficiency <= 1.0

    def test_predict_efficiency_zero_size(self, sample_points):
        """Test predict_efficiency with zero size."""
        curve = PerformanceCurve(
            dtype="bfloat16",
            points=sample_points,
            peak_tflops=1000.0,
        )
        
        efficiency = curve.predict_efficiency(0)
        assert efficiency == 0.0

    def test_predict_tflops(self, sample_points):
        """Test predict_tflops method."""
        curve = PerformanceCurve(
            dtype="bfloat16",
            points=sample_points,
            peak_tflops=1000.0,
            fit_a=0.05,
            fit_b=0.3,
            fit_max=0.85,
        )
        
        tflops = curve.predict_tflops(1024)
        assert tflops > 0
        assert tflops <= 1000.0  # Should not exceed peak

    def test_to_dict(self, sample_points):
        """Test to_dict method."""
        curve = PerformanceCurve(
            dtype="float16",
            points=sample_points,
            peak_tflops=500.0,
            fit_a=0.04,
            fit_b=0.2,
        )
        
        d = curve.to_dict()
        assert d["dtype"] == "float16"
        assert d["peak_tflops"] == 500.0
        assert "points" in d
        assert len(d["points"]) == 5

    def test_str_representation(self, sample_points):
        """Test __str__ method."""
        curve = PerformanceCurve(
            dtype="bfloat16",
            points=sample_points,
            peak_tflops=1000.0,
        )
        
        s = str(curve)
        assert "PerformanceCurve" in s
        assert "bfloat16" in s


# ============================================================================
# CalibrationResult Tests
# ============================================================================


class TestCalibrationResult:
    """Test cases for CalibrationResult class."""

    def test_default_values(self):
        """Test default CalibrationResult values."""
        result = CalibrationResult()
        assert result.gpu_name == "Unknown"
        assert result.gpu_memory_gb == 0.0
        assert result.gpu_count == 0
        assert result.calibrated is False

    def test_custom_values(self):
        """Test CalibrationResult with custom values."""
        result = CalibrationResult(
            gpu_name="H100",
            gpu_memory_gb=80.0,
            gpu_count=8,
            fp32_tflops=67.0,
            fp16_tflops=989.0,
            bf16_tflops=989.0,
            memory_bandwidth_gbps=3350.0,
            calibrated=True,
        )
        assert result.gpu_name == "H100"
        assert result.gpu_memory_gb == 80.0
        assert result.gpu_count == 8
        assert result.bf16_tflops == 989.0
        assert result.calibrated is True

    def test_get_efficiency_no_curve(self):
        """Test get_efficiency returns default when no curve."""
        result = CalibrationResult(calibrated=True)
        
        # Without curves, should return default estimate
        efficiency = result.get_efficiency(1024, "bfloat16")
        assert 0 <= efficiency <= 1.0

    def test_get_efficiency_with_curve(self):
        """Test get_efficiency with a curve."""
        points = [
            PerformancePoint(size=1024, tflops=500.0, efficiency=0.5, time_ms=50.0),
        ]
        curve = PerformanceCurve(
            dtype="bfloat16",
            points=points,
            peak_tflops=1000.0,
            fit_a=0.05,
            fit_b=0.3,
        )
        
        result = CalibrationResult(
            gpu_name="H100",
            bf16_curve=curve,
            calibrated=True,
        )
        
        efficiency = result.get_efficiency(1024, "bfloat16")
        assert 0 <= efficiency <= 1.0

    def test_to_dict(self):
        """Test to_dict method."""
        result = CalibrationResult(
            gpu_name="A100",
            gpu_memory_gb=80.0,
            gpu_count=8,
            fp32_tflops=19.5,
            bf16_tflops=312.0,
            calibrated=True,
        )
        d = result.to_dict()
        
        assert d["gpu_name"] == "A100"
        assert d["gpu_memory_gb"] == 80.0
        assert d["calibrated"] is True

    def test_str_representation_calibrated(self):
        """Test __str__ for calibrated result."""
        result = CalibrationResult(
            gpu_name="V100",
            gpu_memory_gb=32.0,
            gpu_count=4,
            fp32_tflops=15.7,
            calibrated=True,
        )
        s = str(result)
        
        assert "CalibrationResult" in s
        assert "V100" in s

    def test_str_representation_not_calibrated(self):
        """Test __str__ for not calibrated result."""
        result = CalibrationResult(
            calibrated=False,
            error_message="No GPU available",
        )
        s = str(result)
        
        assert "Not calibrated" in s


# ============================================================================
# HardwareCalibrator Tests
# ============================================================================


class TestHardwareCalibrator:
    """Test cases for HardwareCalibrator class."""

    def test_init_default(self):
        """Test HardwareCalibrator initialization with defaults."""
        calibrator = HardwareCalibrator()
        assert calibrator.device_id == 0
        assert calibrator.warmup_iters == 5
        assert calibrator.test_iters == 20

    def test_init_custom(self):
        """Test HardwareCalibrator initialization with custom values."""
        calibrator = HardwareCalibrator(
            device_id=1, warmup_iters=10, test_iters=50
        )
        assert calibrator.device_id == 1
        assert calibrator.warmup_iters == 10
        assert calibrator.test_iters == 50

    def test_result_property_initial(self):
        """Test result property is None initially."""
        calibrator = HardwareCalibrator()
        assert calibrator.result is None

    def test_detect_gpu_info(self):
        """Test detect_gpu_info method."""
        calibrator = HardwareCalibrator()
        
        gpu_name, memory_gb, gpu_count = calibrator.detect_gpu_info()
        
        # Should return some values (may be Unknown if no GPU)
        assert isinstance(gpu_name, str)
        assert isinstance(memory_gb, float)
        assert isinstance(gpu_count, int)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])