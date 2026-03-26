#!/usr/bin/env python3
"""
pdcostmodel - 分布式训练代价模型

接口:
- ModelConfig: 模型配置，from_json() / from_name()
- ParallelConfig: 并行配置 (tp, pp, dp, ep, sharding)
- PDCostModel: 代价模型，predict()
- get_hardware_config: 获取/校准硬件配置
"""

from .config import ModelConfig, ParallelConfig
from .costmodel import PDCostModel
from .calibration import get_hardware_config
try:
    from .entrypoints import (
        load_model_from_real_config,
        predict_from_real_config,
    )
except ImportError:
    load_model_from_real_config = None
    predict_from_real_config = None
from .validation import (
    ParallelValidationIssue,
    ParallelValidationResult,
    validate_parallel_config,
)

__version__ = "0.1.0"
__all__ = [
    "ModelConfig",
    "ParallelConfig",
    "PDCostModel",
    "get_hardware_config",
    "load_model_from_real_config",
    "predict_from_real_config",
    "ParallelValidationIssue",
    "ParallelValidationResult",
    "validate_parallel_config",
]
