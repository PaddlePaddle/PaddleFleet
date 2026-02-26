#!/usr/bin/env python3
"""
pdcostmodel - PaddleFormers 分布式训练代价模型

用于预测不同并行配置下的:
- Step 时间 (训练迭代耗时)
- 显存占用 (每卡峰值显存)

支持的并行策略:
- Tensor Parallel (TP)
- Pipeline Parallel (PP)  
- Data Parallel (DP)
- Sharding (ZeRO Stage 1/2/3)
- Expert Parallel (EP) - MoE 模型
- Sequence Parallel (SP)
- Context Parallel (CP)

使用方式:
    from pdcostmodel import PDCostModel, ModelConfig, ParallelConfig
    
    # 创建模型配置 (以 Qwen3-30B-A3B 为例)
    model_config = ModelConfig.from_name("qwen3-30b-a3b")
    
    # 创建代价模型
    costmodel = PDCostModel(model_config)
    
    # 预测不同并行配置
    parallel = ParallelConfig(tp=8, pp=1, dp=1, ep=8, sharding="stage1")
    result = costmodel.predict(parallel, micro_batch_size=1, seq_len=8192)
    
    print(f"Step Time: {result.step_time_ms:.2f} ms")
    print(f"Memory: {result.memory_gb:.2f} GB")

快速搜索最优配置:
    from pdcostmodel import grid_search
    
    # 一行代码搜索最优配置
    results = grid_search("qwen3-30b-a3b", total_gpus=8)
    
    # 获取最优配置
    best = results.best
    print(f"最优配置: {best.config_str}")
    print(f"吞吐量: {best.tokens_per_second_per_gpu:.0f} tok/s/GPU")
"""

from .config import (
    ModelConfig,
    TrainingConfig,
    ParallelConfig,
    HardwareConfig,
    GPUSpec,
    NetworkSpec,
)
from .memory_model import (
    MemoryModel,
    MemoryBreakdown,
    RecomputeConfig,
    ShardingConfig,
)
from .compute_model import ComputeModel
from .comm_model import CommModel
from .costmodel import PDCostModel, PredictionResult
from .calibration import (
    HardwareCalibrator,
    CalibrationResult,
    quick_calibrate,
    create_calibrated_hardware_config,
)
from .profile_manager import (
    ProfileManager,
    auto_calibrate_or_load,
    precise_calibrate,
    list_saved_profiles,
    get_profile_path,
    PROFILES_DIR,
)
from .gridsearch import (
    GridSearcher,
    GridSearchResult,
    SearchResult,
    grid_search,
)

__version__ = "0.1.0"
__all__ = [
    # 配置类
    "ModelConfig",
    "TrainingConfig", 
    "ParallelConfig",
    "HardwareConfig",
    "GPUSpec",
    "NetworkSpec",
    # 子模型
    "MemoryModel",
    "MemoryBreakdown",
    "RecomputeConfig",
    "ShardingConfig",
    "ComputeModel",
    "CommModel",
    # 主模型
    "PDCostModel",
    "PredictionResult",
    # 校准
    "HardwareCalibrator",
    "CalibrationResult",
    "quick_calibrate",
    "create_calibrated_hardware_config",
    # 校准配置管理
    "ProfileManager",
    "auto_calibrate_or_load",
    "precise_calibrate",
    "list_saved_profiles",
    "get_profile_path",
    "PROFILES_DIR",
    # Grid Search
    "GridSearcher",
    "GridSearchResult",
    "SearchResult",
    "grid_search",
]