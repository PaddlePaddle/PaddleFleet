#!/usr/bin/env python3
"""
硬件校准结果管理器

功能:
1. 保存校准结果到本地 JSON 文件
2. 加载已保存的校准结果
3. 自动检测当前硬件并匹配已有校准
4. 支持多种硬件配置的管理

文件存储位置: pdcost/profiles/
文件命名规则: {gpu_name}_{gpu_count}gpu_{node_count}node.json
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from .config import HardwareConfig, GPUSpec, NetworkSpec


# 获取 profiles 目录路径
PROFILES_DIR = Path(__file__).parent / "profiles"


def get_profile_filename(gpu_name: str, gpu_count: int, node_count: int = 1) -> str:
    """
    生成校准文件名
    
    Args:
        gpu_name: GPU 名称 (如 "NVIDIA H800")
        gpu_count: GPU 数量
        node_count: 节点数量
    
    Returns:
        文件名 (如 "H800_8gpu_1node.json")
    """
    # 清理 GPU 名称，移除空格和特殊字符
    clean_name = gpu_name.replace("NVIDIA ", "").replace(" ", "_").replace("-", "_")
    # 移除多余的下划线
    while "__" in clean_name:
        clean_name = clean_name.replace("__", "_")
    clean_name = clean_name.strip("_")
    
    return f"{clean_name}_{gpu_count}gpu_{node_count}node.json"


def get_profile_path(gpu_name: str, gpu_count: int, node_count: int = 1) -> Path:
    """获取校准文件完整路径"""
    filename = get_profile_filename(gpu_name, gpu_count, node_count)
    return PROFILES_DIR / filename


class ProfileManager:
    """
    硬件校准结果管理器
    
    使用示例:
        manager = ProfileManager()
        
        # 检查是否有已保存的校准
        if manager.has_profile("H800", 8, 1):
            config = manager.load_profile("H800", 8, 1)
        else:
            # 执行校准并保存
            calibrator = HardwareCalibrator()
            result = calibrator.calibrate()
            manager.save_calibration(result, node_count=1)
    """
    
    def __init__(self, profiles_dir: Path = None):
        """
        初始化管理器
        
        Args:
            profiles_dir: 校准文件存储目录 (默认使用 pdcost/profiles/)
        """
        self.profiles_dir = profiles_dir or PROFILES_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
    
    def list_profiles(self) -> List[Dict]:
        """
        列出所有已保存的校准配置
        
        Returns:
            配置列表，每个元素包含 gpu_name, gpu_count, node_count, filepath, calibrated_at
        """
        profiles = []
        
        for filepath in self.profiles_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                profiles.append({
                    "gpu_name": data.get("gpu_name", "Unknown"),
                    "gpu_count": data.get("gpu_count", 0),
                    "node_count": data.get("node_count", 1),
                    "filepath": str(filepath),
                    "filename": filepath.name,
                    "calibrated_at": data.get("calibrated_at", "Unknown"),
                    "bf16_tflops": data.get("bf16_tflops", 0),
                    "memory_gb": data.get("gpu_memory_gb", 0),
                })
            except Exception:
                continue
        
        return profiles
    
    def has_profile(self, gpu_name: str, gpu_count: int, node_count: int = 1) -> bool:
        """
        检查是否存在指定硬件的校准文件
        
        Args:
            gpu_name: GPU 名称
            gpu_count: GPU 数量
            node_count: 节点数量
        
        Returns:
            是否存在
        """
        profile_path = get_profile_path(gpu_name, gpu_count, node_count)
        return profile_path.exists()
    
    def find_matching_profile(self, gpu_name: str = None, gpu_count: int = None, 
                              node_count: int = None) -> Optional[Dict]:
        """
        查找匹配的校准配置
        
        如果提供了 gpu_name，会尝试模糊匹配
        
        Args:
            gpu_name: GPU 名称 (可选，模糊匹配)
            gpu_count: GPU 数量 (可选)
            node_count: 节点数量 (可选)
        
        Returns:
            匹配的配置，如果没找到返回 None
        """
        profiles = self.list_profiles()
        
        for profile in profiles:
            # GPU 名称匹配 (模糊匹配)
            if gpu_name:
                profile_gpu = profile["gpu_name"].upper()
                search_gpu = gpu_name.upper()
                # 提取关键字进行匹配 (如 H800, A100, V100 等)
                keywords = ["H800", "H100", "A100", "A800", "V100", "A10", "L40", "4090", "3090"]
                matched = False
                for kw in keywords:
                    if kw in profile_gpu and kw in search_gpu:
                        matched = True
                        break
                if not matched and search_gpu not in profile_gpu:
                    continue
            
            # GPU 数量匹配
            if gpu_count is not None and profile["gpu_count"] != gpu_count:
                continue
            
            # 节点数量匹配
            if node_count is not None and profile["node_count"] != node_count:
                continue
            
            return profile
        
        return None
    
    def load_profile(self, gpu_name: str, gpu_count: int, node_count: int = 1) -> Optional[Dict]:
        """
        加载校准配置
        
        Args:
            gpu_name: GPU 名称
            gpu_count: GPU 数量
            node_count: 节点数量
        
        Returns:
            校准数据字典，如果不存在返回 None
        """
        profile_path = get_profile_path(gpu_name, gpu_count, node_count)
        
        if not profile_path.exists():
            return None
        
        try:
            with open(profile_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load profile {profile_path}: {e}")
            return None
    
    def load_profile_by_path(self, filepath: str) -> Optional[Dict]:
        """根据文件路径加载校准配置"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load profile {filepath}: {e}")
            return None
    
    def save_calibration(self, calibration_result, node_count: int = 1, 
                         extra_info: Dict = None) -> str:
        """
        保存校准结果到文件
        
        Args:
            calibration_result: CalibrationResult 对象
            node_count: 节点数量
            extra_info: 额外信息 (如测试环境描述)
        
        Returns:
            保存的文件路径
        """
        # 准备保存数据
        data = calibration_result.to_dict()
        data["node_count"] = node_count
        data["calibrated_at"] = datetime.now().isoformat()
        data["pdcost_version"] = "1.0.0"
        
        if extra_info:
            data["extra_info"] = extra_info
        
        # 生成文件路径
        profile_path = get_profile_path(
            calibration_result.gpu_name,
            calibration_result.gpu_count,
            node_count
        )
        
        # 保存
        with open(profile_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(profile_path)
    
    def create_hardware_config(self, profile_data: Dict) -> HardwareConfig:
        """
        从校准数据创建 HardwareConfig
        
        Args:
            profile_data: 校准数据字典
        
        Returns:
            HardwareConfig 对象
        """
        gpu = GPUSpec(
            name=profile_data.get("gpu_name", "Unknown"),
            memory_gb=profile_data.get("gpu_memory_gb", 80.0),
            fp32_tflops=profile_data.get("fp32_tflops", 60.0),
            fp16_tflops=profile_data.get("fp16_tflops", 900.0),
            bf16_tflops=profile_data.get("bf16_tflops", 900.0),
            memory_bandwidth_gbps=profile_data.get("memory_bandwidth_gbps", 2000.0),
        )
        
        node_count = profile_data.get("node_count", 1)
        gpu_count = profile_data.get("gpu_count", 8)
        gpus_per_node = gpu_count // node_count if node_count > 0 else gpu_count
        
        # 确保网络带宽有合理的默认值
        intra_bw = profile_data.get("intra_node_bandwidth_gbps", 0)
        inter_bw = profile_data.get("inter_node_bandwidth_gbps", 0)
        
        # 如果节点内带宽未设置或为0，使用 NVLink 默认带宽
        if intra_bw <= 0:
            intra_bw = 900.0  # H800/H100 NVLink default
        if inter_bw <= 0:
            inter_bw = 200.0  # IB default
        
        network = NetworkSpec(
            intra_node_bandwidth_gbps=intra_bw,
            inter_node_bandwidth_gbps=inter_bw,
        )
        
        return HardwareConfig(
            gpu=gpu,
            network=network,
            num_nodes=node_count,
            gpus_per_node=gpus_per_node,
        )
    
    def delete_profile(self, gpu_name: str, gpu_count: int, node_count: int = 1) -> bool:
        """
        删除校准配置
        
        Returns:
            是否成功删除
        """
        profile_path = get_profile_path(gpu_name, gpu_count, node_count)
        
        if profile_path.exists():
            profile_path.unlink()
            return True
        return False


def _is_valid_precise_calibration(profile_data: Dict) -> bool:
    """
    验证校准配置是否为有效的精确校准
    
    精确校准需要满足:
    1. 有 bf16_curve 字段
    2. bf16_curve 中的 fit_a, fit_b 不能都为 0
    3. points 数量 >= 5
    4. memory_bandwidth_gbps > 0
    """
    if not profile_data:
        return False
    
    bf16_curve = profile_data.get("bf16_curve")
    if not bf16_curve:
        return False
    
    # 检查拟合参数是否有效
    fit_a = bf16_curve.get("fit_a", 0)
    fit_b = bf16_curve.get("fit_b", 0)
    if fit_a == 0 and fit_b == 0:
        return False
    
    # 检查测试点数量
    points = bf16_curve.get("points", [])
    if len(points) < 5:
        return False
    
    # 检查显存带宽
    memory_bw = profile_data.get("memory_bandwidth_gbps", 0)
    if memory_bw <= 0:
        return False
    
    return True


def auto_calibrate_or_load(node_count: int = 1, 
                           force_calibrate: bool = False,
                           precise: bool = False,
                           require_precise: bool = False,
                           verbose: bool = True) -> Tuple[HardwareConfig, bool]:
    """
    自动校准或加载已有配置
    
    1. 检测当前硬件信息
    2. 检查是否有匹配的已保存校准
    3. 如果有，加载并返回
    4. 如果没有，执行校准并保存
    
    Args:
        node_count: 节点数量
        force_calibrate: 强制重新校准
        precise: 是否使用精确校准（多尺寸测试，耗时更长但更准确）
        require_precise: 是否必须使用精确校准（如果已有校准不是精确校准，则重新校准）
        verbose: 是否打印信息
    
    Returns:
        (HardwareConfig, is_from_cache) - 硬件配置和是否来自缓存
    """
    from .calibration import HardwareCalibrator
    
    manager = ProfileManager()
    calibrator = HardwareCalibrator()
    
    # 检测当前 GPU 信息
    gpu_name, memory_gb, gpu_count = calibrator.detect_gpu_info()
    
    if verbose:
        print(f"🔍 检测到硬件: {gpu_name} × {gpu_count} ({memory_gb:.1f} GB)")
    
    # 检查是否有已保存的校准
    if not force_calibrate and manager.has_profile(gpu_name, gpu_count, node_count):
        if verbose:
            print(f"✅ 找到已保存的校准配置，正在加载...")
        
        profile_data = manager.load_profile(gpu_name, gpu_count, node_count)
        if profile_data:
            # 如果要求精确校准，验证校准配置是否有效
            is_precise = _is_valid_precise_calibration(profile_data)
            
            if require_precise and not is_precise:
                if verbose:
                    print(f"⚠️ 已有校准不满足精确校准要求，将重新进行精确校准...")
                # 继续执行下面的校准逻辑
            else:
                config = manager.create_hardware_config(profile_data)
                if verbose:
                    calibrated_at = profile_data.get("calibrated_at", "Unknown")
                    print(f"   校准时间: {calibrated_at}")
                    print(f"   BF16 峰值: {profile_data.get('bf16_tflops', 0):.1f} TFLOPS")
                    if is_precise:
                        bf16_curve = profile_data.get("bf16_curve", {})
                        print(f"   校准类型: 精确校准 (含多尺寸性能曲线)")
                        print(f"      • 测试点数: {len(bf16_curve.get('points', []))}")
                        print(f"      • 显存带宽: {profile_data.get('memory_bandwidth_gbps', 0):.1f} GB/s")
                    else:
                        print(f"   校准类型: 快速校准")
                return config, True
    
    # 需要重新校准
    if verbose:
        if force_calibrate:
            print("🔧 强制重新校准...")
        else:
            print("🔧 未找到校准配置，开始校准...")
        if precise:
            print("📊 使用精确校准模式 (多尺寸测试)")
    
    result = calibrator.calibrate(
        multi_size_test=precise,
        verbose=verbose
    )
    
    # 保存校准结果
    filepath = manager.save_calibration(result, node_count)
    if verbose:
        print(f"\n💾 校准结果已保存: {filepath}")
    
    # 创建配置
    config = calibrator.create_hardware_config(
        num_nodes=node_count,
        gpus_per_node=gpu_count // node_count if node_count > 0 else gpu_count
    )
    
    return config, False


def precise_calibrate(node_count: int = 1, 
                      force: bool = True,
                      verbose: bool = True) -> Tuple[HardwareConfig, Dict]:
    """
    执行精确校准
    
    精确校准包括:
    1. GPU 信息检测
    2. FP32/FP16/BF16 峰值算力测试
    3. 显存带宽测试
    4. 多尺寸 GEMM 性能曲线测试 (关键)
    5. 性能曲线拟合
    
    Args:
        node_count: 节点数量
        force: 强制重新校准 (即使已有缓存)
        verbose: 是否打印详细信息
    
    Returns:
        (HardwareConfig, calibration_data) - 硬件配置和完整校准数据
    """
    from .calibration import HardwareCalibrator
    
    manager = ProfileManager()
    calibrator = HardwareCalibrator()
    
    # 检测硬件
    gpu_name, memory_gb, gpu_count = calibrator.detect_gpu_info()
    
    if verbose:
        print("=" * 70)
        print("🔬 pdcost 精确校准")
        print("=" * 70)
        print(f"\n检测到硬件: {gpu_name} × {gpu_count} ({memory_gb:.1f} GB)")
    
    # 执行精确校准 (包含多尺寸测试)
    result = calibrator.calibrate(
        test_compute=True,
        test_memory=True,
        gemm_size=8192,
        multi_size_test=True,  # 关键: 多尺寸测试
        test_sizes=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384],
        verbose=verbose
    )
    
    # 保存校准结果
    filepath = manager.save_calibration(result, node_count)
    if verbose:
        print(f"\n💾 精确校准结果已保存: {filepath}")
    
    # 创建配置
    config = calibrator.create_hardware_config(
        num_nodes=node_count,
        gpus_per_node=gpu_count // node_count if node_count > 0 else gpu_count
    )
    
    # 返回完整校准数据
    calibration_data = result.to_dict()
    
    if verbose:
        print("\n" + "=" * 70)
        print("📊 精确校准结果摘要")
        print("=" * 70)
        print(f"  GPU: {result.gpu_name} × {result.gpu_count}")
        print(f"  显存: {result.gpu_memory_gb:.1f} GB")
        print(f"  FP32 峰值: {result.fp32_tflops:.1f} TFLOPS")
        print(f"  FP16 峰值: {result.fp16_tflops:.1f} TFLOPS")
        print(f"  BF16 峰值: {result.bf16_tflops:.1f} TFLOPS")
        print(f"  显存带宽: {result.memory_bandwidth_gbps:.1f} GB/s")
        
        if result.bf16_curve:
            print(f"\n📈 BF16 性能曲线:")
            print(f"  拟合公式: efficiency = {result.bf16_curve.fit_a:.4f} * log(size) + {result.bf16_curve.fit_b:.4f}")
            print(f"  峰值效率: {result.bf16_curve.fit_max:.1%}")
            print(f"  测试点数: {len(result.bf16_curve.points)}")
        
        print("=" * 70)
    
    return config, calibration_data


def list_saved_profiles(verbose: bool = True) -> List[Dict]:
    """
    列出所有已保存的校准配置
    
    Returns:
        配置列表
    """
    manager = ProfileManager()
    profiles = manager.list_profiles()
    
    if verbose:
        print("=" * 70)
        print("📋 已保存的硬件校准配置")
        print("=" * 70)
        
        if not profiles:
            print("  (无)")
        else:
            print(f"{'GPU':<20} {'数量':<8} {'节点':<6} {'BF16 TFLOPS':<12} {'校准时间':<20}")
            print("-" * 70)
            for p in profiles:
                print(f"{p['gpu_name']:<20} {p['gpu_count']:<8} {p['node_count']:<6} "
                      f"{p['bf16_tflops']:<12.1f} {p['calibrated_at'][:19]:<20}")
        print("=" * 70)
    
    return profiles