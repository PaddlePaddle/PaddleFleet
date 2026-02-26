#!/usr/bin/env python3
"""
PDCost Grid Search - 并行配置搜索模块

提供自动搜索最优并行配置的功能，支持：
- 自动生成合法的并行配置搜索空间
- 使用 PDCost 模型预测每个配置的性能
- 按吞吐量/时延排序，找出最优配置
- 生成训练配置 YAML 文件

使用示例:
    from pdcostmodel import grid_search, ModelConfig
    
    # 快速搜索
    results = grid_search("qwen3-30b-a3b", total_gpus=8)
    
    # 或使用完整接口
    from pdcostmodel import GridSearcher
    searcher = GridSearcher(model_name="qwen3-30b-a3b", total_gpus=8)
    results = searcher.search()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import json
import os

from .config import ModelConfig, ParallelConfig, HardwareConfig, TrainingConfig
from .costmodel import PDCostModel, PredictionResult
from .profile_manager import auto_calibrate_or_load


@dataclass
class SearchResult:
    """单个配置的搜索结果"""
    rank: int = 0
    config: Dict = field(default_factory=dict)
    config_str: str = ""
    
    # 性能预测
    step_time_ms: float = 0.0
    memory_gb: float = 0.0
    fits_memory: bool = True
    mfu: float = 0.0
    tokens_per_second: float = 0.0
    tokens_per_second_per_gpu: float = 0.0
    
    # 训练参数
    micro_batch_size: int = 1
    seq_len: int = 8192
    gradient_accumulation_steps: int = 16
    global_batch_size: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "config": self.config,
            "config_str": self.config_str,
            "step_time_ms": round(self.step_time_ms, 2),
            "memory_gb": round(self.memory_gb, 2),
            "fits_memory": self.fits_memory,
            "mfu": round(self.mfu, 4),
            "tokens_per_second": round(self.tokens_per_second, 0),
            "tokens_per_second_per_gpu": round(self.tokens_per_second_per_gpu, 0),
            "micro_batch_size": self.micro_batch_size,
            "seq_len": self.seq_len,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_batch_size": self.global_batch_size,
        }


@dataclass
class GridSearchResult:
    """Grid Search 完整结果"""
    model_name: str = ""
    total_gpus: int = 8
    total_configs: int = 0
    valid_configs: int = 0
    results: List[SearchResult] = field(default_factory=list)
    best: Optional[SearchResult] = None
    
    # 搜索配置
    search_space: Dict = field(default_factory=dict)
    training_params: Dict = field(default_factory=dict)
    hardware_info: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "total_gpus": self.total_gpus,
            "total_configs": self.total_configs,
            "valid_configs": self.valid_configs,
            "results": [r.to_dict() for r in self.results],
            "best": self.best.to_dict() if self.best else None,
            "search_space": self.search_space,
            "training_params": self.training_params,
            "hardware_info": self.hardware_info,
        }
    
    def save_json(self, path: str):
        """保存结果到 JSON 文件"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def print_report(self, top_k: int = 10):
        """打印搜索报告"""
        print("\n" + "=" * 110)
        print(f"🚀 PDCost 配置搜索报告 - {self.model_name}")
        print("=" * 110)
        print(
            f"{'排名':<4} {'配置':<35} {'step时延(ms)':<14} {'显存(GB)':<10} "
            f"{'约束':<6} {'MFU':<8} {'tok/s/GPU':<12}"
        )
        print("-" * 110)
        
        for r in self.results[:top_k]:
            fits = "✅" if r.fits_memory else "❌"
            print(
                f"{r.rank:<4} {r.config_str:<35} "
                f"{r.step_time_ms:<14.2f} {r.memory_gb:<10.2f} "
                f"{fits:<6} {r.mfu:<8.1%} "
                f"{r.tokens_per_second_per_gpu:<12,.0f}"
            )
        
        print("-" * 110)
        
        if self.best:
            cfg = self.best.config
            print(f"\n📊 最优配置详情")
            print("-" * 50)
            print(f"   并行策略:")
            print(f"      • TP (Tensor Parallel): {cfg.get('tp', 1)}")
            print(f"      • PP (Pipeline Parallel): {cfg.get('pp', 1)}")
            print(f"      • DP (Data Parallel): {cfg.get('dp', 1)}")
            print(f"      • EP (Expert Parallel): {cfg.get('ep', 1)}")
            print(f"      • Sharding: {cfg.get('sharding', 'stage1')}")
            print(f"   训练参数:")
            print(f"      • per_device_train_batch_size: {self.best.micro_batch_size}")
            print(f"      • gradient_accumulation_steps: {self.best.gradient_accumulation_steps}")
            print(f"      • global_batch_size: {self.best.global_batch_size}")
            print(f"      • max_seq_length: {self.best.seq_len}")
            print(f"   性能预测:")
            print(f"      • step 时延: {self.best.step_time_ms:,.2f} ms")
            print(f"      • 显存占用: {self.best.memory_gb:.2f} GB")
            print(f"      • MFU: {self.best.mfu:.1%}")
            print(f"      • 吞吐量: {self.best.tokens_per_second_per_gpu:,.0f} tok/s/GPU")


class GridSearcher:
    """
    并行配置搜索器
    
    自动搜索最优的分布式训练并行配置。
    
    使用示例:
        # 方式 1: 使用模型名称
        searcher = GridSearcher(model_name="qwen3-30b-a3b", total_gpus=8)
        results = searcher.search()
        results.print_report()
        
        # 方式 2: 使用自定义模型配置
        model_config = ModelConfig(...)
        searcher = GridSearcher(model_config=model_config, total_gpus=8)
        results = searcher.search()
        
        # 方式 3: 自定义搜索空间
        searcher = GridSearcher(
            model_name="qwen3-30b-a3b",
            total_gpus=8,
            tp_candidates=[1, 2, 4, 8],
            pp_candidates=[1, 2, 4],
            ep_candidates=[1, 2, 4, 8],
        )
        results = searcher.search(seq_len=4096, micro_batch_size=2)
    """
    
    def __init__(
        self,
        model_name: str = None,
        model_config: ModelConfig = None,
        hardware_config: HardwareConfig = None,
        total_gpus: int = 8,
        node_count: int = 1,
        tp_candidates: List[int] = None,
        pp_candidates: List[int] = None,
        ep_candidates: List[int] = None,
        sharding_candidates: List[str] = None,
        auto_calibrate: bool = True,
        require_precise: bool = True,
        verbose: bool = True,
    ):
        """
        初始化搜索器
        
        Args:
            model_name: 模型名称 (如 "qwen3-30b-a3b", "llama3-70b")
            model_config: 自定义模型配置 (优先于 model_name)
            hardware_config: 硬件配置 (默认自动校准)
            total_gpus: 总 GPU 数
            node_count: 节点数
            tp_candidates: TP 候选值
            pp_candidates: PP 候选值
            ep_candidates: EP 候选值
            sharding_candidates: Sharding 候选值
            auto_calibrate: 是否自动校准硬件
            require_precise: 是否要求精确校准
            verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.total_gpus = total_gpus
        self.node_count = node_count
        
        # 模型配置
        if model_config is not None:
            self.model_config = model_config
            self.model_name = getattr(model_config, 'model_type', None) or "custom"
        elif model_name is not None:
            self.model_config = ModelConfig.from_name(model_name)
            self.model_name = model_name
        else:
            raise ValueError("必须提供 model_name 或 model_config")
        
        # 硬件配置
        if hardware_config is not None:
            self.hardware_config = hardware_config
        elif auto_calibrate:
            if self.verbose:
                print("📡 正在校准硬件...")
            self.hardware_config, _ = auto_calibrate_or_load(
                node_count=node_count,
                require_precise=require_precise,
                verbose=verbose,
            )
        else:
            self.hardware_config = HardwareConfig()
        
        # 搜索空间配置
        self.tp_candidates = tp_candidates or [1, 2, 4, 8]
        self.pp_candidates = pp_candidates or [1, 2, 4]
        self.sharding_candidates = sharding_candidates or ["stage1", "stage2"]
        
        # EP 候选值
        if ep_candidates is not None:
            self.ep_candidates = ep_candidates
        elif self.model_config.num_experts > 1:
            self.ep_candidates = [1, 2, 4, 8]
        else:
            self.ep_candidates = [1]
        
        # 创建 PDCostModel
        self.costmodel = PDCostModel(
            self.model_config,
            self.hardware_config,
        )
    
    def generate_search_space(self) -> List[Dict]:
        """
        生成合法的并行配置搜索空间
        
        约束条件:
        - TP × PP × DP = total_gpus
        - PP 必须能整除层数
        - EP ≤ DP (EP 是 DP 的子分组)
        - EP ≤ 专家数
        
        Returns:
            合法的配置列表
        """
        configs = []
        num_layers = self.model_config.num_hidden_layers
        num_experts = self.model_config.num_experts
        
        for tp in self.tp_candidates:
            if tp > self.total_gpus:
                continue
            
            for pp in self.pp_candidates:
                if tp * pp > self.total_gpus:
                    continue
                
                # PP 必须能整除层数
                if num_layers % pp != 0:
                    continue
                
                # 计算 DP
                dp = self.total_gpus // (tp * pp)
                if dp < 1:
                    continue
                
                # 验证总 GPU 数
                if tp * pp * dp != self.total_gpus:
                    continue
                
                for ep in self.ep_candidates:
                    # EP 不能超过专家数
                    if num_experts > 1 and ep > num_experts:
                        continue
                    
                    # EP 不能超过 DP
                    if ep > dp:
                        continue
                    
                    for sharding in self.sharding_candidates:
                        configs.append({
                            "tp": tp,
                            "pp": pp,
                            "dp": dp,
                            "ep": ep,
                            "sharding": sharding,
                        })
        
        return configs
    
    def search(
        self,
        micro_batch_size: int = 1,
        seq_len: int = 8192,
        gradient_accumulation_steps: int = 16,
        recompute_granularity: str = "full",
        sort_by: str = "throughput",
        top_k: int = 10,
        print_report: bool = True,
    ) -> GridSearchResult:
        """
        执行配置搜索
        
        Args:
            micro_batch_size: 每设备批次大小
            seq_len: 序列长度
            gradient_accumulation_steps: 梯度累积步数
            recompute_granularity: 重计算粒度 ("none", "selective", "full")
            sort_by: 排序方式 ("throughput" 或 "step_time")
            top_k: 显示前 k 个配置
            print_report: 是否打印报告
        
        Returns:
            GridSearchResult: 搜索结果
        """
        if self.verbose:
            print(f"\n🔍 搜索配置空间...")
        
        # 生成搜索空间
        configs = self.generate_search_space()
        
        if self.verbose:
            print(f"   生成了 {len(configs)} 个候选配置")
        
        # 评估每个配置
        results = []
        for config in configs:
            try:
                parallel = ParallelConfig(
                    tp=config["tp"],
                    pp=config["pp"],
                    dp=config["dp"],
                    ep=config.get("ep", 1),
                    sharding=config.get("sharding", "stage1"),
                )
                
                pred = self.costmodel.predict(
                    parallel,
                    micro_batch_size=micro_batch_size,
                    seq_len=seq_len,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    recompute_granularity=recompute_granularity,
                )
                
                global_bs = micro_batch_size * config["dp"] * gradient_accumulation_steps
                
                results.append(SearchResult(
                    config=config,
                    config_str=str(parallel),
                    step_time_ms=pred.step_time_ms,
                    memory_gb=pred.memory_gb,
                    fits_memory=pred.fits_memory,
                    mfu=pred.mfu,
                    tokens_per_second=pred.tokens_per_second,
                    tokens_per_second_per_gpu=pred.tokens_per_second_per_gpu,
                    micro_batch_size=micro_batch_size,
                    seq_len=seq_len,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    global_batch_size=global_bs,
                ))
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 配置评估失败: {config}, 错误: {e}")
        
        # 过滤满足显存约束的配置
        valid_results = [r for r in results if r.fits_memory]
        
        # 排序
        if sort_by == "throughput":
            valid_results.sort(key=lambda x: x.tokens_per_second_per_gpu, reverse=True)
        else:
            valid_results.sort(key=lambda x: x.step_time_ms)
        
        # 更新排名
        for i, r in enumerate(valid_results):
            r.rank = i + 1
        
        # 构建结果
        search_result = GridSearchResult(
            model_name=self.model_name,
            total_gpus=self.total_gpus,
            total_configs=len(configs),
            valid_configs=len(valid_results),
            results=valid_results[:top_k],
            best=valid_results[0] if valid_results else None,
            search_space={
                "tp_candidates": self.tp_candidates,
                "pp_candidates": self.pp_candidates,
                "ep_candidates": self.ep_candidates,
                "sharding_candidates": self.sharding_candidates,
            },
            training_params={
                "micro_batch_size": micro_batch_size,
                "seq_len": seq_len,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "recompute_granularity": recompute_granularity,
            },
            hardware_info={
                "gpu_name": self.hardware_config.gpu.name,
                "gpu_memory_gb": self.hardware_config.gpu.memory_gb,
                "bf16_tflops": self.hardware_config.gpu.bf16_tflops,
                "num_nodes": self.hardware_config.num_nodes,
                "gpus_per_node": self.hardware_config.gpus_per_node,
            },
        )
        
        if self.verbose:
            print(f"   评估完成: {len(results)} 个配置, {len(valid_results)} 个满足显存约束")
        
        if print_report:
            search_result.print_report(top_k)
        
        return search_result
    
    def generate_yaml_config(self, result: SearchResult = None, output_path: str = None) -> str:
        """
        生成训练配置 YAML 文件
        
        Args:
            result: 搜索结果 (默认使用最优配置)
            output_path: 输出路径
        
        Returns:
            YAML 配置字符串
        """
        if result is None:
            search_result = self.search(print_report=False)
            if search_result.best is None:
                raise ValueError("没有找到满足显存约束的配置")
            result = search_result.best
        
        cfg = result.config
        
        yaml_content = f'''## {self.model_name} 自动生成配置 ##
## PDCost 预测吞吐量: {result.tokens_per_second_per_gpu:.0f} tok/s/GPU ##

# 数据配置
train_dataset_type: erniekit
eval_dataset_type: erniekit
max_seq_len: {result.seq_len}

# 模型配置
model_name_or_path: {self.model_name}
_attn_implementation: flashmask

# 训练配置
do_train: true
per_device_train_batch_size: {result.micro_batch_size}
gradient_accumulation_steps: {result.gradient_accumulation_steps}
# global_batch_size: {result.global_batch_size}

# 并行配置
tensor_model_parallel_size: {cfg['tp']}
sequence_parallel: {'true' if cfg['tp'] > 1 else 'false'}
pipeline_model_parallel_size: {cfg['pp']}
use_expert_parallel: {'true' if cfg.get('ep', 1) > 1 else 'false'}
expert_model_parallel_size: {cfg.get('ep', 1)}

# 重计算配置
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1

# Sharding 配置
sharding: {cfg.get('sharding', 'stage1')}
split_param: true

# 优化器配置
optim: adamw
bf16: true
tensorwise_offload_optimizer: true
'''
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(yaml_content)
            print(f"\n✅ 配置已保存到: {output_path}")
        
        return yaml_content


def grid_search(
    model: Union[str, ModelConfig],
    total_gpus: int = 8,
    micro_batch_size: int = 1,
    seq_len: int = 8192,
    gradient_accumulation_steps: int = 16,
    tp_candidates: List[int] = None,
    pp_candidates: List[int] = None,
    ep_candidates: List[int] = None,
    sharding_candidates: List[str] = None,
    sort_by: str = "throughput",
    top_k: int = 10,
    print_report: bool = True,
    verbose: bool = True,
) -> GridSearchResult:
    """
    快速执行并行配置搜索 (便捷函数)
    
    Args:
        model: 模型名称或 ModelConfig 对象
        total_gpus: 总 GPU 数
        micro_batch_size: 每设备批次大小
        seq_len: 序列长度
        gradient_accumulation_steps: 梯度累积步数
        tp_candidates: TP 候选值 (默认 [1, 2, 4, 8])
        pp_candidates: PP 候选值 (默认 [1, 2, 4])
        ep_candidates: EP 候选值 (默认根据模型决定)
        sharding_candidates: Sharding 候选值 (默认 ["stage1", "stage2"])
        sort_by: 排序方式 ("throughput" 或 "step_time")
        top_k: 显示前 k 个配置
        print_report: 是否打印报告
        verbose: 是否打印详细信息
    
    Returns:
        GridSearchResult: 搜索结果
    
    示例:
        # 快速搜索
        results = grid_search("qwen3-30b-a3b", total_gpus=8)
        
        # 自定义搜索空间
        results = grid_search(
            "llama3-70b",
            total_gpus=32,
            seq_len=4096,
            tp_candidates=[2, 4, 8],
            pp_candidates=[1, 2, 4, 8],
        )
        
        # 获取最优配置
        best = results.best
        print(f"最优配置: {best.config_str}")
        print(f"吞吐量: {best.tokens_per_second_per_gpu:.0f} tok/s/GPU")
    """
    if isinstance(model, str):
        model_name = model
        model_config = None
    else:
        model_name = None
        model_config = model
    
    searcher = GridSearcher(
        model_name=model_name,
        model_config=model_config,
        total_gpus=total_gpus,
        tp_candidates=tp_candidates,
        pp_candidates=pp_candidates,
        ep_candidates=ep_candidates,
        sharding_candidates=sharding_candidates,
        verbose=verbose,
    )
    
    return searcher.search(
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        gradient_accumulation_steps=gradient_accumulation_steps,
        sort_by=sort_by,
        top_k=top_k,
        print_report=print_report,
    )