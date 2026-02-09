#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cost Model EP8 Verify V2 - 真实 EP8 训练验证 + 路由统计采集

功能：
1. 真正启用 EP8 (expert_model_parallel_size=8)
2. Hook 到 MoE 层采集 tokens_per_expert 统计
3. 测量真实训练 step time (forward/backward/scaler_step/step)
4. 输出详细的路由分布和通信量估计

运行方式:
    python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 costmodel_verify_ep8_v2.py

环境变量:
    PROFILE_LAYERS  : 测量层数（默认 4）
    SEQ_LEN         : 序列长度（默认 1024）
    MICRO_BSZ       : micro batch size（默认 1）
    STEPS           : 测量轮数（默认 20）
    WARMUP          : warmup 轮数（默认 3）
"""

import os
import time
import json
import gc
import warnings
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import paddle
import paddle.nn as nn
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

# 延迟导入，确保 PYTHONPATH 已设置
from paddleformers.transformers.qwen3_moe.configuration import Qwen3MoeConfig
from paddleformers.transformers.qwen3_moe.modeling import Qwen3MoeForCausalLMDecapitated


# ============================================================
# Data Classes
# ============================================================
@dataclass
class TimingStats:
    """计时统计"""
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    n: int = 0

    @staticmethod
    def from_list(xs: List[float]) -> "TimingStats":
        if not xs:
            return TimingStats()
        x = np.asarray(xs, dtype="float64")
        return TimingStats(
            mean_ms=float(x.mean()),
            std_ms=float(x.std()),
            min_ms=float(x.min()),
            max_ms=float(x.max()),
            p50_ms=float(np.percentile(x, 50)),
            p90_ms=float(np.percentile(x, 90)),
            n=int(x.size),
        )


@dataclass
class RouterStats:
    """MoE 路由统计（每个 step 采集一次）"""
    layer_id: int = 0
    total_tokens: int = 0
    tokens_per_expert: List[int] = field(default_factory=list)
    expert_utilization: List[float] = field(default_factory=list)
    max_tokens: int = 0
    min_tokens: int = 0
    imbalance_ratio: float = 0.0  # max/mean
    gini_coefficient: float = 0.0


@dataclass
class AggregatedRouterStats:
    """聚合的路由统计"""
    total_tokens: int = 0
    tokens_per_expert_mean: List[float] = field(default_factory=list)
    tokens_per_expert_std: List[float] = field(default_factory=list)
    imbalance_ratio_mean: float = 0.0
    imbalance_ratio_std: float = 0.0
    gini_coefficient_mean: float = 0.0


# ============================================================
# Utils
# ============================================================
def sync():
    if paddle.is_compiled_with_cuda():
        paddle.device.synchronize()


def dtype_nbytes(dtype_str: str) -> int:
    d = dtype_str.lower()
    if d in ("bfloat16", "float16"):
        return 2
    return 4


def count_parameters(model: nn.Layer) -> int:
    total = 0
    for p in model.parameters():
        n = p.numel()
        if hasattr(n, 'item'):
            n = n.item()
        total += int(n)
    return total


def compute_gini(values: np.ndarray) -> float:
    """计算 Gini 系数，衡量分布不均匀程度"""
    if len(values) < 2:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)


# ============================================================
# Router Statistics Hook (GPU-based, no .numpy() in hook)
# ============================================================
class RouterStatsCollector:
    """
    收集 MoE 路由统计的 Hook
    - Hook 注册在 raw model（不是 fleet wrapper）
    - Hook 到 gate/router 子模块的输出
    - GPU 端统计，禁止在 hook 内调用 .numpy()
    - 使用 dist.all_reduce 聚合全局统计
    - 只采样少数层（首/中/末）避免污染计时
    """
    
    def __init__(self, num_experts: int, ep_degree: int, topk: int, num_layers: int):
        self.num_experts = num_experts
        self.ep_degree = ep_degree
        self.topk = topk  # cfg.num_experts_per_tok
        self.num_layers = num_layers
        self.experts_per_rank = num_experts // ep_degree
        
        # 要采样的层（首/中/末）- 使用真实 layer index
        if num_layers <= 3:
            self.sample_layer_indices = list(range(num_layers))
        else:
            mid = num_layers // 2
            self.sample_layer_indices = [0, mid, num_layers - 1]
        
        # layer_name -> real_layer_index 的映射
        self.layer_name_to_index: Dict[str, int] = {}
        
        # 延迟聚合：只保存原始 topk_indices，最后统一用 bincount
        self.raw_indices_per_step: List[Dict[int, paddle.Tensor]] = []
        self.current_step_indices: Dict[int, paddle.Tensor] = {}
        self.hooks = []
        self.enabled = False
        self.rank = dist.get_rank()
    
    def _parse_layer_index(self, name: str) -> Optional[int]:
        """从 module name 解析真实 layer index，如 model.layers.5.mlp.gate -> 5"""
        import re
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return None
    
    def _hook_fn(self, real_layer_index: int):
        """
        创建 hook 函数 - 零计算，只保存 topk_indices
        
        核心改进：Hook 内不做 softmax/one_hot，只保存 gate 输出的 topk_indices
        """
        def hook(module, input, output):
            if not self.enabled:
                return
            if real_layer_index not in self.sample_layer_indices:
                return
            
            # gate 输出是 tuple:
            # (capacity, topk_weights, topk_indices, gates_masked, mask, priorities, aux_loss, z_loss)
            if output is None or not isinstance(output, tuple) or len(output) < 3:
                return
            
            topk_indices = output[2]  # [batch*seq, topk]
            if topk_indices is None:
                return
            
            # 零计算：直接保存 detach 后的 tensor，不做任何计算
            self.current_step_indices[real_layer_index] = topk_indices.detach()
        
        return hook
    
    def register_hooks(self, raw_model: nn.Layer):
        """
        注册 hooks 到 raw model 的 gate 子模块
        使用真实 layer index（从 name 解析）
        """
        hooked_layers = []
        
        for name, module in raw_model.named_modules():
            if name.endswith(".gate") and "mlp" in name:
                real_idx = self._parse_layer_index(name)
                if real_idx is not None:
                    self.layer_name_to_index[name] = real_idx
                    hook = module.register_forward_hook(self._hook_fn(real_idx))
                    self.hooks.append(hook)
                    hooked_layers.append((real_idx, name))
        
        # 按真实 layer index 排序
        hooked_layers.sort(key=lambda x: x[0])
        
        if self.rank == 0:
            if hooked_layers:
                print(f"[RouterStatsCollector] Registered {len(hooked_layers)} gate hooks")
                print(f"[RouterStatsCollector] Sample layer indices: {self.sample_layer_indices}")
                for real_idx, nm in hooked_layers:
                    marker = " [SAMPLED]" if real_idx in self.sample_layer_indices else ""
                    print(f"  Layer {real_idx}: {nm}{marker}")
            else:
                print(f"[RouterStatsCollector] Warning: No gate modules found")
    
    def start_step(self):
        """开始收集"""
        self.current_step_indices = {}
        self.enabled = True
    
    def end_step(self):
        """结束收集"""
        self.enabled = False
        if self.current_step_indices:
            self.raw_indices_per_step.append(self.current_step_indices.copy())
    
    def aggregate_and_compute(self, hidden_size: int, dtype_bytes: int = 2) -> Dict[str, Any]:
        """
        聚合统计并计算最终结果（使用 bincount 代替 one_hot）
        
        指标说明：
        - expert_selections[i]: expert i 被选中的次数（全局，所有 rank 累加）
        - selections_per_rank[r]: rank r 上的 experts 被选中的总次数
        - A2A 通信量基于 selections 估算（每次 selection = 1 个 token 的 hidden_size bytes）
        """
        if not self.raw_indices_per_step:
            return {
                "error": "no stats collected",
                "expert_selections": [],
                "selections_per_rank": [],
            }
        
        # 在 GPU 上用 bincount 统计每个 expert 的 selection 次数（比 one_hot 更高效）
        all_counts = []
        for step_data in self.raw_indices_per_step:
            for layer_idx, topk_indices in step_data.items():
                # topk_indices: [num_tokens, topk]
                flat = topk_indices.reshape([-1]).astype("int64")
                # bincount: [num_experts] - 每个 expert 被选中的次数
                counts = paddle.bincount(flat, minlength=self.num_experts)
                all_counts.append(counts.astype("float32"))
        
        if not all_counts:
            return {"error": "no valid indices", "expert_selections": [], "selections_per_rank": []}
        
        # 平均（跨 steps 和 sampled layers）
        stacked = paddle.stack(all_counts, axis=0)
        avg_counts = paddle.mean(stacked, axis=0)  # [num_experts]
        
        # all_reduce 得到全局统计
        global_counts = avg_counts.clone()
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
        
        result = {}
        if self.rank == 0:
            expert_selections = global_counts.numpy()
            
            # 基础统计
            result["expert_selections"] = expert_selections.tolist()
            result["expert_selections_mean"] = float(expert_selections.mean())
            result["expert_selections_std"] = float(expert_selections.std())
            result["expert_selections_max"] = float(expert_selections.max())
            result["expert_selections_min"] = float(expert_selections.min())
            
            # 按 rank 聚合（假设 contiguous 映射: expert [r*E_r, (r+1)*E_r) -> rank r）
            selections_per_rank = []
            for r in range(self.ep_degree):
                start_exp = r * self.experts_per_rank
                end_exp = (r + 1) * self.experts_per_rank
                rank_selections = expert_selections[start_exp:end_exp].sum()
                selections_per_rank.append(float(rank_selections))
            
            selections_per_rank = np.array(selections_per_rank)
            result["selections_per_rank"] = selections_per_rank.tolist()
            result["selections_per_rank_mean"] = float(selections_per_rank.mean())
            result["selections_per_rank_max"] = float(selections_per_rank.max())
            result["selections_per_rank_min"] = float(selections_per_rank.min())
            
            # Imbalance ratio (基于 selections)
            mean_sel = selections_per_rank.mean()
            max_sel = selections_per_rank.max()
            result["imbalance_ratio"] = float(max_sel / (mean_sel + 1e-10))
            
            # Gini coefficient
            result["gini_coefficient"] = float(compute_gini(selections_per_rank))
            
            # A2A 通信量估算
            # 每次 selection = 发送 1 个 token 的 hidden_state (hidden_size * dtype_bytes)
            a2a_recv_bytes_per_rank = selections_per_rank * hidden_size * dtype_bytes
            result["a2a_estimate"] = {
                "recv_bytes_per_rank_mean": float(a2a_recv_bytes_per_rank.mean()),
                "recv_bytes_per_rank_max": float(a2a_recv_bytes_per_rank.max()),
                "recv_bytes_per_rank_mean_MB": float(a2a_recv_bytes_per_rank.mean() / 1e6),
                "recv_bytes_per_rank_max_MB": float(a2a_recv_bytes_per_rank.max() / 1e6),
                # Forward: dispatch + combine, Backward: dispatch + combine = 4x per layer
                "per_layer_total_4x_MB": float(a2a_recv_bytes_per_rank.mean() * 4 / 1e6),
                "note": "Based on expert selection counts, assuming contiguous expert-to-rank mapping"
            }
            
            # 额外元信息
            result["metadata"] = {
                "topk": self.topk,
                "num_experts": self.num_experts,
                "ep_degree": self.ep_degree,
                "experts_per_rank": self.experts_per_rank,
                "sampled_layers": self.sample_layer_indices,
                "num_steps_collected": len(self.raw_indices_per_step),
                "mapping_assumption": "contiguous: expert [r*E_r, (r+1)*E_r) -> rank r"
            }
        
        return result
    
    def remove_hooks(self):
        """移除 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.raw_indices_per_step = []
        self.current_step_indices = {}


# ============================================================
# Fleet Initialization for EP8
# ============================================================
def init_fleet_ep8(seed: int, ep_degree: int = 8) -> Dict[str, Any]:
    """
    初始化 Fleet with EP8 配置
    
    参考 PaddleFormers trainer/training_args.py 的配置方式
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    
    strategy = fleet.DistributedStrategy()
    
    # 核心: EP8 配置
    # ep_degree=8, 其他并行度为 1
    # moe_sharding_degree = world // ep_degree = 1 (单机8卡)
    moe_sharding_degree = max(1, world // ep_degree)
    
    # 参考 PaddleFormers training_args.py 的 order（不包含 cp 如果 context_parallel 不支持）
    # 当 expert_model_parallel_size > 1:
    # order = ["sharding", "moe_sharding", "pp", "sep", "dp", "ep", "mp"]
    order = ["sharding", "moe_sharding", "pp", "sep", "dp", "ep", "mp"]
    
    # 关键: dense topology world_size 必须等于 moe topology world_size
    # dense_world = dp * sharding * pp * sep * mp
    # moe_world = moe_sharding * ep * pp * mp
    # 对于 EP8 (8卡): dense_world = 8, moe_world = 1 * 8 = 8
    # 
    # EPHybridCommunicateGroup 要求: dp_degree == 1 and sep_degree == 1 in MoE!
    # 因此通过 sharding_degree 来满足 world_size 约束
    hybrid_configs = {
        "dp_degree": 1,  # 在 MoE EP 模式下必须为 1
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": world,  # 用 sharding 来满足 world_size = 8
        "sep_degree": 1,  # 在 MoE EP 模式下必须为 1
        "ep_degree": ep_degree,  # EP8: MoE experts 分布在 8 个 GPU
        "moe_sharding_degree": moe_sharding_degree,  # = 1
        "order": order,
    }
    
    strategy.hybrid_configs = hybrid_configs
    
    if rank == 0:
        print(f"[Fleet] hybrid_configs = {hybrid_configs}")
    
    fleet.init(is_collective=True, strategy=strategy)
    paddle.seed(seed)
    
    return {
        "world_size": world,
        "rank": rank,
        "ep_degree": ep_degree,
        "moe_sharding_degree": moe_sharding_degree,
        "hybrid_configs": hybrid_configs,
    }


# ============================================================
# EP8 Profiler
# ============================================================
class EP8Profiler:
    """EP8 真实训练测量器"""

    def __init__(self, raw_model: nn.Layer, optimizer, router_collector: RouterStatsCollector, dtype: str = "bfloat16"):
        self.dtype = dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.router_collector = router_collector
        self._raw_model = raw_model
        
        # 注册路由统计 hooks 到 raw model（在 Fleet 包装之前）
        router_collector.register_hooks(raw_model)
        
        # Fleet 包装
        self.model = fleet.distributed_model(raw_model)
        self.optimizer = fleet.distributed_optimizer(optimizer)
        self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    def profile(
        self,
        input_ids: paddle.Tensor,
        labels: paddle.Tensor,
        steps: int,
        warmup: int,
    ) -> Dict[str, Any]:
        """
        测量真实 EP8 训练时间
        """
        forward_times = []
        backward_times = []
        scaler_step_times = []
        scaler_update_times = []
        step_times = []

        for step in range(steps):
            self.optimizer.clear_grad()
            
            # 开始路由统计
            if step >= warmup:
                self.router_collector.start_step()
            
            sync()
            step_start = time.time()

            # Forward
            sync()
            t_fwd_start = time.time()
            with paddle.amp.auto_cast(enable=True, level="O2", dtype=self.dtype):
                output = self.model(input_ids=input_ids, labels=labels)
            
            if isinstance(output, tuple):
                loss = output[0]
            elif hasattr(output, 'loss'):
                loss = output.loss
            else:
                loss = output
            if loss.ndim > 0:
                loss = loss.mean()
            sync()
            t_fwd_end = time.time()

            # 结束路由统计（forward 后）
            if step >= warmup:
                self.router_collector.end_step()

            # Backward
            sync()
            t_bwd_start = time.time()
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            sync()
            t_bwd_end = time.time()

            # Scaler step
            sync()
            t_scaler_start = time.time()
            self.scaler.step(self.optimizer)
            sync()
            t_scaler_end = time.time()
            
            # Scaler update
            sync()
            t_update_start = time.time()
            self.scaler.update()
            sync()
            t_update_end = time.time()

            sync()
            step_end = time.time()

            if step >= warmup:
                forward_times.append((t_fwd_end - t_fwd_start) * 1000.0)
                backward_times.append((t_bwd_end - t_bwd_start) * 1000.0)
                scaler_step_times.append((t_scaler_end - t_scaler_start) * 1000.0)
                scaler_update_times.append((t_update_end - t_update_start) * 1000.0)
                step_times.append((step_end - step_start) * 1000.0)
            
            if self.rank == 0 and step < 3:
                print(f"[Step {step}] loss={float(loss):.4f}, step_time={(step_end - step_start)*1000:.2f}ms")

        return {
            "forward": TimingStats.from_list(forward_times),
            "backward": TimingStats.from_list(backward_times),
            "scaler_step": TimingStats.from_list(scaler_step_times),
            "scaler_update": TimingStats.from_list(scaler_update_times),
            "step": TimingStats.from_list(step_times),
        }


# ============================================================
# Main
# ============================================================
def main():
    model_path = os.environ.get(
        "MODEL_PATH", 
        "/root/paddlejob/workspace/env_run/zhangdongqi/Qwen3-30B-A3B-Base"
    )
    profile_layers = int(os.environ.get("PROFILE_LAYERS", "4"))
    seq_len = int(os.environ.get("SEQ_LEN", "1024"))
    micro_bsz = int(os.environ.get("MICRO_BSZ", "1"))
    steps = int(os.environ.get("STEPS", "20"))
    warmup = int(os.environ.get("WARMUP", "3"))
    seed = int(os.environ.get("SEED", "23"))
    dtype = os.environ.get("DTYPE", "bfloat16")
    ep_degree = int(os.environ.get("EP_DEGREE", "8"))
    output_file = os.environ.get("OUTPUT_FILE", "costmodel_ep8_verify.json")

    # 初始化 Fleet with EP8
    fleet_config = init_fleet_ep8(seed, ep_degree)
    rank = fleet_config["rank"]
    world = fleet_config["world_size"]

    if rank == 0:
        print("=" * 70)
        print("EP8 VERIFY - Real EP8 Training with Router Statistics")
        print("=" * 70)
        print(f"[Config] MODEL_PATH     = {model_path}")
        print(f"[Config] PROFILE_LAYERS = {profile_layers}")
        print(f"[Config] SEQ_LEN        = {seq_len}")
        print(f"[Config] MICRO_BSZ      = {micro_bsz}")
        print(f"[Config] STEPS          = {steps}")
        print(f"[Config] WARMUP         = {warmup}")
        print(f"[Config] DTYPE          = {dtype}")
        print(f"[Config] EP_DEGREE      = {ep_degree}")
        print(f"[Config] WORLD_SIZE     = {world}")
        print("=" * 70)

    # 加载模型配置
    if rank == 0:
        print("\n[Step 1] Loading model configuration...")

    cfg = Qwen3MoeConfig.from_pretrained(model_path)
    original_layers = cfg.num_hidden_layers
    cfg.num_hidden_layers = profile_layers
    
    # 启用 EP
    cfg.expert_model_parallel_size = ep_degree
    cfg.use_expert_parallel = True
    
    num_experts = getattr(cfg, "num_experts", 128)
    num_experts_per_tok = getattr(cfg, "num_experts_per_tok", 8)
    experts_per_rank = num_experts // ep_degree

    if rank == 0:
        print(f"[Step 1] Creating model with {profile_layers} layers (original: {original_layers})")
        print(f"[Step 1] MoE config: num_experts={num_experts}, num_experts_per_tok={num_experts_per_tok}")
        print(f"[Step 1] EP config: ep_degree={ep_degree}, experts_per_rank={experts_per_rank}")

    # 创建模型
    model = Qwen3MoeForCausalLMDecapitated(cfg)
    model = model.astype(dtype)
    model.train()

    total_params = count_parameters(model)

    if rank == 0:
        print(f"[Step 1] Params: {total_params:,}")

    # 创建优化器
    if rank == 0:
        print("\n[Step 2] Creating optimizer...")

    optimizer = paddle.optimizer.AdamW(
        learning_rate=1e-5,
        parameters=model.parameters(),
        weight_decay=0.01,
        multi_precision=True,
    )

    # 创建路由统计收集器（传入 topk = num_experts_per_tok）
    router_collector = RouterStatsCollector(
        num_experts=num_experts,
        ep_degree=ep_degree,
        topk=num_experts_per_tok,
        num_layers=profile_layers,
    )

    # 创建输入数据
    if rank == 0:
        print("\n[Step 3] Creating input data...")

    input_ids = paddle.randint(0, cfg.vocab_size, shape=[micro_bsz, seq_len], dtype="int64")
    labels = paddle.randint(0, cfg.vocab_size, shape=[micro_bsz, seq_len], dtype="int64")

    # 测量
    if rank == 0:
        print(f"\n[Step 4] Profiling ({steps} steps, {warmup} warmup)...")

    profiler = EP8Profiler(model, optimizer, router_collector, dtype=dtype)
    timing_results = profiler.profile(input_ids, labels, steps, warmup)

    # 路由统计（全局聚合）
    if rank == 0:
        print("\n[Step 5] Aggregating router statistics (all_reduce)...")

    router_result = router_collector.aggregate_and_compute(cfg.hidden_size, dtype_nbytes(dtype))

    # 清理 hooks
    router_collector.remove_hooks()

    # 自检：打印 EP 配置确认
    if rank == 0:
        print("\n" + "=" * 70)
        print("EP CONFIGURATION SELF-CHECK")
        print("=" * 70)
        print(f"{'cfg.use_expert_parallel':<35} {getattr(cfg, 'use_expert_parallel', 'N/A')}")
        print(f"{'cfg.expert_model_parallel_size':<35} {getattr(cfg, 'expert_model_parallel_size', 'N/A')}")
        print(f"{'cfg.num_experts':<35} {num_experts}")
        print(f"{'cfg.num_experts_per_tok (topk)':<35} {num_experts_per_tok}")
        print(f"{'experts_per_rank':<35} {experts_per_rank}")
        print(f"{'world_size':<35} {world}")
        print(f"{'ep_degree':<35} {ep_degree}")
        print("=" * 70)

    # 输出结果
    if rank == 0:
        timing_sum = (timing_results["forward"].mean_ms + 
                     timing_results["backward"].mean_ms + 
                     timing_results["scaler_step"].mean_ms + 
                     timing_results["scaler_update"].mean_ms)
        framework_overhead = timing_results["step"].mean_ms - timing_sum

        # 提取 router_result 数据（使用新字段名）
        selections_per_rank = router_result.get("selections_per_rank", [])
        a2a_estimate = router_result.get("a2a_estimate", {})
        metadata = router_result.get("metadata", {})

        output = {
            "meta": {
                "mode": "EP8_VERIFY_V2",
                "model_path": model_path,
                "profile_layers": profile_layers,
                "original_layers": original_layers,
                "seq_len": seq_len,
                "micro_bsz": micro_bsz,
                "dtype": dtype,
                "ep_degree": ep_degree,
                "experts_per_rank": experts_per_rank,
                "num_experts": num_experts,
                "num_experts_per_tok": num_experts_per_tok,
                "total_params": total_params,
                "fleet_config": fleet_config,
                "ep_self_check": {
                    "cfg_use_expert_parallel": getattr(cfg, "use_expert_parallel", None),
                    "cfg_expert_model_parallel_size": getattr(cfg, "expert_model_parallel_size", None),
                },
            },
            "timing": {
                "forward_ms": timing_results["forward"].mean_ms,
                "backward_ms": timing_results["backward"].mean_ms,
                "scaler_step_ms": timing_results["scaler_step"].mean_ms,
                "scaler_update_ms": timing_results["scaler_update"].mean_ms,
                "step_ms": timing_results["step"].mean_ms,
                "timing_sum_ms": timing_sum,
                "framework_overhead_ms": framework_overhead,
                "forward_detail": asdict(timing_results["forward"]),
                "backward_detail": asdict(timing_results["backward"]),
                "step_detail": asdict(timing_results["step"]),
            },
            "router_stats": {
                "imbalance_ratio": router_result.get("imbalance_ratio", 0),
                "gini_coefficient": router_result.get("gini_coefficient", 0),
                "selections_per_rank": selections_per_rank,
                "selections_per_rank_mean": router_result.get("selections_per_rank_mean", 0),
                "selections_per_rank_max": router_result.get("selections_per_rank_max", 0),
                "selections_per_rank_min": router_result.get("selections_per_rank_min", 0),
                "expert_selections_sample": router_result.get("expert_selections", [])[:20],
                "metadata": metadata,
            },
            "a2a_estimate": a2a_estimate,
        }

        print("\n" + "=" * 70)
        print(f"EP8 VERIFY RESULTS ({profile_layers} layers)")
        print("=" * 70)
        print(f"{'Component':<20} {'Time (ms)':>12} {'Percent':>10}")
        print("-" * 50)
        print(f"{'Forward':<20} {timing_results['forward'].mean_ms:>12.2f} {timing_results['forward'].mean_ms/timing_results['step'].mean_ms*100:>9.1f}%")
        print(f"{'Backward':<20} {timing_results['backward'].mean_ms:>12.2f} {timing_results['backward'].mean_ms/timing_results['step'].mean_ms*100:>9.1f}%")
        print(f"{'Scaler+Opt':<20} {timing_results['scaler_step'].mean_ms:>12.2f} {timing_results['scaler_step'].mean_ms/timing_results['step'].mean_ms*100:>9.1f}%")
        print(f"{'Scaler Update':<20} {timing_results['scaler_update'].mean_ms:>12.2f}")
        print("-" * 50)
        print(f"{'Sum':<20} {timing_sum:>12.2f}")
        print(f"{'Step (actual)':<20} {timing_results['step'].mean_ms:>12.2f}")
        print(f"{'Framework Overhead':<20} {framework_overhead:>12.2f}")
        print("=" * 70)

        print("\n" + "=" * 70)
        print("ROUTER STATISTICS (Global Aggregated)")
        print("=" * 70)
        print(f"{'Metric':<35} {'Value':>15}")
        print("-" * 55)
        print(f"{'Imbalance Ratio':<35} {router_result.get('imbalance_ratio', 0):>15.3f}")
        print(f"{'Gini Coefficient':<35} {router_result.get('gini_coefficient', 0):>15.3f}")
        print(f"{'Selections per rank (mean)':<35} {router_result.get('selections_per_rank_mean', 0):>15.1f}")
        print(f"{'Selections per rank (max)':<35} {router_result.get('selections_per_rank_max', 0):>15.1f}")
        print(f"{'Selections per rank (min)':<35} {router_result.get('selections_per_rank_min', 0):>15.1f}")
        if selections_per_rank:
            dist_str = ", ".join([f"{x:.0f}" for x in selections_per_rank])
            print(f"{'Selections per rank distribution':<35}")
            print(f"  [{dist_str}]")
        print(f"{'Note':<35} topk={num_experts_per_tok}, each token selects topk experts")
        print("=" * 70)

        print("\n" + "=" * 70)
        print("ALL-TO-ALL COMMUNICATION ESTIMATE")
        print("=" * 70)
        print(f"{'Metric':<40} {'Value':>15}")
        print("-" * 60)
        print(f"{'A2A recv bytes per rank (mean)':<40} {a2a_estimate.get('recv_bytes_per_rank_mean_MB', 0):>12.2f} MB")
        print(f"{'A2A recv bytes per rank (max)':<40} {a2a_estimate.get('recv_bytes_per_rank_max_MB', 0):>12.2f} MB")
        print(f"{'Per layer total (4x: fwd+bwd)':<40} {a2a_estimate.get('per_layer_total_4x_MB', 0):>12.2f} MB")
        print(f"{'Note':<40} {a2a_estimate.get('note', '')}")
        print("=" * 70)

        # 保存
        with open(output_file, "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[Done] Results saved to {output_file}")


if __name__ == "__main__":
    main()
