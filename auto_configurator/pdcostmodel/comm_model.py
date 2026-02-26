#!/usr/bin/env python3
"""
通信模型模块 - 预测 PaddleFormers 分布式训练的通信时间

支持的通信原语:
1. AllReduce - TP 层内同步
2. AllGather/ReduceScatter - ZeRO/Sharding
3. AllToAll - MoE EP
4. P2P Send/Recv - PP 流水线

参考 Galvatron 的通信建模方法
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

from .config import HardwareConfig, ParallelConfig, ModelConfig, TrainingConfig


class CommType(Enum):
    """通信类型"""
    ALLREDUCE = "allreduce"
    ALLGATHER = "allgather"
    REDUCE_SCATTER = "reduce_scatter"
    ALLTOALL = "alltoall"
    P2P = "p2p"
    BROADCAST = "broadcast"


@dataclass
class CommResult:
    """通信预测结果"""
    time_ms: float = 0.0  # 预测时间
    bandwidth_gbps: float = 0.0  # 实际带宽利用
    volume_bytes: int = 0  # 通信量
    latency_ms: float = 0.0  # 延迟
    transfer_ms: float = 0.0  # 传输时间


class CommModel:
    """
    通信模型
    
    支持感知网络拓扑 (节点内/节点间)
    """
    
    def __init__(self, hardware_config: HardwareConfig):
        self.hardware = hardware_config
    
    def _get_bandwidth(self, is_intra_node: bool) -> float:
        """获取带宽 (GB/s)"""
        if is_intra_node:
            return self.hardware.network.intra_node_bandwidth_gbps
        return self.hardware.network.inter_node_bandwidth_gbps
    
    def _get_latency(self, is_intra_node: bool) -> float:
        """获取延迟 (us)"""
        if is_intra_node:
            return self.hardware.network.intra_node_latency_us
        return self.hardware.network.inter_node_latency_us
    
    def _bytes_to_gb(self, bytes_count: int) -> float:
        return bytes_count / (1024 ** 3)
    
    def predict_allreduce(self, data_size_bytes: int, num_gpus: int,
                          is_intra_node: bool = True) -> CommResult:
        """
        预测 AllReduce 通信时间
        
        Ring AllReduce:
        - 通信量 = 2 * (N-1)/N * data_size
        - 延迟 = 2 * (N-1) * α
        """
        if num_gpus <= 1:
            return CommResult()
        
        # Ring AllReduce 通信量
        ring_factor = 2 * (num_gpus - 1) / num_gpus
        comm_volume = int(data_size_bytes * ring_factor)
        
        bandwidth = self._get_bandwidth(is_intra_node)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.allreduce_efficiency
        
        # 延迟: 2 * (N-1) 步
        num_steps = 2 * (num_gpus - 1)
        latency_ms = num_steps * latency_us / 1000.0
        
        # 传输时间
        data_gb = self._bytes_to_gb(comm_volume)
        transfer_ms = data_gb / (bandwidth * efficiency) * 1000.0
        
        total_time_ms = latency_ms + transfer_ms
        
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_gbps=data_gb / (total_time_ms / 1000.0) if total_time_ms > 0 else 0,
            volume_bytes=comm_volume,
            latency_ms=latency_ms,
            transfer_ms=transfer_ms,
        )
    
    def predict_allgather(self, data_size_bytes: int, num_gpus: int,
                          is_intra_node: bool = True) -> CommResult:
        """
        预测 AllGather 通信时间
        
        通信量 = (N-1)/N * data_size
        """
        if num_gpus <= 1:
            return CommResult()
        
        gather_factor = (num_gpus - 1) / num_gpus
        comm_volume = int(data_size_bytes * gather_factor)
        
        bandwidth = self._get_bandwidth(is_intra_node)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.allgather_efficiency
        
        num_steps = num_gpus - 1
        latency_ms = num_steps * latency_us / 1000.0
        
        data_gb = self._bytes_to_gb(comm_volume)
        transfer_ms = data_gb / (bandwidth * efficiency) * 1000.0
        
        total_time_ms = latency_ms + transfer_ms
        
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_gbps=data_gb / (total_time_ms / 1000.0) if total_time_ms > 0 else 0,
            volume_bytes=comm_volume,
            latency_ms=latency_ms,
            transfer_ms=transfer_ms,
        )
    
    def predict_reduce_scatter(self, data_size_bytes: int, num_gpus: int,
                               is_intra_node: bool = True) -> CommResult:
        """
        预测 ReduceScatter 通信时间
        
        用于 ZeRO-2 梯度分片
        通信量 = (N-1)/N * data_size
        """
        if num_gpus <= 1:
            return CommResult()
        
        # 与 AllGather 类似
        scatter_factor = (num_gpus - 1) / num_gpus
        comm_volume = int(data_size_bytes * scatter_factor)
        
        bandwidth = self._get_bandwidth(is_intra_node)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.allreduce_efficiency
        
        num_steps = num_gpus - 1
        latency_ms = num_steps * latency_us / 1000.0
        
        data_gb = self._bytes_to_gb(comm_volume)
        transfer_ms = data_gb / (bandwidth * efficiency) * 1000.0
        
        total_time_ms = latency_ms + transfer_ms
        
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_gbps=data_gb / (total_time_ms / 1000.0) if total_time_ms > 0 else 0,
            volume_bytes=comm_volume,
            latency_ms=latency_ms,
            transfer_ms=transfer_ms,
        )
    
    def predict_alltoall(self, data_size_bytes: int, num_gpus: int,
                         is_intra_node: bool = True,
                         topk: int = 8, num_experts: int = 128) -> CommResult:
        """
        预测 AllToAll 通信时间
        
        用于 MoE EP 的 dispatch 和 combine
        
        AllToAll 特点:
        - 每个 GPU 发送 data_size / num_gpus 给其他每个 GPU
        - 通信量 = data_size * (num_gpus - 1) / num_gpus
        - 所有通信可以并行进行（全双工网络）
        """
        if num_gpus <= 1:
            return CommResult()
        
        # A2A 通信量: 每个 GPU 发送/接收 (N-1)/N 的数据
        a2a_factor = (num_gpus - 1) / num_gpus
        comm_volume = int(data_size_bytes * a2a_factor)
        
        bandwidth = self._get_bandwidth(is_intra_node)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.alltoall_efficiency
        
        # AllToAll 延迟 (并行通信，只算一次启动延迟)
        latency_ms = latency_us / 1000.0
        
        # 传输时间 (全双工网络，所有发送并行)
        data_gb = self._bytes_to_gb(comm_volume)
        transfer_ms = data_gb / (bandwidth * efficiency) * 1000.0
        
        # 负载不均衡因子 (MoE routing 不均衡)
        imbalance_factor = 1.15
        
        total_time_ms = (latency_ms + transfer_ms) * imbalance_factor
        
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_gbps=data_gb / (total_time_ms / 1000.0) if total_time_ms > 0 else 0,
            volume_bytes=comm_volume,
            latency_ms=latency_ms,
            transfer_ms=transfer_ms,
        )
    
    def predict_p2p(self, data_size_bytes: int, is_intra_node: bool = False) -> CommResult:
        """
        预测 P2P (Send/Recv) 通信时间
        
        用于 PP 流水线
        """
        bandwidth = self._get_bandwidth(is_intra_node)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.p2p_efficiency
        
        latency_ms = latency_us / 1000.0
        
        data_gb = self._bytes_to_gb(data_size_bytes)
        transfer_ms = data_gb / (bandwidth * efficiency) * 1000.0
        
        total_time_ms = latency_ms + transfer_ms
        
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_gbps=data_gb / (total_time_ms / 1000.0) if total_time_ms > 0 else 0,
            volume_bytes=data_size_bytes,
            latency_ms=latency_ms,
            transfer_ms=transfer_ms,
        )
    
    def predict_tp_comm(self, activation_size_bytes: int, tp_degree: int) -> CommResult:
        """
        预测 TP 通信时间
        
        TP 在每层需要 AllReduce (通常节点内)
        """
        if tp_degree <= 1:
            return CommResult()
        
        is_intra_node = self.hardware.is_intra_node(tp_degree)
        return self.predict_allreduce(activation_size_bytes, tp_degree, is_intra_node)
    
    def predict_dp_comm(self, gradient_size_bytes: int, dp_degree: int,
                        use_sharding: bool = False) -> CommResult:
        """
        预测 DP 通信时间
        
        - 普通 DP: AllReduce 梯度
        - Sharding: ReduceScatter 梯度
        """
        if dp_degree <= 1:
            return CommResult()
        
        is_intra_node = self.hardware.is_intra_node(dp_degree)
        
        if use_sharding:
            return self.predict_reduce_scatter(gradient_size_bytes, dp_degree, is_intra_node)
        return self.predict_allreduce(gradient_size_bytes, dp_degree, is_intra_node)
    
    def predict_ep_comm(self, token_data_bytes: int, ep_degree: int,
                        topk: int = 8, num_experts: int = 128) -> CommResult:
        """
        预测 EP 通信时间
        
        MoE 的 dispatch + combine 两次 AllToAll
        """
        if ep_degree <= 1:
            return CommResult()
        
        is_intra_node = self.hardware.is_intra_node(ep_degree)
        
        return self.predict_alltoall(
            token_data_bytes, ep_degree, is_intra_node,
            topk=topk, num_experts=num_experts
        )
    
    def predict_pp_comm(self, activation_size_bytes: int, pp_degree: int,
                        num_micro_batches: int) -> CommResult:
        """
        预测 PP 通信时间
        
        每个 micro-batch 需要在 stage 之间传递激活值
        """
        if pp_degree <= 1:
            return CommResult()
        
        # PP 通常跨节点
        is_intra_node = False
        
        # 单次 P2P
        single_p2p = self.predict_p2p(activation_size_bytes, is_intra_node)
        
        # 每个 micro-batch: (pp_degree - 1) 次前向 + (pp_degree - 1) 次后向
        num_comms = 2 * (pp_degree - 1)
        
        return CommResult(
            time_ms=single_p2p.time_ms * num_comms,
            bandwidth_gbps=single_p2p.bandwidth_gbps,
            volume_bytes=single_p2p.volume_bytes * num_comms,
            latency_ms=single_p2p.latency_ms * num_comms,
            transfer_ms=single_p2p.transfer_ms * num_comms,
        )
    
    def predict_sp_comm(self, activation_size_bytes: int, tp_degree: int) -> CommResult:
        """
        预测 SP (Sequence Parallel) 通信时间
        
        使用 AllGather 收集序列维度
        """
        if tp_degree <= 1:
            return CommResult()
        
        is_intra_node = self.hardware.is_intra_node(tp_degree)
        return self.predict_allgather(activation_size_bytes, tp_degree, is_intra_node)
    
    def estimate_step_comm_time(self, 
                                model_config: ModelConfig,
                                training_config: TrainingConfig,
                                parallel: ParallelConfig,
                                num_micro_batches: int) -> Dict[str, float]:
        """
        估算一个 step 的通信时间
        
        Returns:
            各类通信时间详情
        """
        h = model_config.hidden_size
        seq_len = training_config.sequence_length
        micro_bsz = training_config.micro_batch_size
        dtype_bytes = training_config.dtype_bytes
        
        # 激活大小
        activation_size = micro_bsz * seq_len * h * dtype_bytes
        
        # ========== TP 通信 ==========
        # 每层 2 次 AllReduce (Attention 后 + MLP 后)
        layers_per_stage = model_config.num_hidden_layers // parallel.pp
        tp_comm_result = self.predict_tp_comm(activation_size, parallel.tp)
        tp_comm_time = tp_comm_result.time_ms * 2 * layers_per_stage * num_micro_batches
        
        # ========== EP 通信 ==========
        # MoE 层的 dispatch + combine
        moe_layers_per_stage = model_config.num_moe_layers // parallel.pp
        topk = model_config.num_experts_per_tok
        token_data_size = micro_bsz * seq_len * h * topk * dtype_bytes
        
        ep_comm_result = self.predict_ep_comm(
            token_data_size, parallel.ep,
            topk=topk, num_experts=model_config.num_experts
        )
        ep_comm_time = ep_comm_result.time_ms * moe_layers_per_stage * num_micro_batches
        
        # ========== PP 通信 ==========
        pp_comm_result = self.predict_pp_comm(
            activation_size, parallel.pp, num_micro_batches
        )
        pp_comm_time = pp_comm_result.time_ms
        
        # ========== DP/Sharding 通信 ==========
        # 梯度同步 (step 结束时)
        # 参数量估算
        param_count = model_config.estimate_parameters()["total"] // (parallel.tp * parallel.pp)
        grad_size = param_count * dtype_bytes
        
        use_sharding = parallel.sharding_stage.value != "none"
        dp_degree = parallel.effective_sharding_degree if use_sharding else parallel.dp
        
        dp_comm_result = self.predict_dp_comm(grad_size, dp_degree, use_sharding)
        dp_comm_time = dp_comm_result.time_ms
        
        # ========== SP 通信 ==========
        sp_comm_time = 0.0
        if parallel.sp and parallel.tp > 1:
            sp_comm_result = self.predict_sp_comm(activation_size, parallel.tp)
            sp_comm_time = sp_comm_result.time_ms * layers_per_stage * num_micro_batches
        
        return {
            "tp_comm_time_ms": tp_comm_time,
            "ep_comm_time_ms": ep_comm_time,
            "pp_comm_time_ms": pp_comm_time,
            "dp_comm_time_ms": dp_comm_time,
            "sp_comm_time_ms": sp_comm_time,
            "total_comm_time_ms": tp_comm_time + ep_comm_time + pp_comm_time + dp_comm_time + sp_comm_time,
        }