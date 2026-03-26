#!/usr/bin/env python3
"""
通信模型模块 - 预测 PaddleFormers 分布式训练的通信时间

支持的通信原语:
1. AllReduce - TP 层内同步
2. AllGather/ReduceScatter - ZeRO/Sharding
3. AllToAll - MoE EP
4. P2P Send/Recv - PP 流水线

"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

from ..config import (
    HardwareConfig,
    ParallelConfig,
    ModelConfig,
    TrainingConfig,
    TRANSFORMER_DENSE_LAYER_KIND,
    TRANSFORMER_MOE_LAYER_KIND,
    INPUT_EMBEDDING_LAYER_KIND,
    OUTPUT_HEAD_LAYER_KIND,
)
from ..stage_layout import (
    resolve_chunk_ranges,
    resolve_stage_chunk_ranges,
    resolve_stage_layer_indices,
)
from .pipeline_schedule import simulate_1f1b_makespan


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

    DEFAULT_DP_BUCKET_BYTES = 40 * 1024 * 1024
    MAX_DP_BUCKET_BYTES = 256 * 1024 * 1024
    
    def __init__(self, hardware_config: HardwareConfig):
        self.hardware = hardware_config

    def _get_virtual_pipeline_size(self, parallel: ParallelConfig) -> int:
        raw_value = getattr(parallel, "vpp", 1)
        try:
            return max(1, int(raw_value))
        except Exception:
            return 1

    def _chunk_layer_ranges(self, model_config: ModelConfig,
                            parallel: ParallelConfig) -> list[tuple[int, int]]:
        return resolve_chunk_ranges(int(model_config.num_hidden_layers), parallel)

    def _stage_chunk_ranges(self, model_config: ModelConfig,
                            parallel: ParallelConfig) -> list[list[tuple[int, int]]]:
        return resolve_stage_chunk_ranges(int(model_config.num_hidden_layers), parallel)

    def _stage_layer_indices(self, model_config: ModelConfig,
                             parallel: ParallelConfig,
                             stage_id: int) -> list[int]:
        return resolve_stage_layer_indices(
            int(model_config.num_hidden_layers), parallel, stage_id
        )

    def _is_moe_layer(self, model_config: ModelConfig, layer_idx: int) -> bool:
        return model_config.is_moe_layer(layer_idx)

    def _transformer_layer_kind(self,
                                model_config: ModelConfig,
                                layer_idx: int) -> str:
        return model_config.transformer_layer_kind(layer_idx)

    def _estimate_stage_parameter_counts_per_gpu(self,
                                                 model_config: ModelConfig,
                                                 parallel: ParallelConfig) -> list[int]:
        h = max(1, int(model_config.hidden_size))
        ffn = max(1, int(model_config.intermediate_size))
        moe_ffn = max(1, int(model_config.moe_intermediate_size))
        v = max(1, int(model_config.vocab_size))
        tp = max(1, int(parallel.tp))
        ep = max(1, int(parallel.ep))
        pp = max(1, int(parallel.pp))

        q_size = h
        kv_size = max(1, int(model_config.num_key_value_heads * model_config.head_dim))
        attention_params = q_size * h + 2 * kv_size * h + h * h
        dense_mlp_params = 3 * h * ffn
        router_params = h * max(1, int(model_config.num_experts))
        expert_params = 3 * h * moe_ffn * max(1, int(model_config.num_experts))
        layernorm_params = 2 * h
        embedding_params = v * h

        stage_parameter_templates = {
            TRANSFORMER_DENSE_LAYER_KIND: (
                math.ceil(attention_params / tp) +
                layernorm_params +
                math.ceil(dense_mlp_params / tp)
            ),
            TRANSFORMER_MOE_LAYER_KIND: (
                math.ceil(attention_params / tp) +
                layernorm_params +
                router_params +
                math.ceil(expert_params / ep)
            ),
            INPUT_EMBEDDING_LAYER_KIND: math.ceil(embedding_params / tp),
            OUTPUT_HEAD_LAYER_KIND: math.ceil(embedding_params / tp) + 2 * h,
        }

        stage_param_counts = []
        for stage_id in range(pp):
            stage_params = 0
            if stage_id == 0:
                stage_params += stage_parameter_templates[INPUT_EMBEDDING_LAYER_KIND]
            if stage_id == pp - 1:
                stage_params += stage_parameter_templates[OUTPUT_HEAD_LAYER_KIND]

            for layer_idx in self._stage_layer_indices(model_config, parallel, stage_id):
                stage_params += stage_parameter_templates[
                    self._transformer_layer_kind(model_config, layer_idx)
                ]

            stage_param_counts.append(max(0, int(stage_params)))
        return stage_param_counts

    def _group_topology(self, num_gpus: int) -> tuple[int, int]:
        local_degree = min(max(1, int(self.hardware.gpus_per_node)), max(1, int(num_gpus)))
        num_nodes = max(1, math.ceil(max(1, int(num_gpus)) / local_degree))
        return local_degree, num_nodes

    def _predict_hierarchical_allreduce(self, data_size_bytes: int,
                                        num_gpus: int) -> CommResult:
        local_degree, num_nodes = self._group_topology(num_gpus)
        if num_nodes <= 1:
            return self.predict_allreduce(data_size_bytes, num_gpus, is_intra_node=True)

        local_reduce = self.predict_reduce_scatter(data_size_bytes, local_degree, True)
        inter_shard_bytes = max(1, math.ceil(data_size_bytes / local_degree))
        inter_reduce = self.predict_allreduce(inter_shard_bytes, num_nodes, False)
        local_gather = self.predict_allgather(data_size_bytes, local_degree, True)
        total_time_ms = (
            local_reduce.time_ms +
            inter_reduce.time_ms +
            local_gather.time_ms
        )
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_gbps=min(
                local_reduce.bandwidth_gbps or float("inf"),
                inter_reduce.bandwidth_gbps or float("inf"),
                local_gather.bandwidth_gbps or float("inf"),
            ),
            volume_bytes=(
                local_reduce.volume_bytes +
                inter_reduce.volume_bytes +
                local_gather.volume_bytes
            ),
            latency_ms=(
                local_reduce.latency_ms +
                inter_reduce.latency_ms +
                local_gather.latency_ms
            ),
            transfer_ms=(
                local_reduce.transfer_ms +
                inter_reduce.transfer_ms +
                local_gather.transfer_ms
            ),
        )

    def _predict_hierarchical_reduce_scatter(self, data_size_bytes: int,
                                             num_gpus: int) -> CommResult:
        local_degree, num_nodes = self._group_topology(num_gpus)
        if num_nodes <= 1:
            return self.predict_reduce_scatter(
                data_size_bytes, num_gpus, is_intra_node=True
            )

        local_reduce = self.predict_reduce_scatter(data_size_bytes, local_degree, True)
        inter_shard_bytes = max(1, math.ceil(data_size_bytes / local_degree))
        inter_reduce = self.predict_reduce_scatter(inter_shard_bytes, num_nodes, False)
        total_time_ms = local_reduce.time_ms + inter_reduce.time_ms
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_gbps=min(
                local_reduce.bandwidth_gbps or float("inf"),
                inter_reduce.bandwidth_gbps or float("inf"),
            ),
            volume_bytes=local_reduce.volume_bytes + inter_reduce.volume_bytes,
            latency_ms=local_reduce.latency_ms + inter_reduce.latency_ms,
            transfer_ms=local_reduce.transfer_ms + inter_reduce.transfer_ms,
        )

    def _estimate_dp_bucket_bytes(self, gradient_size_bytes: int) -> int:
        return int(
            min(
                self.MAX_DP_BUCKET_BYTES,
                max(self.DEFAULT_DP_BUCKET_BYTES, gradient_size_bytes / 32.0),
            )
        )

    def _simulate_overlap_exposed_tail(self,
                                       raw_comm_ms: float,
                                       bucket_count: int,
                                       stage_backward_ms: float,
                                       overlap_horizon_ms: float) -> float:
        if raw_comm_ms <= 0:
            return 0.0
        if bucket_count <= 1 or stage_backward_ms <= 0 or overlap_horizon_ms <= 0:
            return raw_comm_ms

        per_bucket_comm_ms = raw_comm_ms / bucket_count
        comm_stream_free_ms = 0.0
        finish_time_ms = 0.0
        for bucket_idx in range(bucket_count):
            ready_time_ms = stage_backward_ms * (bucket_idx + 1) / bucket_count
            start_time_ms = max(comm_stream_free_ms, ready_time_ms)
            finish_time_ms = start_time_ms + per_bucket_comm_ms
            comm_stream_free_ms = finish_time_ms

        return max(0.0, finish_time_ms - overlap_horizon_ms)

    def _estimate_dp_overlap(self,
                             model_config: ModelConfig,
                             training_config: TrainingConfig,
                             parallel: ParallelConfig,
                             dp_degree: int,
                             use_sharding: bool,
                             stage_backward_time_ms: Optional[List[float]]) -> Dict[str, float]:
        stage_param_counts = self._estimate_stage_parameter_counts_per_gpu(
            model_config, parallel
        )
        if not stage_param_counts:
            return {
                "raw_time_ms": 0.0,
                "exposed_time_ms": 0.0,
                "hidden_time_ms": 0.0,
                "peak_stage": 0,
                "stage_raw_time_ms": [],
                "stage_exposed_time_ms": [],
            }

        overlap_enabled = False
        if use_sharding:
            overlap_enabled = bool(
                training_config.stage1_overlap or
                training_config.enable_sharding_comm_overlap
            )
        else:
            overlap_enabled = dp_degree > 1

        if not stage_backward_time_ms or len(stage_backward_time_ms) != len(stage_param_counts):
            stage_backward_time_ms = [0.0] * len(stage_param_counts)

        stage_raw_times = []
        stage_exposed_times = []
        for stage_id, param_count in enumerate(stage_param_counts):
            gradient_size_bytes = max(
                1,
                int(param_count) * max(1, int(training_config.dtype_bytes)),
            )
            raw_result = self.predict_dp_comm(
                gradient_size_bytes, dp_degree, use_sharding
            )
            raw_time_ms = raw_result.time_ms
            stage_raw_times.append(raw_time_ms)

            if not overlap_enabled:
                stage_exposed_times.append(raw_time_ms)
                continue

            bucket_bytes = self._estimate_dp_bucket_bytes(gradient_size_bytes)
            bucket_count = max(1, math.ceil(gradient_size_bytes / max(1, bucket_bytes)))
            local_backward_ms = max(0.0, float(stage_backward_time_ms[stage_id]))
            overlap_horizon_ms = sum(
                max(0.0, float(value))
                for value in stage_backward_time_ms[:stage_id + 1]
            )
            exposed_tail_ms = self._simulate_overlap_exposed_tail(
                raw_time_ms,
                bucket_count,
                local_backward_ms,
                overlap_horizon_ms,
            )
            stage_exposed_times.append(exposed_tail_ms)

        raw_peak = max(stage_raw_times) if stage_raw_times else 0.0
        exposed_peak = max(stage_exposed_times) if stage_exposed_times else 0.0
        peak_stage = (
            max(range(len(stage_exposed_times)), key=stage_exposed_times.__getitem__)
            if stage_exposed_times else 0
        )
        return {
            "raw_time_ms": raw_peak,
            "exposed_time_ms": exposed_peak,
            "hidden_time_ms": max(0.0, raw_peak - exposed_peak),
            "peak_stage": peak_stage,
            "stage_raw_time_ms": stage_raw_times,
            "stage_exposed_time_ms": stage_exposed_times,
        }
    
    def _get_bandwidth(self, is_intra_node: bool,
                       data_size_bytes: int = 0) -> float:
        """获取有效带宽 (GB/s)，支持消息大小感知"""
        return self.hardware.network.get_effective_bandwidth(
            is_intra_node, data_size_bytes
        )
    
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
        
        bandwidth = self._get_bandwidth(is_intra_node, comm_volume)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.allreduce_efficiency
        
        # 延迟: 2 * (N-1) 步
        num_steps = 2 * (num_gpus - 1)
        latency_ms = num_steps * latency_us / 1000.0
        
        # 传输时间
        data_gb = self._bytes_to_gb(comm_volume)
        effective_bw = bandwidth * efficiency
        if effective_bw <= 0:
            effective_bw = 0.1  # 防止除零
        transfer_ms = data_gb / effective_bw * 1000.0
        
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
        
        bandwidth = self._get_bandwidth(is_intra_node, comm_volume)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.allgather_efficiency
        
        num_steps = num_gpus - 1
        latency_ms = num_steps * latency_us / 1000.0
        
        data_gb = self._bytes_to_gb(comm_volume)
        effective_bw = bandwidth * efficiency
        if effective_bw <= 0:
            effective_bw = 0.1  # 防止除零
        transfer_ms = data_gb / effective_bw * 1000.0
        
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
        
        bandwidth = self._get_bandwidth(is_intra_node, comm_volume)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.allreduce_efficiency
        
        num_steps = num_gpus - 1
        latency_ms = num_steps * latency_us / 1000.0
        
        data_gb = self._bytes_to_gb(comm_volume)
        effective_bw = bandwidth * efficiency
        if effective_bw <= 0:
            effective_bw = 0.1  # 防止除零
        transfer_ms = data_gb / effective_bw * 1000.0
        
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
        
        bandwidth = self._get_bandwidth(is_intra_node, comm_volume)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.alltoall_efficiency
        
        # AllToAll 延迟 (并行通信，只算一次启动延迟)
        latency_ms = latency_us / 1000.0
        
        # 传输时间 (全双工网络，所有发送并行)
        data_gb = self._bytes_to_gb(comm_volume)
        effective_bw = bandwidth * efficiency
        if effective_bw <= 0:
            effective_bw = 0.1  # 防止除零，使用最小带宽
        transfer_ms = data_gb / effective_bw * 1000.0
        
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
        bandwidth = self._get_bandwidth(is_intra_node, data_size_bytes)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.p2p_efficiency
        
        latency_ms = latency_us / 1000.0
        
        data_gb = self._bytes_to_gb(data_size_bytes)
        effective_bw = bandwidth * efficiency
        if effective_bw <= 0:
            effective_bw = 0.1  # 防止除零
        transfer_ms = data_gb / effective_bw * 1000.0
        
        total_time_ms = latency_ms + transfer_ms
        
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_gbps=data_gb / (total_time_ms / 1000.0) if total_time_ms > 0 else 0,
            volume_bytes=data_size_bytes,
            latency_ms=latency_ms,
            transfer_ms=transfer_ms,
        )

    def _is_pipeline_boundary_intra_node(self, boundary_idx: int, stage_width: int) -> bool:
        """
        判断相邻两个 PP stage 的边界是否完全位于同一节点内。

        近似假设单个 model-parallel group 按 rank 连续映射，stage 宽度约等于 TP 度数。
        只要该边界上的任一 TP lane 跨节点，就按节点间 P2P 处理。
        """
        gpus_per_node = max(1, int(self.hardware.gpus_per_node))
        width = max(1, int(stage_width))
        left_stage_base = max(0, int(boundary_idx)) * width
        right_stage_base = left_stage_base + width

        for lane in range(width):
            src_rank = left_stage_base + lane
            dst_rank = right_stage_base + lane
            if (src_rank // gpus_per_node) != (dst_rank // gpus_per_node):
                return False
        return True
    
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

        if use_sharding:
            return self._predict_hierarchical_reduce_scatter(
                gradient_size_bytes, dp_degree
            )
        return self._predict_hierarchical_allreduce(gradient_size_bytes, dp_degree)
    
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
                        num_micro_batches: int, stage_width: int = 1,
                        stage_forward_time_ms: Optional[List[float]] = None,
                        stage_backward_time_ms: Optional[List[float]] = None) -> CommResult:
        """
        预测 PP 通信时间

        1F1B 流水线调度中：
        - 稳态阶段：P2P 通信与计算高度重叠，暴露时间接近零
        - 启动阶段 (warm-up)：前 pp-1 个 micro-batch 的前向 P2P 无法重叠
        - 排空阶段 (drain)：最后 pp-1 个 micro-batch 的后向 P2P 无法重叠
        - 非重叠 P2P 次数 = 2 * (pp-1) 次（每次跨 1 个 stage 边界）
        """
        if pp_degree <= 1:
            return CommResult()

        boundary_results = []
        for boundary_idx in range(pp_degree - 1):
            is_intra_node = self._is_pipeline_boundary_intra_node(boundary_idx, stage_width)
            boundary_results.append(
                self.predict_p2p(activation_size_bytes, is_intra_node)
            )
        if not boundary_results:
            return CommResult()

        total_volume = activation_size_bytes * 2 * num_micro_batches * (pp_degree - 1)
        bottleneck_bw = min(r.bandwidth_gbps for r in boundary_results)
        total_latency_ms = 2.0 * num_micro_batches * sum(
            r.latency_ms for r in boundary_results
        )
        total_transfer_ms = 2.0 * num_micro_batches * sum(
            r.transfer_ms for r in boundary_results
        )

        if (
            stage_forward_time_ms is not None
            and stage_backward_time_ms is not None
            and len(stage_forward_time_ms) == pp_degree
            and len(stage_backward_time_ms) == pp_degree
        ):
            boundary_time_ms = [r.time_ms for r in boundary_results]
            compute_only_ms = simulate_1f1b_makespan(
                stage_forward_time_ms,
                stage_backward_time_ms,
                num_micro_batches,
            )
            with_p2p_ms = simulate_1f1b_makespan(
                stage_forward_time_ms,
                stage_backward_time_ms,
                num_micro_batches,
                boundary_time_ms,
                boundary_time_ms,
            )
            total_time_ms = max(0.0, with_p2p_ms - compute_only_ms)
        else:
            warmup_and_drain_time_ms = 2.0 * sum(r.time_ms for r in boundary_results)
            steady_micro_batches = max(0, num_micro_batches - (pp_degree - 1))
            avg_boundary_latency_ms = (
                sum(r.latency_ms for r in boundary_results) / len(boundary_results)
            )
            steady_exposed_per_mb = avg_boundary_latency_ms * 2
            total_time_ms = (
                warmup_and_drain_time_ms +
                steady_exposed_per_mb * steady_micro_batches
            )

        return CommResult(
            time_ms=total_time_ms,
            bandwidth_gbps=bottleneck_bw,
            volume_bytes=total_volume,
            latency_ms=total_latency_ms,
            transfer_ms=total_transfer_ms,
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
                                num_micro_batches: int,
                                stage_forward_time_ms: Optional[List[float]] = None,
                                stage_backward_time_ms: Optional[List[float]] = None) -> Dict[str, float]:
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
        
        tp_comm_result = self.predict_tp_comm(activation_size, parallel.tp)

        # ========== EP 通信 ==========
        # MoE 层的 dispatch + combine
        topk = model_config.num_experts_per_tok
        token_data_size = micro_bsz * seq_len * h * topk * dtype_bytes

        ep_comm_result = self.predict_ep_comm(
            token_data_size, parallel.ep,
            topk=topk, num_experts=model_config.num_experts
        )
        
        # ========== PP 通信 ==========
        pp_comm_result = self.predict_pp_comm(
            activation_size,
            parallel.pp,
            num_micro_batches,
            stage_width=max(1, parallel.tp),
            stage_forward_time_ms=stage_forward_time_ms,
            stage_backward_time_ms=stage_backward_time_ms,
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
        dp_overlap_result = self._estimate_dp_overlap(
            model_config,
            training_config,
            parallel,
            dp_degree,
            use_sharding,
            stage_backward_time_ms,
        )
        dp_comm_time = dp_overlap_result["raw_time_ms"]
        dp_exposed_time = dp_overlap_result["exposed_time_ms"]
        
        # ========== SP 通信 ==========
        sp_comm_time = 0.0
        stage_sp_comm_micro_ms = [0.0] * max(1, int(parallel.pp))
        if parallel.sp and parallel.tp > 1:
            sp_comm_result = self.predict_sp_comm(activation_size, parallel.tp)

        layer_comm_templates = {
            TRANSFORMER_DENSE_LAYER_KIND: {
                "tp": tp_comm_result.time_ms * 2,
                "ep": 0.0,
                "sp": (
                    sp_comm_result.time_ms
                    if parallel.sp and parallel.tp > 1
                    else 0.0
                ),
            },
            TRANSFORMER_MOE_LAYER_KIND: {
                "tp": tp_comm_result.time_ms * 2,
                "ep": ep_comm_result.time_ms,
                "sp": (
                    sp_comm_result.time_ms
                    if parallel.sp and parallel.tp > 1
                    else 0.0
                ),
            },
        }

        stage_tp_comm_micro_ms = []
        stage_ep_comm_micro_ms = []
        stage_sp_comm_micro_ms = []
        for stage_id in range(max(1, int(parallel.pp))):
            stage_tp_time = 0.0
            stage_ep_time = 0.0
            stage_sp_time = 0.0
            for layer_idx in self._stage_layer_indices(model_config, parallel, stage_id):
                template = layer_comm_templates[
                    self._transformer_layer_kind(model_config, layer_idx)
                ]
                stage_tp_time += template["tp"]
                stage_ep_time += template["ep"]
                stage_sp_time += template["sp"]
            stage_tp_comm_micro_ms.append(stage_tp_time)
            stage_ep_comm_micro_ms.append(stage_ep_time)
            stage_sp_comm_micro_ms.append(stage_sp_time)

        tp_comm_time = (
            max(stage_tp_comm_micro_ms) * num_micro_batches
            if stage_tp_comm_micro_ms else 0.0
        )
        ep_comm_time = (
            max(stage_ep_comm_micro_ms) * num_micro_batches
            if stage_ep_comm_micro_ms else 0.0
        )
        sp_comm_time = (
            max(stage_sp_comm_micro_ms) * num_micro_batches
            if stage_sp_comm_micro_ms else 0.0
        )

        return {
            "tp_comm_time_ms": tp_comm_time,
            "ep_comm_time_ms": ep_comm_time,
            "pp_comm_time_ms": pp_comm_time,
            "dp_comm_time_ms": dp_comm_time,
            "dp_exposed_comm_time_ms": dp_exposed_time,
            "dp_hidden_comm_time_ms": dp_overlap_result["hidden_time_ms"],
            "dp_overlap_peak_stage": dp_overlap_result["peak_stage"],
            "dp_stage_raw_comm_time_ms": dp_overlap_result.get("stage_raw_time_ms", []),
            "dp_stage_exposed_comm_time_ms": dp_overlap_result.get("stage_exposed_time_ms", []),
            "stage_tp_comm_micro_ms": stage_tp_comm_micro_ms,
            "stage_ep_comm_micro_ms": stage_ep_comm_micro_ms,
            "stage_sp_comm_micro_ms": stage_sp_comm_micro_ms,
            "sp_comm_time_ms": sp_comm_time,
            "total_comm_time_ms": tp_comm_time + ep_comm_time + pp_comm_time + dp_comm_time + sp_comm_time,
        }
