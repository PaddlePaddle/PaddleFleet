#!/usr/bin/env python3
"""
配置搜索脚本 - 搜索所有可运行的并行配置，输出最佳配置
"""

import json
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# 添加 pdcostmodel 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pdcostmodel import ModelConfig, PDCostModel, ParallelConfig
from pdcostmodel.config import TrainingConfig
from pdcostmodel.calibration import get_hardware_config

# 日志目录
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(log_file: str = None) -> logging.Logger:
    """配置日志"""
    logger = logging.getLogger("search_configs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 文件 handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"search_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file


def search_best_config(
    model_config_path: str = 'Qwen3-30B-A3B-Base/config.json',
    total_gpus: int = None,
    node_count: int = 1,
    tp_values: list = None,
    pp_values: list = None,
    ep_values: list = None,
    sharding_values: list = None,
    mbs_values: list = None,
    gas_values: list = None,
    seq_len_values: list = None,
    force_calibrate: bool = False,
    log_file: str = None,
):
    """
    搜索最佳配置
    
    Returns:
        best_config: 最佳配置字典
    """
    logger, log_path = setup_logger(log_file)
    
    # 加载模型
    model = ModelConfig.from_json(model_config_path)
    
    # 加载硬件校准数据
    hardware = get_hardware_config(
        node_count=node_count,
        force_calibrate=force_calibrate,
        verbose=False
    )
    
    # 记录硬件信息
    logger.info("=" * 60)
    logger.info("硬件信息")
    logger.info("=" * 60)
    logger.info(f"GPU: {hardware.gpu.name}")
    logger.info(f"GPU 数量: {hardware.gpus_per_node}")
    logger.info(f"显存: {hardware.gpu.memory_gb:.1f} GB")
    logger.info(f"BF16 峰值: {hardware.gpu.bf16_tflops:.1f} TFLOPS")
    logger.info(f"显存带宽: {hardware.gpu.memory_bandwidth_gbps:.1f} GB/s")
    logger.info(f"节点数: {node_count}")
    
    # 从校准数据获取 GPU 数量
    if total_gpus is None:
        total_gpus = hardware.gpus_per_node * node_count
    logger.info(f"总 GPU 数: {total_gpus}")
    
    training = TrainingConfig(micro_batch_size=1, sequence_length=8192, dtype='bfloat16')
    costmodel = PDCostModel(model, hardware, training)
    
    # 搜索空间
    tp_values = tp_values or [1, 2, 4, 8]
    pp_values = pp_values or [1, 2, 4, 8]
    ep_values = ep_values or [1, 2, 4, 8]
    sharding_values = sharding_values or ['stage1', 'stage2']
    mbs_values = mbs_values or [1, 2]
    gas_values = gas_values or [8, 16, 32, 64]
    seq_len_values = seq_len_values or [256, 512, 1024, 2048, 4096, 8192]
    
    # 记录搜索空间
    logger.info("")
    logger.info("=" * 60)
    logger.info("搜索空间")
    logger.info("=" * 60)
    logger.info(f"TP: {tp_values}")
    logger.info(f"PP: {pp_values}")
    logger.info(f"EP: {ep_values}")
    logger.info(f"Sharding: {sharding_values}")
    logger.info(f"MBS: {mbs_values}")
    logger.info(f"GAS: {gas_values}")
    logger.info(f"seq_len: {seq_len_values}")
    
    num_experts = model.num_experts
    best_config = None
    best_throughput = 0
    total_combinations = 0
    valid_count = 0
    
    # 遍历所有组合
    for tp in tp_values:
        for pp in pp_values:
            if total_gpus % (tp * pp) != 0:
                continue
            dp = total_gpus // (tp * pp)
            if dp < 1:
                continue
            
            for ep in ep_values:
                if ep > num_experts or num_experts % ep != 0 or ep > total_gpus:
                    continue
                    
                for sharding in sharding_values:
                    for mbs in mbs_values:
                        for gas in gas_values:
                            for seq_len in seq_len_values:
                                total_combinations += 1
                                use_offload = dp > 1
                                
                                try:
                                    parallel = ParallelConfig(
                                        tp=tp, pp=pp, dp=dp, ep=ep, 
                                        sharding=sharding
                                    )
                                    
                                    result = costmodel.predict_calibrated(
                                        parallel,
                                        seq_len=seq_len,
                                        micro_batch_size=mbs,
                                        gradient_accumulation_steps=gas,
                                        tensorwise_offload_optimizer=use_offload,
                                        tensorwise_offload_ratio=0.95
                                    )
                                    
                                    if result.fits_memory:
                                        valid_count += 1
                                        if result.tokens_per_second_per_gpu > best_throughput:
                                            best_throughput = result.tokens_per_second_per_gpu
                                            mb = result.memory_breakdown
                                            best_config = {
                                                'tp': tp,
                                                'pp': pp,
                                                'dp': dp,
                                                'ep': ep,
                                                'sharding': sharding,
                                                'micro_batch_size': mbs,
                                                'gradient_accumulation_steps': gas,
                                                'seq_len': seq_len,
                                                'use_offload': use_offload,
                                                'step_time_s': round(result.step_time_ms / 1000, 2),
                                                'tokens_per_second_per_gpu': round(result.tokens_per_second_per_gpu, 0),
                                                'tokens_per_step': result.tokens_per_step,
                                                'global_batch_size': dp * mbs * gas,
                                                'allocated_memory_gb': round(mb.allocated_memory_gb, 2),
                                                'reserved_memory_gb': round(mb.reserved_memory_gb, 2),
                                                'mfu': round(result.mfu * 100, 2),
                                            }
                                except Exception:
                                    pass
    
    # 记录搜索结果
    logger.info("")
    logger.info("=" * 60)
    logger.info("搜索结果")
    logger.info("=" * 60)
    logger.info(f"总组合数: {total_combinations}")
    logger.info(f"可运行配置数: {valid_count}")
    
    # 记录最佳配置
    if best_config:
        logger.info("")
        logger.info("=" * 60)
        logger.info("最佳配置")
        logger.info("=" * 60)
        logger.info(f"TP: {best_config['tp']}")
        logger.info(f"PP: {best_config['pp']}")
        logger.info(f"DP: {best_config['dp']}")
        logger.info(f"EP: {best_config['ep']}")
        logger.info(f"Sharding: {best_config['sharding']}")
        logger.info(f"MBS: {best_config['micro_batch_size']}")
        logger.info(f"GAS: {best_config['gradient_accumulation_steps']}")
        logger.info(f"seq_len: {best_config['seq_len']}")
        logger.info(f"use_offload: {best_config['use_offload']}")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("预测结果")
        logger.info("=" * 60)
        logger.info(f"吞吐量: {best_config['tokens_per_second_per_gpu']:.0f} tok/s/GPU")
        logger.info(f"Step 时间: {best_config['step_time_s']:.2f} s")
        logger.info(f"Global Batch Size: {best_config['global_batch_size']}")
        logger.info(f"分配显存: {best_config['allocated_memory_gb']:.2f} GB")
        logger.info(f"预留显存: {best_config['reserved_memory_gb']:.2f} GB")
        logger.info(f"MFU: {best_config['mfu']:.2f}%")
    else:
        logger.info("未找到可运行的配置")
    
    logger.info("")
    logger.info(f"日志已保存: {log_path}")
    
    return best_config


if __name__ == '__main__':
    import argparse
    
    def parse_int_list(s):
        if not s:
            return None
        return [int(x.strip()) for x in s.split(',')]
    
    def parse_str_list(s):
        if not s:
            return None
        return [x.strip() for x in s.split(',')]
    
    parser = argparse.ArgumentParser(description='搜索最佳并行配置')
    parser.add_argument('--model', type=str, default='Qwen3-30B-A3B-Base/config.json',
                        help='模型配置文件路径')
    parser.add_argument('--gpus', type=int, default=None, help='GPU 总数 (默认自动检测)')
    parser.add_argument('--nodes', type=int, default=1, help='节点数')
    parser.add_argument('--force-calibrate', action='store_true', help='强制重新校准')
    parser.add_argument('--log', type=str, default=None, help='日志文件路径')
    
    # 搜索空间参数
    parser.add_argument('--tp', type=parse_int_list, default=None, help='TP 搜索空间')
    parser.add_argument('--pp', type=parse_int_list, default=None, help='PP 搜索空间')
    parser.add_argument('--ep', type=parse_int_list, default=None, help='EP 搜索空间')
    parser.add_argument('--sharding', type=parse_str_list, default=None, help='Sharding 搜索空间')
    parser.add_argument('--mbs', type=parse_int_list, default=None, help='MBS 搜索空间')
    parser.add_argument('--gas', type=parse_int_list, default=None, help='GAS 搜索空间')
    parser.add_argument('--seq-len', type=parse_int_list, default=None, help='序列长度搜索空间')
    
    args = parser.parse_args()
    
    best = search_best_config(
        model_config_path=args.model,
        total_gpus=args.gpus,
        node_count=args.nodes,
        tp_values=args.tp,
        pp_values=args.pp,
        ep_values=args.ep,
        sharding_values=args.sharding,
        mbs_values=args.mbs,
        gas_values=args.gas,
        seq_len_values=args.seq_len,
        force_calibrate=args.force_calibrate,
        log_file=args.log,
    )