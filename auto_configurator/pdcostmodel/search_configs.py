#!/usr/bin/env python3
"""
配置搜索脚本 - 搜索所有可运行的并行配置

搜索空间:
- TP: 1, 2, 4, 8
- PP: 1, 2, 4, 8
- DP: 自动计算 (total_gpus / tp / pp)
- EP: 1, 2, 4, 8, 16, 32, 64, 128
- Sharding: stage1, stage2
- MBS: 1, 2
- GAS: 8, 16, 32, 64
- seq_len: 2048, 4096, 8192

约束:
- tp * pp * dp == total_gpus
- ep <= num_experts (128)
- ep 整除 num_experts
- 显存 <= GPU 显存容量
- tensorwise_offload 需要 dp > 1
"""

import json
import sys
import os

# 添加 pdcost 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdcost import ModelConfig, PDCostModel, ParallelConfig
from pdcost.config import TrainingConfig, HardwareConfig, GPUSpec


def search_all_configs(
    model_config_path: str = 'Qwen3-30B-A3B-Base/config.json',
    total_gpus: int = 8,
    gpu_memory_gb: float = 79.6,
    output_file: str = 'all_runnable_configs.json'
):
    """
    搜索所有可运行配置
    
    Args:
        model_config_path: 模型配置文件路径
        total_gpus: GPU 总数
        gpu_memory_gb: 每卡显存容量 (GB)
        output_file: 输出文件路径
    """
    # 加载模型
    model = ModelConfig.from_json(model_config_path)
    hardware = HardwareConfig(
        gpu=GPUSpec(name='H800', memory_gb=gpu_memory_gb, bf16_tflops=788.0),
        num_nodes=1, 
        gpus_per_node=total_gpus
    )
    training = TrainingConfig(micro_batch_size=1, sequence_length=8192, dtype='bfloat16')
    costmodel = PDCostModel(model, hardware, training)
    
    # 搜索空间定义
    tp_values = [1, 2, 4, 8]
    pp_values = [1, 2, 4, 8]
    ep_values = [1, 2, 4, 8]
    sharding_values = ['stage1', 'stage2']
    mbs_values = [1, 2]
    gas_values = [8, 16, 32, 64]
    seq_len_values = [256, 512, 1024, 2048, 4096, 8192]
    
    num_experts = model.num_experts  # 128
    
    all_configs = []
    valid_configs = []
    invalid_configs = []
    
    total_combinations = 0
    
    print("=" * 100)
    print("🔍 配置搜索开始")
    print("=" * 100)
    print(f"\n搜索空间:")
    print(f"  TP: {tp_values}")
    print(f"  PP: {pp_values}")
    print(f"  EP: {ep_values}")
    print(f"  Sharding: {sharding_values}")
    print(f"  MBS: {mbs_values}")
    print(f"  GAS: {gas_values}")
    print(f"  seq_len: {seq_len_values}")
    
    # 遍历所有组合
    for tp in tp_values:
        for pp in pp_values:
            # 计算 DP
            if total_gpus % (tp * pp) != 0:
                continue
            dp = total_gpus // (tp * pp)
            if dp < 1:
                continue
            
            for ep in ep_values:
                # EP 约束
                # 1. EP 不能超过专家总数
                if ep > num_experts:
                    continue
                # 2. EP 需要整除专家数
                if num_experts % ep != 0:
                    continue
                # 3. EP 不能超过实际 GPU 数量 (关键约束!)
                #    专家只能分布到实际存在的 GPU 上
                if ep > total_gpus:
                    continue
                    
                for sharding in sharding_values:
                    for mbs in mbs_values:
                        for gas in gas_values:
                            for seq_len in seq_len_values:
                                total_combinations += 1
                                
                                config = {
                                    'tp': tp,
                                    'pp': pp,
                                    'dp': dp,
                                    'ep': ep,
                                    'sharding': sharding,
                                    'micro_batch_size': mbs,
                                    'gradient_accumulation_steps': gas,
                                    'seq_len': seq_len,
                                }
                                
                                # tensorwise_offload 需要 dp > 1
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
                                    
                                    mb = result.memory_breakdown
                                    
                                    config.update({
                                        'use_offload': use_offload,
                                        'step_time_s': round(result.step_time_ms / 1000, 2),
                                        'tokens_per_second_per_gpu': round(result.tokens_per_second_per_gpu, 0),
                                        'tokens_per_step': result.tokens_per_step,
                                        'global_batch_size': dp * mbs * gas,
                                        'allocated_memory_gb': round(mb.allocated_memory_gb, 2),
                                        'reserved_memory_gb': round(mb.reserved_memory_gb, 2),
                                        'mfu': round(result.mfu * 100, 2),
                                        'fits_memory': result.fits_memory,
                                    })
                                    
                                    all_configs.append(config)
                                    
                                    if result.fits_memory:
                                        valid_configs.append(config)
                                    else:
                                        config['reject_reason'] = 'OOM'
                                        invalid_configs.append(config)
                                        
                                except Exception as e:
                                    config['reject_reason'] = str(e)
                                    invalid_configs.append(config)
    
    # 按吞吐量降序排序
    valid_configs.sort(key=lambda x: x['tokens_per_second_per_gpu'], reverse=True)
    
    # 添加排名
    for i, cfg in enumerate(valid_configs):
        cfg['rank'] = i + 1
    
    print(f"\n搜索结果:")
    print(f"  总组合数: {total_combinations}")
    print(f"  可运行配置: {len(valid_configs)}")
    print(f"  不可运行配置: {len(invalid_configs)}")
    
    # 保存结果
    output = {
        'search_params': {
            'total_gpus': total_gpus,
            'gpu_memory_gb': gpu_memory_gb,
            'model': model_config_path,
            'tp_values': tp_values,
            'pp_values': pp_values,
            'ep_values': ep_values,
            'sharding_values': sharding_values,
            'mbs_values': mbs_values,
            'gas_values': gas_values,
            'seq_len_values': seq_len_values,
        },
        'summary': {
            'total_combinations': total_combinations,
            'valid_count': len(valid_configs),
            'invalid_count': len(invalid_configs),
        },
        'valid_configs': valid_configs,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 打印 Top 10
    print("\n" + "=" * 100)
    print("🏆 Top 10 最优配置 (吞吐量最高)")
    print("=" * 100)
    print(f"{'排名':<6} {'tok/s/GPU':<12} {'step(s)':<10} {'alloc(GB)':<12} {'reserv(GB)':<12} "
          f"{'tp':<4} {'pp':<4} {'dp':<4} {'ep':<4} {'sharding':<8} {'mbs':<4} {'gas':<4} {'seq':<6}")
    print("-" * 100)
    
    for cfg in valid_configs[:10]:
        print(f"#{cfg['rank']:<5} {cfg['tokens_per_second_per_gpu']:<12.0f} {cfg['step_time_s']:<10.2f} "
              f"{cfg['allocated_memory_gb']:<12.2f} {cfg['reserved_memory_gb']:<12.2f} "
              f"{cfg['tp']:<4} {cfg['pp']:<4} {cfg['dp']:<4} {cfg['ep']:<4} {cfg['sharding']:<8} "
              f"{cfg['micro_batch_size']:<4} {cfg['gradient_accumulation_steps']:<4} {cfg['seq_len']:<6}")
    
    # 打印 Bottom 10
    print("\n" + "=" * 100)
    print("😢 Bottom 10 最差配置 (吞吐量最低)")
    print("=" * 100)
    print(f"{'排名':<6} {'tok/s/GPU':<12} {'step(s)':<10} {'alloc(GB)':<12} {'reserv(GB)':<12} "
          f"{'tp':<4} {'pp':<4} {'dp':<4} {'ep':<4} {'sharding':<8} {'mbs':<4} {'gas':<4} {'seq':<6}")
    print("-" * 100)
    
    for cfg in valid_configs[-10:]:
        print(f"#{cfg['rank']:<5} {cfg['tokens_per_second_per_gpu']:<12.0f} {cfg['step_time_s']:<10.2f} "
              f"{cfg['allocated_memory_gb']:<12.2f} {cfg['reserved_memory_gb']:<12.2f} "
              f"{cfg['tp']:<4} {cfg['pp']:<4} {cfg['dp']:<4} {cfg['ep']:<4} {cfg['sharding']:<8} "
              f"{cfg['micro_batch_size']:<4} {cfg['gradient_accumulation_steps']:<4} {cfg['seq_len']:<6}")
    
    return valid_configs


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='搜索所有可运行的并行配置')
    parser.add_argument('--model', type=str, default='Qwen3-30B-A3B-Base/config.json',
                        help='模型配置文件路径')
    parser.add_argument('--gpus', type=int, default=8, help='GPU 总数')
    parser.add_argument('--memory', type=float, default=79.6, help='每卡显存 (GB)')
    parser.add_argument('--output', type=str, default='all_runnable_configs.json',
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    search_all_configs(
        model_config_path=args.model,
        total_gpus=args.gpus,
        gpu_memory_gb=args.memory,
        output_file=args.output
    )