#!/usr/bin/env python3
"""搜索示例：使用 GridSearch 生成搜索空间 + PDCostModel 进行性能预测"""

import logging
from core.grid_search import GPTGridSearch, GridSearchConfig
from pdcostmodel import ModelConfig, PDCostModel, ParallelConfig, get_hardware_config

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def search(model_name: str = "qwen3-30b-a3b", total_gpus: int = 8):
    """
    使用 GridSearch 生成搜索空间，PDCostModel 进行性能预测
    
    Args:
        model_name: 预设模型名称 (如 "qwen3-30b-a3b") 或 config.json 路径
        total_gpus: 总 GPU 数量
    """
    # 1. 加载模型配置和硬件
    if model_name.endswith('.json'):
        model = ModelConfig.from_json(model_name)
    else:
        model = ModelConfig.from_name(model_name)
    hardware = get_hardware_config(verbose=False)
    costmodel = PDCostModel(model, hardware)
    
    # 2. 使用 GPTGridSearch 生成搜索空间
    model_size_b = model.estimate_parameters()["total"] / 1e9
    grid_search = GPTGridSearch(
        model_size_in_b=model_size_b,
        valid_pp=[1, 2, 4],
        seq_length=8192,
        gpu_memory_gb=int(hardware.gpu.memory_gb),
    )
    grid_search.init_params()
    
    # 3. 构建搜索空间
    search_space = GridSearchConfig(
        tp=grid_search.tp,
        pp=grid_search.pp,
        cp=grid_search.cp,
        ep=[1, 2, 4, 8] if model.num_experts > 1 else [1],
        mbs=grid_search.mbs,
        gbs=grid_search.gbs,
        min_model_parallel=grid_search.min_model_parallel,
        max_model_parallel=grid_search.max_model_parallel,
        sharding=grid_search.sharding,
        gas=grid_search.gas,
        seq_len=grid_search.seq_len_list,
    )
    
 
    # 4. 遍历搜索空间，使用 PDCostModel 预测
    results = []
    for tp in search_space.tp:
        for pp in search_space.pp:
            if tp * pp > total_gpus:
                continue
            dp = total_gpus // (tp * pp)
            if dp < 1 or tp * pp * dp != total_gpus:
                continue
            
            for ep in search_space.ep:
                # EP 必须 <= DP
                if ep > dp:
                    continue
                
                for sharding in search_space.sharding:
                    for mbs in search_space.mbs:
                        for gas in search_space.gas:
                            for seq_len in search_space.seq_len:
                                try:
                                    # 构建并行配置
                                    parallel = ParallelConfig(
                                        tp=tp, pp=pp, dp=dp, ep=ep, sharding=sharding
                                    )
                                    
                                    # 使用 PDCostModel 预测性能
                                    result = costmodel.predict_calibrated(
                                        parallel,
                                        micro_batch_size=mbs,
                                        seq_len=seq_len,
                                        gradient_accumulation_steps=gas,
                                        tensorwise_offload_optimizer=True,
                                        tensorwise_offload_ratio=0.95,
                                    )
                                    
                                    # 过滤显存不足的配置
                                    if not result.fits_memory:
                                        continue
                                    
                                    results.append({
                                        'tp': tp, 'pp': pp, 'dp': dp, 'ep': ep,
                                        'sharding': sharding, 'mbs': mbs, 'gas': gas,
                                        'seq_len': seq_len,
                                        'tps': result.tokens_per_second_per_gpu,
                                        'step_time_s': result.step_time_ms / 1000,
                                        'memory_gb': result.memory_gb,
                                        'mfu': result.mfu,
                                    })
                                except Exception:
                                    continue
    
    # 5. 按吞吐量排序
    results.sort(key=lambda x: -x['tps'])
    
    return results[0] if results else None


if __name__ == "__main__":
    result = search("qwen3-30b-a3b")
    if result:
        print(f"\n最佳配置: tp={result['tp']}, pp={result['pp']}, dp={result['dp']}, ep={result['ep']}, sharding={result['sharding']}")
        print(f"seq_len={result['seq_len']}, mbs={result['mbs']}, gas={result['gas']}")
        print(f"吞吐量: {result['tps']:.0f} tok/s/GPU")
        print(f"步长时间: {result['step_time_s']:.2f} s/step")
        print(f"显存占用: {result['memory_gb']:.2f} GB")
        print(f"MFU: {result['mfu']:.2f}")
    else:
        print("\n没有找到合适的配置.")