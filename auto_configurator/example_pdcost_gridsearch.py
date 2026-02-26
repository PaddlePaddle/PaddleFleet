#!/usr/bin/env python3
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
示例: 使用 PDCost Grid Search 接口进行并行配置搜索

本示例展示如何使用 pdcostmodel 的 Grid Search 接口:
1. 使用 grid_search() 快速搜索 - 一行代码完成
2. 使用 GridSearcher 类 - 更细致的控制
3. 保存搜索结果到 JSON
4. 生成训练配置 YAML 文件

使用方法:
    python example_pdcost_gridsearch.py
"""

import sys
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入 pdcostmodel Grid Search 接口
from pdcostmodel import (
    grid_search,
    GridSearcher,
    GridSearchResult,
    ModelConfig,
)


def demo_quick_search():
    """
    演示 1: 使用 grid_search() 快速搜索
    
    最简单的使用方式，一行代码完成搜索
    """
    print("=" * 80)
    print("演示 1: 使用 grid_search() 快速搜索")
    print("=" * 80)
    
    # 一行代码搜索最优配置
    results = grid_search(
        model="qwen3-30b-a3b",  # 模型名称
        total_gpus=8,           # 总 GPU 数
        seq_len=8192,           # 序列长度
        micro_batch_size=1,     # 每卡 batch size
        top_k=5,                # 显示前 5 个配置
    )
    
    # 获取最优配置
    if results.best:
        print(f"\n✅ 最优配置: {results.best.config_str}")
        print(f"   吞吐量: {results.best.tokens_per_second_per_gpu:.0f} tok/s/GPU")
        print(f"   显存: {results.best.memory_gb:.2f} GB")
    
    return results


def demo_gridsearcher_class():
    """
    演示 2: 使用 GridSearcher 类进行更细致的控制
    
    适合需要自定义搜索空间、多次搜索的场景
    """
    print("\n" + "=" * 80)
    print("演示 2: 使用 GridSearcher 类")
    print("=" * 80)
    
    # 创建搜索器，自定义搜索空间
    searcher = GridSearcher(
        model_name="qwen3-30b-a3b",
        total_gpus=8,
        tp_candidates=[1, 2, 4, 8],
        pp_candidates=[1, 2, 4],
        ep_candidates=[1, 2, 4, 8],
        sharding_candidates=["stage1", "stage2"],
        verbose=True,
    )
    
    # 执行搜索
    print("\n🔍 执行搜索...")
    results = searcher.search(
        micro_batch_size=1,
        seq_len=8192,
        gradient_accumulation_steps=16,
        recompute_granularity="full",
        top_k=10,
        print_report=True,
    )
    
    return searcher, results


def demo_save_results(results: GridSearchResult):
    """
    演示 3: 保存搜索结果到 JSON
    """
    print("\n" + "=" * 80)
    print("演示 3: 保存搜索结果")
    print("=" * 80)
    
    # 保存到 JSON 文件
    output_path = "search_results.json"
    results.save_json(output_path)
    print(f"✅ 搜索结果已保存到: {output_path}")
    
    # 也可以获取字典格式
    result_dict = results.to_dict()
    print(f"   总配置数: {result_dict['total_configs']}")
    print(f"   有效配置数: {result_dict['valid_configs']}")


def demo_generate_yaml(searcher: GridSearcher):
    """
    演示 4: 生成训练配置 YAML 文件
    """
    print("\n" + "=" * 80)
    print("演示 4: 生成训练配置 YAML")
    print("=" * 80)
    
    # 生成 YAML 配置文件
    yaml_content = searcher.generate_yaml_config(
        output_path="best_config.yaml"
    )
    
    print("\n📄 生成的 YAML 配置:")
    print("-" * 40)
    # 只打印前 20 行
    lines = yaml_content.strip().split('\n')
    for line in lines[:20]:
        print(f"   {line}")
    if len(lines) > 20:
        print(f"   ... (共 {len(lines)} 行)")


def demo_custom_model():
    """
    演示 5: 使用自定义模型配置
    """
    print("\n" + "=" * 80)
    print("演示 5: 使用自定义模型配置")
    print("=" * 80)
    
    # 创建自定义模型配置 (类似 LLaMA 8B)
    custom_model = ModelConfig(
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_experts=1,  # Dense 模型
        vocab_size=128256,
    )
    
    print(f"   自定义 Dense 模型配置:")
    print(f"   层数: {custom_model.num_hidden_layers}")
    print(f"   Hidden Size: {custom_model.hidden_size}")
    print(f"   专家数: {custom_model.num_experts} (Dense)")
    
    # 使用自定义模型搜索
    results = grid_search(
        model=custom_model,
        total_gpus=8,
        seq_len=4096,
        top_k=3,
        print_report=True,
        verbose=False,
    )
    
    return results


def demo_compare_seq_lengths():
    """
    演示 6: 比较不同序列长度的最优配置
    """
    print("\n" + "=" * 80)
    print("演示 6: 比较不同序列长度")
    print("=" * 80)
    
    searcher = GridSearcher(
        model_name="qwen3-30b-a3b",
        total_gpus=8,
        verbose=False,
    )
    
    seq_lengths = [2048, 4096, 8192]
    
    print(f"\n{'序列长度':<12} {'最优配置':<35} {'吞吐量':<15} {'显存':<10}")
    print("-" * 75)
    
    for seq_len in seq_lengths:
        results = searcher.search(
            seq_len=seq_len,
            print_report=False,
        )
        if results.best:
            print(
                f"{seq_len:<12} {results.best.config_str:<35} "
                f"{results.best.tokens_per_second_per_gpu:<15,.0f} "
                f"{results.best.memory_gb:<10.2f}"
            )


def main():
    """主函数 - 运行所有演示"""
    print("=" * 80)
    print("PDCost Grid Search 接口演示")
    print("=" * 80)
    
    # 演示 1: 快速搜索
    results = demo_quick_search()
    
    # 演示 2: GridSearcher 类
    searcher, results = demo_gridsearcher_class()
    
    # 演示 3: 保存结果
    demo_save_results(results)
    
    # 演示 4: 生成 YAML
    demo_generate_yaml(searcher)
    
    # 演示 5: 自定义模型
    demo_custom_model()
    
    # 演示 6: 比较序列长度
    demo_compare_seq_lengths()
    
    print("\n" + "=" * 80)
    print("✅ 所有演示完成！")
    print("=" * 80)
    
    print("\n📚 API 快速参考:")
    print("-" * 40)
    print("   from pdcostmodel import grid_search, GridSearcher")
    print("")
    print("   # 快速搜索")
    print("   results = grid_search('qwen3-30b-a3b', total_gpus=8)")
    print("")
    print("   # 使用搜索器类")
    print("   searcher = GridSearcher(model_name='qwen3-30b-a3b', total_gpus=8)")
    print("   results = searcher.search(seq_len=8192)")
    print("   searcher.generate_yaml_config(output_path='config.yaml')")


if __name__ == "__main__":
    main()