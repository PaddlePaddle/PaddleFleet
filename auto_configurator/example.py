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
Example usage of AutoConfigurator with PaddleFleet.

This script demonstrates how to use AutoConfigurator to generate
optimal training configurations for large model training in PaddleFleet.
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_configurator import (
    AutoConfigurator,
    PaddleFleetRecipe,
    generate_configs,
    get_results,
)
from paddlefleet.models.gpt.gpt_config import GPTConfig
from paddlefleet.transformer import TransformerConfig


def create_example_gpt_config() -> GPTConfig:
    """Create an example GPT model configuration.

    Returns:
        GPTConfig with example parameters
    """
    return GPTConfig(
        # Model architecture
        num_hidden_layers=24,
        hidden_size=2048,
        num_attention_heads=32,
        intermediate_size=8192,
        max_sequence_length=4096,
        vocab_size=32000,
        # Parallel settings
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        # Training settings
        fp16=True,
        bf16=False,
        params_dtype="float16",
    )


def create_example_bert_config() -> TransformerConfig:
    """Create an example BERT model configuration.

    Returns:
        TransformerConfig with example parameters
    """
    return TransformerConfig(
        # Model architecture
        num_hidden_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=30000,
        # Parallel settings
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        # Training settings
        fp16=True,
        bf16=False,
        params_dtype="float16",
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AutoConfigurator example for PaddleFleet"
    )

    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["gpt", "bert", "t5"],
        default="gpt",
        help="Model type to configure",
    )

    # Hardware
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="Number of compute nodes"
    )
    parser.add_argument(
        "--gpus_per_node", type=int, default=8, help="Number of GPUs per node"
    )
    parser.add_argument(
        "--gpu_memory",
        type=int,
        choices=[40, 80],
        default=80,
        help="GPU memory in GB",
    )

    # Parallel search
    parser.add_argument(
        "--tp",
        type=str,
        default="auto",
        help="Tensor parallel sizes (comma-separated or 'auto')",
    )
    parser.add_argument(
        "--pp",
        type=str,
        default="auto",
        help="Pipeline parallel sizes (comma-separated or 'auto')",
    )
    parser.add_argument(
        "--mbs",
        type=str,
        default="auto",
        help="Micro batch sizes (comma-separated or 'auto')",
    )
    parser.add_argument(
        "--cp",
        type=str,
        default="1",
        help="Context parallel sizes (comma-separated)",
    )
    parser.add_argument(
        "--ep",
        type=str,
        default="1",
        help="Expert parallel sizes (comma-separated)",
    )

    # Training
    parser.add_argument(
        "--seq_length",
        type=int,
        default=4096,
        choices=[2048, 4096, 8192, 16384, 32768],
        help="Sequence length",
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=2048, help="Global batch size"
    )
    parser.add_argument(
        "--max_steps_per_run",
        type=int,
        default=50,
        help="Max steps per grid search run",
    )
    parser.add_argument(
        "--max_training_days",
        type=float,
        default=2.0,
        help="Expected training days",
    )
    parser.add_argument(
        "--num_tokens_in_b",
        type=int,
        default=1400,
        help="Number of tokens in billions",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=32000, help="Vocabulary size"
    )

    # Model calculation
    parser.add_argument(
        "--calculate_model_size",
        action="store_true",
        help="Auto-calculate model architecture",
    )

    # Output
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./auto_config_logs",
        help="Directory for saving logs",
    )
    parser.add_argument(
        "--get_results",
        action="store_true",
        help="Generate results summary from logs",
    )
    parser.add_argument(
        "--run_number",
        type=int,
        default=1,
        help="Which configuration to run (1-indexed)",
    )

    return parser.parse_args()


def main():
    """Main entry point for AutoConfigurator example."""
    args = parse_args()

    # Parse parallel sizes
    tp_sizes = (
        [int(x) for x in args.tp.split(",")] if args.tp != "auto" else "auto"
    )
    pp_sizes = (
        [int(x) for x in args.pp.split(",")] if args.pp != "auto" else "auto"
    )
    mbs_sizes = (
        [int(x) for x in args.mbs.split(",")] if args.mbs != "auto" else "auto"
    )
    cp_sizes = [int(x) for x in args.cp.split(",")]
    ep_sizes = [int(x) for x in args.ep.split(",")]

    # Create model config based on type
    if args.model_type == "gpt":
        model_config = create_example_gpt_config()
        model_config.max_sequence_length = args.seq_length
    elif args.model_type == "bert":
        model_config = create_example_bert_config()
    else:
        raise ValueError(
            f"Model type {args.model_type} not yet implemented in example"
        )

    # Create recipe
    recipe = PaddleFleetRecipe(
        model_config=model_config,
        parallel_config=None,
        micro_batch_size=1,  # Will be set by AutoConfigurator
        global_batch_size=args.global_batch_size,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.gpus_per_node,
        max_steps=args.max_steps_per_run,
        log_dir=args.log_dir,
    )

    # Create AutoConfigurator
    runner = AutoConfigurator(
        recipe=recipe,
        path_to_logs=args.log_dir,
        mode="pretrain",
        gpu_memory_gb=args.gpu_memory,
        tensor_parallel_sizes=tp_sizes,
        pipeline_parallel_sizes=pp_sizes,
        micro_batch_sizes=mbs_sizes,
        context_parallel_sizes=cp_sizes,
        expert_parallel_sizes=ep_sizes,
        num_tokens_in_b=args.num_tokens_in_b,
        tflops_per_gpu=140,  # Default TFLOPS for A100
        max_minutes_per_run=30,
        max_training_days=args.max_training_days,
        max_steps_per_run=args.max_steps_per_run,
        vocab_size=args.vocab_size,
        calculate_model_size=args.calculate_model_size,
    )

    # Generate configurations
    base_config, configs = generate_configs(runner)

    print("\n" + "=" * 70)
    print("AutoConfigurator Configuration Generation Complete")
    print("=" * 70)
    print(f"Model Type: {runner.model_type}")
    print(f"Model Size: {runner.model_size_in_b:.2f}B")
    print(f"Sequence Length: {runner.seq_length}")
    print(f"Global Batch Size: {runner.global_batch_size}")
    print(
        f"GPU Configuration: {runner.gpu_count} GPUs ({args.num_nodes} nodes x {args.gpus_per_node})"
    )
    print(f"Generated {len(configs)} candidate configurations")
    print("=" * 70)

    # List configurations
    print("\nGenerated Configurations:")
    print("-" * 70)
    for i, (name, config) in enumerate(configs.items(), 1):
        print(f"\n{i}. {name}")
        print(
            f"   TP: {config.tensor_parallel_size}, PP: {config.pipeline_parallel_size}"
        )
        print(
            f"   CP: {config.context_parallel_size}, EP: {config.expert_parallel_size}"
        )
        print(f"   MBS: {config.micro_batch_size}")
        print(f"   Max Steps: {config.max_steps}")
        print(f"   Log Dir: {config.log_dir}")

    print("\n" + "-" * 70)

    # Generate results if requested
    if args.get_results:
        print("\nGenerating results summary from logs...")
        get_results(
            base_config=base_config,
            runner_config=runner,
            path_to_save=args.log_dir,
            output_top_n=10,
        )

    # Instructions for running configurations
    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("To run a specific configuration, use --run_number flag:")
    print(f"  python {__file__} --run_number <config_number>")
    print("\nExample:")
    print(f"  python {__file__} --run_number 1")
    print("=" * 70)


if __name__ == "__main__":
    main()
