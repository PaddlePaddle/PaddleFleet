# AutoConfigurator for PaddleFleet

AutoConfigurator migrated from NVIDIA NeMo to PaddlePaddle's PaddleFleet distributed training framework, optimized for NVIDIA H100 80GB GPUs.

## Overview

AutoConfigurator provides automatic configuration generation for large model training, including:

- **Model architecture inference**: Automatically infer optimal model parameters (hidden_size, attention_heads, etc.) based on model size
- **Grid search for parallel strategies**: Generate candidate configurations with different TP/PP/CP/EP/MBS combinations
- **Configuration validation**: Ensure all configurations are valid for the given hardware constraints
- **Performance analysis**: Calculate TFLOPS and rank configurations by training efficiency

## Installation

The AutoConfigurator is located at `<PaddleFleetRoot>/auto_configurator/`

### Directory Structure

```
auto_configurator/
├── __init__.py                   # Public API exports
├── autoconfigurator.py           # AutoConfigurator class + generate_configs API
├── paddlefleet_adapters.py       # Adapters to bridge with PaddleFleet's config system
├── main.py                       # CLI main entry (aligned with NeMo auto_config.py)
├── auto_search.py                # 自动搜索最优并行策略并 benchmark (支持 --base_yaml 指定任意模型)
├── run_top3_qwen30b.py           # Qwen3-30B-A3B Top-3 运行脚本
├── README.md                     # 本文档
├── CHANGELOG.md                  # 变更记录
├── ARCHIVE.md                    # 项目存档概览
├── core/
│   ├── __init__.py               # Core module exports
│   ├── model_size.py             # Model size calculation and architecture inference
│   ├── grid_search.py            # Grid search generation for parallel strategies
│   ├── performance.py            # TFLOPS calculation formulas
│   ├── log_parser.py             # Training log parser (core parsing logic)
│   └── results.py                # Results aggregation and TFLOPS summary
├── utils/
│   ├── __init__.py               # Utility module exports
│   ├── model_presets.py          # Model preset configurations (12 unique + 9 aliases)
│   ├── cli_args.py               # Command-line argument parser
│   ├── training_runner.py        # Training command builder and runner
│   ├── results_formatter.py      # Results display and CSV export
│   └── qwen3_moe_30b.yaml       # Qwen3-MoE-30B 基础配置 YAML
└── tests/
    ├── __init__.py               # Test package init
    ├── test_adapters.py          # Adapter unit tests (50 tests)
    ├── test_grid_search.py       # Grid search unit tests (45 tests)
    ├── test_model_size.py        # Model size unit tests (23 tests)
    ├── test_performance.py       # TFLOPS unit tests (10 tests)
    ├── test_integration.py       # Mock integration tests (32 tests)
    └── test_e2e_integration.py   # Real GPTConfig E2E integration tests (42 tests)
```

## Quick Start

### 1. Basic Usage

```python
import paddle
from paddlefleet.transformer import TransformerConfig
from paddlefleet.models.gpt.gpt_config import GPTConfig
from auto_configurator import (
    AutoConfigurator,
    PaddleFleetRecipe,
    generate_configs,
)

# Create model configuration
model_config = GPTConfig(
    num_hidden_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    intermediate_size=16384,
    max_sequence_length=4096,
    vocab_size=32000,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)

# Create recipe
recipe = PaddleFleetRecipe(
    model_config=model_config,
    micro_batch_size=4,
    global_batch_size=2048,
    num_nodes=4,
    num_gpus_per_node=8,
)

# Create AutoConfigurator
runner = AutoConfigurator(
    recipe=recipe,
    path_to_logs="./logs",
    gpu_memory_gb=80,
    tensor_parallel_sizes="auto",
    pipeline_parallel_sizes="auto",
    micro_batch_sizes=[1, 2, 4, 8],
    max_steps_per_run=100,
    max_training_days=7,
    num_tokens_in_b=1400,
    vocab_size=32000,
)

# Generate configurations
base_config, configs = generate_configs(runner)
print(f"Generated {len(configs)} candidate configurations")
```

### 2. Using the CLI Script

The `main.py` script is aligned with NeMo's AutoConfigurator design. It supports multiple model types with presets and custom configurations.

```bash
# View help
python main.py --help

# List available model presets
python main.py --list_presets

# Use preset model (LLaMA-2 7B) - generate only
python main.py --model_type llama --model_size 7b

# Run benchmark with preset model
python main.py \
    --model_type llama \
    --model_size 7b \
    --batch_mode

# MoE model (Mixtral 8x7B)
python main.py \
    --model_type mixtral \
    --model_size 8x7b \
    --moe \
    --batch_mode

# Custom model parameters
python main.py \
    --num_layers 32 \
    --hidden_size 4096 \
    --num_heads 32 \
    --num_kv_heads 8 \
    --seq_length 8192 \
    --vocab_size 128256 \
    --tensor_parallel_sizes 1,2,4 \
    --pipeline_parallel_sizes 1,2 \
    --batch_mode

# Run single configuration
python main.py \
    --model_type llama \
    --model_size 7b \
    --run_number 1

# Collect results from existing logs
python main.py \
    --model_type llama \
    --model_size 7b \
    --get_results

# Dry run (print commands only)
python main.py \
    --model_type llama \
    --model_size 7b \
    --batch_mode \
    --dry_run
```

**Supported Model Presets:**

| Preset | Layers | Hidden | Heads | Seq Len | MoE |
|---------|--------|--------|--------|---------|-----|
| `gpt3_175b` | 96 | 12288 | 96 | 2048 | No |
| `llama2_7b` | 32 | 4096 | 32 | 4096 | No |
| `llama2_70b` | 80 | 8192 | 64 | 4096 | No |
| `llama3_8b` | 32 | 4096 | 32 | 8192 | No |
| `llama3_70b` | 80 | 8192 | 64 | 8192 | No |
| `qwen2_7b` | 28 | 3584 | 28 | 32768 | No |
| `qwen2_72b` | 80 | 8192 | 64 | 32768 | No |
| `qwen3_moe_30b` | 48 | 2048 | 32 | 8192 | Yes |
| `mixtral_8x7b` | 32 | 4096 | 32 | 32768 | Yes |
| `mixtral_8x22b` | 56 | 6144 | 48 | 32768 | Yes |
| `gemma2_9b` | 42 | 3584 | 16 | 8192 | No |
| `gemma2_27b` | 46 | 4608 | 32 | 8192 | No |

**Preset Aliases:** For convenience, shorter alias names are also supported:

| Alias | Maps to |
|-------|---------|
| `llama_7b` | `llama2_7b` |
| `llama_70b` | `llama2_70b` |
| `qwen_7b` | `qwen2_7b` |
| `qwen_72b` | `qwen2_72b` |
| `qwen3_30b` | `qwen3_moe_30b` |
| `mixtral_7b` | `mixtral_8x7b` |
| `mixtral_22b` | `mixtral_8x22b` |
| `gemma_9b` | `gemma2_9b` |
| `gemma_27b` | `gemma2_27b` |

## API Reference

### AutoConfigurator

Main class for configuration generation.

```python
@dataclass
class AutoConfigurator:
    recipe: PaddleFleetRecipe           # Training recipe with model and hardware configs
    path_to_logs: str                    # Directory for saving logs

    # Hardware constraints
    gpu_memory_gb: Optional[int] = 80          # 80 GB (H100)
    tensor_parallel_sizes: Optional[List[int]] = None   # None or "auto" or explicit list
    pipeline_parallel_sizes: Optional[List[int]] = None  # None or "auto" or explicit list
    micro_batch_sizes: Optional[List[int]] = None        # None or "auto" or explicit list
    context_parallel_sizes: Optional[List[int]] = None
    expert_parallel_sizes: Optional[List[int]] = None

    # Training constraints
    num_tokens_in_b: Optional[int] = 1400    # Dataset size in billions
    tflops_per_gpu: Optional[int] = 989         # BF16 TFLOPS per GPU (H100)
    max_steps_per_run: Optional[int] = 50        # Grid search steps per config
    max_training_days: Optional[int] = 2            # Expected training days
    vocab_size: Optional[int] = 32000          # Tokenizer vocab size

    # Model calculation
    calculate_model_size: Optional[bool] = False  # Auto-calculate architecture
```

### PaddleFleetRecipe

Dataclass containing all configuration for training.

```python
@dataclass
class PaddleFleetRecipe:
    model_config: object                   # TransformerConfig or subclass
    parallel_config: object | None      # Optional separate parallel config
    micro_batch_size: int = 1
    global_batch_size: int = 512
    num_nodes: int = 1
    num_gpus_per_node: int = 8
    max_steps: int | None = None
    log_dir: str | None = None

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.num_gpus_per_node
```

### Public API Functions

#### `generate_configs(runner, max_configs=None, scoring_fn=None)`

Generate all candidate configurations via grid search.

**Parameters:**
- `runner`: AutoConfigurator instance
- `max_configs` (int | None): Maximum number of configs to return. Only takes effect when `scoring_fn` is provided. None means no limit.
- `scoring_fn` (Callable[[GeneratedConfig], float] | None): Optional scoring function. When provided, configs are scored, deduplicated by parallel strategy (TP, PP, CP, EP) keeping the best MBS variant per group, and truncated to `max_configs`.

**Returns:**
- Tuple of (base_config, configs_dict)

**Example:**
```python
from auto_configurator import generate_configs, AutoConfigurator, PaddleFleetRecipe

runner = AutoConfigurator(recipe=recipe, path_to_logs="./logs")
base_config, configs = generate_configs(runner)

# Iterate through generated configs
for name, config in configs.items():
    print(f"Config: {name}")
    print(f"  TP={config.tensor_parallel_size}, PP={config.pipeline_parallel_size}")

# With scoring function for top-N selection
def my_scoring_fn(cfg):
    return cfg.expert_parallel_size / cfg.tensor_parallel_size

base_config, top3 = generate_configs(runner, max_configs=3, scoring_fn=my_scoring_fn)
```

#### `estimate_model_size(...) -> float`

Estimate model size based on training constraints.

**Parameters:**
- `gpu_count`: Number of GPUs
- `max_training_days`: Training time constraint
- `model_size_in_b`: Known model size (optional, default `None`)
- `tflops_per_gpu`: Expected TFLOPS per GPU (default `989`)
- `num_tokens_in_b`: Dataset size in billions (default `300`; note: `AutoConfigurator` class uses `1400`)
- `model_name`: Model type (default `"gpt"`)

**Returns:**
- Estimated model size in billions of parameters

**Example:**
```python
from auto_configurator import estimate_model_size

# Estimate model size for 7 days on 64 H100s
size = estimate_model_size(
    gpu_count=64,
    max_training_days=7,
    tflops_per_gpu=989,
    num_tokens_in_b=300,
    model_name="gpt"
)
print(f"Estimated model size: {size:.2f}B")
```

#### `get_results(base_config, runner_config, path_to_save, output_top_n=10)`

Generate performance summary from training logs.

**Parameters:**
- `base_config`: Base configuration object
- `runner_config`: AutoConfigurator instance
- `path_to_save`: Directory containing training logs
- `output_top_n`: Number of top configs to display

**Example:**
```python
from auto_configurator import get_results, AutoConfigurator

runner = AutoConfigurator(...)
get_results(
    base_config=runner.recipe.model_config,
    runner_config=runner,
    path_to_save="./logs",
    output_top_n=10,
)
```

## NeMo vs PaddleFleet Parameter Mapping

| NeMo Parameter | PaddleFleet Parameter |
|----------------|---------------------|
| `num_layers` | `num_hidden_layers` |
| `hidden_size` | `hidden_size` |
| `num_attention_heads` | `num_attention_heads` |
| `ffn_hidden_size` | `intermediate_size` |
| `seq_length` | `max_sequence_length` |
| `tensor_model_parallel_size` | `tensor_model_parallel_size` |
| `pipeline_model_parallel_size` | `pipeline_model_parallel_size` |
| `virtual_pipeline_model_parallel_size` | `virtual_pipeline_model_parallel_size` |
| `context_parallel_size` | `context_parallel_size` |
| `expert_model_parallel_size` | `expert_model_parallel_size` |

## Supported Models

Currently, only GPT-based models are supported in PaddleFleet:

- `gpt`, `llama`, `qwen`, `mixtral`, `mistral`, `gemma`, `glm`

**Note:** T5/mT5 and BERT models are not currently supported in PaddleFleet.
The interfaces for these models are preserved for future extension.

## Configuration Search Space

AutoConfigurator uses heuristic rules based on model size and hardware:

### Grid Search for 80GB GPUs (seq_length=2048)

| Model Size | TP | PP | MBS | GBS | Min MP | Max MP |
|-----------|-----|-----|------|------|--------|--------|
| <= 1B | 1,2 | 1 | 1,2,4,8 | 256 | 1 | 8 |
| <= 4B | 1,2,4 | 1 | 1,2,4,8 | 1024 | 1 | 8 |
| <= 8B | 1,2,4 | 1 | 1,2,4,8 | 2048 | 1 | 8 |
| <= 13B | 1,2,4,8 | 1 | 1,2,4,8 | 2048 | 4 | 8 |
| <= 23B | 1,2,4 | 1..4 | 1,2,4 | 2048 | 4 | 8 |
| <= 45B | 2,4,8 | 1..4 | 1,2,4 | 2048 | 8 | 32 |
| <= 95B | 2,4,8 | 1..8 | 1,2,4,8 | 2048 | 8 | 64 |

### Grid Search for 80GB GPUs (seq_length=4096)

| Model Size | TP | PP | MBS | GBS | Min MP | Max MP |
|-----------|-----|-----|------|------|--------|--------|
| <= 1B | 1,2,4 | 1 | 1,2,4,8 | 128 | 1 | 8 |
| <= 4B | 1,2,4 | 1 | 1,2,4,8 | 512 | 1 | 8 |
| <= 8B | 1,2,4 | 1..2 | 1,2,4 | 1024 | 1 | 8 |
| <= 13B | 1,2,4,8 | 1 | 1,2,4,8 | 1024 | 4 | 8 |

### Grid Search for 80GB GPUs (seq_length=8192)

| Model Size | TP | PP | MBS | GBS |
|-----------|-----|-----|------|------|
| <= 1B | 1,2 | 1..2 | 1,2,4 | 64 |
| <= 4B | 1,2,4 | 1..2 | 1,2,4 | 128 |

### Grid Search for 80GB GPUs (seq_length=16384)

| Model Size | TP | PP | MBS | GBS |
|-----------|-----|-----|------|------|
| <= 1B | 2,4 | 1 | 1,2 | 32 |
| <= 4B | 2,4 | 1..2 | 1 | 64 |

### Grid Search for 80GB GPUs (seq_length=32768)

| Model Size | TP | PP | MBS | GBS |
|-----------|-----|-----|------|------|
| <= 1B | 2,4 | 1..2 | 1 | 16 |
| <= 4B | 2,4 | 1..2 | 1 | 32 |

### Grid Search for 40GB GPUs

| Model Size | TP | PP | MBS | GBS | Min MP | Max MP |
|-----------|-----|-----|------|------|--------|--------|
| <= 1B | 1,2,4 | 1 | 1,2,4,8 | 256 | 1 | 8 |
| <= 4B | 1,2,4,8 | 1 | 1,2,4,8 | 1024 | 1 | 8 |
| <= 8B | 2,4,8 | 1,2 | 1,2,4 | 2048 | 2 | 8 |
| <= 13B | 4,8 | 1,2,4 | 1,2,4 | 2048 | 4 | 32 |

Note: PP values shown as "1..N" means valid pipeline parallel sizes up to N (i.e., values that evenly divide `num_layers`). "Min MP" and "Max MP" are the minimum and maximum allowed total model parallelism (TP × PP × CP × EP); configurations outside this range are filtered out.

## Validation Rules

AutoConfigurator validates all configurations:

1. **Model parallelism**: `total_parallel = TP × PP × CP × EP` must be within bounds
2. **Attention heads**: `num_attention_heads % TP == 0`
3. **Pipeline layers**: `num_layers × multiplier % PP == 0` (where multiplier=1 for GPT-based models)
4. **Batch size**: `GBS % (MBS × GPUs / MP) == 0`
5. **Sequence length**: Must be a positive multiple of 1024 (up to 1048576)

## Performance Calculation

TFLOPS formulas implemented (same as NeMo):

### GPT-based Models
```
Model FLOPs = (24·B·s·H² + 4·B·s²·H) × (3×L) + (6·B·s·H·V)
```

## Integration with PaddleFleet Training

To use generated configurations with PaddleFleet training:

```python
from paddlefleet import ModelParallelConfig
from paddlefleet.parallel_state import initialize_model_parallel
from auto_configurator import AutoConfigurator, generate_configs, PaddleFleetRecipe

# 1. Generate configurations
runner = AutoConfigurator(...)
base_config, configs = generate_configs(runner)

# 2. Select a configuration
selected_config = configs[list(configs.keys())[args.run_number - 1]]

# 3. Apply configuration to model
base_config.tensor_model_parallel_size = selected_config.tensor_parallel_size
base_config.pipeline_model_parallel_size = selected_config.pipeline_parallel_size
base_config.context_parallel_size = selected_config.context_parallel_size

# 4. Initialize parallel state
mp_config = ModelParallelConfig(
    tensor_model_parallel_size=base_config.tensor_model_parallel_size,
    pipeline_model_parallel_size=base_config.pipeline_model_parallel_size,
    context_parallel_size=base_config.context_parallel_size,
)

# 5. Start training
initialize_model_parallel(
    hcg=...  # Get your HCG from PaddlePaddle
    virtual_pipeline_model_parallel_size=base_config.virtual_pipeline_model_parallel_size,
)

# Your PaddleFleet training code here...
```

## Advanced Usage

### Custom Parallel Search

```python
runner = AutoConfigurator(
    recipe=recipe,
    path_to_logs="./logs",
    # Custom search space
    tensor_parallel_sizes=[1, 2, 4, 8],
    pipeline_parallel_sizes=[1, 2, 4, 8],
    micro_batch_sizes=[2, 4, 8],
    context_parallel_sizes=[1, 2, 4],
    expert_parallel_sizes=[1, 2, 4],
)
```

### Auto-Calculate Model Architecture

```python
runner = AutoConfigurator(
    recipe=recipe,
    path_to_logs="./logs",
    calculate_model_size=True,  # Enable auto-calculation
    # ... other params
)

# AutoConfigurator will infer optimal:
# - num_layers
# - hidden_size
# - num_attention_heads
# - intermediate_size
```

## Limitations

1. **GPU Types**: Currently optimized for NVIDIA H100 80GB
2. **Model Types**: Currently only GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma, glm) are supported. T5/mT5 and BERT are not supported in PaddleFleet.
3. **TFLOPS Estimation**: Assumes ideal conditions; actual performance may vary

## License

Apache License 2.0

## References

- NVIDIA NeMo: https://github.com/NVIDIA/NeMo
- PaddleFleet: https://github.com/PaddlePaddle/PaddleFleet
- PaddlePaddle: https://github.com/PaddlePaddle/PaddlePaddle
