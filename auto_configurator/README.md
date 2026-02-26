# AutoConfigurator for PaddleFleet

AutoConfigurator migrated from NVIDIA NeMo to PaddlePaddle's PaddleFleet distributed training framework.

## Overview

AutoConfigurator provides automatic configuration generation for large model training, including:

- **Model architecture inference**: Automatically infer optimal model parameters (hidden_size, attention_heads, etc.) based on model size
- **Grid search for parallel strategies**: Generate candidate configurations with different TP/PP/CP/EP/MBS combinations
- **Configuration validation**: Ensure all configurations are valid for the given hardware constraints
- **Performance analysis**: Calculate TFLOPS and rank configurations by training efficiency

## Installation

The AutoConfigurator is located at `/root/paddlejob/gpfs/xuxinyi/fleet/PaddleFleet/auto_configurator/`

### Directory Structure

```
auto_configurator/
├── __init__.py              # Main module with AutoConfigurator class and public API
├── paddlefleet_adapters.py  # Adapters to bridge with PaddleFleet's config system
├── core/
│   ├── __init__.py         # Core module exports
│   ├── model_size.py         # Model size calculation and architecture inference
│   ├── grid_search.py        # Grid search generation for parallel strategies
│   └── performance.py        # TFLOPS calculation formulas
└── example.py               # Example usage script
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

### 2. Using the Example Script

```bash
# Run AutoConfigurator with default settings
python auto_configurator/example.py --model_type gpt --num_nodes 4 --gpus_per_node 8

# Generate results from existing logs
python auto_configurator/example.py --model_type gpt --get_results

# Specify custom parallel search space
python auto_configurator/example.py \
    --model_type gpt \
    --tp "1,2,4,8" \
    --pp "1,2,4" \
    --mbs "1,2,4" \
    --gpu_memory 80
```

## API Reference

### AutoConfigurator

Main class for configuration generation.

```python
@dataclass
class AutoConfigurator:
    recipe: PaddleFleetRecipe           # Training recipe with model and hardware configs
    path_to_logs: str                    # Directory for saving logs

    # Hardware constraints
    gpu_memory_gb: Optional[int] = 80          # 40 or 80 GB
    tensor_parallel_sizes: Optional[List[int]] = "auto"
    pipeline_parallel_sizes: Optional[List[int]] = "auto"
    micro_batch_sizes: Optional[List[int]] = "auto"
    context_parallel_sizes: Optional[List[int]] = [1]
    expert_parallel_sizes: Optional[List[int]] = [1]

    # Training constraints
    num_tokens_in_b: Optional[int] = 1400    # Dataset size in billions
    tflops_per_gpu: Optional[int] = 140         # TFLOPS per GPU
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

#### `generate_configs(runner: AutoConfigurator) -> Tuple[object, Dict[str, object]]`

Generate all candidate configurations via grid search.

**Parameters:**
- `runner`: AutoConfigurator instance

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
```

#### `estimate_model_size(...) -> float`

Estimate model size based on training constraints.

**Parameters:**
- `gpu_count`: Number of GPUs
- `max_training_days`: Training time constraint
- `model_size_in_b`: Known model size (optional)
- `tflops_per_gpu`: Expected TFLOPS per GPU
- `num_tokens_in_b`: Dataset size in billions
- `model_name`: Model type

**Returns:**
- Estimated model size in billions of parameters

**Example:**
```python
from auto_configurator import estimate_model_size

# Estimate model size for 7 days on 64 A100s
size = estimate_model_size(
    gpu_count=64,
    max_training_days=7,
    tflops_per_gpu=140,
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

### Grid Search for 80GB GPUs

| Model Size | Seq Len | TP | PP | MBS |
|-----------|---------|-----|-----|------|
| <= 1B | 2048 | 1,2 | [1,2,4,8] |
| <= 1B | 4096 | 1,2,4 | [1,2,4,8] |
| <= 4B | 2048 | 1,2,4 | [1,2,4,8] |

### Grid Search for 40GB GPUs

Similar search space adjusted for 40GB memory constraints.

## Validation Rules

AutoConfigurator validates all configurations:

1. **Model parallelism**: `total_parallel = TP × PP × CP × EP` must be within bounds
2. **Attention heads**: `num_attention_heads % TP == 0`
3. **Pipeline layers**: `num_layers × multiplier % PP == 0` (where multiplier=1 for GPT-based models)
4. **Batch size**: `GBS % (MBS × GPUs / MP) == 0`
5. **Sequence length**: Must be supported for model type:
   - GPT-based: `[2048, 4096, 8192, 16384, 32768]`

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

1. **GPU Types**: Currently optimized for NVIDIA A100 40GB/80GB
2. **Model Types**: Currently only GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma) are supported. T5/mT5 and BERT are not supported in PaddleFleet.
3. **TFLOPS Estimation**: Assumes ideal conditions; actual performance may vary

## License

Apache License 2.0

## References

- NVIDIA NeMo: https://github.com/NVIDIA/NeMo
- PaddleFleet: https://github.com/PaddlePaddle/PaddleFleet
- PaddlePaddle: https://github.com/PaddlePaddle/PaddlePaddle
