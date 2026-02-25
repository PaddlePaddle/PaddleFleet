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
PaddleFleet Adapters for AutoConfigurator.

This module provides adapter classes to bridge AutoConfigurator with PaddleFleet's
configuration system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# ============================================================================
# Base Adapter Interfaces
# ============================================================================


class ModelConfigAdapter(ABC):
    """Abstract base class for model configuration adapters.

    Provides a unified interface for accessing model configuration parameters
    regardless of the underlying framework.
    """

    @abstractmethod
    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        pass

    @abstractmethod
    def set_num_layers(self, value: int):
        """Set number of transformer layers."""
        pass

    @abstractmethod
    def get_hidden_size(self) -> int:
        """Get hidden dimension size."""
        pass

    @abstractmethod
    def set_hidden_size(self, value: int):
        """Set hidden dimension size."""
        pass

    @abstractmethod
    def get_num_attention_heads(self) -> int:
        """Get number of attention heads."""
        pass

    @abstractmethod
    def set_num_attention_heads(self, value: int):
        """Set number of attention heads."""
        pass

    @abstractmethod
    def get_ffn_hidden_size(self) -> int:
        """Get FFN hidden size (intermediate_size)."""
        pass

    @abstractmethod
    def set_ffn_hidden_size(self, value: int):
        """Set FFN hidden size (intermediate_size)."""
        pass

    @abstractmethod
    def get_seq_length(self) -> int:
        """Get sequence length (max_sequence_length)."""
        pass

    @abstractmethod
    def set_seq_length(self, value: int):
        """Set sequence length (max_sequence_length)."""
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass

    @abstractmethod
    def set_vocab_size(self, value: int):
        """Set vocabulary size."""
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Get model type identifier (e.g., 'gpt', 'bert', 't5')."""
        pass


class ParallelStrategyAdapter(ABC):
    """Abstract base class for parallel strategy adapters.

    Provides a unified interface for configuring parallelism strategies
    regardless of the underlying framework.
    """

    @abstractmethod
    def get_tensor_parallel_size(self) -> int:
        """Get tensor parallel size."""
        pass

    @abstractmethod
    def set_tensor_parallel_size(self, value: int):
        """Set tensor parallel size."""
        pass

    @abstractmethod
    def get_pipeline_parallel_size(self) -> int:
        """Get pipeline parallel size."""
        pass

    @abstractmethod
    def set_pipeline_parallel_size(self, value: int):
        """Set pipeline parallel size."""
        pass

    @abstractmethod
    def get_virtual_pipeline_size(self) -> int | None:
        """Get virtual pipeline parallel size."""
        pass

    @abstractmethod
    def set_virtual_pipeline_size(self, value: int | None):
        """Set virtual pipeline parallel size."""
        pass

    @abstractmethod
    def get_context_parallel_size(self) -> int:
        """Get context parallel size."""
        pass

    @abstractmethod
    def set_context_parallel_size(self, value: int):
        """Set context parallel size."""
        pass

    @abstractmethod
    def get_expert_parallel_size(self) -> int:
        """Get expert parallel size."""
        pass

    @abstractmethod
    def set_expert_parallel_size(self, value: int):
        """Set expert parallel size."""
        pass


class DataConfigAdapter(ABC):
    """Abstract base class for data configuration adapters.

    Provides a unified interface for data/batch configuration.
    """

    @abstractmethod
    def get_micro_batch_size(self) -> int:
        """Get micro batch size."""
        pass

    @abstractmethod
    def set_micro_batch_size(self, value: int):
        """Set micro batch size."""
        pass

    @abstractmethod
    def get_global_batch_size(self) -> int:
        """Get global batch size."""
        pass

    @abstractmethod
    def set_global_batch_size(self, value: int):
        """Set global batch size."""
        pass


class TrainingConfigAdapter(ABC):
    """Abstract base class for training configuration adapters.

    Provides a unified interface for training-related configuration.
    """

    @abstractmethod
    def get_num_nodes(self) -> int:
        """Get number of nodes."""
        pass

    @abstractmethod
    def get_num_gpus_per_node(self) -> int:
        """Get number of GPUs per node."""
        pass

    @abstractmethod
    def get_max_steps(self) -> int | None:
        """Get maximum training steps."""
        pass

    @abstractmethod
    def set_max_steps(self, value: int):
        """Set maximum training steps."""
        pass

    @abstractmethod
    def get_recompute_config(self) -> dict:
        """Get activation recompute configuration."""
        pass

    @abstractmethod
    def set_recompute_config(
        self,
        granularity: str | None = None,
        method: str | None = None,
        num_layers: int | None = None,
        modules: list[str] | None = None,
    ):
        """Set activation recompute configuration.

        Args:
            granularity: 'full' or 'selective'
            method: 'uniform' or 'block'
            num_layers: Number of layers for recompute
            modules: List of module names to recompute
        """
        pass


class CombinedConfigAdapter(ABC):
    """Combined adapter that provides access to all config aspects."""

    @abstractmethod
    def get_model_config(self) -> ModelConfigAdapter:
        """Get model configuration adapter."""
        pass

    @abstractmethod
    def get_parallel_strategy(self) -> ParallelStrategyAdapter:
        """Get parallel strategy adapter."""
        pass

    @abstractmethod
    def get_data_config(self) -> DataConfigAdapter:
        """Get data configuration adapter."""
        pass

    @abstractmethod
    def get_training_config(self) -> TrainingConfigAdapter:
        """Get training configuration adapter."""
        pass


# ============================================================================
# PaddleFleet Specific Adapters
# ============================================================================


class PaddleFleetModelConfigAdapter(ModelConfigAdapter):
    """Adapter for PaddleFleet TransformerConfig/GPTConfig."""

    def __init__(self, config):
        self._config = config

    def get_num_layers(self) -> int:
        return self._config.num_hidden_layers

    def set_num_layers(self, value: int):
        self._config.num_hidden_layers = value

    def get_hidden_size(self) -> int:
        return self._config.hidden_size

    def set_hidden_size(self, value: int):
        self._config.hidden_size = value

    def get_num_attention_heads(self) -> int:
        return self._config.num_attention_heads

    def set_num_attention_heads(self, value: int):
        self._config.num_attention_heads = value

    def get_ffn_hidden_size(self) -> int:
        return self._config.intermediate_size

    def set_ffn_hidden_size(self, value: int):
        self._config.intermediate_size = value

    def get_seq_length(self) -> int:
        # PaddleFleet uses max_sequence_length
        return getattr(self._config, "max_sequence_length", 2048)

    def set_seq_length(self, value: int):
        self._config.max_sequence_length = value

    def get_vocab_size(self) -> int:
        # Check for vocab_size in GPTConfig
        if hasattr(self._config, "vocab_size"):
            return self._config.vocab_size
        return getattr(self._config, "vocab_size", 32000)

    def set_vocab_size(self, value: int):
        if hasattr(self._config, "vocab_size"):
            self._config.vocab_size = value
        else:
            self._config.vocab_size = value

    def get_model_type(self) -> str:
        config_name = self._config.__class__.__name__.lower()
        if "gpt" in config_name:
            return "gpt"
        elif "bert" in config_name:
            return "bert"
        elif "t5" in config_name:
            return "t5"
        else:
            return "gpt"  # Default to GPT-based


class PaddleFleetParallelConfigAdapter(ParallelStrategyAdapter):
    """Adapter for PaddleFleet ModelParallelConfig."""

    def __init__(self, config):
        self._config = config

    def get_tensor_parallel_size(self) -> int:
        return self._config.tensor_model_parallel_size

    def set_tensor_parallel_size(self, value: int):
        self._config.tensor_model_parallel_size = value

    def get_pipeline_parallel_size(self) -> int:
        return self._config.pipeline_model_parallel_size

    def set_pipeline_parallel_size(self, value: int):
        self._config.pipeline_model_parallel_size = value

    def get_virtual_pipeline_size(self) -> int | None:
        return self._config.virtual_pipeline_model_parallel_size

    def set_virtual_pipeline_size(self, value: int | None):
        self._config.virtual_pipeline_model_parallel_size = value

    def get_context_parallel_size(self) -> int:
        return self._config.context_parallel_size

    def set_context_parallel_size(self, value: int):
        self._config.context_parallel_size = value

    def get_expert_parallel_size(self) -> int:
        return self._config.expert_model_parallel_size

    def set_expert_parallel_size(self, value: int):
        self._config.expert_model_parallel_size = value


class PaddleFleetDataConfigAdapter(DataConfigAdapter):
    """Adapter for PaddleFleet data configuration.

    In PaddleFleet, batch sizes are typically handled via command-line args
    or passed directly to the training function. This adapter provides
    a consistent interface.
    """

    def __init__(self, micro_batch_size: int = 1, global_batch_size: int = 512):
        self._micro_batch_size = micro_batch_size
        self._global_batch_size = global_batch_size

    def get_micro_batch_size(self) -> int:
        return self._micro_batch_size

    def set_micro_batch_size(self, value: int):
        self._micro_batch_size = value

    def get_global_batch_size(self) -> int:
        return self._global_batch_size

    def set_global_batch_size(self, value: int):
        self._global_batch_size = value


class PaddleFleetTrainingConfigAdapter(TrainingConfigAdapter):
    """Adapter for PaddleFleet training configuration."""

    def __init__(
        self,
        num_nodes: int = 1,
        num_gpus_per_node: int = 8,
        max_steps: int | None = None,
    ):
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self._max_steps = max_steps
        self._recompute_granularity = None
        self._recompute_method = None
        self._recompute_num_layers = None
        self._recompute_modules = None

    def get_num_nodes(self) -> int:
        return self._num_nodes

    def get_num_gpus_per_node(self) -> int:
        return self._num_gpus_per_node

    def get_max_steps(self) -> int | None:
        return self._max_steps

    def set_max_steps(self, value: int):
        self._max_steps = value

    def get_recompute_config(self) -> dict:
        return {
            "granularity": self._recompute_granularity,
            "method": self._recompute_method,
            "num_layers": self._recompute_num_layers,
            "modules": self._recompute_modules,
        }

    def set_recompute_config(
        self,
        granularity: str | None = None,
        method: str | None = None,
        num_layers: int | None = None,
        modules: list[str] | None = None,
    ):
        self._recompute_granularity = granularity
        self._recompute_method = method
        self._recompute_num_layers = num_layers
        self._recompute_modules = modules


class PaddleFleetCombinedConfigAdapter(CombinedConfigAdapter):
    """Combined adapter for PaddleFleet configuration.

    This adapter provides a unified interface to access model, parallel,
    data, and training configurations in PaddleFleet.
    """

    def __init__(
        self,
        model_config,
        parallel_config=None,
        data_config=None,
        training_config=None,
    ):
        self._model_adapter = PaddleFleetModelConfigAdapter(model_config)
        self._data_adapter = data_config
        self._training_adapter = training_config

        if parallel_config is not None:
            # Use the provided parallel_config directly
            self._parallel_adapter = PaddleFleetParallelConfigAdapter(
                parallel_config
            )
        else:
            # Try to get parallel config from model_config if it's a TransformerConfig
            if hasattr(model_config, "tensor_model_parallel_size"):
                self._parallel_adapter = PaddleFleetParallelConfigAdapter(
                    model_config
                )
            else:
                self._parallel_adapter = None

        if data_config is None:
            self._data_adapter = PaddleFleetDataConfigAdapter()

        if training_config is None:
            self._training_adapter = PaddleFleetTrainingConfigAdapter()

    def get_model_config(self) -> ModelConfigAdapter:
        return self._model_adapter

    def get_parallel_strategy(self) -> ParallelStrategyAdapter:
        return self._parallel_adapter

    def get_data_config(self) -> DataConfigAdapter:
        return self._data_adapter

    def get_training_config(self) -> TrainingConfigAdapter:
        return self._training_adapter


# ============================================================================
# Data Classes for Configuration
# ============================================================================


@dataclass
class PaddleFleetRecipe:
    """Recipe configuration for PaddleFleet training.

    This dataclass holds all the necessary configuration objects
    for training with AutoConfigurator.
    """

    model_config: object
    """The model configuration object (TransformerConfig or subclass)."""

    parallel_config: object | None = None
    """Optional separate parallel configuration object."""

    micro_batch_size: int = 1
    """Micro batch size for training."""

    global_batch_size: int = 512
    """Global batch size for training."""

    num_nodes: int = 1
    """Number of compute nodes."""

    num_gpus_per_node: int = 8
    """Number of GPUs per node."""

    max_steps: int | None = None
    """Maximum number of training steps."""

    log_dir: str | None = None
    """Directory for saving training logs."""

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs."""
        return self.num_nodes * self.num_gpus_per_node


def create_paddlefleet_adapter(
    recipe: PaddleFleetRecipe,
) -> CombinedConfigAdapter:
    """Factory function to create a combined config adapter from a recipe.

    Args:
        recipe: PaddleFleetRecipe object containing all configuration

    Returns:
        CombinedConfigAdapter instance for the recipe
    """
    return PaddleFleetCombinedConfigAdapter(
        model_config=recipe.model_config,
        parallel_config=recipe.parallel_config,
        data_config=PaddleFleetDataConfigAdapter(
            micro_batch_size=recipe.micro_batch_size,
            global_batch_size=recipe.global_batch_size,
        ),
        training_config=PaddleFleetTrainingConfigAdapter(
            num_nodes=recipe.num_nodes,
            num_gpus_per_node=recipe.num_gpus_per_node,
            max_steps=recipe.max_steps,
        ),
    )
