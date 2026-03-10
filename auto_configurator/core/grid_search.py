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
Grid search generation for AutoConfigurator in PaddleFleet.

This module provides grid search functionality for parallel strategy exploration,
generating candidate configurations with different TP/PP/CP/EP/MBS combinations.

Note: T5/mT5 and BERT models are currently not supported in PaddleFleet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .model_size import GPT_BASED_MODELS

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Data Classes
# ============================================================================


@dataclass
class GridSearchConfig:
    """Grid search configuration for a specific model and hardware setup.

    This dataclass defines the search space for parallel strategies
    and recommended training parameters.

    Attributes:
        tp: List of tensor parallel sizes to search
        pp: List of pipeline parallel sizes to search
        cp: List of context parallel sizes
        ep: List of expert parallel sizes
        mbs: List of micro batch sizes
        gbs: Global batch size
        min_model_parallel: Minimum model parallelism
        max_model_parallel: Maximum model parallelism
    """

    tp: list[int]
    pp: list[int]
    cp: list[int]
    ep: list[int]
    mbs: list[int]
    gbs: int
    min_model_parallel: int
    max_model_parallel: int


@dataclass
class GeneratedConfig:
    """A generated candidate configuration.

    This dataclass represents a single candidate configuration
    generated from the grid search.

    Attributes:
        name: Unique name for this configuration
        tensor_parallel_size: Tensor parallel size
        pipeline_parallel_size: Pipeline parallel size
        virtual_pipeline_size: Virtual pipeline size
        context_parallel_size: Context parallel size
        expert_parallel_size: Expert parallel size
        micro_batch_size: Micro batch size
        global_batch_size: Global batch size
        max_steps: Maximum training steps
        recompute_granularity: Activation recompute granularity
        recompute_method: Activation recompute method
        recompute_num_layers: Number of layers to recompute
        recompute_modules: Modules to recompute
        log_dir: Directory for saving logs
    """

    name: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    virtual_pipeline_size: int | None
    context_parallel_size: int
    expert_parallel_size: int
    micro_batch_size: int
    global_batch_size: int
    max_steps: int
    recompute_granularity: str | None
    recompute_method: str | None
    recompute_num_layers: int | None
    recompute_modules: list | None
    log_dir: str


# ============================================================================
# Grid Search Classes
# ============================================================================


@dataclass
class GPTGridSearch:
    """Grid search rules for GPT-based models on 40GB/80GB GPUs.

    Implements heuristic rules similar to NeMo's GPT3GridSearch,
    adapted for PaddleFleet's configuration system.

    Args:
        model_size_in_b: Model size in billions
        valid_pp: List of valid pipeline parallel sizes
        seq_length: Training sequence length
        gpu_memory_gb: GPU memory (40 or 80)
    """

    model_size_in_b: float
    valid_pp: list[int]
    seq_length: int
    gpu_memory_gb: int

    # Default search space
    tp: list[int] = None
    pp: list[int] = None
    cp: list[int] = None
    ep: list[int] = None
    mbs: list[int] = None
    gbs: int = 1024
    min_model_parallel: int = 1
    max_model_parallel: int = 8

    def init_params(self):
        """Initialize search space based on model size and hardware."""
        model_size = self.model_size_in_b

        # Default values
        self.tp = [1, 2, 4, 8]
        self.pp = [1]
        self.cp = [1]
        self.ep = [1]
        self.mbs = [1, 2, 4, 8]

        if self.gpu_memory_gb == 80:
            self._init_80gb_params(model_size)
        else:  # 40GB
            self._init_40gb_params(model_size)

        logger.info(
            f"Grid search space: TP={self.tp}, PP={self.pp}, "
            f"MBS={self.mbs}, GBS={self.gbs}"
        )

    def _init_80gb_params(self, model_size: float):
        """Initialize search space for 80GB GPUs."""
        if self.seq_length == 2048:
            if model_size <= 1.0:
                self.tp, self.gbs = [1, 2], 256
            elif model_size <= 4.0:
                self.tp, self.gbs = [1, 2, 4], 1024
            elif model_size <= 8.0:
                self.tp, self.gbs = [1, 2, 4], 2048
            elif model_size <= 13.0:
                (
                    self.tp,
                    self.gbs,
                    self.min_model_parallel,
                    self.max_model_parallel,
                ) = ([1, 2, 4, 8], 2048, 4, 8)
            elif model_size <= 23.0:
                (
                    self.tp,
                    self.pp,
                    self.mbs,
                    self.min_model_parallel,
                    self.max_model_parallel,
                    self.gbs,
                ) = (
                    [1, 2, 4],
                    [x for x in self.valid_pp if 1 <= x <= 4],
                    [1, 2, 4],
                    4,
                    8,
                    2048,
                )
            elif model_size <= 45.0:
                (
                    self.tp,
                    self.pp,
                    self.mbs,
                    self.min_model_parallel,
                    self.max_model_parallel,
                    self.gbs,
                ) = (
                    [2, 4, 8],
                    [x for x in self.valid_pp if 1 <= x <= 4],
                    [1, 2, 4],
                    8,
                    32,
                    2048,
                )
            elif model_size <= 95:
                (
                    self.tp,
                    self.pp,
                    self.mbs,
                    self.min_model_parallel,
                    self.max_model_parallel,
                    self.gbs,
                ) = (
                    [2, 4, 8],
                    [x for x in self.valid_pp if 1 <= x <= 8],
                    [1, 2, 4, 8],
                    8,
                    64,
                    2048,
                )
            # Larger models follow similar patterns...

        elif self.seq_length == 4096:
            if model_size <= 1.0:
                self.tp, self.mbs, self.gbs = [1, 2, 4], [1, 2, 4, 8], 128
            elif model_size <= 4.0:
                self.tp, self.mbs, self.gbs = [1, 2, 4], [1, 2, 4, 8], 512
            elif model_size <= 8.0:
                self.tp, self.pp, self.mbs, self.gbs = (
                    [1, 2, 4],
                    [x for x in self.valid_pp if 1 <= x <= 2],
                    [1, 2, 4],
                    1024,
                )
            elif model_size <= 13.0:
                (
                    self.tp,
                    self.mbs,
                    self.gbs,
                    self.min_model_parallel,
                    self.max_model_parallel,
                ) = ([1, 2, 4, 8], [1, 2, 4, 8], 1024, 4, 8)
            # ... additional cases

        # Add support for 8192, 16384, 32768 sequence lengths
        elif self.seq_length == 8192:
            self._init_80gb_8192(model_size)
        elif self.seq_length == 16384:
            self._init_80gb_16384(model_size)
        elif self.seq_length == 32768:
            self._init_80gb_32768(model_size)

    def _init_40gb_params(self, model_size: float):
        """Initialize search space for 40GB GPUs."""
        if model_size <= 1.0:
            self.tp, self.mbs, self.gbs = [1, 2, 4], [1, 2, 4, 8], 256
        elif model_size <= 4.0:
            self.tp, self.mbs, self.gbs = [1, 2, 4, 8], [1, 2, 4, 8], 1024
        elif model_size <= 8.0:
            self.tp, self.pp, self.mbs, self.min_model_parallel, self.gbs = (
                [2, 4, 8],
                [1, 2],
                [1, 2, 4],
                2,
                2048,
            )
        elif model_size <= 13.0:
            (
                self.tp,
                self.pp,
                self.mbs,
                self.min_model_parallel,
                self.max_model_parallel,
                self.gbs,
            ) = ([4, 8], [1, 2, 4], [1, 2, 4], 4, 32, 2048)
        # ... additional cases

    def _init_80gb_8192(self, model_size: float):
        """Initialize for 80GB GPUs with 8192 seq length."""
        if model_size <= 1.0:
            self.tp, self.pp, self.mbs, self.gbs = [1, 2], [1, 2], [1, 2, 4], 64
        elif model_size <= 4.0:
            self.tp, self.pp, self.mbs, self.gbs = (
                [1, 2, 4],
                [1, 2],
                [1, 2, 4],
                128,
            )
        # ... additional cases

    def _init_80gb_16384(self, model_size: float):
        """Initialize for 80GB GPUs with 16384 seq length."""
        if model_size <= 1.0:
            self.tp, self.mbs, self.gbs = [2, 4], [1, 2], [1, 2], 32
        elif model_size <= 4.0:
            self.tp, self.pp, self.mbs, self.gbs = [2, 4], [1, 2], [1], 64
        # ... additional cases

    def _init_80gb_32768(self, model_size: float):
        """Initialize for 80GB GPUs with 32768 seq length."""
        if model_size <= 1.0:
            self.tp, self.pp, self.mbs, self.gbs = [2, 4], [1, 2], [1], 16
        elif model_size <= 4.0:
            self.tp, self.pp, self.mbs, self.gbs = [2, 4], [1, 2], [1], 32
        # ... additional cases


@dataclass
class T5GridSearch:
    """Grid search rules for T5/mT5 models on 40GB/80GB GPUs.

    Note: T5/mT5 models are currently not supported in PaddleFleet.
    This interface is provided for future extension.

    Args:
        model_size_in_b: Model size in billions
        valid_pp: List of valid pipeline parallel sizes
        seq_length: Training sequence length
        gpu_memory_gb: GPU memory (40 or 80)
    """

    model_size_in_b: float
    valid_pp: list[int]
    seq_length: int
    gpu_memory_gb: int

    tp: list[int] = None
    pp: list[int] = None
    cp: list[int] = None
    ep: list[int] = None
    mbs: list[int] = None
    gbs: int = 1920
    min_model_parallel: int = 1
    max_model_parallel: int = 8

    def init_params(self):
        """Initialize search space based on model size and hardware.

        Raises:
            NotImplementedError: T5/mT5 models are not currently supported.
        """
        raise NotImplementedError(
            "T5/mT5 models are currently not supported in PaddleFleet. "
            "Please use GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma, glm)."
        )


@dataclass
class BertGridSearch:
    """Grid search rules for BERT models on 40GB/80GB GPUs.

    Note: BERT models are currently not supported in PaddleFleet.
    This interface is provided for future extension.

    Args:
        model_size_in_b: Model size in billions
        valid_pp: List of valid pipeline parallel sizes
        seq_length: Training sequence length
        gpu_memory_gb: GPU memory (40 or 80)
    """

    model_size_in_b: float
    valid_pp: list[int]
    seq_length: int
    gpu_memory_gb: int

    tp: list[int] = None
    pp: list[int] = None
    cp: list[int] = None
    ep: list[int] = None
    mbs: list[int] = None
    gbs: int = 1920
    min_model_parallel: int = 1
    max_model_parallel: int = 8

    def init_params(self):
        """Initialize search space based on model size and hardware.

        Raises:
            NotImplementedError: BERT models are not currently supported.
        """
        raise NotImplementedError(
            "BERT models are currently not supported in PaddleFleet. "
            "Please use GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma, glm)."
        )


# ============================================================================
# Grid Search Generator
# ============================================================================


def get_grid_search_params(
    model_type: str,
    model_size_in_b: float,
    num_layers: int,
    seq_length: int,
    gpu_memory_gb: int,
    tensor_parallel_sizes: list[int] | str,
    pipeline_parallel_sizes: list[int] | str,
    micro_batch_sizes: list[int] | str,
    context_parallel_sizes: list[int],
    expert_parallel_sizes: list[int],
    min_model_parallel_size: int | str,
    max_model_parallel_size: int | str,
    global_batch_size: int,
) -> GridSearchConfig:
    """Get grid search parameters for the given configuration.

    Selects the appropriate search class (GPTGridSearch, T5GridSearch, BertGridSearch)
    based on model type, and allows override via explicit parameters.

    Args:
        model_type: Type of model (gpt, bert, t5)
        model_size_in_b: Model size in billions
        num_layers: Number of transformer layers
        seq_length: Sequence length
        gpu_memory_gb: GPU memory (40 or 80)
        tensor_parallel_sizes: TP sizes or "auto"
        pipeline_parallel_sizes: PP sizes or "auto"
        micro_batch_sizes: MBS sizes or "auto"
        context_parallel_sizes: CP sizes
        expert_parallel_sizes: EP sizes
        min_model_parallel_size: Min parallelism or "auto"
        max_model_parallel_size: Max parallelism or "auto"
        global_batch_size: Global batch size

    Returns:
        GridSearchConfig with search space parameters
    """
    # Calculate valid PP sizes (must divide num_layers)
    multiplier = 1 if model_type.lower() in GPT_BASED_MODELS else 2
    valid_pp = [1] + [
        multiplier * x for x in range(1, num_layers + 1) if num_layers % x == 0
    ]

    # Create appropriate search class
    if model_type.lower() in GPT_BASED_MODELS:
        search_class = GPTGridSearch
    elif model_type.lower() in ["t5", "mt5"]:
        raise NotImplementedError(
            "T5/mT5 models are currently not supported in PaddleFleet. "
            "Please use GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma, glm)."
        )
    elif model_type.lower() == "bert":
        raise NotImplementedError(
            "BERT models are currently not supported in PaddleFleet. "
            "Please use GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma, glm)."
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Initialize search class
    params = search_class(
        model_size_in_b=model_size_in_b,
        valid_pp=valid_pp,
        seq_length=seq_length,
        gpu_memory_gb=gpu_memory_gb,
    )
    params.init_params()

    # Override with explicit parameters if provided
    if tensor_parallel_sizes != "auto":
        params.tp = tensor_parallel_sizes
    if pipeline_parallel_sizes != "auto":
        params.pp = pipeline_parallel_sizes
    if context_parallel_sizes is not None:
        params.cp = context_parallel_sizes
    if expert_parallel_sizes is not None:
        params.ep = expert_parallel_sizes
    if micro_batch_sizes != "auto":
        params.mbs = micro_batch_sizes
    if global_batch_size is not None:
        params.gbs = global_batch_size
    if min_model_parallel_size != "auto":
        params.min_model_parallel = min_model_parallel_size
    if max_model_parallel_size != "auto":
        params.max_model_parallel = max_model_parallel_size

    return GridSearchConfig(
        tp=params.tp,
        pp=params.pp,
        cp=params.cp,
        ep=params.ep,
        mbs=params.mbs,
        gbs=params.gbs,
        min_model_parallel=params.min_model_parallel,
        max_model_parallel=params.max_model_parallel,
    )


def generate_grid_search_configs(
    runner_config,
    adapter,
) -> dict[str, GeneratedConfig]:
    """Generate grid search configurations.

    Generates all valid combinations of parallel strategies
    and creates GeneratedConfig objects for each.

    Args:
        runner_config: AutoConfigurator instance
        adapter: CombinedConfigAdapter for config access

    Returns:
        Dictionary mapping config names to GeneratedConfig objects
    """
    model_config = adapter.get_model_config()
    parallel_config = adapter.get_parallel_strategy()
    data_config = adapter.get_data_config()
    training_config = adapter.get_training_config()

    # Get grid search parameters
    grid_params = get_grid_search_params(
        model_type=runner_config.model_type,
        model_size_in_b=runner_config.model_size_in_b,
        num_layers=model_config.get_num_layers(),
        seq_length=runner_config.seq_length,
        gpu_memory_gb=runner_config.gpu_memory_gb,
        tensor_parallel_sizes=runner_config.tensor_parallel_sizes,
        pipeline_parallel_sizes=runner_config.pipeline_parallel_sizes,
        micro_batch_sizes=runner_config.micro_batch_sizes,
        context_parallel_sizes=runner_config.context_parallel_sizes,
        expert_parallel_sizes=runner_config.expert_parallel_sizes,
        min_model_parallel_size=runner_config.min_model_parallel_size,
        max_model_parallel_size=runner_config.max_model_parallel_size,
        global_batch_size=runner_config.global_batch_size,
    )

    # Get model family for multiplier
    model_name = runner_config.model_type.lower()
    multiplier = 1 if model_name in GPT_BASED_MODELS else 2

    # Generate valid TP/PP/CP/EP combinations
    configs = {}
    valid_tp_pp_list = []

    num_layers = model_config.get_num_layers()
    num_gpus = (
        training_config.get_num_nodes()
        * training_config.get_num_gpus_per_node()
    )
    att_heads = model_config.get_num_attention_heads()

    for tp in grid_params.tp:
        for pp in grid_params.pp:
            for cp in grid_params.cp:
                for ep in grid_params.ep:
                    for mbs in grid_params.mbs:
                        # Validate parallel constraints
                        model_parallelism = (
                            (tp * pp * cp * ep) if (cp and ep) else (tp * pp)
                        )

                        # Check: GBS divisible by (MBS * GPUs / MP)
                        mod_gbs = (
                            grid_params.gbs
                            % (mbs * num_gpus / model_parallelism)
                            if model_parallelism > 0
                            else 0
                        )
                        # Check: attention heads divisible by TP
                        mod_att_heads = att_heads % tp
                        # Check: layers divisible by PP (with multiplier)
                        mod_layers = (multiplier * num_layers) % pp

                        # Check: model parallelism in bounds
                        if (
                            grid_params.min_model_parallel is not None
                            and grid_params.max_model_parallel is not None
                            and (
                                model_parallelism
                                < grid_params.min_model_parallel
                                or model_parallelism
                                > grid_params.max_model_parallel
                            )
                        ):
                            continue

                        # Check: EP/CP compatibility
                        mod_cp = cp if cp else 1
                        mod_ep = ep if ep else 1
                        if not (
                            mod_cp // mod_ep == mod_cp
                            or mod_ep // mod_cp == mod_ep
                        ):
                            continue

                        # Skip duplicates
                        if (tp, pp, cp, ep) in valid_tp_pp_list:
                            continue

                        # Valid configuration
                        valid_tp_pp_list.append((tp, pp, cp, ep))

    logger.info(
        f"Generated {len(valid_tp_pp_list)} valid parallel configurations"
    )

    # Generate configs
    for tp, pp, cp, ep in valid_tp_pp_list:
        # Determine virtual pipeline and activation recompute parameters
        virtual_pipelines, act_layers, num_mbs_act, act_per_pipe = (
            _get_activation_params(
                tp, pp, cp, ep, num_layers, runner_config.model_type
            )
        )

        # Generate config for each activation checkpoint setting
        if act_layers[0] is not None:
            for act in act_layers:
                for num_mbs in num_mbs_act:
                    for act_pipe in act_per_pipe:
                        config = _create_config(
                            runner_config,
                            tp,
                            pp,
                            cp,
                            ep,
                            virtual_pipelines,
                            mbs,
                            act,
                            num_mbs,
                            act_pipe,
                            grid_params,
                        )
                        if config:
                            configs[config.name] = config
        else:
            config = _create_config(
                runner_config,
                tp,
                pp,
                cp,
                ep,
                virtual_pipelines,
                grid_params.mbs[0] if grid_params.mbs else 1,
                None,
                None,
                None,
                grid_params,
            )
            if config:
                configs[config.name] = config

    logger.info(f"Total candidate configurations: {len(configs)}")
    return configs


def _get_activation_params(
    tp: int, pp: int, cp: int, ep: int, num_layers: int, model_type: str
) -> tuple:
    """Get activation checkpointing parameters for given parallel config.

    Returns virtual pipeline size and activation recompute settings.
    """
    # Default: no recompute
    virtual_pipelines = None
    act_layers = [None]
    num_mbs_act = [None]
    act_per_pipe = [None]

    # For interleaved pipeline (PP > 2 with GPT models)
    if model_type.lower() in GPT_BASED_MODELS and pp > 2:
        virtual_pipelines = num_layers // pp

    return virtual_pipelines, act_layers, num_mbs_act, act_per_pipe


def _create_config(
    runner_config,
    tp: int,
    pp: int,
    cp: int,
    ep: int,
    virtual_pipelines: int,
    mbs: int,
    act: int | None,
    num_mbs_act: int | None,
    act_per_pipe: int | None,
    grid_params: GridSearchConfig,
) -> GeneratedConfig | None:
    """Create a GeneratedConfig from parameters.

    Validates the configuration and creates the config object.
    """
    model_config = runner_config.adapter.get_model_config()
    training_config = runner_config.adapter.get_training_config()

    num_layers = model_config.get_num_layers()
    num_gpus = (
        training_config.get_num_nodes()
        * training_config.get_num_gpus_per_node()
    )
    att_heads = model_config.get_num_attention_heads()
    model_name = runner_config.model_type.lower()
    multiplier = 1 if model_name in GPT_BASED_MODELS else 2

    # Validate constraints
    mod_gbs = (
        grid_params.gbs % (mbs * num_gpus / (tp * pp)) if (tp * pp) > 0 else 0
    )
    mod_att_heads = att_heads % tp
    mod_layers = (multiplier * num_layers) % pp

    if mod_gbs != 0 or mod_att_heads != 0 or mod_layers != 0:
        return None

    # Generate config name
    name = (
        f"{runner_config.model_type}_{runner_config.model_size_in_b}b_"
        f"{training_config.get_num_nodes()}nodes_"
        f"tp_{tp}_pp_{pp}_cp_{cp}_ep_{ep}_mbs_{mbs}"
    )
    if virtual_pipelines:
        name += f"_vp_{virtual_pipelines}"

    # Determine log directory
    log_dir = f"{runner_config.path_to_logs}/{name}"

    return GeneratedConfig(
        name=name,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        virtual_pipeline_size=virtual_pipelines,
        context_parallel_size=cp if cp else 1,
        expert_parallel_size=ep if ep else 1,
        micro_batch_size=mbs,
        global_batch_size=grid_params.gbs,
        max_steps=runner_config.max_steps_per_run,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=act,
        recompute_modules=None,
        log_dir=log_dir,
    )
