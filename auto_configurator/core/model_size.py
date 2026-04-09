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
Core utilities for AutoConfigurator in PaddleFleet.

This module contains:
- Model size calculation
- Architecture parameter inference (hidden_size, attention_heads, etc.)
- Grid search generation for parallel strategies

Note: T5/mT5 and BERT models are currently not supported in PaddleFleet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Model families for calculation
GPT_BASED_MODELS = [
    "gpt",
    "llama",
    "llama2",
    "llama3",
    "qwen",
    "qwen2",
    "qwen3",
    "mixtral",
    "mistral",
    "gemma",
    "glm",
]


# ============================================================================
# Model Size Calculation
# ============================================================================


def calculate_model_size(
    vocab_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    ffn_size: int | None = None,
    model_name: str = "gpt",
) -> float:
    """Calculate model size in billions of parameters.

    Args:
        vocab_size: Vocabulary size
        seq_length: Sequence length (for embedding calculation)
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        ffn_size: FFN intermediate size (defaults to 4*hidden_size)
        model_name: Model type for formula selection

    Returns:
        Model size in billions of parameters

    Raises:
        NotImplementedError: If model_name is t5, mt5, or bert
    """
    if ffn_size is None:
        ffn_size = 4 * hidden_size

    if model_name.lower() in GPT_BASED_MODELS:
        # GPT-style formula (from NeMo's calculate_model_size)
        # Formula: ~12 * L * H^2 * (1 + (13/(12*H)) + ((V+S)/(12*L*H)))
        # This approximates: attention weights + MLP + embeddings
        # Using floating point arithmetic for precision

        model_size = (
            12.0
            * num_layers
            * hidden_size
            * hidden_size
            * (
                1.0
                + (13.0 / (12.0 * hidden_size))
                + (
                    (vocab_size + seq_length)
                    / (12.0 * num_layers * hidden_size)
                )
            )
        ) / 1e9
    elif model_name.lower() in ["t5", "mt5"]:
        # T5/mT5 models are not currently supported in PaddleFleet
        raise NotImplementedError(
            "T5/mT5 models are currently not supported in PaddleFleet. "
            "Please use GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma, glm)."
        )
    elif model_name.lower() == "bert":
        # BERT models are not currently supported in PaddleFleet
        raise NotImplementedError(
            "BERT models are currently not supported in PaddleFleet. "
            "Please use GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma, glm)."
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model_size


# ============================================================================
# Architecture Parameter Inference
# ============================================================================


@dataclass
class ModelSizeParams:
    """Calculates model architecture parameters from model size.

    This class implements the same inference rules as NeMo's ModelSizeParams,
    adapted for PaddleFleet's configuration structure.

    Args:
        model_size_in_b: Desired model size in billions
        vocab_size: Tokenizer vocabulary size
        seq_length: Training sequence length
        model_name: Model type (gpt, llama, qwen, mixtral, mistral, gemma)

    Attributes:
        layers: Number of transformer layers
        hidden_size: Hidden dimension
        num_attention_heads: Number of attention heads
        ffn_size: FFN intermediate size
        kv_channels: KV channels for GQA (optional)
        learning_rate: Recommended learning rate

    Raises:
        NotImplementedError: If model_name is t5, mt5, or bert
    """

    model_size_in_b: float
    vocab_size: int
    seq_length: int
    model_name: str

    # Output parameters
    layers: int = None
    hidden_size: int = None
    num_attention_heads: int = None
    ffn_size: int = None
    kv_channels: int = None
    learning_rate: float = None

    def init_params(self):
        """Initialize model architecture parameters based on model size.

        Uses rule-based lookup tables similar to NeMo's approach to infer
        hidden_size, attention_heads, and learning rate from model size.
        Then searches for the optimal number of layers.

        Raises:
            NotImplementedError: If model_name is t5, mt5, or bert
        """
        model_name = self.model_name.lower()
        model_size = self.model_size_in_b

        # Infer hidden_size, attention_heads, and learning rate
        if model_name in GPT_BASED_MODELS:
            self._infer_gpt_params(model_size)
        elif model_name in ["t5", "mt5"]:
            raise NotImplementedError(
                "T5/mT5 models are currently not supported in PaddleFleet. "
                "Please use GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma)."
            )
        elif model_name == "bert":
            raise NotImplementedError(
                "BERT models are currently not supported in PaddleFleet. "
                "Please use GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma)."
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Set ffn_size if not set
        if self.ffn_size is None:
            self.ffn_size = 4 * self.hidden_size

        # Search for optimal number of layers
        self._find_num_layers()

        logger.info(
            f"Inferred architecture: {self.layers} layers, "
            f"hidden_size={self.hidden_size}, "
            f"num_attention_heads={self.num_attention_heads}, "
            f"ffn_size={self.ffn_size}, "
            f"learning_rate={self.learning_rate}"
        )

    def _infer_gpt_params(self, model_size: float):
        """Infer parameters for GPT-based models."""
        if model_size < 0.25:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                768,
                12,
                6e-4,
            )
        elif model_size <= 0.5:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                1024,
                16,
                3e-4,
            )
        elif model_size < 1:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                1536,
                16,
                2.5e-4,
            )
        elif model_size < 2:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                2048,
                16,
                2e-4,
            )
        elif model_size < 3:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                2560,
                32,
                1.6e-4,
            )
        elif model_size < 4.5:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                3072,
                32,
                1.4e-4,
            )
        elif model_size < 8:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                4096,
                32,
                1.2e-4,
            )
        elif model_size < 15:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                5120,
                40,
                1e-4,
            )
        elif model_size < 25:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                6144,
                48,
                1e-4,
            )
        elif model_size < 52:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                8192,
                64,
                0.8e-4,
            )
        elif model_size < 105:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                10240,
                80,
                0.7e-4,
            )
        elif model_size < 205:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                12288,
                96,
                0.6e-4,
            )
        elif model_size < 405:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                20480,
                128,
                0.5e-4,
            )
        elif model_size < 805:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                20480,
                128,
                0.4e-4,
            )
        elif model_size < 1105:
            self.hidden_size, self.num_attention_heads, self.learning_rate = (
                25600,
                160,
                0.3e-4,
            )
        else:
            raise ValueError("Model size for GPT must be < 1.1T parameters")

    def _find_num_layers(self):
        """Find optimal number of layers to match target model size.

        Searches through multiples of 2, 4, 5, 16, and all valid numbers
        to find layer count that produces a model size closest to target.
        """
        model_name = self.model_name.lower()
        # Multiplier for encoder-decoder models (not applicable to GPT-based models)
        multiplier = 1 if model_name in GPT_BASED_MODELS else 2

        # Set ffn_size for calculation
        calc_ffn = (
            self.ffn_size if self.ffn_size is not None else 4 * self.hidden_size
        )
        target_size = self.model_size_in_b
        margin = 0.01

        # Try powers of 2
        self.layers = self._try_power_of_two(
            target_size, margin, multiplier, calc_ffn
        )

        # Try multiples of 16
        if self.layers is None:
            self.layers = self._try_multiples(
                target_size, 16, margin, multiplier, calc_ffn
            )

        # Try multiples of 2
        if self.layers is None:
            self.layers = self._try_multiples(
                target_size, 2, margin, multiplier, calc_ffn
            )

        # Try multiples of 5
        if self.layers is None:
            self.layers = self._try_multiples(
                target_size, 5, margin, multiplier, calc_ffn
            )

        # Try any valid number
        if self.layers is None:
            for layers in range(1, 200):
                estimated = calculate_model_size(
                    vocab_size=self.vocab_size,
                    seq_length=self.seq_length,
                    hidden_size=self.hidden_size,
                    num_layers=layers,
                    ffn_size=calc_ffn,
                    model_name=model_name,
                )
                if (
                    target_size * (1.0 - margin)
                    < estimated
                    < target_size * (1.0 + margin)
                ):
                    self.layers = layers
                    break

        if self.layers is None:
            raise ValueError(
                "Could not find valid number of layers for model size"
            )

    def _try_power_of_two(
        self, target_size: float, margin: float, multiplier: int, ffn_size: int
    ) -> int | None:
        """Try layer counts that are powers of 2."""
        model_name = self.model_name.lower()

        for p in range(1, 10):
            layers = 2**p
            estimated = calculate_model_size(
                vocab_size=self.vocab_size,
                seq_length=self.seq_length,
                hidden_size=self.hidden_size,
                num_layers=layers,
                ffn_size=ffn_size,
                model_name=model_name,
            )
            if (
                target_size * (1.0 - margin)
                < estimated
                < target_size * (1.0 + margin)
            ):
                return layers

        return None

    def _try_multiples(
        self,
        target_size: float,
        step: int,
        margin: float,
        multiplier: int,
        ffn_size: int,
        max_layers: int = 200,
    ) -> int | None:
        """Try layer counts that are multiples of step."""
        model_name = self.model_name.lower()

        for layers in range(step, max_layers + 1, step):
            estimated = calculate_model_size(
                vocab_size=self.vocab_size,
                seq_length=self.seq_length,
                hidden_size=self.hidden_size,
                num_layers=layers,
                ffn_size=ffn_size,
                model_name=model_name,
            )
            if (
                target_size * (1.0 - margin)
                < estimated
                < target_size * (1.0 + margin)
            ):
                return layers

        return None


# ============================================================================
# Model Size Estimation
# ============================================================================


def estimate_model_size(
    gpu_count: int,
    max_training_days: float,
    model_size_in_b: float | None = None,
    tflops_per_gpu: int = 989,  # H100 BF16 peak TFLOPS
    num_tokens_in_b: int = 300,
    model_name: str = "gpt",
) -> float:
    """Estimates model size to train given constraints.

    If model_size is provided, estimates the time to train it.
    If not provided, estimates the model size that can be trained.

    Args:
        gpu_count: Number of GPUs to use
        max_training_days: Number of days to train
        model_size_in_b: Model size in billions (if known)
        tflops_per_gpu: Estimated TFLOPS per GPU
        num_tokens_in_b: Number of tokens in dataset (billions)
        model_name: Model type for penalty adjustment (unused - kept for API compatibility)

    Returns:
        Estimated model size in billions of parameters
    """
    # Calculate model size if not provided
    if model_size_in_b is None:
        model_size_in_b = (
            (max_training_days * 3600 * 24 * gpu_count * tflops_per_gpu * 1e12)
            / (8 * num_tokens_in_b * 1e9)
            / 1e9
        )
        model_size_in_b = round(model_size_in_b, 2)
    else:
        # Calculate training time if model size is provided
        max_training_days = (
            model_size_in_b * 1e9 * 8 * num_tokens_in_b * 1e9
        ) / (3600 * 24 * gpu_count * tflops_per_gpu * 1e12)
        max_training_days = round(max_training_days, 2)

    logger.info(
        f"You can train a {model_size_in_b}B parameter model in "
        f"{max_training_days} days using {gpu_count} GPUs. "
        f"Assuming {num_tokens_in_b}B tokens, {tflops_per_gpu} TFLOPS/GPU."
    )

    return model_size_in_b
