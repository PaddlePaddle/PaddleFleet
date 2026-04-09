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
Performance calculation utilities for AutoConfigurator in PaddleFleet.

This module provides TFLOPS calculation for model performance evaluation.
Note: T5/mT5 and BERT models are currently not supported in PaddleFleet.
"""

from .model_size import GPT_BASED_MODELS

# ============================================================================
# TFLOPS Calculation
# ============================================================================


def calculate_tflops(
    model_name: str,
    gbs: int,
    enc_seq_len: int,
    dec_seq_len: int,
    hidden_size: int,
    ffn_size: int,
    num_layers: int,
    vocab: int,
    num_nodes: int,
    gpus_per_node: int,
    time_per_step: float,
) -> tuple[float, float]:
    """Calculate model and hardware TFLOPS for each model.

    Implements the same formulas as NeMo's calculate_tflops:
    - GPT-based: Model FLOPs = (24BsH^2 + 4Bss^2H) x (3xL) + 6BsHV

    Note: T5/mT5 and BERT models are currently not supported in PaddleFleet.

    Args:
        model_name: Model type (gpt, llama, qwen, mixtral, mistral, gemma)
        gbs: Global batch size
        enc_seq_len: Encoder sequence length
        dec_seq_len: Decoder sequence length
        hidden_size: Hidden dimension
        ffn_size: FFN intermediate size
        num_layers: Number of layers
        vocab: Vocabulary size
        num_nodes: Number of compute nodes
        gpus_per_node: GPUs per node
        time_per_step: Time per step in seconds

    Returns:
        Tuple of (aggregate_tflops, per_gpu_tflops)

    Raises:
        NotImplementedError: If model_name is t5, mt5, or bert
    """
    num_gpus = num_nodes * gpus_per_node

    if model_name.lower() in GPT_BASED_MODELS:
        # GPT-3 formula
        # Model FLOPs = (24 * B * s * H^2 + 4 * B * s^2 * H) * (3 * num_layers) + (6 * B * s * H * V)
        model_flops = (
            (
                24 * gbs * enc_seq_len * hidden_size * hidden_size
                + 4 * gbs * enc_seq_len * enc_seq_len * hidden_size
            )
            * (3 * num_layers)
            + (6 * gbs * enc_seq_len * hidden_size * vocab)
        ) / time_per_step

        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = (model_flops / num_gpus) / 1e12

    elif model_name.lower() in ["t5", "mt5"]:
        # T5/mT5 models are not currently supported in PaddleFleet
        raise NotImplementedError(
            "T5/mT5 models are currently not supported in PaddleFleet. "
            "Please use GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma)."
        )

    elif model_name.lower() == "bert":
        # BERT models are not currently supported in PaddleFleet
        raise NotImplementedError(
            "BERT models are currently not supported in PaddleFleet. "
            "Please use GPT-based models (gpt, llama, qwen, mixtral, mistral, gemma)."
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return round(model_tflops, 2), round(model_tflops_per_gpu, 2)
