# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
Internal Medicine — Model Health Monitoring System.

Unified API for monitoring model health during training across backends:
- Megatron-Bridge (PyTorch)
- PaddleFleet (PaddlePaddle)

Monitors:
- qk_stats: Attention QK statistics (entropy, max logits, sink weights)
- moe_health: MoE health metrics (router, experts, tokens, shared)
- ple_health: Per-Layer Embedding health (Megatron only)

Usage:
    from paddlefleet.internal_medicine import (
        setup_internal_medicine,
        training_logs,
    )

    monitor_dict = {}
    model = setup_internal_medicine(
        model,
        monitors=['qk_stats', 'moe_health'],
        monitor_dict=monitor_dict,
    )

    # In train_step wrapper
    for monitor in monitor_dict.values():
        monitor.step()

    # Read metrics
    metrics = training_logs.gather_and_aggregate()
"""

from .core.registry import detect_backend, get_backend_setup_fn
from .core.training_logs import TrainingLogs, training_logs

__all__ = [
    "setup_internal_medicine",
    "training_logs",
    "TrainingLogs",
    "detect_backend",
]


def setup_internal_medicine(
    model,
    monitors: list[str] | None = None,
    monitor_dict: dict | None = None,
    monitor_interval: int = 1,
    verbose: bool = False,
    backend: str | None = None,
    hook_timing_enabled: bool = False,
    **kwargs,
):
    """
    Unified setup function for all internal medicine monitors.

    Args:
        model: Model or list of models to monitor
        monitors: List of monitor names. Options:
            - 'qk_stats': QK attention statistics
            - 'moe_health': MoE health metrics
            - 'ple_health': PLE health (Megatron only)
            - 'massive_act': Massive activation metrics
            - 'all': Enable all available monitors. None defaults to all.
        monitor_dict: Dict to store monitor instances
        monitor_interval: Steps between monitoring
        verbose: Print debug information
        backend: Force a specific backend ('megatron' or 'paddlefleet').
            Auto-detected if None.
        hook_timing_enabled: Enable lightweight per-hook CPU wall-time diagnostics
            for backends that support it.
        **kwargs: Per-monitor kwargs, keyed by monitor name.
            e.g. qk_stats={'causal': True}

    Returns:
        model (unchanged, monitors are registered via hooks)
    """
    setup_fn = get_backend_setup_fn(backend)
    return setup_fn(
        model,
        monitors=monitors,
        monitor_dict=monitor_dict,
        monitor_interval=monitor_interval,
        verbose=verbose,
        hook_timing_enabled=hook_timing_enabled,
        **kwargs,
    )
