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
QK Stats Monitor for Megatron-Bridge + Transformer Engine.
Migrated from src/internal_medicine/qk_logits/.
"""

import logging

import torch
from torch import nn

from .base import TorchProbe
from .sink_head_metrics import compute_sink_head_classification
from .triton_kernels import compute_qk_stats, compute_qk_stats_packed

logger = logging.getLogger(__name__)

# Above this seq_len, skip QK stats to avoid OOM in pytorch fallback path
_MAX_SEQ_LEN_FOR_QK = 8192


_LAYER_METRICS = (
    "max",
    "mean",
    "entropy_avg",
    "sink",
    "entropy_min",
    "entropy_max",
    "sink_head_ratio",
    "sink_head_max",
    "sink_nonsink_gap",
)


class QKStatsMonitor(TorchProbe):
    METRIC_PREFIX = "qk_stats"
    MAX_AGGREGATED = {"max", "entropy_max", "sink_head_max"}
    MIN_AGGREGATED = {"entropy_min"}

    def __init__(
        self,
        causal: bool = True,
        use_triton: bool = True,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        hook_timing_enabled: bool = False,
        sink_head_threshold: float = 0.3,
        exclude_families=None,
    ):
        super().__init__(
            exclude_families=exclude_families,
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            hook_timing_enabled=hook_timing_enabled,
        )
        self.causal = causal
        self.use_triton = use_triton
        self.sink_head_threshold = sink_head_threshold
        self.tp_size = 1
        self.tp_rank = 0
        self.tp_group = None

    def register_hooks(self, model: nn.Module):
        self._init_parallel_state()
        targets = self._prepare_layers(model)
        if not targets:
            return
        self.allocate_buffers(next(model.parameters()).device)
        self._attach_hooks(targets)

    def _init_parallel_state(self):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.tp_size = (
                    parallel_state.get_tensor_model_parallel_world_size()
                )
                self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                self.tp_group = parallel_state.get_tensor_model_parallel_group()
        except ImportError:
            pass

    def _prepare_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
        """Discover attention layers and declare their metric keys. No hooks attached."""
        attention_layers = self._find_attention_layers(model)
        if len(attention_layers) == 0:
            logger.warning("[QKMonitor] No attention layers found!")
            return []
        if self.verbose:
            logger.info(
                f"[QKMonitor] Found {len(attention_layers)} attention layers. TP={self.tp_size}"
            )

        for layer_idx, _ in attention_layers:
            for name in _LAYER_METRICS:
                self.declare_layer_metric(layer_idx, name)
        return attention_layers

    def _attach_hooks(self, targets: list[tuple[int, nn.Module]]):
        for layer_idx, attention_module in targets:
            if hasattr(attention_module, "core_attention"):
                hook = (
                    attention_module.core_attention.register_forward_pre_hook(
                        self.timed_hook(
                            "compute", self._make_compute_hook(layer_idx)
                        ),
                        with_kwargs=True,
                    )
                )
                self.hooks.append(hook)
        logger.info(f"[QKMonitor] Registered {len(self.hooks)} hooks.")

    def _find_attention_layers(
        self, model: nn.Module
    ) -> list[tuple[int, nn.Module]]:
        attention_layers = []
        if hasattr(model, "module"):
            model = model.module
        layers = None
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            layers = model.decoder.layers
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        if layers is None:
            return []
        for local_idx, layer in enumerate(layers):
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers))
            attn = None
            if hasattr(layer, "self_attention"):
                attn = layer.self_attention
            elif hasattr(layer, "attention"):
                attn = layer.attention
            if attn:
                attention_layers.append((global_idx, attn))
        return attention_layers

    def _make_compute_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs=None):
            if not self._should_monitor():
                return
            try:
                if len(args) < 2:
                    return
                query, key = args[0].detach(), args[1].detach()

                # Packed (THD) sequences carry cu_seqlens via packed_seq_params;
                # honor per-sequence boundaries instead of treating the whole
                # packed row as one giant sequence.
                cu_seqlens = None
                psp = (kwargs or {}).get("packed_seq_params")
                if psp is not None:
                    cu_seqlens = getattr(psp, "cu_seqlens_q", None)
                    if cu_seqlens is None:
                        cu_seqlens = getattr(psp, "cu_seqlens_q_padded", None)

                is_thd = query.dim() == 3

                # Learned per-head sink logit (learnable/off-by-one softmax);
                # None for vanilla. `module` is core_attention. Keep on-device.
                attn_sink = getattr(module, "softmax_offset", None)
                if attn_sink is not None:
                    attn_sink = attn_sink.detach()

                with torch.no_grad():
                    if is_thd and cu_seqlens is not None:
                        stats = compute_qk_stats_packed(
                            query,
                            key,
                            cu_seqlens.detach(),
                            causal=self.causal,
                            use_triton=self.use_triton,
                            attn_sink=attn_sink,
                        )
                    else:
                        if is_thd:
                            query = query.unsqueeze(1)
                            key = key.unsqueeze(1)
                        seq_len = query.shape[0]
                        if (
                            seq_len > _MAX_SEQ_LEN_FOR_QK
                            and not self.use_triton
                        ):
                            return
                        stats = compute_qk_stats(
                            query,
                            key,
                            causal=self.causal,
                            use_triton=self.use_triton,
                            attn_sink=attn_sink,
                        )

                # NOTE: TP cross-rank aggregation is intentionally NOT done here.
                # gather_and_aggregate() at flush time pools across all ranks
                # using max-for-max / min-for-min / mean-for-others; that yields
                # the global value for max/min metrics directly, and a balanced
                # mean for the rest. Doing dist.all_reduce inside this hook used
                # to compete with the EP a2a stream and was the dominant
                # monitor-induced slowdown.

                local_head_entropy = stats["entropy_per_head"].mean(dim=0)
                sink_per_head = stats["sink_per_head"]
                sink_local = (
                    sink_per_head.mean(dim=0)
                    if sink_per_head.dim() > 1
                    else sink_per_head
                )
                sink_class = compute_sink_head_classification(
                    sink_local, threshold=self.sink_head_threshold
                )

                self.record_layer_metric(layer_idx, "max", stats["max_global"])
                self.record_layer_metric(
                    layer_idx, "mean", stats["mean_global"]
                )
                self.record_layer_metric(
                    layer_idx, "entropy_avg", stats["entropy_global"]
                )
                self.record_layer_metric(
                    layer_idx, "sink", stats["sink_global"]
                )
                self.record_layer_metric(
                    layer_idx, "entropy_min", local_head_entropy.min()
                )
                self.record_layer_metric(
                    layer_idx, "entropy_max", local_head_entropy.max()
                )
                self.record_layer_metric(
                    layer_idx, "sink_head_ratio", sink_class["sink_head_ratio"]
                )
                self.record_layer_metric(
                    layer_idx, "sink_head_max", sink_class["sink_head_max"]
                )
                self.record_layer_metric(
                    layer_idx,
                    "sink_nonsink_gap",
                    sink_class["sink_nonsink_gap"],
                )
            except Exception as e:
                if self.verbose:
                    logger.error(f"[QKMonitor] Error layer {layer_idx}: {e}")

        return hook_fn


def setup_qk_monitor(
    model: nn.Module,
    causal: bool = True,
    use_triton: bool = True,
    verbose: bool = False,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    sink_head_threshold: float = 0.3,
    hook_timing_enabled: bool = False,
    monitor_dict: dict | None = None,
    exclude_families=None,
) -> nn.Module:
    monitor = QKStatsMonitor(
        exclude_families=exclude_families,
        causal=causal,
        use_triton=use_triton,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        hook_timing_enabled=hook_timing_enabled,
        sink_head_threshold=sink_head_threshold,
    )
    models = [model] if not isinstance(model, list) else model
    monitor._init_parallel_state()
    chunk_targets = []
    for m in models:
        chunk_targets.append((m, monitor._prepare_layers(m)))
    if any(targets for _, targets in chunk_targets):
        device = next((p.device for m in models for p in m.parameters()), None)
        assert device is not None, (
            "no parameters across model chunks; cannot pick a device"
        )
        monitor.allocate_buffers(device)
        for _, targets in chunk_targets:
            monitor._attach_hooks(targets)
    logger.info(
        f"[QKMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers."
    )
    if monitor_dict is not None:
        monitor_dict["qk_stats"] = monitor
    return model
