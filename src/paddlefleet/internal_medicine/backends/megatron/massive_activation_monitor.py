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
Massive Activation Monitor for Megatron-Bridge.

Monitors massive activations in post-residual hidden states — extreme outlier
values that appear in a few channels and persist across intermediate layers
via the residual connection.
"""

import logging

import torch
from torch import nn

from .base import TorchProbe
from .massive_activation_metrics import (
    DEFAULT_ABSOLUTE_THRESHOLDS,
    _threshold_key,
    compute_grad_gain_bounds,
    compute_hidden_spectral_entropy,
    compute_logit_lens_entropy,
    compute_per_channel_max,
    compute_post_norm_cosine_stability,
    compute_post_norm_sparsity,
    compute_spectral_norm_bounds,
    summarize_per_channel_max,
)

logger = logging.getLogger(__name__)


class MassiveActivationMonitor(TorchProbe):
    """Monitor massive activations in the residual stream."""

    METRIC_PREFIX = "massive_act"
    MAX_AGGREGATED = {
        "channel_max",
        "channel_median",
        "channel_p95",
        "channel_p99",
        "channel_max_ratio",
        "topk_channel_norm",
        "activation_rms",
        "massive_act_channel_count",
        "spectral_norm_max",
        "lipschitz_max",
    }
    MIN_AGGREGATED = {
        "spectral_norm_min",
        "lipschitz_min",
    }

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        spike_threshold_multiplier: float = 100.0,
        topk_channels: int = 3,
        sparsity_epsilon: float = 0.01,
        cosine_sample_pairs: int = 256,
        sample_layers: list[int] | None = None,
        absolute_thresholds: tuple[float, ...] = DEFAULT_ABSOLUTE_THRESHOLDS,
        log_post_norm_metrics: bool = True,
        log_activation_rms: bool = True,
        log_lipschitz: bool = True,
        log_logit_lens_entropy: bool = False,
        logit_lens_chunk_size: int = 1024,
        logit_lens_apply_final_norm: bool = True,
        logit_lens_layers: list[int] | None = None,
        log_logit_lens_cross_entropy: bool = False,
        log_hidden_spectral_entropy: bool = False,
        hook_timing_enabled: bool = False,
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
        self.spike_threshold_multiplier = spike_threshold_multiplier
        self.topk_channels = topk_channels
        self.sparsity_epsilon = sparsity_epsilon
        self.cosine_sample_pairs = cosine_sample_pairs
        self.sample_layers = set(sample_layers) if sample_layers else None
        self.absolute_thresholds = tuple(absolute_thresholds)
        self.log_post_norm_metrics = log_post_norm_metrics
        # Single knob for the residual-scale/gain group: activation_rms plus the
        # per-token spectral-norm bounds. activation_rms is derived from the same
        # per-token pre-RMS the spectral hook computes, so they share one flag.
        self.log_activation_rms = log_activation_rms
        # Per-layer Lipschitz constant estimate, captured in the backward pass: the
        # per-token gradient-gain ratio ‖∂L/∂x‖/‖∂L/∂y‖ bounds the layer Jacobian's
        # singular values (max -> σ_max lower bound = Lipschitz const; min -> σ_min
        # upper bound). Rides a forward hook that registers tensor grad hooks on the
        # input/output hidden states — module full_backward_hook cannot see the input
        # grad because Megatron calls layers with all-keyword args.
        self.log_lipschitz = log_lipschitz
        # Logit-lens predictive entropy H(p) of each layer's residual, projected
        # through the LM head. Opt-in / default off: the projection is ~one LM-head
        # forward per monitored layer per monitored step. Only attached on the PP
        # stage that owns the head weight (non-head stages are a clean no-op).
        self.log_logit_lens_entropy = log_logit_lens_entropy
        self.logit_lens_chunk_size = logit_lens_chunk_size
        self.logit_lens_apply_final_norm = logit_lens_apply_final_norm
        self.logit_lens_layers = (
            set(logit_lens_layers) if logit_lens_layers else None
        )
        # Per-layer cross-entropy via the same logit-lens projection: CE = log_z − l[label]
        # against the ground-truth next token. The final layer's CE equals the LM loss up
        # to loss-mask weighting (the mask is applied outside the model forward, so this
        # token-mean is UNWEIGHTED). Labels are captured from the model forward's kwargs by
        # a top-level pre-hook on the head-owning chunk. Shares the logit-lens hook/head.
        self.log_logit_lens_cross_entropy = log_logit_lens_cross_entropy
        # Set by the label-capture forward pre-hook each monitored forward; read by the
        # logit-lens hook (which fires after it) to align labels to the flattened tokens.
        self._captured_labels = None
        # Head-owning model chunks that need a label-capture pre-hook (for cross-entropy).
        self._label_capture_models: list = []
        # Matrix (von Neumann) entropy of the post-RMSNorm hidden spectrum — a
        # representation-diversity / rank-collapse signal computed via eigvalsh of the
        # smaller Gram (no LM head, no full SVD). SET-LEVEL nonlinear quantity, so the
        # cross-shard/microbatch mean is only approximate (accepted by design). Rides the
        # input_layernorm hook, so it only records where an input_layernorm exists.
        self.log_hidden_spectral_entropy = log_hidden_spectral_entropy
        # global_idx -> (lm_head_weight, final_norm) for head-owning, eligible layers.
        # Populated in _prepare_layers; read in _attach_hooks to decide the entropy hook.
        self._layer_head_info: dict[int, tuple] = {}
        self.MAX_AGGREGATED = self.MAX_AGGREGATED | {
            f"channel_count_gt_{_threshold_key(t)}"
            for t in self.absolute_thresholds
        }
        self.tp_size = 1
        self.tp_group = None
        self._warned_per_channel_aggregate = False
        self._post_norm_failed_layers: set[int] = set()

    def _layer_metric_names(self) -> tuple[str, ...]:
        names = [
            "channel_max",
            "channel_median",
            "channel_p95",
            "channel_p99",
            "channel_max_ratio",
            "topk_channel_norm",
            "massive_act_channel_count",
        ]
        if self.log_post_norm_metrics:
            names.extend(["post_norm_sparsity", "post_norm_cosine"])
        if self.log_hidden_spectral_entropy:
            names.append("hidden_spectral_entropy")
        if self.log_activation_rms:
            names.extend(
                [
                    "activation_rms",
                    "activation_rms_std",
                    "spectral_norm_max",
                    "spectral_norm_min",
                ]
            )
        if self.log_lipschitz:
            names.extend(["lipschitz_max", "lipschitz_min"])
        for t in self.absolute_thresholds:
            names.append(f"channel_count_gt_{_threshold_key(t)}")
        return tuple(names)

    def _logit_lens_metric_names(self) -> tuple[str, ...]:
        """Logit-lens metric names to declare on head-owning eligible layers.

        Gated by the two logit-lens flags: entropy/logsumexp share the projection with
        cross-entropy, so a layer may declare any subset depending on which are enabled.
        """
        names = []
        if self.log_logit_lens_entropy:
            names.extend(
                ["logit_lens_entropy_mean", "logit_lens_logsumexp_mean"]
            )
        if self.log_logit_lens_cross_entropy:
            names.append("logit_lens_cross_entropy_mean")
        return tuple(names)

    def _logit_lens_enabled(self) -> bool:
        return self.log_logit_lens_entropy or self.log_logit_lens_cross_entropy

    def register_hooks(self, model: nn.Module, layer_offset: int = 0):
        """Register forward hooks. Single-chunk path.

        For multi-chunk models, prefer the two-phase setup in
        ``setup_massive_activation_monitor`` so all chunks declare keys before
        ``allocate_buffers`` locks the schema.
        """
        self._init_parallel_state()
        targets = self._prepare_layers(model, layer_offset=layer_offset)
        if not targets:
            return
        self.allocate_buffers(next(model.parameters()).device)
        self._attach_hooks(targets, model=model)

    def _init_parallel_state(self):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                self.tp_size = (
                    parallel_state.get_tensor_model_parallel_world_size()
                )
                self.tp_group = parallel_state.get_tensor_model_parallel_group()
        except ImportError:
            pass

    def _prepare_layers(
        self, model: nn.Module, layer_offset: int = 0
    ) -> list[tuple[int, nn.Module]]:
        layers = self._find_transformer_layers(model)
        if not layers:
            logger.warning("[MassiveActMonitor] No transformer layers found!")
            return []

        targets: list[tuple[int, nn.Module]] = []
        for local_idx, layer in layers:
            global_idx = self._resolve_layer_idx(
                layer, local_idx, len(layers), layer_offset
            )
            if self.sample_layers and global_idx not in self.sample_layers:
                continue
            targets.append((global_idx, layer))

        # Resolve the LM head once per chunk for the logit-lens metrics (entropy and/or
        # cross-entropy share it). Returns (None, None) on PP stages that don't own the
        # head weight, which keeps the metric a clean no-op there (no weight broadcast).
        head_weight, head_norm = (None, None)
        if self._logit_lens_enabled():
            head_weight, head_norm = self._resolve_lm_head(model)

        head_owned_eligible = False
        for global_idx, _ in targets:
            for name in self._layer_metric_names():
                self.declare_layer_metric(global_idx, name)
            # Logit-lens keys are declared ONLY on the head-owning chunk, and only for
            # the (optionally logit_lens_layers-filtered) eligible layers. Non-head
            # PP stages declare nothing here and attach no logit-lens hook.
            if head_weight is not None and (
                self.logit_lens_layers is None
                or global_idx in self.logit_lens_layers
            ):
                self._layer_head_info[global_idx] = (head_weight, head_norm)
                head_owned_eligible = True
                for name in self._logit_lens_metric_names():
                    self.declare_layer_metric(global_idx, name)
        # Cross-entropy needs the forward's `labels`; capture them once per head-owning
        # chunk via a top-level pre-hook (attached in _attach_hooks).
        if (
            self.log_logit_lens_cross_entropy
            and head_owned_eligible
            and not any(m is model for m in self._label_capture_models)
        ):
            self._label_capture_models.append(model)
        return targets

    def _attach_hooks(
        self,
        targets: list[tuple[int, nn.Module]],
        model: nn.Module | None = None,
    ):
        # Cross-entropy: capture the forward's `labels` kwarg once per head-owning chunk.
        # A model-level pre-hook fires before the layer hooks, so _captured_labels is
        # fresh when the logit-lens hook reads it. Attaches only where CE is enabled and
        # this chunk owns the head (recorded in _prepare_layers).
        if model is not None and any(
            m is model for m in self._label_capture_models
        ):
            cap_hook = model.register_forward_pre_hook(
                self._make_label_capture_hook(), with_kwargs=True
            )
            self.hooks.append(cap_hook)
        registered = 0
        for global_idx, layer in targets:
            norm_layer = getattr(layer, "input_layernorm", None)
            if norm_layer is not None:
                hook = norm_layer.register_forward_hook(
                    self.timed_hook(
                        "input_layernorm",
                        self._make_input_layernorm_hook(global_idx),
                    ),
                    with_kwargs=True,
                )
            else:
                hook = layer.register_forward_pre_hook(
                    self.timed_hook(
                        "residual", self._make_residual_hook(global_idx)
                    ),
                    with_kwargs=True,
                )
            self.hooks.append(hook)
            # Spectral-norm bounds need BOTH the layer input and output, so they
            # ride a dedicated forward hook on the layer itself (the residual/
            # input_layernorm hooks above only see the input).
            if self.log_activation_rms:
                spec_hook = layer.register_forward_hook(
                    self.timed_hook(
                        "spectral_norm",
                        self._make_spectral_norm_hook(global_idx),
                    ),
                    with_kwargs=True,
                )
                self.hooks.append(spec_hook)
            # Lipschitz / gradient-gain bounds are a backward-pass quantity, but we
            # attach from a forward hook that registers tensor grad hooks on the input
            # and output hidden states. A module full_backward_hook cannot recover the
            # input gradient here (Megatron calls layers all-keyword, so grad_input is
            # empty). The forward-hook grad guard also selects the recompute pass under
            # activation checkpointing, so the grad hooks fire exactly once.
            if self.log_lipschitz:
                grad_hook = layer.register_forward_hook(
                    self.timed_hook(
                        "grad_gain", self._make_grad_gain_hook(global_idx)
                    ),
                    with_kwargs=True,
                )
                self.hooks.append(grad_hook)
            # Logit-lens metrics (entropy/logsumexp and/or cross-entropy) ride their own
            # forward hook on the layer output, but only for layers on the head-owning
            # chunk (populated in _prepare_layers). The weight captured here is a live
            # Parameter, so each step reads the current (optimizer-updated) weight
            # without a per-hook lookup.
            head_info = self._layer_head_info.get(global_idx)
            if head_info is not None:
                weight, final_norm = head_info
                ll_hook = layer.register_forward_hook(
                    self.timed_hook(
                        "logit_lens",
                        self._make_logit_lens_hook(
                            global_idx, weight, final_norm
                        ),
                    ),
                    with_kwargs=True,
                )
                self.hooks.append(ll_hook)
            registered += 1
        logger.info(f"[MassiveActMonitor] Registered {registered} hooks.")

    def _find_transformer_layers(
        self, model: nn.Module
    ) -> list[tuple[int, nn.Module]]:
        if hasattr(model, "module"):
            model = model.module

        layers = None
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            layers = model.decoder.layers
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "decoder") and hasattr(lm.decoder, "layers"):
                layers = lm.decoder.layers

        if layers is None:
            return []

        return list(enumerate(layers))

    def _resolve_lm_head(self, model: nn.Module):
        """Return ``(lm_head_weight, final_norm)`` for this model chunk, else ``(None, None)``.

        Only the PP stage that owns the LM head (the last stage, or a tied-embedding
        stage) exposes the output weight; every other stage returns ``(None, None)``
        so the logit-lens metric is a clean no-op there — no weight is broadcast.
        Uses the tie-safe ``shared_embedding_or_output_weight()`` accessor when present
        (handles tied input/output embeddings), falling back to ``output_layer.weight``.
        The final norm (``decoder.final_layernorm``) is what the LM head was trained on,
        so intermediate residuals are normed through it before projection.
        """
        if hasattr(model, "module"):
            model = model.module

        weight = None
        getter = getattr(model, "shared_embedding_or_output_weight", None)
        if callable(getter):
            try:
                weight = getter()
            except Exception:
                weight = None
        if weight is None:
            output_layer = getattr(model, "output_layer", None)
            weight = (
                getattr(output_layer, "weight", None)
                if output_layer is not None
                else None
            )
        if weight is None:
            return None, None

        final_norm = None
        decoder = getattr(model, "decoder", None)
        if decoder is not None:
            final_norm = getattr(decoder, "final_layernorm", None)
        return weight, final_norm

    def _extract_hidden_states(self, args, kwargs=None):
        if args:
            return args[0]
        if kwargs:
            for name in ("hidden_states", "input", "x"):
                if name in kwargs:
                    return kwargs[name]
        return None

    def _first_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, tuple | list):
            for item in value:
                tensor = self._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _make_input_layernorm_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs, output):
            if not self._should_monitor():
                return
            try:
                hidden_states = self._extract_hidden_states(args, kwargs)
                if hidden_states is None:
                    return
                normalized = self._first_tensor(output)

                with torch.no_grad():
                    self._compute_residual_metrics(
                        layer_idx, hidden_states.detach()
                    )
                    if self.log_post_norm_metrics and normalized is not None:
                        self._compute_post_norm_metrics(
                            layer_idx, normalized.detach()
                        )
                    if (
                        self.log_hidden_spectral_entropy
                        and normalized is not None
                    ):
                        self._compute_spectral_entropy(
                            layer_idx, normalized.detach()
                        )
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[MassiveActMonitor] Error at layer {layer_idx}: {e}"
                    )

        return hook_fn

    def _make_residual_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs=None):
            if not self._should_monitor():
                return
            try:
                hidden_states = self._extract_hidden_states(args, kwargs)
                if hidden_states is None:
                    return

                with torch.no_grad():
                    self._compute_residual_metrics(
                        layer_idx, hidden_states.detach()
                    )
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[MassiveActMonitor] Error at layer {layer_idx}: {e}"
                    )

        return hook_fn

    def _make_spectral_norm_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs, output):
            if not self._should_monitor():
                return
            try:
                pre = self._extract_hidden_states(args, kwargs)
                post = self._first_tensor(output)
                if pre is None or post is None:
                    return

                with torch.no_grad():
                    self._compute_spectral_norm(
                        layer_idx, pre.detach(), post.detach()
                    )
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[MassiveActMonitor] Spectral-norm error at layer {layer_idx}: {e}"
                    )

        return hook_fn

    def _compute_spectral_norm(
        self, layer_idx: int, pre: torch.Tensor, post: torch.Tensor
    ):
        if pre.shape[-1] == 0 or pre.numel() == 0 or post.numel() == 0:
            return
        # pre is the layer input residual — the same tensor the input_layernorm /
        # residual hook sees — so activation_rms is derived here for free from the
        # per-token pre-RMS instead of squaring the input a second time.
        tensor_metrics = compute_spectral_norm_bounds(
            pre, post, include_activation_rms=True
        )
        for name, val in tensor_metrics.items():
            self.record_layer_metric(layer_idx, name, val)

    def _make_grad_gain_hook(self, layer_idx: int):
        """Forward hook that captures the layer's backward gradient-gain (Lipschitz).

        It registers tensor grad hooks on the input (``pre``) and output (``post``)
        hidden states. In the backward pass ``post``'s hook fires first (output->input)
        and stashes ∂L/∂y; ``pre``'s hook then reads it and records the per-token ratio
        ‖∂L/∂x‖/‖∂L/∂y‖ bounds. The per-invocation ``state`` dict isolates concurrent
        microbatch/pipeline backwards.
        """

        def hook_fn(module, args, kwargs, output):
            # _should_monitor() requires grad enabled, so the initial no-grad forward
            # under activation recompute is skipped and only the recompute (grad-enabled)
            # forward registers the grad hooks — they fire exactly once.
            if not self._should_monitor():
                return
            try:
                pre = self._extract_hidden_states(args, kwargs)
                post = self._first_tensor(output)
                if (
                    pre is None
                    or post is None
                    or not pre.requires_grad
                    or not post.requires_grad
                ):
                    return

                state: dict[str, torch.Tensor] = {}

                def save_grad_out(grad):
                    state["dy"] = grad.detach()

                def record_grad_gain(grad):
                    grad_out = state.pop("dy", None)
                    if grad_out is None:
                        return
                    with torch.no_grad():
                        tensor_metrics = compute_grad_gain_bounds(
                            grad_in=grad.detach(), grad_out=grad_out
                        )
                    for name, val in tensor_metrics.items():
                        self.record_layer_metric(layer_idx, name, val)

                post.register_hook(save_grad_out)
                pre.register_hook(record_grad_gain)
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[MassiveActMonitor] Grad-gain error at layer {layer_idx}: {e}"
                    )

        return hook_fn

    def _make_label_capture_hook(self):
        """Model-level forward pre-hook: stash the forward's ``labels`` kwarg.

        Fires before the layer forward hooks, so the logit-lens hook reads a fresh
        reference each monitored forward. Labels are ``[b, s]`` (Megatron passes them as
        a keyword); ``None`` on eval/inference forwards without labels — cross-entropy is
        then simply not emitted that step. Cheap: stashes one small int tensor reference.
        """

        def hook_fn(module, args, kwargs):
            self._captured_labels = kwargs.get("labels") if kwargs else None

        return hook_fn

    def _aligned_labels(self, post: torch.Tensor):
        """Align captured labels to ``post.reshape(-1, h)`` row order (seq-major).

        ``post`` is ``[s, b, h]``; Megatron labels are ``[b, s]`` → transpose to ``[s, b]``
        then flatten so row ``r`` maps to ``(pos=r//b, batch=r%b)``, matching the hidden
        flatten. Returns ``None`` on any orientation mismatch (compute_fn then skips CE).
        """
        labels = self._captured_labels
        if labels is None or post.dim() != 3:
            return None
        s, b = post.shape[0], post.shape[1]
        if labels.dim() == 2 and labels.shape[0] == b and labels.shape[1] == s:
            return labels.transpose(0, 1).reshape(-1)
        if labels.dim() == 2 and labels.shape[0] == s and labels.shape[1] == b:
            return labels.reshape(-1)
        if labels.dim() == 1 and labels.numel() == s * b:
            return labels.reshape(-1)
        return None

    def _make_logit_lens_hook(
        self, layer_idx: int, lm_head_weight: torch.Tensor, final_norm
    ):
        """Forward hook: predictive entropy + logsumexp (+ optional cross-entropy) of this
        layer's output via the logit lens.

        Projects the layer output residual through the LM head to vocab logits and records
        the token-mean entropy, logsumexp (log-partition), and — when cross-entropy is
        enabled and labels were captured — the token-mean cross-entropy against the
        ground-truth next token. Chunked over tokens, so the full ``[tokens, vocab]``
        logits are never materialized. Attached only on the head-owning PP stage; the
        weight is a live Parameter captured at attach time. Vocab-parallel TP is not
        supported yet — ``compute_logit_lens_entropy`` asserts ``tp_size <= 1``.
        """
        norm = final_norm if self.logit_lens_apply_final_norm else None
        want_entropy = self.log_logit_lens_entropy
        want_ce = self.log_logit_lens_cross_entropy

        def hook_fn(module, args, kwargs, output):
            if not self._should_monitor():
                return
            try:
                post = self._first_tensor(output)
                if post is None:
                    return
                labels_flat = self._aligned_labels(post) if want_ce else None
                with torch.no_grad():
                    tensor_metrics = compute_logit_lens_entropy(
                        post.detach(),
                        lm_head_weight,
                        final_norm=norm,
                        chunk_size=self.logit_lens_chunk_size,
                        tp_size=self.tp_size,
                        labels=labels_flat,
                        want_entropy=want_entropy,
                    )
                for name, val in tensor_metrics.items():
                    self.record_layer_metric(layer_idx, name, val)
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[MassiveActMonitor] Logit-lens error at layer {layer_idx}: {e}"
                    )

        return hook_fn

    def _compute_residual_metrics(
        self, layer_idx: int, hidden_states: torch.Tensor
    ):
        per_channel_max = compute_per_channel_max(hidden_states)
        per_channel_max = self._aggregate_per_channel_max(per_channel_max)
        tensor_metrics = summarize_per_channel_max(
            per_channel_max,
            threshold_multiplier=self.spike_threshold_multiplier,
            k=self.topk_channels,
            absolute_thresholds=self.absolute_thresholds,
        )

        for name, val in tensor_metrics.items():
            self.record_layer_metric(layer_idx, name, val)

    def _compute_post_norm_metrics(
        self, layer_idx: int, normalized: torch.Tensor
    ):
        try:
            sparsity = compute_post_norm_sparsity(
                normalized, epsilon=self.sparsity_epsilon
            )
            self.record_layer_metric(layer_idx, "post_norm_sparsity", sparsity)

            cosine = compute_post_norm_cosine_stability(
                normalized, num_sample_pairs=self.cosine_sample_pairs
            )
            self.record_layer_metric(layer_idx, "post_norm_cosine", cosine)
        except Exception as e:
            if self.verbose and layer_idx not in self._post_norm_failed_layers:
                logger.warning(
                    f"[MassiveActMonitor] Post-norm metrics disabled at layer {layer_idx}: {e}"
                )
                self._post_norm_failed_layers.add(layer_idx)

    def _compute_spectral_entropy(
        self, layer_idx: int, normalized: torch.Tensor
    ):
        if normalized.shape[-1] == 0 or normalized.numel() == 0:
            return
        entropy = compute_hidden_spectral_entropy(normalized)
        self.record_layer_metric(layer_idx, "hidden_spectral_entropy", entropy)

    def _aggregate_per_channel_max(
        self, per_channel_max: torch.Tensor
    ) -> torch.Tensor:
        """TP all-reduce on the per-channel-max vector.

        This collective is unavoidable for correctness (TP shards the channel
        dim across ranks). It runs once per hook on a length-H vector, which
        is much smaller than the QK-monitor collectives we eliminated.
        """
        if self.tp_size <= 1 or self.tp_group is None:
            return per_channel_max
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.all_reduce(
                    per_channel_max, op=dist.ReduceOp.MAX, group=self.tp_group
                )
        except Exception as e:
            if self.verbose and not self._warned_per_channel_aggregate:
                logger.warning(
                    f"[MassiveActMonitor] TP per-channel aggregation failed; using local values: {e}"
                )
                self._warned_per_channel_aggregate = True
        return per_channel_max


def setup_massive_activation_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    spike_threshold_multiplier: float = 100.0,
    topk_channels: int = 3,
    sparsity_epsilon: float = 0.01,
    cosine_sample_pairs: int = 256,
    sample_layers: list[int] | None = None,
    absolute_thresholds: tuple[float, ...] = DEFAULT_ABSOLUTE_THRESHOLDS,
    log_post_norm_metrics: bool = True,
    log_activation_rms: bool = True,
    log_lipschitz: bool = True,
    log_logit_lens_entropy: bool = False,
    logit_lens_chunk_size: int = 1024,
    logit_lens_apply_final_norm: bool = True,
    logit_lens_layers: list[int] | None = None,
    log_logit_lens_cross_entropy: bool = False,
    log_hidden_spectral_entropy: bool = False,
    hook_timing_enabled: bool = False,
    monitor_dict: dict | None = None,
    exclude_families=None,
):
    monitor = MassiveActivationMonitor(
        exclude_families=exclude_families,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        spike_threshold_multiplier=spike_threshold_multiplier,
        topk_channels=topk_channels,
        sparsity_epsilon=sparsity_epsilon,
        cosine_sample_pairs=cosine_sample_pairs,
        sample_layers=sample_layers,
        absolute_thresholds=absolute_thresholds,
        log_post_norm_metrics=log_post_norm_metrics,
        log_activation_rms=log_activation_rms,
        log_lipschitz=log_lipschitz,
        log_logit_lens_entropy=log_logit_lens_entropy,
        logit_lens_chunk_size=logit_lens_chunk_size,
        logit_lens_apply_final_norm=logit_lens_apply_final_norm,
        logit_lens_layers=logit_lens_layers,
        log_logit_lens_cross_entropy=log_logit_lens_cross_entropy,
        log_hidden_spectral_entropy=log_hidden_spectral_entropy,
        hook_timing_enabled=hook_timing_enabled,
    )

    models = [model] if not isinstance(model, list) else model
    monitor._init_parallel_state()
    chunk_targets = []
    layer_offset = 0
    for m in models:
        targets = monitor._prepare_layers(m, layer_offset=layer_offset)
        chunk_targets.append((m, targets))
        layer_offset += len(monitor._find_transformer_layers(m))
    if any(targets for _, targets in chunk_targets):
        device = next((p.device for m in models for p in m.parameters()), None)
        assert device is not None, (
            "no parameters across model chunks; cannot pick a device"
        )
        monitor.allocate_buffers(device)
        for m, targets in chunk_targets:
            monitor._attach_hooks(targets, model=m)
    logger.info(
        f"[MassiveActMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers."
    )

    if monitor_dict is not None:
        monitor_dict["massive_act"] = monitor

    return model
