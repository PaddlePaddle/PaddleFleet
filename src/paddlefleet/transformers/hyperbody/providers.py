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

"""HF-style config -> ``GPTConfig`` providers for the unified HyperBody model.

This module makes ``transformers/hyperbody`` self-contained: it copies the two
provider paths the unified model needs (previously imported from
``transformers/hyperbody_decoder`` and ``transformers/hyperencoder``) so the
unified modeling no longer reaches into those two packages.

* :class:`HyperEncoderConfig` -- a **transient** config used by
  ``modeling._build_encoder_view`` to strip the ``encoder_`` prefix off the flat
  :class:`HyperBodyConfig` into a plain HF-style config, then hand it to
  :class:`HyperEncoderProvider`. It never travels with a checkpoint on its own.
* :class:`HyperEncoderProvider` -- ``HyperEncoderConfig`` -> ``GPTConfig`` view.
  The unified model only uses ``from_config`` + attribute reads (it drives the
  encoder trunk via the spec builders in ``models/hyperbody``); it never calls
  ``provide()`` (the standalone ``HyperEncoderModel`` is replaced by the
  frontend/bridge wrapper layers), so ``provide()`` raises.
* :class:`HyperBodyDecoderModelProvider` + :func:`build_hyperbody_decoder_model`
  -- the decoder ``GPTConfig`` view. The unified model uses ``from_config`` to
  get a correctly-typed decoder view; ``provide()`` remains functional for
  completeness (it assembles a standalone decoder ``GPTModel``).

The layer specs come from ``paddlefleet.models.hyperbody`` (the self-contained
component package), NOT from ``models/hyperbody_decoder`` or
``models/hyperencoder``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import paddle
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import build_spec_layer
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.gpt_config import GPTConfig
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_spec
from paddlefleet.models.hyperbody import get_hyperbody_decoder_layer_specs

from ..configuration_utils import PretrainedConfig
from ..gpt_provider import GPTModel, GPTModelProvider
from ..model_provider import ModelProviderMixin

logger = logging.getLogger(__name__)

__all__ = [
    "HyperEncoderConfig",
    "HyperEncoderProvider",
    "HyperBodyDecoderModelProvider",
    "build_hyperbody_decoder_model",
]

# HyperEncoder geometry defaults (mirrored here so the provider path is
# self-contained). These match the production HyperEncoder values.
_ENC_VOCAB_SIZE = 129280
_ENC_HIDDEN_SIZE = 1280
_ENC_NUM_HEADS = 10
_ENC_FFN_HIDDEN = 6848
_ENC_MOE_FFN_HIDDEN = 896
_ENC_NUM_MOE_EXPERTS = 64
_ENC_NUM_LAYERS = 12

class HyperEncoderConfig(PretrainedConfig):
    """Transient HF-style config for the encoder view.

    Used by ``modeling._build_encoder_view`` to strip the ``encoder_`` prefix off
    the flat :class:`HyperBodyConfig` into a plain HF-style config, then hand it
    to :class:`HyperEncoderProvider`. It never travels with a checkpoint on its
    own. Field defaults are the fallback when a value is not provided; the flat
    config always overrides them.

    Transfer of fields to the Fleet side uses the framework's generic mechanism
    (``TransformerConfig.from_config``: ``object.__new__`` + ``register_attributes``
    + ``__post_init__``). Non-default architecture values are the field defaults
    of :class:`HyperEncoderProvider`, and derivations live in its ``__post_init__``.
    """

    model_type = "hyperencoder"

    def __init__(
        self,
        vocab_size=_ENC_VOCAB_SIZE,
        hidden_size=_ENC_HIDDEN_SIZE,
        intermediate_size=_ENC_FFN_HIDDEN,
        num_hidden_layers=_ENC_NUM_LAYERS,
        num_attention_heads=_ENC_NUM_HEADS,
        num_key_value_heads=_ENC_NUM_HEADS,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        attention_dropout=0.0,
        hidden_dropout_prob=0.0,
        attention_bias=False,
        moe_intermediate_size=_ENC_MOE_FFN_HIDDEN,
        n_routed_experts=_ENC_NUM_MOE_EXPERTS,
        num_experts_per_tok=6,
        n_shared_experts=2,
        first_k_dense_replace=1,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        scoring_func="softmax",
        topk_method="greedy",
        tie_word_embeddings=False,
        # ---- HyperEncoder-specific geometry (exposed here as config fields) ----
        hyperencoder_query_lengths=(256, 8192),
        hyperencoder_seq_align: int = 128,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_bias = attention_bias

        # MoE
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.first_k_dense_replace = first_k_dense_replace
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.topk_method = topk_method

        # HyperEncoder-specific geometry. json only carries a list, so normalize to a tuple.
        ql = tuple(int(v) for v in hyperencoder_query_lengths)
        if len(ql) != 2:
            raise ValueError(
                f"hyperencoder_query_lengths must be (short, long), got {ql}"
            )
        self.hyperencoder_query_lengths = ql
        self.hyperencoder_seq_align = int(hyperencoder_seq_align)

        # `tie_word_embeddings` is popped from kwargs by the base class (default
        # True), so it must be put back into kwargs rather than set directly --
        # a direct setattr would be overwritten by the base class.
        kwargs.setdefault("tie_word_embeddings", tie_word_embeddings)
        super().__init__(**kwargs)


def build_hyperbody_decoder_model(config, *, num_stages: int, loss_fn=None):
    """Assembly: fleet layer spec -> ``get_gpt_spec`` -> ``build_spec_layer``.

    A narrowed version of ``paddlefleet.gpt_builders.gpt_builder``. ``config`` is
    an already-converted ``GPTConfig`` (i.e. :class:`HyperBodyDecoderModelProvider`
    itself). Every branch this model does not use (MTP, head/tail EmptyLayer,
    ``separate_mtp_headloss``, ringmoe subgroups, meta-device init) is explicitly
    rejected rather than silently skipped.

    Args:
        num_stages: number of pipeline stages, equal to ``pipeline_model_parallel_size``.
        loss_fn: defaults to ``LanguageLoss(config)`` (matching ``gpt_builder``).
    """
    # These branches are unused by this model; silently skipping would fail silently later.
    if getattr(config, "mtp_num_layers", None):
        raise NotImplementedError("HyperBody decoder has no MTP layers.")
    if getattr(config, "separate_mtp_headloss", False):
        raise NotImplementedError("HyperBody decoder does not use separate_mtp_headloss.")
    if config.num_empty_layers_add_in_head or config.num_empty_layers_add_in_tail:
        raise NotImplementedError("HyperBody decoder inserts no EmptyLayer (pp split relies on seg_method).")
    if getattr(config, "moe_token_dispatcher_type", None) == "ringmoe":
        raise NotImplementedError("ringmoe needs world-level subgroup init, not supported by this model.")
    if getattr(config, "init_model_with_meta_device", False):
        raise NotImplementedError("HyperBody decoder does not use meta-device init.")

    gpt_spec = get_gpt_spec(
        config=config,
        head_empty_layers_spec=[],
        # The only line that differs from gpt_builder: spec comes from the fleet hyperbody component.
        transformer_layers_spec=get_hyperbody_decoder_layer_specs(config),
        tail_empty_layers_spec=[],
        mtp_layers_spec=None,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        max_sequence_length=config.max_sequence_length,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rope_theta,
        swa_rotary_base=config.swa_rope_theta,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
    )
    return build_spec_layer(
        gpt_spec,
        loss_fn=LanguageLoss(config) if loss_fn is None else loss_fn,
        num_stages=num_stages,
        # Same split criterion as GPTModelProvider.provide().
        seg_method="layer:TransformerLayer|EmptyLayer",
    )


@dataclass
class HyperBodyDecoderModelProvider(GPTModelProvider):
    """Landing point for HF-style config -> ``GPTConfig`` (``GPTModelProvider`` is itself a ``GPTConfig``).

    Only the switches that **must be pinned** are listed here. Geometry (num layers /
    hidden / experts / topk ...) and other numerics are injected off the flat
    :class:`HyperBodyConfig` via ``TransformerConfig.register_attributes`` and are
    not redeclared here.

    WARNING: ``from_config`` goes through ``object.__new__`` + ``register_attributes``
    and **does not run the dataclass ``__init__``**. The defaults below still take
    effect because a dataclass field with no ``default_factory`` is a class
    attribute. So do not change them to ``field(default_factory=...)``.
    """

    # ---- attention: MLA fully off, pure MHA (Paddle must turn it off explicitly) ----
    multi_latent_attention: bool = False
    use_qk_norm: bool = False

    # ---- position embedding: plain RoPE, no scaling ----
    # WARNING: must explicitly write None back: ``GPTConfig`` overrides the inherited
    # ``rope_scaling: dict = None`` to ``float = 1.0``, and ``gpt_provider`` only
    # checks ``is not None``, so ``1.0`` falls into the ``"mscale_all_dim" in
    # self.rope_scaling`` check and blows up with
    # ``TypeError: argument of type 'float' is not iterable``.
    rope_scaling: dict = None

    # ---- FFN / norm ----
    gated_linear_unit: bool = True
    normalization: str = "RMSNorm"

    # ---- MoE ----
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_shared_expert_overlap: bool = True

    # ---- fusion switches ----
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = False
    cross_entropy_loss_fusion: bool = False

    # ---- misc ----
    # GPTModelProvider defaults tie_word_embeddings to True; not overriding would
    # silently tie embedding and lm_head.
    tie_word_embeddings: bool = False
    # Recompute disabled.
    recompute_granularity: str = None

    # ``dtype`` -> ``params_dtype`` is already handled by ``_process_attribute``;
    # writing it here explicitly only keeps the same shape as other fleet providers.
    transform_rules = {
        **GPTModelProvider.transform_rules,
        "dtype": "params_dtype",
    }

    def provide(self, pre_process=None, post_process=None, vp_stage=None, loss_fn=None) -> GPTModel:
        """Override the parent's ``provide()``, swapping assembly for :func:`build_hyperbody_decoder_model`.

        The parent's version calls ``gpt_builder`` (which picks its own layer spec);
        this model needs the spec provided by fleet ``hyperbody``, so it assembles
        by itself.
        """
        # Parent's rope flattening: ``GPTConfig`` may carry a ``rope_parameters`` wrapper.
        if getattr(self, "rope_parameters", None):
            if self.rope_parameters.get("rope_type", "default") != "default":
                self.rope_type = self.rope_parameters["rope_type"]
            if "rope_theta" in self.rope_parameters:
                self.rope_theta = self.rope_parameters["rope_theta"]
        if isinstance(self.rope_scaling, dict) and "mscale_all_dim" in self.rope_scaling:
            self.mscale_all_dim = self.rope_scaling["mscale_all_dim"]

        fleet_model = build_hyperbody_decoder_model(
            self,
            num_stages=self.pipeline_model_parallel_size,
            loss_fn=loss_fn,
        )
        # Convert FleetGPTModel into formers' GPTModel so it inherits PretrainedModel's
        # methods (attribute-by-attribute copy, same technique as parent
        # gpt_provider.provide()).
        model = GPTModel.__new__(GPTModel)
        for attr_name in dir(fleet_model):
            if not attr_name.startswith("__"):
                try:
                    setattr(model, attr_name, getattr(fleet_model, attr_name))
                except Exception:  # read-only attribute / property, just skip
                    pass
        return model


@dataclass
class HyperEncoderProvider(GPTConfig, ModelProviderMixin["HyperEncoderModel"]):
    """``HyperEncoderConfig`` -> ``GPTConfig`` view for the encoder trunk.

    The unified model only uses ``from_config`` + attribute reads (it drives the
    encoder trunk via the spec builders in ``models/hyperbody``); it never calls
    ``provide()`` (the standalone ``HyperEncoderModel`` is replaced by the
    frontend/bridge wrapper layers), so ``provide()`` raises.

    ``from_config`` uses ``object.__new__`` and does NOT run the dataclass
    ``__init__`` -- field defaults take effect because a simple default becomes a
    class attribute. So this class's fields must NOT use
    ``field(default_factory=...)``.
    """

    # ---- Structure ----
    gated_linear_unit: bool = True
    normalization: str = "RMSNorm"
    use_bias: bool = False  # add_bias_linear=False
    # ---- Positional encoding ----
    position_embedding_type: str = "rope"
    rotary_percent: float = 1.0
    # ---- MoE ----
    moe_token_dispatcher_type: str = "alltoall"
    moe_expert_fusion: bool = False  # moe_grouped_gemm=False
    moe_router_load_balancing_type: str = "seq_aux_loss"
    routed_scaling_factor_learnable: bool = False
    # ---- dtype ----
    params_dtype: paddle.dtype = paddle.bfloat16
    bf16: bool = True
    attention_softmax_in_fp32: bool = False
    # ---- Fusion switches ----
    masked_softmax_fusion: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = False
    # ---- Kernel-selection compatibility ----
    # Must stay False. It is a bundled switch that would force
    # `attention_softmax_in_fp32=True`, which conflicts with this model's
    # intended False.
    use_accuracy_compatible: bool = False

    # ---- HyperEncoder-specific geometry (config fields here) ----
    hyperencoder_query_lengths: tuple = (256, 8192)
    hyperencoder_seq_align: int = 128
    #: Output-projection width for encoder -> LLM.
    language_hidden_size: int = 1280

    # ---- Attention backend + packed-decoder path ----
    #: ``"dp"`` (dense per-layer mask) or ``"triton"`` (packed prefix-LM core).
    hyperencoder_attn_backend: str = "dp"
    #: Run the trunk as a single packed call. Requires the ``triton`` backend.
    hyperencoder_packed_decoder: bool = False

    # ---- Triton prefix-LM kernel tuning ----
    hyperencoder_triton_block_m: int = 64
    hyperencoder_triton_block_n: int = 64
    hyperencoder_triton_fwd_warps: int = 4
    hyperencoder_triton_fwd_stages: int = 2
    hyperencoder_triton_bwd_warps: int = 4
    hyperencoder_triton_bwd_stages: int = 2
    #: LRU cap for the exec-plan cache (0 disables caching).
    hyperencoder_triton_plan_cache_size: int = 64

    # No `transform_rules`: `HyperEncoderConfig` field names are exactly the same
    # as on the Fleet side, so none needs renaming.

    def __post_init__(self) -> None:
        """Four derivations plus one divisibility check.

        ``from_config`` runs ``register_attributes`` then this method, so the
        values seen here are already those from ``config.json`` / yaml;
        derivations must go here (class defaults would be overwritten).
        """
        if getattr(self, "params_dtype", None) is None:
            self.params_dtype = paddle.bfloat16

        # `hidden_act` can only be set here, NOT as a class field default --
        # `from_config` uses `object.__new__`, and a plain function on a class
        # attribute becomes a bound method.
        self.hidden_act = F.silu  # activation_func

        # Restore the Fleet defaults that ``register_attributes`` would flip via
        # the ``LlmMetaConfig`` defaults (these switch kernels).
        self.moe_router_fusion = False
        self.situ_glu_fusion = False
        self.fp32_residual_connection = False
        self.moe_expert_capacity_factor = None
        self.moe_subbatch_token_num_before_dispatch = None
        self.train_mtp_only = False
        self.pad_token_id = 0

        # Detach recompute first, so the parent's post_init sees "recompute off".
        for name, pinned, where in (
            ("recompute_method", "uniform", "recompute_method"),
            ("recompute_num_layers", 1, "recompute_num_layers"),
        ):
            got = getattr(self, name, None)
            if got is not None and got != pinned:
                raise ValueError(
                    f"{name}={got!r} is not allowed ({where} is pinned to {pinned!r}). "
                    "HyperEncoder recompute has only one degree of freedom: "
                    "`recompute_granularity in (None, 'full')`."
                )
        rg = getattr(self, "recompute_granularity", None)
        if rg not in (None, "full"):
            raise ValueError(
                f"recompute_granularity={rg!r} is not supported; only 'full' is "
                "allowed, set None to disable."
            )
        if getattr(self, "sequence_parallel", False):
            raise ValueError(
                "Do not set sequence_parallel explicitly: it is derived from "
                "`(tp_size > 1)`. Set only tensor_model_parallel_size."
            )

        _recompute_on = rg is not None
        self.recompute_granularity = None

        ql = tuple(int(v) for v in self.hyperencoder_query_lengths)
        if len(ql) != 2 or ql[0] <= 0 or ql[1] <= 0:
            raise ValueError(
                f"hyperencoder_query_lengths must be two positive integers, got {ql}"
            )
        self.hyperencoder_query_lengths = ql
        # seq_length = max_position_embeddings = long_q + 8192.
        self.max_sequence_length = ql[1] + 8192

        self.head_dim = self.hidden_size // self.num_attention_heads

        # Validate the attention-backend / packed-decoder / Triton-tuning fields
        # here so a bad config fails at construction rather than deep inside a
        # kernel launch.
        from paddlefleet.models.hyperencoder.attn_backend import (
            use_packed_decoder,
        )

        use_packed_decoder(self)
        for _name in (
            "hyperencoder_triton_block_m",
            "hyperencoder_triton_block_n",
            "hyperencoder_triton_fwd_warps",
            "hyperencoder_triton_fwd_stages",
            "hyperencoder_triton_bwd_warps",
            "hyperencoder_triton_bwd_stages",
        ):
            _v = int(getattr(self, _name))
            if _v <= 0:
                raise ValueError(
                    f"{_name} must be a positive integer, got {_v}"
                )
            setattr(self, _name, _v)
        _cache = int(self.hyperencoder_triton_plan_cache_size)
        if _cache < 0:
            raise ValueError(
                f"hyperencoder_triton_plan_cache_size must be >= 0, got {_cache}"
            )
        self.hyperencoder_triton_plan_cache_size = _cache
        # Only block_m == block_n is a validated configuration.
        if self.hyperencoder_triton_block_m != self.hyperencoder_triton_block_n:
            raise ValueError(
                "hyperencoder_triton_block_m must equal "
                "hyperencoder_triton_block_n (only symmetric block shapes are "
                f"supported), got {self.hyperencoder_triton_block_m} vs "
                f"{self.hyperencoder_triton_block_n}"
            )

        # ETP == TP; SP is derived from the real tp_size.
        tp = int(self.tensor_model_parallel_size or 1)
        self.expert_tensor_parallel_size = tp
        self.sequence_parallel = tp > 1

        # The encoder only supports PP=1, so VPP is meaningless. Normalize a
        # possible 1 from LlmMetaConfig to None; raise for any other value.
        vpp = getattr(self, "virtual_pipeline_model_parallel_size", None)
        if vpp not in (None, 1):
            raise ValueError(
                f"virtual_pipeline_model_parallel_size={vpp} is not supported: encoder only supports PP=1"
            )
        self.virtual_pipeline_model_parallel_size = None

        self._check_divisibility()

        super().__post_init__()

        # `first_k_dense_replace`'s only purpose is to let the parent derive
        # `moe_layer_freq`. After derivation it is reset to None.
        self.first_k_dense_replace = None

        # ---- Recompute. See the docstring: must be set after super() ----
        if _recompute_on:
            self.recompute_granularity = "full"
            self.recompute_method = "uniform"
            self.recompute_num_layers = 1

    def _check_divisibility(self) -> None:
        """The divisibility assertions for the model geometry."""
        tp = int(self.tensor_model_parallel_size or 1)
        ep = int(self.expert_model_parallel_size or 1)
        pp = int(self.pipeline_model_parallel_size or 1)
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("num_attention_heads", self.num_attention_heads),
            ("ffn_hidden_size", self.intermediate_size),
            ("moe_ffn_hidden_size", self.moe_intermediate_size),
        ):
            if value % tp != 0:
                raise ValueError(
                    f"encoder {name}={value} must be divisible by encoder TP={tp}"
                )
        if self.n_routed_experts % ep != 0:
            raise ValueError(
                f"encoder num_moe_experts={self.n_routed_experts} must be divisible by encoder EP={ep}"
            )
        if pp != 1:
            raise ValueError(
                "HyperEncoder currently supports encoder PP=1 only"
            )

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Not used by the unified model.

        The unified HyperBody model drives the encoder trunk via the spec
        builders in ``models/hyperbody`` and wraps the pipeline-hostile parts in
        the frontend/bridge layers; it only calls ``from_config`` + attribute
        reads on this view and never builds a standalone ``HyperEncoderModel``.
        """
        raise NotImplementedError(
            "HyperEncoderProvider.provide() is not available in the unified "
            "HyperBody model: the encoder trunk is built from layer specs and "
            "the frontend/bridge wrapper layers, not a standalone HyperEncoderModel."
        )
