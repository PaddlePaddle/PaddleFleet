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

"""Nested (HF ``sub_configs``) configuration for the unified HyperBody model.

Two DIFFERENT backbones live in one archive, so the config is split into a thin
global shell plus two independent sub-configs -- **no field is shared across the
seam except by explicit injection**:

* :class:`HyperBodyConfig` (top level) -- ONLY the fields that physically MUST be
  global: the parallelism topology (TP/PP/VPP/EP/CP/SP share one set of Fleet
  process groups; they cannot differ per region inside a single ``PipelineLayer``),
  the shared MoE/fusion runtime knobs both regions read, the single loss
  coefficient, the cross-seam token ids, and the dtype/init policy switches.
  It carries ``sub_configs = {"decoder_config", "encoder_config"}``.
* :class:`HyperBodyDecoderConfig` (``decoder_config``) -- the language-model
  geometry + MoE topology + vocab. Bare field names identical to the Fleet
  decoder provider.
* :class:`HyperBodyEncoderConfig` (``encoder_config``) -- the HyperEncoder
  geometry + MoE topology + vocab + the hyperencoder-specific query/backend
  knobs.

SENSITIVE-INFO POLICY: model geometry (dims / layer counts / head counts), MoE
topology (routed/shared experts, top-k), vocab size and the encoder query
lengths are **REQUIRED** -- they carry no code default and MUST be supplied by
``config.json``. Passing ``None`` (i.e. leaving them out) raises. Only benign,
non-structural policy knobs (dropout=0, eps, fusion switches, defaults that do
not leak architecture) keep functional defaults.

LOSS-EQUIVALENCE CONTRACT: :func:`modeling._build_decoder_view` reconstructs the
decoder provider's input namespace as ``{top-level globals} + {decoder_config
geometry}`` -- exactly the field set the previous flat config fed the provider --
so the materialized decoder/encoder views (and therefore the training loss) are
bit-identical to the pre-nesting flat config. See that function for the merge.

WHY geometry-as-plain-attrs is safe: ``PretrainedConfig.__init__`` routes only
``LlmMetaConfig._get_init()`` keys through ``set_expected_keys`` -- and that set
EXCLUDES ``model_conf`` (``hidden_size`` / ``num_hidden_layers`` /
``num_attention_heads`` / ``num_key_value_heads`` / ``num_experts_per_tok`` /
``intermediate_size`` / ``n_routed_experts``). So assigning them as plain
attributes is not clobbered by ``super().__init__``. The global model_attributes
switches (normalization / gated_linear_unit / position_embedding_type / ...)
ARE in ``_get_init``, so they must go through the top-level ``super().__init__``.
"""

from __future__ import annotations

from ..configuration_utils import PretrainedConfig

__all__ = [
    "HyperBodyConfig",
    "HyperBodyDecoderConfig",
    "HyperBodyEncoderConfig",
    "CONTEXT_TOKEN",
    "IM_PATCH_TOKEN",
    "AUDIO_PATCH_TOKEN",
    "VIDEO_TOKEN_SENTINEL",
]

# ---------------------------------------------------------------------------
# Special tokens (single source of truth, inlined so this package is
# self-contained). ``CONTEXT_TOKEN``: wherever <context> appears in input_ids,
# the encoder latents are spliced into the decoder embedding stream.
# ``IM_PATCH_TOKEN`` / ``AUDIO_PATCH_TOKEN``: image / audio patch placeholders
# consumed by the encoder frontend when building context embeds. These are
# special-token constants (not model geometry), so they keep code defaults.
# ---------------------------------------------------------------------------
IM_PATCH_TOKEN = 128815
AUDIO_PATCH_TOKEN = 128829
CONTEXT_TOKEN = 128830

# Sentinel for the (unused) video placeholder. ``gpt_embedding`` reads
# ``video_token_id`` unconditionally; -1 never matches a real token id, so the
# video branch stays inert.
VIDEO_TOKEN_SENTINEL = -1


def _require(value, name: str, owner: str):
    """Enforce the sensitive-info policy: geometry/topology/vocab MUST come from
    ``config.json``. A missing (``None``) value is a hard error rather than a
    silent code default, so architecture is never baked into the source.
    """
    if value is None:
        raise ValueError(
            f"{owner}.{name} is required and must be provided by config.json "
            f"(model geometry / MoE topology / vocab is sensitive and carries "
            f"no code default)."
        )
    return value


def build_moe_layer_freq(num_hidden_layers: int) -> list[int]:
    """Per-layer dense/MoE 0-1 table: ``[0] + [1]*(L-1)`` (layer 0 dense, rest MoE).

    WARNING: must be a list. Passing an int makes Paddle take ``i % N``, which is
    not the intended per-layer semantics.
    """
    if num_hidden_layers < 1:
        raise ValueError(
            f"num_hidden_layers must be >= 1, got {num_hidden_layers}"
        )
    return [0] + [1] * (num_hidden_layers - 1)


class HyperBodyDecoderConfig(PretrainedConfig):
    r"""Language-model (decoder) sub-config for the unified HyperBody archive.

    Bare field names identical to the Fleet decoder provider so
    ``modeling._build_decoder_view`` can merge them straight into the provider's
    input namespace. Geometry / MoE topology / vocab are REQUIRED (raise if
    absent); benign policy knobs keep functional defaults.
    """

    model_type = "hyperbody_decoder"
    base_config_key = "decoder_config"

    def __init__(
        self,
        # === REQUIRED geometry / topology / vocab (no code default) ===
        vocab_size: int | None = None,
        hidden_size: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        intermediate_size: int | None = None,
        n_routed_experts: int | None = None,
        moe_intermediate_size: int | None = None,
        num_experts_per_tok: int | None = None,
        n_shared_experts: int | None = None,
        # === optional geometry derivations ===
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        moe_layer_freq: list[int] | None = None,
        first_k_dense_replace: None = None,
        # === benign policy knobs (functional defaults) ===
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        rope_theta: float = 10000,
        rotary_percent: float = 1.0,
        max_position_embeddings: int = 8192,
        attention_dropout: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        use_bias: bool = False,
        attention_bias: bool = False,
        n_group: int = 1,
        topk_group: int = 1,
        scoring_func: str = "softmax",
        topk_method: str = "greedy",
        norm_topk_prob: bool = False,
        routed_scaling_factor: float = 1.0,
        routed_scaling_factor_learnable: bool = False,
        attention_softmax_in_fp32: bool = False,
        variable_seq_lengths: bool = True,
        **kwargs,
    ):
        # ---- required (sensitive) ----
        self.vocab_size = _require(vocab_size, "vocab_size", "decoder_config")
        self.hidden_size = _require(
            hidden_size, "hidden_size", "decoder_config"
        )
        self.num_hidden_layers = _require(
            num_hidden_layers, "num_hidden_layers", "decoder_config"
        )
        self.num_attention_heads = _require(
            num_attention_heads, "num_attention_heads", "decoder_config"
        )
        self.intermediate_size = _require(
            intermediate_size, "intermediate_size", "decoder_config"
        )
        self.n_routed_experts = _require(
            n_routed_experts, "n_routed_experts", "decoder_config"
        )
        self.moe_intermediate_size = _require(
            moe_intermediate_size, "moe_intermediate_size", "decoder_config"
        )
        self.num_experts_per_tok = _require(
            num_experts_per_tok, "num_experts_per_tok", "decoder_config"
        )
        self.n_shared_experts = _require(
            n_shared_experts, "n_shared_experts", "decoder_config"
        )
        # ---- derived geometry ----
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else self.num_attention_heads
        )
        self.head_dim = (
            head_dim
            if head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )
        self.moe_layer_freq = (
            build_moe_layer_freq(self.num_hidden_layers)
            if moe_layer_freq is None
            else list(moe_layer_freq)
        )
        self.first_k_dense_replace = first_k_dense_replace
        # ---- benign policy ----
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rotary_percent = rotary_percent
        self.max_position_embeddings = max_position_embeddings
        self.max_sequence_length = max_position_embeddings
        self.seq_length = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_bias = use_bias
        self.attention_bias = attention_bias
        self.n_group = n_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.routed_scaling_factor_learnable = routed_scaling_factor_learnable
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.variable_seq_lengths = variable_seq_lengths
        super().__init__(**kwargs)

    def to_diff_dict(self, saving_file=False):
        # REQUIRED-geometry policy: a bare ``HyperBodyDecoderConfig()`` (which the
        # base ``to_diff_dict`` builds to compute a default-diff) raises. Emit the
        # full dict instead -- geometry always differs from any base anyway.
        return self.to_dict(saving_file=saving_file)


class HyperBodyEncoderConfig(PretrainedConfig):
    r"""HyperEncoder sub-config for the unified HyperBody archive.

    Bare geometry field names plus the hyperencoder-specific query/backend knobs.
    ``modeling._build_encoder_view`` reads these (``config.encoder_config.*``) to
    populate the transient ``HyperEncoderConfig`` that drives the encoder spec
    builders. Geometry / MoE topology / vocab / query lengths are REQUIRED.
    """

    model_type = "hyperbody_encoder"
    base_config_key = "encoder_config"

    def __init__(
        self,
        # === REQUIRED geometry / topology / vocab / query (no code default) ===
        vocab_size: int | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        moe_intermediate_size: int | None = None,
        n_routed_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        n_shared_experts: int | None = None,
        first_k_dense_replace: int | None = None,
        hyperencoder_query_lengths: tuple[int, int] | None = None,
        # === optional geometry derivation ===
        num_key_value_heads: int | None = None,
        # === benign policy knobs (functional defaults) ===
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000,
        attention_dropout: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        attention_bias: bool = False,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = False,
        scoring_func: str = "softmax",
        topk_method: str = "greedy",
        hyperencoder_seq_align: int = 128,
        hyperencoder_attn_backend: str = "dp",
        hyperencoder_packed_decoder: bool = False,
        **kwargs,
    ):
        # ---- required (sensitive) ----
        self.vocab_size = _require(vocab_size, "vocab_size", "encoder_config")
        self.hidden_size = _require(
            hidden_size, "hidden_size", "encoder_config"
        )
        self.intermediate_size = _require(
            intermediate_size, "intermediate_size", "encoder_config"
        )
        self.num_hidden_layers = _require(
            num_hidden_layers, "num_hidden_layers", "encoder_config"
        )
        self.num_attention_heads = _require(
            num_attention_heads, "num_attention_heads", "encoder_config"
        )
        self.moe_intermediate_size = _require(
            moe_intermediate_size, "moe_intermediate_size", "encoder_config"
        )
        self.n_routed_experts = _require(
            n_routed_experts, "n_routed_experts", "encoder_config"
        )
        self.num_experts_per_tok = _require(
            num_experts_per_tok, "num_experts_per_tok", "encoder_config"
        )
        self.n_shared_experts = _require(
            n_shared_experts, "n_shared_experts", "encoder_config"
        )
        self.first_k_dense_replace = _require(
            first_k_dense_replace, "first_k_dense_replace", "encoder_config"
        )
        ql = _require(
            hyperencoder_query_lengths,
            "hyperencoder_query_lengths",
            "encoder_config",
        )
        ql = tuple(int(v) for v in ql)
        if len(ql) != 2:
            raise ValueError(
                f"encoder_config.hyperencoder_query_lengths must be (short, long), got {ql}"
            )
        self.hyperencoder_query_lengths = ql
        # ---- derived geometry ----
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else self.num_attention_heads
        )
        # ---- benign policy ----
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_bias = attention_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.hyperencoder_seq_align = int(hyperencoder_seq_align)
        self.hyperencoder_attn_backend = hyperencoder_attn_backend
        self.hyperencoder_packed_decoder = hyperencoder_packed_decoder
        # The 'triton' frontend installs a PrefixLMTritonCore that only accepts a
        # packed decoder layout (it errors on a dense attention_mask and needs the
        # prefix_lm_layout). ``attn_backend.use_packed_decoder`` only rejects
        # packed+dp, NOT triton+non-packed, so guard that combination here (as the
        # config once did via __post_init__). Case-insensitive to mirror the
        # backend normalization elsewhere.
        if (
            str(hyperencoder_attn_backend).lower() == "triton"
            and not hyperencoder_packed_decoder
        ):
            raise ValueError(
                "encoder_config.hyperencoder_attn_backend='triton' requires "
                "hyperencoder_packed_decoder=True (the triton PrefixLM core only "
                "supports a packed decoder layout)."
            )
        super().__init__(**kwargs)

    def to_diff_dict(self, saving_file=False):
        # REQUIRED-geometry policy: a bare ``HyperBodyEncoderConfig()`` raises, so
        # skip the base default-diff and emit the full dict.
        return self.to_dict(saving_file=saving_file)


# Explicit allowlist of the DECODER geometry fields the decoder provider needs
# from ``decoder_config``. ``modeling._build_decoder_view`` pulls exactly these
# (NOT decoder_config's llm_meta pollution) so the merged provider namespace ==
# the previous flat config's namespace, guaranteeing loss equivalence.
DECODER_VIEW_KEYS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "hidden_act",
    "rms_norm_eps",
    "initializer_range",
    "use_cache",
    "rope_theta",
    "rotary_percent",
    "max_position_embeddings",
    "max_sequence_length",
    "seq_length",
    "attention_dropout",
    "hidden_dropout_prob",
    "use_bias",
    "attention_bias",
    "n_routed_experts",
    "moe_intermediate_size",
    "num_experts_per_tok",
    "n_shared_experts",
    "moe_layer_freq",
    "first_k_dense_replace",
    "n_group",
    "topk_group",
    "scoring_func",
    "topk_method",
    "norm_topk_prob",
    "routed_scaling_factor",
    "routed_scaling_factor_learnable",
    "attention_softmax_in_fp32",
    "variable_seq_lengths",
)


class HyperBodyConfig(PretrainedConfig):
    r"""Top-level (global) config for the unified HyperBody model.

    Holds ONLY the fields that must be global (see module docstring): parallelism
    topology, shared MoE/fusion runtime, single loss coefficient, cross-seam
    token ids, dtype/init policy. Nests ``decoder_config`` + ``encoder_config``.

    The routed ``super().__init__`` list mirrors the previous flat config exactly
    so the top-level ``__dict__`` reproduces the old global field state
    bit-for-bit; combined with the ``decoder_config`` geometry merge in
    ``modeling._build_decoder_view`` this keeps the training loss identical.
    """

    model_type = "hyperbody"
    # Composite config (nests two PretrainedConfig sub-configs). This tells the
    # base ``to_diff_dict`` NOT to instantiate a bare ``self.__class__()`` for the
    # default-diff -- an empty HyperBodyConfig would construct empty sub-configs
    # and correctly raise (geometry is REQUIRED). HF marks every sub_configs
    # config this way (e.g. Qwen2_5_VLConfig); serialization stays full-fidelity
    # because sub-configs are emitted via their own ``to_dict``.
    is_composition = True
    sub_configs = {
        "decoder_config": HyperBodyDecoderConfig,
        "encoder_config": HyperBodyEncoderConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # ===================== SUB-CONFIGS =====================
        decoder_config=None,
        encoder_config=None,
        # ===================== GLOBAL: cross-seam token ids =====================
        image_token_id: int = CONTEXT_TOKEN,
        video_token_id: int = VIDEO_TOKEN_SENTINEL,
        hyperbody_context_token_id: int = CONTEXT_TOKEN,
        # ===================== GLOBAL: dtype / init policy (plain) =====================
        bf16: bool = True,
        calculate_per_token_loss: bool = False,
        bias_activation_fusion: bool = True,
        masked_softmax_fusion: bool = True,
        bias_dropout_fusion: bool = True,
        cross_entropy_loss_fusion: bool = False,
        use_cpu_initialization: bool = False,
        pp_seg_method: str = "layer:TransformerLayer|EmptyLayer",
        # ===================== GLOBAL: routed llm_meta (via super) =====================
        # architecture switches (both regions; encoder provider re-pins its own)
        multi_latent_attention: bool = False,
        use_qk_norm: bool = False,
        normalization: str = "RMSNorm",
        gated_linear_unit: bool = True,
        position_embedding_type: str = "rope",
        fp32_residual_connection: bool = False,
        apply_rope_fusion: bool = False,
        # shared MoE runtime
        moe_token_dispatcher_type: str = "alltoall",
        moe_expert_fusion: bool = True,
        moe_router_load_balancing_type: str = "seq_aux_loss",
        moe_shared_expert_overlap: bool = True,
        router_aux_loss_coef: float = 0.001,
        # single loss / accuracy target
        use_accuracy_compatible: bool = False,
        tie_word_embeddings: bool = False,
        # parallelism topology (overridden at runtime by yaml)
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        sequence_parallel: bool = False,
        **kwargs,
    ):
        # ---- materialize sub-configs (HF sub_configs pattern) ----
        if isinstance(decoder_config, dict):
            self.decoder_config = self.sub_configs["decoder_config"](
                **decoder_config
            )
        elif isinstance(decoder_config, HyperBodyDecoderConfig):
            self.decoder_config = decoder_config
        else:
            # No silent geometry: an absent decoder_config would raise inside the
            # sub-config's required-field checks, which is the intended behavior.
            self.decoder_config = self.sub_configs["decoder_config"]()

        if isinstance(encoder_config, dict):
            self.encoder_config = self.sub_configs["encoder_config"](
                **encoder_config
            )
        elif isinstance(encoder_config, HyperBodyEncoderConfig):
            self.encoder_config = encoder_config
        else:
            self.encoder_config = self.sub_configs["encoder_config"]()

        # ---- global plain fields (set before super, mirroring old flat order) ----
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.hyperbody_context_token_id = hyperbody_context_token_id
        self.bf16 = bf16
        self.calculate_per_token_loss = calculate_per_token_loss
        self.bias_activation_fusion = bias_activation_fusion
        self.masked_softmax_fusion = masked_softmax_fusion
        self.bias_dropout_fusion = bias_dropout_fusion
        self.cross_entropy_loss_fusion = cross_entropy_loss_fusion
        self.use_cpu_initialization = use_cpu_initialization
        self.pp_seg_method = pp_seg_method

        # WARNING: llm_meta fields must go through kwargs so set_expected_keys does
        # not reset them to LlmMetaConfig defaults. This list is IDENTICAL to the
        # previous flat config's routing, so top-level __dict__ reproduces the old
        # global state exactly.
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            multi_latent_attention=multi_latent_attention,
            use_qk_norm=use_qk_norm,
            normalization=normalization,
            gated_linear_unit=gated_linear_unit,
            position_embedding_type=position_embedding_type,
            fp32_residual_connection=fp32_residual_connection,
            apply_rope_fusion=apply_rope_fusion,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_fusion=moe_expert_fusion,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_shared_expert_overlap=moe_shared_expert_overlap,
            router_aux_loss_coef=router_aux_loss_coef,
            use_accuracy_compatible=use_accuracy_compatible,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            context_parallel_size=context_parallel_size,
            sequence_parallel=sequence_parallel,
            **kwargs,
        )

    def __getattr__(self, key):
        """Convenience read-through to ``decoder_config`` for decoder geometry.

        Only invoked when normal attribute lookup FAILS -- i.e. for keys NOT in
        this object's ``__dict__``. All global llm_meta keys ARE in ``__dict__``
        (set by ``set_expected_keys``), so they resolve normally and NEVER reach
        here; this only surfaces decoder geometry (``hidden_size`` /
        ``num_hidden_layers`` / ...) that lives solely in ``decoder_config``. It
        can therefore never leak ``decoder_config``'s llm_meta defaults over a
        real global value. Encoder geometry is intentionally NOT forwarded (read
        it explicitly via ``config.encoder_config.*``) to avoid decoder/encoder
        ambiguity for the shared bare names.
        """
        dec = self.__dict__.get("decoder_config")
        if dec is not None and key in dec.__dict__:
            return getattr(dec, key)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {key!r}"
        )

    def to_diff_dict(self, saving_file=False):
        """Serialize, emitting the two sub-configs as FULL dicts.

        The base ``to_diff_dict`` diffs each ``sub_configs`` entry via
        ``recursive_diff_dict``, which builds a baseline by calling
        ``sub_config.__class__()`` with NO args. Under the sensitive-info policy
        the sub-configs REQUIRE geometry (a bare ``HyperBodyDecoderConfig()``
        correctly raises), so that baseline construction would crash during any
        ``repr``/save. We therefore serialize the top-level globals via the base
        logic (``is_composition=True`` already suppresses the top-level bare
        ``self.__class__()``) and drop in each sub-config's complete ``to_dict``
        -- which is fully round-trippable and never needs a bare instance.
        """
        config_dict = self.to_dict(saving_file=saving_file)
        default_config_dict = PretrainedConfig().to_dict(
            saving_file=saving_file
        )

        serializable_config_dict = {}
        for key, value in config_dict.items():
            if key in self.sub_configs:
                # ``value`` is already the sub-config's full to_dict() (see
                # PretrainedConfig.to_dict nested-config handling).
                serializable_config_dict[key] = value
                continue
            if key == "tie_word_embeddings":
                serializable_config_dict[key] = value
                continue
            if key == "quantization_config":
                q = self.quantization_config.to_diff_dict()
                if len(q) > 0:
                    serializable_config_dict[key] = q
                continue
            if (
                key not in default_config_dict
                or key == "paddlefleet_version"
                or value != default_config_dict[key]
            ):
                serializable_config_dict[key] = value

        self._remove_keys_not_serialized(serializable_config_dict, saving_file)
        serializable_config_dict.pop("_unsavable_keys", None)
        return serializable_config_dict
