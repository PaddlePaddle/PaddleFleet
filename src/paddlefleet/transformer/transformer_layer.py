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
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    ScheduleNode,
    build_spec_layer,
)
from paddle.distributed.fleet.utils import recompute
from paddlefleet_ops import is_deep_ep_available

from paddlefleet.parallel_state import (
    get_context_parallel_world_size,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.recompute_utils import (
    has_recovered,
    install_recompute_p2p_overlap,
    keep_indexer_grad_path,
    module_needs_recompute,
    need_full_recompute,
)
from paddlefleet.tensor_parallel import RecomputeWithoutOutput
from paddlefleet.transformer.dsv4_hybrid_attention import DSv4HybridAttention
from paddlefleet.transformer.identity_op import IdentityFuncOp, IdentityOp
from paddlefleet.transformer.kimi_delta_attention import KimiDeltaAttention
from paddlefleet.transformer.mlp import MLP
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.multi_latent_attention import MultiLatentAttention
from paddlefleet.transformer.utils import profile
from paddlefleet.utils import log_single_rank

if is_deep_ep_available():
    if paddle.is_compiled_with_cuda():
        from paddlefleet_ops import deep_ep
    else:
        from paddle.distributed.communication import deep_ep

if TYPE_CHECKING:
    from paddlefleet.packed_seq_params import PackedSeqParams
    from paddlefleet.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


def is_mtp_shared_last_layer(config, layer_number, is_mtp_layer):
    """Whether this transformer layer is the MTP-shared backbone last layer.

    When ``mtp_shared_last_layer`` is enabled, the backbone's last transformer
    layer shares (aliases) its weights with the MTP layer. Those shared params
    must use dedicated "no_hook" colors so the sharding-stage1 optimizer places
    them in their own comm buffers and reduces them synchronously instead of via
    the per-param overlap hook (which would fire from multiple detached autograd
    graphs and break the comm buffer bookkeeping). See MuonShardingOptimizer.

    Returns False when sharing is off, when sharding stage1 comm-overlap is
    disabled, when MTP is absent, for the MTP layer itself (its params are
    aliases owned by the backbone), or for non-last layers.
    """
    if not getattr(config, "mtp_shared_last_layer", False):
        return False
    # The no-hook color only exists to keep the sharding-stage1 comm-overlap
    # per-param backward hook from double-firing on the aliased shared params.
    # With stage1 overlap off there is no such hook, so no re-coloring is needed.
    if not getattr(config, "stage1_overlap", False):
        return False
    # Only re-color when MTP is actually present in the model.
    if (getattr(config, "num_nextn_predict_layers", 0) or 0) <= 0:
        return False
    if is_mtp_layer:
        return False
    last_layer_number = (
        config.num_hidden_layers
        - 1
        + getattr(config, "num_empty_layers_add_in_head", 0)
    )
    return layer_number == last_layer_number


def tensors_clone(outputs):
    """
    The tensors required for recompute_forward need to be cloned to prevent them from being released prematurely and becoming inaccessible.
    """
    if isinstance(outputs, paddle.Tensor):
        return outputs.clone()
    elif isinstance(outputs, (tuple, list)):
        res = []
        for item in outputs:
            if isinstance(item, paddle.Tensor):
                res_item = item.clone()
                res.append(res_item)
            else:
                if isinstance(item, dict):
                    res_item = tensors_clone(item)
                    res.append(res_item)
                else:
                    res.append(item)
        if isinstance(outputs, tuple):
            return tuple(res)
        else:
            return res
    elif isinstance(outputs, dict):
        res = {}
        for key, value in outputs.items():
            res[key] = value.clone()
        return res
    else:
        raise ValueError(
            f"Unsupported data type:{type(outputs)} in tensors_clone"
        )


@dataclass
class TransformerLayerSublayersSpec:
    """
    Configuration class for specifying the sublayers_spec of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (LayerSpec | type): Specification for the input layer normalization.
        self_attn (LayerSpec | type): Specification for the self-attention mechanism.
        self_attn_bda (LayerSpec | type): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm (LayerSpec | type): Specification for the layer
            normalization before cross-attention.
        cross_attention (LayerSpec | type): Specification for the cross-attention mechanism.
        cross_attn_bda (LayerSpec | type): Specification for the bias-dropout-add operation
            after cross-attention.
        post_attention_layernorm (LayerSpec | type): Specification for the layer normalization
            before the MLP.
        mlp (LayerSpec | type): Specification for the MLP in Dense layer.
        mlp_bda (LayerSpec | type): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    input_layernorm: LayerSpec | type = IdentityOp
    self_attention_hyper_connection: LayerSpec | type = IdentityOp
    self_attn: LayerSpec | type = IdentityOp
    self_attn_bda: LayerSpec | type = IdentityFuncOp

    pre_cross_attn_layernorm: LayerSpec | type = IdentityOp
    cross_attention: LayerSpec | type = IdentityOp
    cross_attn_bda: LayerSpec | type = IdentityFuncOp

    post_attention_layernorm: LayerSpec | type = IdentityOp
    mlp_hyper_connection: LayerSpec | type = IdentityOp
    mlp: LayerSpec | type = IdentityOp
    mlp_bda: LayerSpec | type = IdentityFuncOp

    block_attn_res: LayerSpec | type = IdentityOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: dict[str, str] = field(default_factory=dict)


class TransformerLayer(nn.Layer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    _gpt_model_use_experimental_version = False
    _LOG_LAYER_MD5 = os.environ.get("LOG_LAYER_MD5", "0") == "1"
    _skip_mtp_probes = (
        False  # Set True during MTP forward to suppress MD5 probes
    )

    @staticmethod
    def _log_md5(tensor, name, layer_idx):
        """Log MD5 of a tensor for precision alignment debugging."""
        if (
            TransformerLayer._LOG_LAYER_MD5
            and TransformerLayer._gpt_model_use_experimental_version
        ):
            if TransformerLayer._skip_mtp_probes:
                return  # Skip MTP passes — EC has no MTP
            data = tensor.cast("float32").numpy().tobytes()
            md5 = hashlib.md5(data).hexdigest()
            rank = (
                paddle.distributed.get_rank()
                if paddle.distributed.is_initialized()
                else 0
            )
            print(
                f"[MD5 Probe] Rank={rank} Layer={layer_idx} {name} MD5={md5} shape={list(tensor.shape)}",
                flush=True,
            )

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: TransformerLayerSublayersSpec,
        layer_number: int = 1,
        hidden_dropout_prob: float | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__()

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.config = config
        # Every recompute span this layer registers has to be visible to the pp
        # scheduler, so install here rather than in one subclass: the base is the
        # only place every transformer variant passes through.
        install_recompute_p2p_overlap(config)
        TransformerLayer._gpt_model_use_experimental_version = (
            config.gpt_model_use_experimental_version
        )

        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        self.hidden_dropout_prob = (
            config.hidden_dropout_prob
            if hidden_dropout_prob is None
            else hidden_dropout_prob
        )

        norm_input_parallel = (
            self.config.sequence_parallel
            and self.config.tensor_model_parallel_size > 1
        )
        # [Layer 1: Input Layernorm] Optional Layernorm on the input data
        self.input_layernorm = build_spec_layer(
            sublayers_spec.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            input_is_parallel=norm_input_parallel,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[
                    self.layer_number
                ]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        attention_optional_kwargs["pg_collection"] = pg_collection

        # [Layer 2: SelfAttention]
        self.self_attn = build_spec_layer(
            sublayers_spec.self_attn,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Layer 3: BiasDropoutFusion]
        self.self_attn_bda = build_spec_layer(sublayers_spec.self_attn_bda)

        # [Layer 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_spec_layer(
            sublayers_spec.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            input_is_parallel=norm_input_parallel,
        )

        # [Layer 5: CrossAttention]
        self.cross_attention = build_spec_layer(
            sublayers_spec.cross_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Layer 6: BiasDropoutFusion]
        self.cross_attn_bda = build_spec_layer(
            sublayers_spec.cross_attn_bda, config=self.config
        )

        # [Layer 7: Pre MLP] Optional Layernorm before MLP
        self.post_attention_layernorm = build_spec_layer(
            sublayers_spec.post_attention_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            input_is_parallel=norm_input_parallel,
        )
        # [Layer 8: MLP block]
        additional_mlp_kwargs = {}

        # MLP expects tp_group but MoELayer expects pg_collection to be passed in.
        # We can change MLP to accept pg_collection but it makes the logic implicit
        # The conditional below is to make the logic explicit
        # if sublayers_spec.mlp is not a LayerSpec,we dont have to handle passing additional kwargs
        if isinstance(sublayers_spec.mlp, LayerSpec):
            if isinstance(sublayers_spec.mlp.layer, type) and issubclass(
                sublayers_spec.mlp.layer, MoELayer
            ):
                additional_mlp_kwargs["pg_collection"] = pg_collection
            elif sublayers_spec.mlp.layer == MLP:
                assert hasattr(pg_collection, "tp"), (
                    "TP process group is required for MLP in TransformerLayer"
                )
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unknown MLP type: {type(sublayers_spec.mlp)}. Using default kwargs.",
                )

        self.mlp = build_spec_layer(
            sublayers_spec.mlp, config=self.config, **additional_mlp_kwargs
        )
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(
                self.layer_number, is_mtp_layer=self.is_mtp_layer
            )

        # [Layer 9: BiasDropoutFusion]
        self.mlp_bda = build_spec_layer(sublayers_spec.mlp_bda)

        self.full_recompute = False
        self.recompute_input_layernorm = False
        self.recompute_post_attention_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == "full":
            self.full_recompute = need_full_recompute(
                self.layer_number, self.config
            )
        elif self.config.recompute_granularity == "selective":
            if module_needs_recompute("norm", self.layer_number, self.config):
                # Both norms share the "norm" entry; each is skipped when it has
                # been specialised away to an IdentityOp.
                self.recompute_input_layernorm = not isinstance(
                    self.input_layernorm, IdentityOp
                )
                self.recompute_post_attention_layernorm = not isinstance(
                    self.post_attention_layernorm, IdentityOp
                )
            self.recompute_mlp = module_needs_recompute(
                "mlp", self.layer_number, self.config
            )

        # [Layer 10: Block Attention Residuals] Optional
        self.attn_res_block_size = None
        if self.config.block_attention_residuals:
            assert self.recompute_mlp is False, (
                "block_attention_residuals cannot use selective recompute mlp."
            )
            if self.full_recompute:
                offload_settings = getattr(
                    self.config,
                    "decoderlayer_act_offload_settings",
                    {"type": "", "value": ""},
                ) or {"type": "", "value": ""}
                if offload_settings.get("type", ""):
                    raise ValueError(
                        "block_attention_residuals with full_recompute does not "
                        "support decoderlayer_act_offload_settings. Please "
                        "disable activation offload or block_attention_residuals."
                    )
            if self._should_skip_block_attn_res():
                # MTP layers do not use attention residual — use IdentityOp
                # to avoid creating params.
                self.block_attn_res_before_attention = IdentityOp()
                self.block_attn_res_before_mlp = IdentityOp()
            else:
                self.block_attn_res_before_attention = build_spec_layer(
                    sublayers_spec.block_attn_res, config=self.config
                )
                self.block_attn_res_before_mlp = build_spec_layer(
                    sublayers_spec.block_attn_res, config=self.config
                )
            self.attn_res_block_size = self.config.attn_res_block_size

        if hasattr(self.mlp, "rr_recompute_update"):
            self.mlp.rr_recompute_update(
                in_full_recompute=self.full_recompute,
                in_mlp_recompute=self.recompute_mlp,
            )

        self._mark_shared_no_hook_params()

    def _compute_act_offload_kwargs(self):
        """Compute activation offload kwargs based on decoderlayer_act_offload_settings."""
        decoderlayer_act_offload_settings = self.config.get(
            "decoderlayer_act_offload_settings", {"type": "", "value": ""}
        ) or {"type": "", "value": ""}
        setting_type = decoderlayer_act_offload_settings["type"]
        offload_value = decoderlayer_act_offload_settings["value"]
        offload_kwargs = {}
        if "mod" == setting_type:
            assert isinstance(offload_value, (list, tuple))
            v1, v2 = offload_value
            offload_kwargs["offload_indices"] = (
                [0] if self.layer_number % v1 == v2 else []
            )
        elif "layer_idxs" == setting_type:
            offload_kwargs["offload_indices"] = (
                [0] if self.layer_number in offload_value else []
            )
        return offload_kwargs

    def _mark_shared_no_hook_params(self):
        """Tag the MTP-shared transformer layer's dense params with a no-hook color.

        When ``mtp_shared_last_layer`` is enabled, the backbone's last
        transformer layer shares its weights with the MTP layer: the very same
        parameter tensors are reused in a second, detached autograd graph. Under
        sharding stage1 comm-overlap, the per-param backward hook that drives
        gradient communication would then fire from multiple graphs (e.g. under
        FP8 manual backward or recompute), breaking the comm buffer's check-in
        bookkeeping (``add_grad`` assert / duplicate reduce).

        To avoid this, the shared params live in dedicated "no_hook" color
        groups. MoE expert params are colored at creation time in
        ``MoELayer.set_layer_number`` (Paddle forbids reassigning ``color``), so
        here we only color the remaining plain dense params, which carry no
        color yet, with ``dense_weight_no_hook`` (default sharding group, same as
        plain dense params with color=None).
        """
        if not is_mtp_shared_last_layer(
            self.config, self.layer_number, self.is_mtp_layer
        ):
            return

        for p in self.parameters():
            color = getattr(p, "color", None)
            # MoE experts are already colored (moe_weight_no_hook) at creation
            # and Paddle forbids reassigning color, so skip anything already
            # colored; only uncolored dense params need the dense no-hook color.
            if isinstance(color, dict) or color not in (None, -1):
                continue
            p.color = {"color": "dense_weight_no_hook"}

    def _should_skip_block_attn_res(self):
        """Determine if this layer should skip block attention residuals.

        MTP layers should NOT do attention residual — they use standard
        residual connections instead.
        """
        if self.is_mtp_layer:
            return True
        return False

    def _is_block_boundary(self):
        """Determine if this layer is a block boundary for attention residuals.

        Each block spans ``attn_res_block_size`` transformer layers, and the
        layer whose index is a multiple of that span closes the previous
        block. This matches Kimi K3's ``layer_idx % attn_res_block_size == 0``.

        ``self.layer_number`` is the *physical* index, which the spec builders
        shift by ``num_empty_layers_add_in_head`` so the empty head layers
        occupy the leading slots (see
        ``gpt_layer_specs.get_gpt_decoder_layers_spec``). The schedule is
        defined on the logical decoder index, so undo that shift here --
        otherwise the whole block layout slides by the offset and logical
        layer 0 never opens a block.
        """
        block_span = self.attn_res_block_size
        if block_span <= 0:
            raise ValueError(
                "attn_res_block_size must be at least 1 when "
                "block_attention_residuals is enabled."
            )
        head_offset = (
            getattr(self.config, "num_empty_layers_add_in_head", 0) or 0
        )
        return (self.layer_number - head_offset) % block_span == 0

    def _forward_impl_block_attn_res_split_recompute(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        attn_mask_startend_row_indices: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        swa_rotary_pos_emb: Tensor | None = None,
        swa_rotary_pos_cos: Tensor | None = None,
        swa_rotary_pos_sin: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        input_ids: Tensor | None = None,
        origin_input_ids: Tensor | None = None,
        blocks: list | None = None,
        cu_seqlens: Tensor | None = None,
    ):
        """Forward with block_attention_residuals + full_recompute.

        block_attn_res runs outside recompute (PyLayer handles its own
        gradient checkpointing internally); attention and MLP each get
        their own recompute wrapper.
        """
        if blocks is None:
            blocks = []
        partial_block = hidden_states

        # --- block_attn_res_before_attention (NOT recomputed) ---
        hidden_states = self.block_attn_res_before_attention(
            partial_block, blocks
        )

        # Block boundary check
        if self._is_block_boundary():
            blocks.append(partial_block)
            partial_block = None

        # --- Attention (recomputed) ---
        # Clone tensors that may be modified in-place during attention
        _attn_mask_clone = (
            attn_mask_startend_row_indices.clone()
            if attn_mask_startend_row_indices is not None
            else None
        )
        _rotary_pos_emb_clone = (
            rotary_pos_emb.clone() if rotary_pos_emb is not None else None
        )
        _rotary_pos_cos_clone = (
            rotary_pos_cos.clone() if rotary_pos_cos is not None else None
        )
        _rotary_pos_sin_clone = (
            rotary_pos_sin.clone() if rotary_pos_sin is not None else None
        )
        _swa_rotary_pos_emb_clone = (
            swa_rotary_pos_emb.clone()
            if swa_rotary_pos_emb is not None
            else None
        )
        _swa_rotary_pos_cos_clone = (
            swa_rotary_pos_cos.clone()
            if swa_rotary_pos_cos is not None
            else None
        )
        _swa_rotary_pos_sin_clone = (
            swa_rotary_pos_sin.clone()
            if swa_rotary_pos_sin is not None
            else None
        )
        _position_ids_clone = (
            position_ids.clone() if position_ids is not None else None
        )

        def _recompute_attention(hidden_states):
            hs, ctx = self._forward_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=_attn_mask_clone,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=_rotary_pos_emb_clone,
                rotary_pos_cos=_rotary_pos_cos_clone,
                rotary_pos_sin=_rotary_pos_sin_clone,
                swa_rotary_pos_emb=_swa_rotary_pos_emb_clone,
                swa_rotary_pos_cos=_swa_rotary_pos_cos_clone,
                swa_rotary_pos_sin=_swa_rotary_pos_sin_clone,
                position_ids=_position_ids_clone,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                block_attention_residuals=True,
                in_recompute=True,
                input_ids=input_ids,
                cu_seqlens=cu_seqlens,
            )
            if ctx is None:
                return hs
            return hs, ctx

        attn_result = recompute(_recompute_attention, hidden_states)

        if isinstance(attn_result, tuple):
            hidden_states, context = attn_result
        else:
            hidden_states = attn_result
            context = None

        # Accumulate attn output into partial_block
        if (
            partial_block is not None
            and partial_block.dtype != hidden_states.dtype
        ):
            partial_block = partial_block.to(hidden_states.dtype)
        partial_block = (
            partial_block + hidden_states
            if partial_block is not None
            else hidden_states
        )

        # --- block_attn_res_before_mlp (NOT recomputed) ---
        hidden_states = self.block_attn_res_before_mlp(partial_block, blocks)

        # --- MLP (recomputed) ---
        def _recompute_mlp(hidden_states):
            return self._forward_mlp(
                hidden_states,
                block_attention_residuals=True,
                input_ids=input_ids,
                origin_input_ids=origin_input_ids,
            )

        mlp_out = recompute(_recompute_mlp, hidden_states)

        # Accumulate mlp output into partial_block
        output = partial_block + mlp_out

        if context is not None:
            return output, context
        return output

    def build_schedule_node(self):
        return TransformerLayerNode(
            self,
            self.config,
            name="TransformerLayerNode",
            layer_number=self.layer_number,
        )

    @property
    def transformer_layer_weights(self):
        return self.named_parameters()

    def _docmask_meta_kwargs(self):
        """Hook for the shared CSA document-mask metadata; opted out by default.

        Overridden by ``HyperConnectionTransformerLayer``, which is the layer
        class the DSv4-hybrid models use. Returning ``{}`` means ``_forward_impl``
        is called with exactly the arguments it was called with before, so every
        other layer class keeps building its own ``CSADocMaskMetadata``.
        """
        return {}

    def forward(
        self,
        dict_args: dict,
    ):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        dict_args.pop("dynamic_inference_decode_only", None)
        keys = tuple(dict_args.keys())
        values = tuple(dict_args.values())

        is_mtp = dict_args.pop("is_mtp", False)
        TransformerLayer._skip_mtp_probes = (
            is_mtp  # Suppress MD5 probes for MTP passes
        )
        mtp_input = None
        mtp_ids = None
        # Under use_erndata the data pipeline emits length-L
        # tensors (input_ids / labels / position_ids / attn_mask) and
        # GPTEmbedding produces mtp_emb_res as (K+1) length-L blocks — the
        # per-depth left-shift is performed inline via roll_tensor. That means
        # the main decoder here runs at seq_len = L (not L-K), so the ernie5
        # L+K path below must be skipped for position_ids / input_ids / mask
        # trims.
        _mtp_is_megatron = getattr(self.config, "use_erndata", False)
        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not is_mtp
            and not self.config.mtp_load_weight_only
            and not self.config.enable_mtp_magic_send
            and not self.config.separate_mtp_input
        ):
            # process hidden_states
            hidden_states_concat = dict_args["hidden_states"]
            tensor_list = paddle.split(
                hidden_states_concat, self.config.num_nextn_predict_layers + 1
            )
            hidden_states = tensor_list[0]
            mtp_input = tuple(tensor_list[1:])
            dict_args["hidden_states"] = hidden_states

            # process position_ids
            # Under use_erndata position_ids is [B, L] already
            # (roll happens inside MTP layer), so DO NOT strip K positions.
            if (
                not self.config.gpt_model_use_experimental_version
                and not _mtp_is_megatron
            ):
                if "position_ids" in dict_args.keys():
                    position_ids = dict_args["position_ids"]
                    # Slice the sequence axis, which is the last one for both
                    # [B, S] and mRoPE's [3, B, S].
                    decoder_ids = position_ids[
                        ..., : -self.config.num_nextn_predict_layers
                    ]
                    mtp_ids = position_ids[
                        ..., -self.config.num_nextn_predict_layers :
                    ]
                    dict_args["position_ids"] = decoder_ids

            # process rotary_pos_emb: trim to main decoder sequence length
            # With SP: rotary_pos_emb is [S, B, head_dim], seq is dim 0
            # Without SP: rotary_pos_emb is [B, S, head_dim] or [1, S, 1, head_dim], seq is dim 1
            # Compute main_seq_len from the split hidden_states (after AllGather for SP)
            if self.config.sequence_parallel:
                main_seq_len = (
                    hidden_states.shape[0]
                    * self.config.tensor_model_parallel_size
                )
            else:
                main_seq_len = hidden_states.shape[1]
            rotary_pos_emb_full = None
            if (
                "rotary_pos_emb" in dict_args.keys()
                and dict_args["rotary_pos_emb"] is not None
            ):
                rotary_pos_emb_full = dict_args["rotary_pos_emb"]
                if self.config.sequence_parallel:
                    dict_args["rotary_pos_emb"] = rotary_pos_emb_full[
                        :main_seq_len
                    ]
                else:
                    dict_args["rotary_pos_emb"] = rotary_pos_emb_full[
                        :, :main_seq_len
                    ]
            # rotary_pos_cos/sin are [B, S, head_dim] (not transposed)
            rotary_pos_cos_full = None
            if (
                "rotary_pos_cos" in dict_args.keys()
                and dict_args["rotary_pos_cos"] is not None
            ):
                rotary_pos_cos_full = dict_args["rotary_pos_cos"]
                dict_args["rotary_pos_cos"] = rotary_pos_cos_full[
                    :, :main_seq_len
                ]
            rotary_pos_sin_full = None
            if (
                "rotary_pos_sin" in dict_args.keys()
                and dict_args["rotary_pos_sin"] is not None
            ):
                rotary_pos_sin_full = dict_args["rotary_pos_sin"]
                dict_args["rotary_pos_sin"] = rotary_pos_sin_full[
                    :, :main_seq_len
                ]

            # process input_ids (for MoE padding mask): split into main and mtp parts
            mtp_input_ids = None
            if (
                "input_ids" in dict_args.keys()
                and dict_args["input_ids"] is not None
            ):
                full_input_ids = dict_args["input_ids"]

                # In EB dataflow and CP size > 1，shape of hidden_states is [b, s/cp, h]
                # but input_ids' shape is [b, s], so we need to get full seq_len here
                seq_lens = hidden_states.shape[
                    0 if self.config.sequence_parallel else 1
                ]
                if get_context_parallel_world_size() > 1:
                    seq_lens *= get_context_parallel_world_size()

                if full_input_ids.shape[-1] > seq_lens:
                    decoder_input_ids = full_input_ids[
                        :, : -self.config.num_nextn_predict_layers
                    ].contiguous()
                    mtp_input_ids = full_input_ids[
                        :, -self.config.num_nextn_predict_layers :
                    ].contiguous()
                    dict_args["input_ids"] = decoder_input_ids
            if (
                not self.config.experimental_dataflow
                and not _mtp_is_megatron
                and "attn_mask_startend_row_indices" in dict_args.keys()
            ):
                # Old dataflow (ernie5 L+K path): main mask contains mtp parts
                # appended along seq dim (total length L+K), split into main
                # [B,1,L,1] + mtp [B,1,K,1] and hand main to the backbone.
                attn_mask_startend_row_indices = dict_args[
                    "attn_mask_startend_row_indices"
                ]
                attn_mask_startend_row_indices_decoder = (
                    attn_mask_startend_row_indices[
                        :, :, : -self.config.num_nextn_predict_layers, :
                    ]
                )
                attn_mask_startend_row_indices_mtp = (
                    attn_mask_startend_row_indices[
                        :, :, -self.config.num_nextn_predict_layers :, :
                    ]
                )
                dict_args["attn_mask_startend_row_indices"] = (
                    attn_mask_startend_row_indices_decoder
                )
            else:
                # New dataflow (experimental_dataflow=True): main mask is already main-seq only,
                # mtp masks are in mtp_startend_row_indices_all and will be used by MTP layer directly.
                # Megatron style: mask is already length-L; MTP layer will derive per-depth mask
                # from cu_seqlens_q, so leave main mask untouched here.
                attn_mask_startend_row_indices_mtp = None

        if self.config.block_attention_residuals and "blocks" not in dict_args:
            dict_args["blocks"] = []

        # For block_attention_residuals: handle boundary logic OUTSIDE
        # recompute so that blocks list mutation doesn't happen twice
        # during backward re-execution.
        skip_block_attn_res = (
            self._should_skip_block_attn_res()
            if self.config.block_attention_residuals
            else True
        )
        if self.config.block_attention_residuals and skip_block_attn_res:
            # Remove blocks from dict_args so that _forward_impl does not
            # receive unused tensors that cause backward errors.
            dict_args.pop("blocks", None)
        # Shared CSA document-mask metadata (see _docmask_meta_kwargs). Taken HERE,
        # outside the recompute wrapper below: `forward` runs exactly once per
        # (layer, micro-batch) whatever the recompute granularity is, while
        # `_forward_impl` may be replayed. Empty dict for every layer class that
        # does not opt in.
        docmask_meta_kwargs = self._docmask_meta_kwargs()

        if self.full_recompute or (not has_recovered()):
            hidden_states = dict_args["hidden_states"]
            hidden_states = keep_indexer_grad_path(hidden_states, self.config)
            attention_mask = dict_args.get("attention_mask", None)
            attn_mask_startend_row_indices = dict_args.get(
                "attn_mask_startend_row_indices", None
            )
            context = dict_args.get("context", None)
            context_mask = dict_args.get("context_mask", None)
            rotary_pos_emb = dict_args.get("rotary_pos_emb", None)
            rotary_pos_cos = dict_args.get("rotary_pos_cos", None)
            rotary_pos_sin = dict_args.get("rotary_pos_sin", None)
            swa_rotary_pos_emb = dict_args.get("swa_rotary_pos_emb", None)
            swa_rotary_pos_cos = dict_args.get("swa_rotary_pos_cos", None)
            swa_rotary_pos_sin = dict_args.get("swa_rotary_pos_sin", None)
            position_ids = dict_args.get("position_ids", None)
            attention_bias = dict_args.get("attention_bias", None)
            packed_seq_params = dict_args.get("packed_seq_params", None)
            input_ids = dict_args.get("input_ids", None)
            offload_kwargs = self._compute_act_offload_kwargs()
            origin_input_ids = dict_args.get("origin_input_ids", None)
            # Only forward this when the embedding actually produced one:
            # recompute(use_reentrant=True) flattens kwargs into positional args,
            # so an unexpected key would overflow a _forward_impl override that
            # only takes it through **kwargs.
            cu_seqlens_kwargs = (
                {"cu_seqlens": dict_args["cu_seqlens"]}
                if "cu_seqlens" in dict_args
                else {}
            )

            if (
                self.config.block_attention_residuals
                and not skip_block_attn_res
            ):
                # block_attention_residuals + full_recompute:
                # attn_res runs outside recompute (PyLayer handles its own
                # gradient checkpointing); attention and MLP each get their
                # own recompute wrapper.
                outputs = self._forward_impl_block_attn_res_split_recompute(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    swa_rotary_pos_emb=swa_rotary_pos_emb,
                    swa_rotary_pos_cos=swa_rotary_pos_cos,
                    swa_rotary_pos_sin=swa_rotary_pos_sin,
                    position_ids=position_ids,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    input_ids=input_ids,
                    origin_input_ids=origin_input_ids,
                    blocks=dict_args.get("blocks", []),
                    **cu_seqlens_kwargs,
                )
            else:
                outputs = recompute(
                    self._forward_impl,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices.clone()
                    if attn_mask_startend_row_indices is not None
                    else None,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb.clone()
                    if rotary_pos_emb is not None
                    else None,
                    rotary_pos_cos=rotary_pos_cos.clone()
                    if rotary_pos_cos is not None
                    else None,
                    rotary_pos_sin=rotary_pos_sin.clone()
                    if rotary_pos_sin is not None
                    else None,
                    swa_rotary_pos_emb=swa_rotary_pos_emb.clone()
                    if swa_rotary_pos_emb is not None
                    else None,
                    swa_rotary_pos_cos=swa_rotary_pos_cos.clone()
                    if swa_rotary_pos_cos is not None
                    else None,
                    swa_rotary_pos_sin=swa_rotary_pos_sin.clone()
                    if swa_rotary_pos_sin is not None
                    else None,
                    position_ids=position_ids.clone()
                    if position_ids is not None
                    else None,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    input_ids=input_ids,
                    origin_input_ids=origin_input_ids,
                    **cu_seqlens_kwargs,
                    **docmask_meta_kwargs,
                    **offload_kwargs,
                )
        else:
            outputs = self._forward_impl(**dict_args, **docmask_meta_kwargs)

        if isinstance(outputs, tuple):
            output, context = outputs[0], outputs[1]
        else:
            output, context = outputs, None

        rst = OrderedDict()
        rst = {"hidden_states": output}
        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not is_mtp
            and not self.config.mtp_load_weight_only
            and not self.config.enable_mtp_magic_send
            and not self.config.separate_mtp_input
        ):
            hidden_states_concat = paddle.concat([output, *mtp_input])
            rst["hidden_states"] = hidden_states_concat
            # Under use_erndata the L+K position-ids split was
            # skipped up front, so there is nothing to concat back either.
            if (
                not self.config.gpt_model_use_experimental_version
                and not _mtp_is_megatron
            ):
                if "position_ids" in dict_args.keys() and mtp_ids is not None:
                    position_ids = paddle.concat(
                        [dict_args["position_ids"], mtp_ids], axis=-1
                    )
                    dict_args["position_ids"] = position_ids

            # Restore rotary_pos_emb/cos/sin to full length for next layer
            if rotary_pos_emb_full is not None:
                dict_args["rotary_pos_emb"] = rotary_pos_emb_full
            if rotary_pos_cos_full is not None:
                dict_args["rotary_pos_cos"] = rotary_pos_cos_full
            if rotary_pos_sin_full is not None:
                dict_args["rotary_pos_sin"] = rotary_pos_sin_full

            # Restore input_ids: concatenate main and mtp parts back
            if mtp_input_ids is not None and "input_ids" in dict_args.keys():
                dict_args["input_ids"] = paddle.concat(
                    [dict_args["input_ids"], mtp_input_ids], axis=1
                )

            if (
                not self.config.experimental_dataflow
                and "attn_mask_startend_row_indices" in dict_args.keys()
            ):
                if attn_mask_startend_row_indices_mtp is not None:
                    attn_mask_startend_row_indices = paddle.concat(
                        [
                            dict_args["attn_mask_startend_row_indices"],
                            attn_mask_startend_row_indices_mtp,
                        ],
                        axis=2,
                    )
                else:
                    # alignment mode: MTP split was skipped
                    attn_mask_startend_row_indices = dict_args[
                        "attn_mask_startend_row_indices"
                    ]
                dict_args["attn_mask_startend_row_indices"] = (
                    attn_mask_startend_row_indices
                )

            # New dataflow (experimental_dataflow=True): mtp_startend_row_indices_all passes through
            # dict_args unchanged and will be consumed by MTP layer directly
        if context is not None:
            rst["context"] = context
        rst = {**dict_args, **rst}
        return rst

    def _forward_impl(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        attn_mask_startend_row_indices: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        swa_rotary_pos_emb: Tensor | None = None,
        swa_rotary_pos_cos: Tensor | None = None,
        swa_rotary_pos_sin: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        input_ids: Tensor | None = None,
        origin_input_ids: Tensor | None = None,
        blocks: list | tuple | None = None,
        cu_seqlens: Tensor | None = None,
        docmask_mb_idx: int = -1,
        **kwargs,
    ):
        def need_do_attention():
            # need_do_prefill = forward_meta.max_len_tensor_cpu[1] > 0
            # need_do_decode = forward_meta.max_len_tensor_cpu[2] > 0
            # in fastdeploy mode , not need_do_prefill and not need_do_decode,
            # core_attention will return none, so pass self attention
            if (
                getattr(self, "training", True)
                or not self.config.multi_latent_attention
            ):
                return True
            if hasattr(self, "self_attn") and hasattr(
                self.self_attn, "core_attention"
            ):
                core_attn = self.self_attn.core_attention
                if hasattr(core_attn, "config") and hasattr(
                    core_attn.config, "forward_meta"
                ):
                    fm = core_attn.config.forward_meta
                    return not (
                        fm.max_len_tensor_cpu[1] <= 0
                        and fm.max_len_tensor_cpu[2] <= 0
                    )
                return True
            else:
                return True

        timer_name = "moe-mlp" if isinstance(self.mlp, MoELayer) else "mlp"
        if (
            self.config.block_attention_residuals
            and not self._should_skip_block_attn_res()
        ):
            if blocks is None:
                blocks = []
            elif isinstance(blocks, tuple):
                blocks = list(blocks)
            partial_block = hidden_states

            # Before attention: block attnres
            hidden_states = self.block_attn_res_before_attention(
                partial_block, blocks
            )

            # Block boundary: append current repr and reset partial_block
            if self._is_block_boundary():
                blocks.append(partial_block)
                partial_block = None

            # Self-attention (skip internal bda residual)
            with profile("attn"):
                if need_do_attention():
                    hidden_states, context = self._forward_attention(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        swa_rotary_pos_emb=swa_rotary_pos_emb,
                        swa_rotary_pos_cos=swa_rotary_pos_cos,
                        swa_rotary_pos_sin=swa_rotary_pos_sin,
                        position_ids=position_ids,
                        attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params,
                        block_attention_residuals=True,
                        in_recompute=self.full_recompute,
                        input_ids=input_ids,
                        cu_seqlens=cu_seqlens,
                        docmask_mb_idx=docmask_mb_idx,
                        **kwargs,
                    )

            # Accumulate attn output into partial_block
            if (
                partial_block is not None
                and partial_block.dtype != hidden_states.dtype
            ):
                partial_block = partial_block.to(hidden_states.dtype)
            partial_block = (
                partial_block + hidden_states
                if partial_block is not None
                else hidden_states
            )

            # Before MLP: block attnres
            hidden_states = self.block_attn_res_before_mlp(
                partial_block, blocks
            )

            # MLP (skip internal bda residual)
            with profile(timer_name):
                mlp_out = self._forward_mlp(
                    hidden_states,
                    block_attention_residuals=True,
                    input_ids=input_ids,
                    origin_input_ids=origin_input_ids,
                )

            # Accumulate mlp output into partial_block
            output = partial_block + mlp_out
        else:
            self._log_md5(hidden_states, "input", self.layer_number)
            with profile("attn"):
                if need_do_attention():
                    hidden_states, context = self._forward_attention(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        swa_rotary_pos_emb=swa_rotary_pos_emb,
                        swa_rotary_pos_cos=swa_rotary_pos_cos,
                        swa_rotary_pos_sin=swa_rotary_pos_sin,
                        position_ids=position_ids,
                        attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params,
                        in_recompute=self.full_recompute,
                        input_ids=input_ids,
                        cu_seqlens=cu_seqlens,
                        docmask_mb_idx=docmask_mb_idx,
                        **kwargs,
                    )
            self._log_md5(
                hidden_states, "post_attn_residual", self.layer_number
            )
            with profile(timer_name):
                output = self._forward_mlp(
                    hidden_states,
                    input_ids=input_ids,
                    origin_input_ids=origin_input_ids,
                )
            self._log_md5(output, "layer_output", self.layer_number)
        if context is not None:
            return output, context
        return output

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        attn_mask_startend_row_indices: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rope_freqs_cis: Tensor | None = None,
        swa_rotary_pos_emb: Tensor | None = None,
        swa_rotary_pos_cos: Tensor | None = None,
        swa_rotary_pos_sin: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        in_recompute: bool = False,
        is_first_fwd: bool = False,
        block_attention_residuals: bool = False,
        input_ids: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor | None): Mask tensor for self-attention.
            context (Tensor | None): Context tensor for cross-attention.
            context_mask (Tensor | None): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor | None): Rotary positional embeddings.
            rotary_pos_cos (Tensor | None): Rotary embedding cosine.
            rotary_pos_sin (Tensor | None): Rotary embedding sine.
            rope_freqs_cis (Tensor | None): Rotary embedding frequency.
            swa_rotary_pos_emb (Tensor | None): Sliding Window Rotary positional embeddings.
            swa_rotary_pos_cos (Tensor | None): Sliding Window Rotary embedding cosine.
            swa_rotary_pos_sin (Tensor | None): Sliding Window Rotary embedding sine.
            attention_bias (Tensor | None): Bias tensor for Q * K.T.
            packed_seq_params (object, optional): Parameters for packed sequence processing.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            input_layernorm_output = recompute(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        self._log_md5(
            input_layernorm_output, "input_layernorm_out", self.layer_number
        )

        extra_kwargs = {}
        # Both indexer-bearing attentions need ``input_ids`` to build the
        # indexer-loss row mask: ``attn_mask_startend_row_indices`` cannot
        # express the trailing padding of a packed sequence, so only
        # ``input_ids != pad_token_id`` identifies the pad rows. The MLA branch
        # forwards it on to its core attention only when that core is the
        # non-absorbed-MQA one.
        if input_ids is not None and isinstance(
            self.self_attn,
            (DSv4HybridAttention, MultiLatentAttention, KimiDeltaAttention),
        ):
            extra_kwargs["input_ids"] = input_ids
        if isinstance(self.self_attn, KimiDeltaAttention):
            # Built once per step by the embedding; None makes KDA build its own.
            extra_kwargs["cu_seqlens"] = cu_seqlens
        if "shared_kv" in kwargs:
            extra_kwargs["shared_kv"] = kwargs["shared_kv"]

        if isinstance(self.self_attn, MultiLatentAttention):
            attention_output_with_bias = self.self_attn(
                input_layernorm_output,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
                in_recompute=in_recompute,
                past_key_values=kwargs.get("past_key_values"),
                layer_idx=self.layer_number,
                use_cache=kwargs.get("use_cache", False),
                **extra_kwargs,
            )
        elif rope_freqs_cis is not None:
            attention_output_with_bias = self.self_attn(
                input_layernorm_output,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                rope_freqs_cis=rope_freqs_cis,
                position_ids=position_ids,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                in_recompute=in_recompute,
                past_key_values=kwargs.get("past_key_values"),
                layer_idx=self.layer_number,
                use_cache=kwargs.get("use_cache", False),
                **extra_kwargs,
            )
        else:
            attention_output_with_bias = self.self_attn(
                input_layernorm_output,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                swa_rotary_pos_emb=swa_rotary_pos_emb,
                swa_rotary_pos_cos=swa_rotary_pos_cos,
                swa_rotary_pos_sin=swa_rotary_pos_sin,
                position_ids=position_ids,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                in_recompute=in_recompute,
                past_key_values=kwargs.get("past_key_values"),
                layer_idx=self.layer_number,
                use_cache=kwargs.get("use_cache", False),
                **extra_kwargs,
            )

        with paddle.enable_grad():
            if block_attention_residuals:
                attn_out, attn_bias = attention_output_with_bias
                if attn_bias is not None:
                    attn_out = attn_out + attn_bias
                hidden_states = paddle.nn.functional.dropout(
                    attn_out, p=self.hidden_dropout_prob, training=self.training
                )
                # hidden_states = attn_out
            else:
                hidden_states = self.self_attn_bda(
                    self.training, self.config.bias_dropout_fusion
                )(
                    attention_output_with_bias,
                    residual,
                    self.hidden_dropout_prob,
                )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(
            hidden_states
        )

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
        )

        if (
            isinstance(attention_output_with_bias, dict)
            and "context" in attention_output_with_bias
        ):
            context = attention_output_with_bias["context"]

        with paddle.enable_grad():
            residual.stop_gradient = False
            hidden_states = self.cross_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout_prob)

        # manually mark tensors that requires gradient in the first forward
        if is_first_fwd:
            hidden_states.stop_gradient = False

        return hidden_states, context

    def _forward_mlp(
        self,
        hidden_states,
        is_first_fwd=False,
        block_attention_residuals=False,
        input_ids=None,
        origin_input_ids=None,
        **kwargs,
    ):
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        if self.recompute_post_attention_layernorm:
            post_attention_layernorm_output = recompute(
                self.post_attention_layernorm, hidden_states
            )
        else:
            post_attention_layernorm_output = self.post_attention_layernorm(
                hidden_states
            )

        self._log_md5(
            post_attention_layernorm_output,
            "post_attn_layernorm_out",
            self.layer_number,
        )

        if self.recompute_mlp:
            _mlp_input_ids = (
                input_ids if isinstance(self.mlp, MoELayer) else None
            )
            _mlp_origin_input_ids = (
                origin_input_ids if isinstance(self.mlp, MoELayer) else None
            )

            def recompute_handler(
                post_attention_layernorm_output,
                _mlp_input_ids=None,
                _mlp_origin_input_ids=None,
            ):
                if _mlp_input_ids is not None:
                    mlp_output, bias = self.mlp(
                        post_attention_layernorm_output,
                        input_ids=_mlp_input_ids,
                        origin_input_ids=_mlp_origin_input_ids,
                    )
                else:
                    mlp_output, bias = self.mlp(post_attention_layernorm_output)
                if bias is None:
                    return mlp_output
                return mlp_output, bias

            mlp_output_with_bias = recompute(
                recompute_handler,
                post_attention_layernorm_output,
                _mlp_input_ids,
                _mlp_origin_input_ids,
            )
            if not isinstance(mlp_output_with_bias, tuple):
                mlp_output_with_bias = (
                    mlp_output_with_bias,
                    None,
                )  # bias is None
        else:
            if isinstance(self.mlp, MoELayer) and input_ids is not None:
                mlp_output_with_bias = self.mlp(
                    post_attention_layernorm_output,
                    input_ids=input_ids,
                    origin_input_ids=origin_input_ids,
                )
            else:
                mlp_output_with_bias = self.mlp(post_attention_layernorm_output)

        # Log MLP raw output before BDA
        if (
            TransformerLayer._LOG_LAYER_MD5
            and TransformerLayer._gpt_model_use_experimental_version
        ):
            _mlp_tensor = (
                mlp_output_with_bias[0]
                if isinstance(mlp_output_with_bias, tuple)
                else mlp_output_with_bias
            )
            self._log_md5(_mlp_tensor, "mlp_out", self.layer_number)

        with paddle.enable_grad():
            if block_attention_residuals:
                mlp_out, mlp_bias = mlp_output_with_bias
                if mlp_bias is not None:
                    mlp_out = mlp_out + mlp_bias
                hidden_states = paddle.nn.functional.dropout(
                    mlp_out, p=self.hidden_dropout_prob, training=self.training
                )
            else:
                hidden_states = self.mlp_bda(
                    self.training,
                    self.config.bias_dropout_fusion,
                )(
                    mlp_output_with_bias,
                    residual,
                    self.hidden_dropout_prob,
                )

        if is_first_fwd:
            hidden_states.stop_gradient = False

        return hidden_states

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        if isinstance(self.mlp, MoELayer):
            self.mlp.fp8_quant_weight(
                batch_mode=batch_mode, quant_transpose=quant_transpose
            )
        # Pre-quantize non-MoE fp8 Linear sublayers (attention projections,
        # dense MLP, shared expert, indexer). Each Linear.fp8_quant_weight
        # is a no-op when the layer is bf16.
        from paddlefleet.tensor_parallel.layers import (
            ColumnParallelLinear,
            Linear,
            RowParallelLinear,
        )

        seen = set()
        for m in self.sublayers(include_self=False):
            if not isinstance(
                m, (Linear, ColumnParallelLinear, RowParallelLinear)
            ):
                continue
            # MoE experts are handled by self.mlp.fp8_quant_weight above.
            if getattr(m, "is_expert", False):
                continue
            if id(m) in seen:
                continue
            seen.add(id(m))
            quant_fn = getattr(m, "fp8_quant_weight", None)
            if quant_fn is not None:
                quant_fn(batch_mode=batch_mode, quant_transpose=quant_transpose)

    def clear_fp8_quant_weight(self):
        if isinstance(self.mlp, MoELayer):
            self.mlp.clear_fp8_quant_weight()
        # Symmetric to fp8_quant_weight above: drop the per-Linear fp8
        # cache stashed on non-MoE Linear weights, otherwise post-optimizer
        # forwards keep using the pre-step quantized weight.
        from paddlefleet.tensor_parallel.layers import (
            ColumnParallelLinear,
            Linear,
            RowParallelLinear,
        )

        seen = set()
        for m in self.sublayers(include_self=False):
            if not isinstance(
                m, (Linear, ColumnParallelLinear, RowParallelLinear)
            ):
                continue
            # MoE experts are handled by self.mlp.clear_fp8_quant_weight above.
            if getattr(m, "is_expert", False):
                continue
            if id(m) in seen:
                continue
            seen.add(id(m))
            clear_fn = getattr(m, "clear_fp8_quant_weight", None)
            if clear_fn is not None:
                clear_fn()

    def use_fp8(self):
        if isinstance(self.mlp, MoELayer):
            return self.mlp.use_fp8()
        else:
            return self.config.fp8 is not None


class HyperConnectionTransformerLayer(TransformerLayer):
    """Transformer layer with Manifold-Constrained Hyper-Connections (mHC).

    Replaces the single residual stream with n parallel residual streams,
    using learned mappings H_pre, H_post, and H_res for aggregation,
    expansion, and mixing respectively.

    Input/output shape: [..., n*C] where n = num_residual_streams, C = hidden_size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: TransformerLayerSublayersSpec,
        layer_number: int = 1,
        hidden_dropout_prob: float | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            hidden_dropout_prob=hidden_dropout_prob,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        assert (
            sublayers_spec.self_attention_hyper_connection is not IdentityOp
        ), (
            "HyperConnectionTransformerLayer requires self_attention_hyper_connection. "
            "Use TransformerLayer instead if hyper connections are not needed."
        )
        assert sublayers_spec.mlp_hyper_connection is not IdentityOp, (
            "HyperConnectionTransformerLayer requires mlp_hyper_connection. "
            "Use TransformerLayer instead if hyper connections are not needed."
        )
        assert not config.block_attention_residuals, (
            "HyperConnectionTransformerLayer does not support block_attention_residuals."
        )

        # mHC treats attention and MLP as two independent layers (paper Fig. 3).
        # HyperConnectionModule uses this index to rotate the one-hot H_pre bias
        # across the n residual streams (mhc_single_stream_init only), so the two
        # sub-layers must get distinct numbers or half of the streams would
        # never be a "home stream".
        #
        # The rotation runs over the layer's logical position in the stack, which
        # is not ``self.layer_number``:
        #  - decoder layers are numbered index + num_empty_layers_add_in_head
        #    (get_gpt_decoder_layers_spec), so the offset is subtracted to keep
        #    the phase independent of the empty-head-layer count;
        #  - MTP layers are numbered by their own 0-based index and carry no such
        #    offset (get_gpt_mtp_layers_spec), so subtracting it would shift
        #    their phase and, with empty head layers, make the index negative.
        #    They keep reading and writing the n streams the backbone wrote, so
        #    they continue the decoder's rotation rather than restarting it --
        #    restarting would hand the first MTP sub-layer the same home stream
        #    as the last decoder sub-layer whenever n is odd.
        if self.is_mtp_layer:
            mhc_logical_index = (
                self.config.num_hidden_layers + self.layer_number
            )
        else:
            head_offset = (
                getattr(self.config, "num_empty_layers_add_in_head", 0) or 0
            )
            mhc_logical_index = self.layer_number - head_offset
        mhc_sublayer_base = 2 * mhc_logical_index

        self.self_attention_hyper_connection = build_spec_layer(
            sublayers_spec.self_attention_hyper_connection,
            config=self.config,
            layer_number=mhc_sublayer_base,
        )
        self.mlp_hyper_connection = build_spec_layer(
            sublayers_spec.mlp_hyper_connection,
            config=self.config,
            layer_number=mhc_sublayer_base + 1,
        )

        # The hyper-connection submodules are created after super().__init__()
        # (which already ran _mark_shared_no_hook_params on the base params), so
        # their params (mapping_proj.weight, alpha_pre/post/res, bias, ...) are
        # still uncolored. Under mtp_shared_last_layer these params are also
        # aliased into the MTP layer, so re-run the no-hook coloring now. It is
        # idempotent: already-colored base params are skipped.
        self._mark_shared_no_hook_params()

        # Consumer identity for the shared CSA document-mask metadata
        # (config.csa_share_docmask_meta): one forward counter per consumer, so
        # virtual-pipeline interleaving across chunks cannot mix them up.
        self._docmask_meta_key = (int(layer_number), bool(is_mtp_layer))
        if getattr(config, "csa_share_docmask_meta", False) or getattr(
            config, "mqa_share_docmask_meta", False
        ):
            from paddlefleet.transformer.doc_mask_meta_registry import (
                doc_mask_meta_registry,
            )

            doc_mask_meta_registry.register(self._docmask_meta_key)

        # mHC forward recompute config
        self.recompute_mhc_forward = (
            config.recompute_granularity == "selective"
            and module_needs_recompute("mhc_forward", self.layer_number, config)
        )

    def _fused_h_res_h_post_bda(
        self,
        hyper_connection,
        h_res,
        original_residual,
        h_post,
        layer_output_with_bias,
        enable_recompute,
    ):
        """Run fused_h_res_h_post_bda, optionally in a RecomputeWithoutOutput span.

        On the fp32 fast path the span keeps the up-casts that
        ``FusedHPostBDA.save_for_backward`` would otherwise pin -- the [..., n, C]
        residual and the [..., C] layer output, i.e. two more activation-sized
        buffers per half-layer -- out of the live set. On the sequential path
        (active dropout) it hides that path's dropout mask instead, which is a
        byte per element of the [..., n*C] output. Either way the span body runs
        under ``no_grad``, where an inner PyLayer's ``save_for_backward`` retains
        nothing, and backward re-runs the kernel from the low-precision inputs.

        ``hyper_connection.bda_span_pays_off`` vetoes the configurations where
        neither applies; ``enable_recompute`` is the caller's own switch on top
        of that.

        Returns ``(output, span)``. ``output`` is the [..., n*C] result; when
        ``span`` is not None the caller MUST hand it to
        ``_cast_and_discard_fused_bda`` -- see there for why the discard is not
        optional.
        """
        bda_kwargs = {
            "dropout_prob": self.hidden_dropout_prob,
            "training": self.training,
            "fused": self.config.bias_dropout_fusion,
        }
        # Only wrap when the call actually retains something the span can hide;
        # ``bda_span_pays_off`` owns that predicate because it depends on which
        # path ``fused_h_res_h_post_bda`` takes.
        if not hyper_connection.bda_span_pays_off(
            self.hidden_dropout_prob, self.training
        ):
            enable_recompute = False
        if not enable_recompute:
            output = hyper_connection.fused_h_res_h_post_bda(
                h_res=h_res,
                original_residual=original_residual,
                h_post=h_post,
                layer_output_with_bias=layer_output_with_bias,
                **bda_kwargs,
            )
            return output, None

        x, bias = layer_output_with_bias

        def _fused(h_res, original_residual, h_post, x, bias):
            return hyper_connection.fused_h_res_h_post_bda(
                h_res=h_res,
                original_residual=original_residual,
                h_post=h_post,
                layer_output_with_bias=(x, bias),
                **bda_kwargs,
            )

        span = RecomputeWithoutOutput()
        output = span.recompute(
            _fused,
            # h_res/h_post are outputs of the mHC-aggregate span, which clears
            # them with Tensor._clear_data(). detach() shares the holder, so the
            # alias that save_for_backward keeps would be emptied too and the
            # replay would read a dangling tensor. They are [..., n, n] and
            # [..., n], i.e. smaller than the residual by a factor of C, so clone
            # instead of depending on the hook registration order of the two
            # spans.
            h_res.clone(),
            original_residual,
            h_post.clone(),
            x,
            bias,
            # No dropout on the fast path, so the replay is deterministic
            # without paying for an RNG-state snapshot.
            preserve_rng_state=self.hidden_dropout_prob > 0.0 and self.training,
            share_grad_holder=True,
        )
        return output, span

    @staticmethod
    def _cast_and_discard_fused_bda(output, ori_dtype, span):
        """Cast the BDA result back to ``ori_dtype`` and close ``span``.

        The fp32 [..., n*C] BDA result is dead the moment it is cast: cast
        backward only needs the grad, not the input data. But the span pins it
        through ``self.outputs``, so without the discard the span would trade the
        two ``save_for_backward`` up-casts for a retained fp32 output and give
        most of the saving back. The discard is also what registers
        ``_recompute``, and ``_recompute`` is the only place that sets
        ``ctx.inputs``/``ctx.outputs`` for
        ``RecomputeWithoutOutputFunction.backward`` -- so it is mandatory, not an
        optimization. The hook goes on the cast result because that is the first
        tensor whose grad is produced after the fp32 data is dead; the cast
        result itself must stay live for the whole half-layer, so the span cannot
        be extended to swallow the cast.
        """
        casted = output.to(ori_dtype)
        if span is None:
            return casted
        if casted is output:
            # Tensor.to() is an identity when the dtype already matches, so the
            # span output and the tensor the rest of the layer holds would be the
            # same object and the discard would clear live data. Reachable when
            # the residual is already fp32 (fp32 training); give the caller its
            # own copy so the span still owns something discardable.
            casted = output.clone()
        span.discard_output_and_register_recompute(casted)
        return casted

    def _docmask_meta_kwargs(self):
        """Micro-batch slot for the shared CSA document-mask metadata, or ``{}``.

        Overrides the base opt-out hook: the DSv4-hybrid models run on this layer
        class, so this is where the sharing is enabled. Returns ``{}`` when
        ``config.csa_share_docmask_meta`` is off, so the switched-off path calls
        ``_forward_impl`` with exactly the arguments it did before.

        Called from ``TransformerLayer.forward``, i.e. outside the recompute
        wrapper: the counter must advance exactly once per (layer, micro-batch),
        whereas ``_forward_impl`` may be replayed by recompute.
        """
        if not (
            getattr(self.config, "csa_share_docmask_meta", False)
            or getattr(self.config, "mqa_share_docmask_meta", False)
        ):
            return {}
        from paddlefleet.transformer.doc_mask_meta_registry import (
            doc_mask_meta_registry,
        )

        return {
            "docmask_mb_idx": doc_mask_meta_registry.advance(
                self._docmask_meta_key, self.training
            )
        }

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        attn_mask_startend_row_indices: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rope_freqs_cis: Tensor | None = None,
        swa_rotary_pos_emb: Tensor | None = None,
        swa_rotary_pos_cos: Tensor | None = None,
        swa_rotary_pos_sin: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        in_recompute: bool = False,
        is_first_fwd: bool = False,
        **kwargs,
    ):
        """mHC attention forward: aggregate → layernorm → attention → fused_h_res_h_post_bda."""
        # Save n-stream residual for H_res mixing
        original_residual = hidden_states
        ori_dtype = hidden_states.dtype

        # mHC: aggregate n-stream → 1-stream
        if self.recompute_mhc_forward and self.training:
            self._attn_mhc_recompute = RecomputeWithoutOutput()
            aggregated, h_res, h_post = self._attn_mhc_recompute.recompute(
                self.self_attention_hyper_connection,
                hidden_states,
                preserve_rng_state=False,
                share_grad_holder=True,
            )
        else:
            aggregated, h_res, h_post = self.self_attention_hyper_connection(
                hidden_states
            )
        aggregated = aggregated.to(ori_dtype)

        # LayerNorm on aggregated single stream
        if self.recompute_input_layernorm:
            input_layernorm_output = recompute(self.input_layernorm, aggregated)
        else:
            input_layernorm_output = self.input_layernorm(aggregated)

        self._log_md5(
            input_layernorm_output, "input_layernorm_out", self.layer_number
        )

        # Self-attention
        extra_kwargs = {}
        if kwargs.get("input_ids") is not None and isinstance(
            self.self_attn, (DSv4HybridAttention, KimiDeltaAttention)
        ):
            extra_kwargs["input_ids"] = kwargs["input_ids"]
        if isinstance(self.self_attn, KimiDeltaAttention):
            # Built once per step by the embedding; None makes KDA build its own.
            extra_kwargs["cu_seqlens"] = kwargs.get("cu_seqlens")
        # Micro-batch slot for the shared document-mask metadata, decided in
        # ``forward`` (outside recompute) and only read here. This override is
        # the ``_forward_attention`` that runs whenever enable_hyper_connections
        # is set, which is the case for the DSv4-hybrid models. Both attention
        # classes derive their own mask group, so an MTP layer simply misses the
        # lookup (the trainer prebuilds no MTP group) and builds privately.
        if isinstance(
            self.self_attn, (DSv4HybridAttention, MultiLatentAttention)
        ):
            extra_kwargs["docmask_mb_idx"] = kwargs.get("docmask_mb_idx", -1)

        if isinstance(self.self_attn, MultiLatentAttention):
            attention_output_with_bias = self.self_attn(
                input_layernorm_output,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
                in_recompute=in_recompute,
                past_key_values=kwargs.get("past_key_values"),
                layer_idx=self.layer_number,
                use_cache=kwargs.get("use_cache", False),
                **extra_kwargs,
            )
        elif rope_freqs_cis is not None:
            attention_output_with_bias = self.self_attn(
                input_layernorm_output,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                rope_freqs_cis=rope_freqs_cis,
                position_ids=position_ids,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                in_recompute=in_recompute,
                past_key_values=kwargs.get("past_key_values"),
                layer_idx=self.layer_number,
                use_cache=kwargs.get("use_cache", False),
                **extra_kwargs,
            )
        else:
            attention_output_with_bias = self.self_attn(
                input_layernorm_output,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                swa_rotary_pos_emb=swa_rotary_pos_emb,
                swa_rotary_pos_cos=swa_rotary_pos_cos,
                swa_rotary_pos_sin=swa_rotary_pos_sin,
                position_ids=position_ids,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                in_recompute=in_recompute,
                past_key_values=kwargs.get("past_key_values"),
                layer_idx=self.layer_number,
                use_cache=kwargs.get("use_cache", False),
                **extra_kwargs,
            )

        # mHC: fused H_res + H_post + bias-dropout-add
        hidden_states, fused_span = self._fused_h_res_h_post_bda(
            self.self_attention_hyper_connection,
            h_res,
            original_residual,
            h_post,
            attention_output_with_bias,
            self.recompute_mhc_forward and self.training,
        )
        # Discard mhc.forward outputs (aggregated, h_res, h_post) after fused_bda consumed them
        if self.recompute_mhc_forward and self.training:
            self._attn_mhc_recompute.discard_output_and_register_recompute(
                hidden_states
            )
            self._attn_mhc_recompute = None
        hidden_states = self._cast_and_discard_fused_bda(
            hidden_states, ori_dtype, fused_span
        )

        # Cross attention (unchanged)
        residual = hidden_states
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(
            hidden_states
        )
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
        )
        if (
            isinstance(attention_output_with_bias, dict)
            and "context" in attention_output_with_bias
        ):
            context = attention_output_with_bias["context"]

        with paddle.enable_grad():
            residual.stop_gradient = False
            hidden_states = self.cross_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout_prob)

        if is_first_fwd:
            hidden_states.stop_gradient = False

        return hidden_states, context

    def _forward_mlp(
        self,
        hidden_states,
        is_first_fwd=False,
        input_ids=None,
        **kwargs,
    ):
        """mHC MLP forward: aggregate → layernorm → MLP → fused_h_res_h_post_bda."""
        # Save n-stream residual for H_res mixing
        original_residual = hidden_states
        ori_dtype = hidden_states.dtype

        # mHC: aggregate n-stream → 1-stream
        if self.recompute_mhc_forward and self.training:
            self._mlp_mhc_recompute = RecomputeWithoutOutput()
            aggregated, h_res, h_post = self._mlp_mhc_recompute.recompute(
                self.mlp_hyper_connection,
                hidden_states,
                preserve_rng_state=False,
                share_grad_holder=True,
            )
        else:
            aggregated, h_res, h_post = self.mlp_hyper_connection(hidden_states)
        aggregated = aggregated.to(ori_dtype)

        # LayerNorm on aggregated single stream
        if self.recompute_post_attention_layernorm:
            post_attention_layernorm_output = recompute(
                self.post_attention_layernorm, aggregated
            )
        else:
            post_attention_layernorm_output = self.post_attention_layernorm(
                aggregated
            )

        self._log_md5(
            post_attention_layernorm_output,
            "post_attn_layernorm_out",
            self.layer_number,
        )

        # MLP
        if self.recompute_mlp:
            _mlp_input_ids = (
                input_ids if isinstance(self.mlp, MoELayer) else None
            )

            def recompute_handler(
                post_attention_layernorm_output, _mlp_input_ids=None
            ):
                if _mlp_input_ids is not None:
                    mlp_output, bias = self.mlp(
                        post_attention_layernorm_output,
                        input_ids=_mlp_input_ids,
                    )
                else:
                    mlp_output, bias = self.mlp(post_attention_layernorm_output)
                if bias is None:
                    return mlp_output
                return mlp_output, bias

            mlp_output_with_bias = recompute(
                recompute_handler,
                post_attention_layernorm_output,
                _mlp_input_ids,
            )
            if not isinstance(mlp_output_with_bias, tuple):
                mlp_output_with_bias = (mlp_output_with_bias, None)
        else:
            if isinstance(self.mlp, MoELayer) and input_ids is not None:
                mlp_output_with_bias = self.mlp(
                    post_attention_layernorm_output, input_ids=input_ids
                )
            else:
                mlp_output_with_bias = self.mlp(post_attention_layernorm_output)

        # mHC: fused H_res + H_post + bias-dropout-add
        hidden_states, fused_span = self._fused_h_res_h_post_bda(
            self.mlp_hyper_connection,
            h_res,
            original_residual,
            h_post,
            mlp_output_with_bias,
            self.recompute_mhc_forward and self.training,
        )
        # Discard mhc.forward outputs (aggregated, h_res, h_post) after fused_bda consumed them
        if self.recompute_mhc_forward and self.training:
            self._mlp_mhc_recompute.discard_output_and_register_recompute(
                hidden_states
            )
            self._mlp_mhc_recompute = None
        hidden_states = self._cast_and_discard_fused_bda(
            hidden_states, ori_dtype, fused_span
        )

        if is_first_fwd:
            hidden_states.stop_gradient = False

        return hidden_states


class HySparseTransformerLayer(TransformerLayer):
    """Transformer layer with cross-layer KV sharing."""

    def _mtp_enabled(self, is_mtp):
        return (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not is_mtp
            and not self.config.mtp_load_weight_only
            and not self.config.enable_mtp_magic_send
            and not self.config.separate_mtp_input
        )

    def _mtp_split(self, dict_args, is_mtp):
        """Split MTP-stacked tensors into main-decoder parts.

        Mirrors ``TransformerLayer.forward`` (the base class does this inline).
        MTP concatenates the main hidden state and the ``num_nextn_predict_layers``
        shifted hidden states along the batch dimension; input_ids / position_ids /
        masks carry their MTP parts along the seq dimension. Split so the layer
        body sees a consistent (main-decoder) batch/seq, mutating ``dict_args`` in
        place. Returns a context dict for :meth:`_mtp_restore`, or ``None`` when
        MTP is not active.
        """
        if not self._mtp_enabled(is_mtp):
            return None
        n = self.config.num_nextn_predict_layers
        # Under use_erndata position_ids / masks are already
        # main-decoder length L (per-doc shifting happens inside the MTP layer
        # via roll_tensor), so the L+K -> L seq-dim trims below must be
        # skipped. Mirrors the ``_mtp_is_megatron`` guards in
        # ``TransformerLayer.forward``.
        _mtp_is_megatron = getattr(self.config, "use_erndata", False)
        ctx = {
            "mtp_ids": None,
            "mtp_input_ids": None,
            "rotary_pos_emb_full": None,
            "rotary_pos_cos_full": None,
            "rotary_pos_sin_full": None,
            "attn_mask_mtp": None,
        }

        # hidden_states: split along batch dim -> main + mtp parts
        tensor_list = paddle.split(dict_args["hidden_states"], n + 1)
        hidden_states = tensor_list[0]
        ctx["mtp_input"] = tuple(tensor_list[1:])
        dict_args["hidden_states"] = hidden_states

        # position_ids: split along seq dim (ernie5 L+K path only)
        if (
            not self.config.gpt_model_use_experimental_version
            and not _mtp_is_megatron
        ):
            if (
                "position_ids" in dict_args
                and dict_args["position_ids"] is not None
            ):
                position_ids = dict_args["position_ids"]
                dict_args["position_ids"] = position_ids[:, :-n]
                ctx["mtp_ids"] = position_ids[:, -n:]

        # rotary_pos_emb/cos/sin: trim to main-decoder seq length
        if self.config.sequence_parallel:
            main_seq_len = (
                hidden_states.shape[0] * self.config.tensor_model_parallel_size
            )
        else:
            main_seq_len = hidden_states.shape[1]
        if (
            "rotary_pos_emb" in dict_args
            and dict_args["rotary_pos_emb"] is not None
        ):
            ctx["rotary_pos_emb_full"] = dict_args["rotary_pos_emb"]
            if self.config.sequence_parallel:
                dict_args["rotary_pos_emb"] = ctx["rotary_pos_emb_full"][
                    :main_seq_len
                ]
            else:
                dict_args["rotary_pos_emb"] = ctx["rotary_pos_emb_full"][
                    :, :main_seq_len
                ]
        if (
            "rotary_pos_cos" in dict_args
            and dict_args["rotary_pos_cos"] is not None
        ):
            ctx["rotary_pos_cos_full"] = dict_args["rotary_pos_cos"]
            dict_args["rotary_pos_cos"] = ctx["rotary_pos_cos_full"][
                :, :main_seq_len
            ]
        if (
            "rotary_pos_sin" in dict_args
            and dict_args["rotary_pos_sin"] is not None
        ):
            ctx["rotary_pos_sin_full"] = dict_args["rotary_pos_sin"]
            dict_args["rotary_pos_sin"] = ctx["rotary_pos_sin_full"][
                :, :main_seq_len
            ]

        # input_ids: split along seq dim (only when it carries mtp tokens)
        if "input_ids" in dict_args and dict_args["input_ids"] is not None:
            full_input_ids = dict_args["input_ids"]
            seq_lens = hidden_states.shape[
                0 if self.config.sequence_parallel else 1
            ]
            if get_context_parallel_world_size() > 1:
                seq_lens *= get_context_parallel_world_size()
            if full_input_ids.shape[-1] > seq_lens:
                dict_args["input_ids"] = full_input_ids[:, :-n].contiguous()
                ctx["mtp_input_ids"] = full_input_ids[:, -n:].contiguous()

        # attn_mask_startend_row_indices: split along seq dim (old dataflow,
        # ernie5 L+K path only -- under megatron the mask is already length L)
        if (
            not self.config.experimental_dataflow
            and not _mtp_is_megatron
            and "attn_mask_startend_row_indices" in dict_args
            and dict_args["attn_mask_startend_row_indices"] is not None
        ):
            mask = dict_args["attn_mask_startend_row_indices"]
            dict_args["attn_mask_startend_row_indices"] = mask[:, :, :-n, :]
            ctx["attn_mask_mtp"] = mask[:, :, -n:, :]

        return ctx

    def _mtp_restore(self, dict_args, output, ctx):
        """Re-stack MTP outputs and restore full-length auxiliary tensors.

        Inverse of :meth:`_mtp_split`. Returns the batch-stacked hidden state and
        restores ``dict_args`` (input_ids / position_ids / rotary / mask) to full
        length for the next layer.
        """
        hidden_states_concat = paddle.concat([output, *ctx["mtp_input"]])

        if not self.config.gpt_model_use_experimental_version:
            if "position_ids" in dict_args and ctx["mtp_ids"] is not None:
                dict_args["position_ids"] = paddle.concat(
                    [dict_args["position_ids"], ctx["mtp_ids"]], axis=1
                )
        if ctx["rotary_pos_emb_full"] is not None:
            dict_args["rotary_pos_emb"] = ctx["rotary_pos_emb_full"]
        if ctx["rotary_pos_cos_full"] is not None:
            dict_args["rotary_pos_cos"] = ctx["rotary_pos_cos_full"]
        if ctx["rotary_pos_sin_full"] is not None:
            dict_args["rotary_pos_sin"] = ctx["rotary_pos_sin_full"]
        if ctx["mtp_input_ids"] is not None and "input_ids" in dict_args:
            dict_args["input_ids"] = paddle.concat(
                [dict_args["input_ids"], ctx["mtp_input_ids"]], axis=1
            )
        if (
            not self.config.experimental_dataflow
            and "attn_mask_startend_row_indices" in dict_args
            and ctx["attn_mask_mtp"] is not None
        ):
            dict_args["attn_mask_startend_row_indices"] = paddle.concat(
                [
                    dict_args["attn_mask_startend_row_indices"],
                    ctx["attn_mask_mtp"],
                ],
                axis=2,
            )
        return hidden_states_concat

    def forward(
        self,
        dict_args: dict,
    ):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        dict_args.pop("dynamic_inference_decode_only", None)
        keys = tuple(dict_args.keys())
        values = tuple(dict_args.values())

        is_mtp = dict_args.pop("is_mtp", False)
        TransformerLayer._skip_mtp_probes = (
            is_mtp  # Suppress MD5 probes for MTP passes
        )

        # MTP stacks the main decoder + shifted hidden states along the batch
        # dim (see MTP module's paddle.concat(axis=0)); the base
        # TransformerLayer.forward splits them off so attention / MoE router see
        # a consistent batch, then re-stacks. HySparseTransformerLayer must do
        # the same, otherwise the batch-2 stacked hidden reaches the MoE router
        # with batch-1 input_ids and trips its shape assertion. Split now,
        # restore after the layer body runs.
        mtp_ctx = self._mtp_split(dict_args, is_mtp)

        if self.full_recompute or (not has_recovered()):

            def dict_args_get_clone(key):
                """Clone is necessary for some args."""
                value = dict_args.get(key, None)
                return value.clone() if value is not None else None

            # Mirror the base TransformerLayer recompute path: recompute both
            # when full_recompute is set AND inside the RECOVER_STEP recovery
            # window (not has_recovered()), so recovering training keeps the
            # same reduced activation footprint. Activation-offload settings are
            # threaded via _compute_act_offload_kwargs (consumed by recompute).
            offload_kwargs = self._compute_act_offload_kwargs()
            outputs = recompute(
                self._forward_impl,
                hidden_states=dict_args["hidden_states"],
                attention_mask=dict_args.get("attention_mask", None),
                attn_mask_startend_row_indices=dict_args_get_clone(
                    "attn_mask_startend_row_indices"
                ),
                context=dict_args.get("context", None),
                context_mask=dict_args.get("context_mask", None),
                rotary_pos_emb=dict_args_get_clone("rotary_pos_emb"),
                rotary_pos_cos=dict_args_get_clone("rotary_pos_cos"),
                rotary_pos_sin=dict_args_get_clone("rotary_pos_sin"),
                swa_rotary_pos_emb=dict_args_get_clone("swa_rotary_pos_emb"),
                swa_rotary_pos_cos=dict_args_get_clone("swa_rotary_pos_cos"),
                swa_rotary_pos_sin=dict_args_get_clone("swa_rotary_pos_sin"),
                position_ids=dict_args_get_clone("position_ids"),
                attention_bias=dict_args.get("attention_bias", None),
                packed_seq_params=dict_args.get("packed_seq_params", None),
                input_ids=dict_args.get("input_ids", None),
                origin_input_ids=dict_args.get("origin_input_ids", None),
                shared_key=dict_args.get("shared_key", None),
                shared_block_indices=dict_args.get(
                    "shared_block_indices", None
                ),
                **offload_kwargs,
            )
        else:
            outputs = self._forward_impl(**dict_args)

        if isinstance(outputs, tuple):
            output, shared_key, shared_block_indices = outputs
        else:
            output, shared_key, shared_block_indices = outputs, None, None

        rst = OrderedDict()
        rst = {"hidden_states": output}
        if mtp_ctx is not None:
            # Re-stack main + mtp hidden along batch and restore full-length
            # auxiliary tensors (input_ids / position_ids / rotary / mask) into
            # dict_args for the next layer.
            rst["hidden_states"] = self._mtp_restore(dict_args, output, mtp_ctx)
        if shared_key is not None:
            rst["shared_key"] = shared_key
            rst["shared_block_indices"] = shared_block_indices
        rst = {**dict_args, **rst}
        return rst

    def _forward_impl(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        attn_mask_startend_row_indices: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        swa_rotary_pos_emb: Tensor | None = None,
        swa_rotary_pos_cos: Tensor | None = None,
        swa_rotary_pos_sin: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        input_ids: Tensor | None = None,
        shared_key: Tensor | None = None,
        shared_block_indices: Tensor | None = None,
        origin_input_ids: Tensor | None = None,
        **kwargs,
    ):
        timer_name = "moe-mlp" if isinstance(self.mlp, MoELayer) else "mlp"

        # 使用统一的 shared_kv 参数处理输入输出:
        # 1. 对于 swa 层是输入, 只消费 shared_kv，不生产;
        # 2. 对于 full 层是输出, 只生产 shared_kv, 不消费.
        if self.self_attn.is_swa:
            if shared_key is None or shared_block_indices is None:
                raise ValueError(
                    f"HySparse SWA layer (layer_number={self.layer_number}) "
                    "requires shared KV latent and top-k block indices from a "
                    "preceding full-attention layer, but none were provided. "
                    "Ensure the first backbone attention layer is a full "
                    "(non-SWA) attention layer so it can produce the shared "
                    "state -- e.g. set window_attn_skip_freq so that layer 0 "
                    "is full attention rather than SWA."
                )
            shared_kv = [shared_key, shared_block_indices]
        else:
            shared_kv = []

        self._log_md5(hidden_states, "input", self.layer_number)
        with profile("attn"):
            hidden_states, context = self._forward_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                swa_rotary_pos_emb=swa_rotary_pos_emb,
                swa_rotary_pos_cos=swa_rotary_pos_cos,
                swa_rotary_pos_sin=swa_rotary_pos_sin,
                position_ids=position_ids,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                in_recompute=self.full_recompute,
                input_ids=input_ids,
                shared_kv=shared_kv,
                **kwargs,
            )
        assert context is None, (
            "HySparseTransformerLayer doesn't support cross-attention."
        )
        self._log_md5(hidden_states, "post_attn_residual", self.layer_number)
        with profile(timer_name):
            output = self._forward_mlp(
                hidden_states,
                input_ids=input_ids,
                origin_input_ids=origin_input_ids,
            )
        self._log_md5(output, "layer_output", self.layer_number)

        if (not self.self_attn.is_swa) and shared_kv:
            shared_key, shared_block_indices = shared_kv
            if self.training and not paddle.is_grad_enabled():
                shared_key.stop_gradient = False
            return output, shared_key, shared_block_indices
        return output


class TransformerLayerWithOverlap(TransformerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.recompute_mlp
        assert not self.recompute_input_layernorm
        assert not self.recompute_post_attention_layernorm
        if isinstance(self.mlp, MoELayer):
            assert not self.mlp.gate.norm_topk_prob, (
                "By enabling `forward_backward_overlap_scheduler`, you should not use `norm_topk_prob` in TopKRouter."
            )
            assert self.mlp.expert_model_parallel_size > 1, (
                "By enabling `forward_backward_overlap_scheduler`, you should use expert parallel."
            )
            if self.mlp.moe_token_dispatcher_type not in (
                "deepep",
                "hybridep",
            ):
                raise ValueError(
                    f"TransformerLayerWithOverlap "
                    f"(forward_backward_overlap_scheduler) requires "
                    f"moe_token_dispatcher_type='deepep' or 'hybridep', but "
                    f"got '{self.mlp.moe_token_dispatcher_type}'. The "
                    f"'{self.mlp.moe_token_dispatcher_type}' dispatcher does "
                    f"not implement the overlap dataflow contract "
                    f"(_comm_manager, token_dispatch_overlap, dispatched_* "
                    f"metadata) required by the overlap scheduler. Please "
                    f"either switch to deepep/hybridep or disable "
                    f"forward_backward_overlap_scheduler."
                )

    def compute_attention(self, dict_args, is_first_fwd=False):
        with profile("attn"):
            return self._forward_attention(
                **dict_args, is_first_fwd=is_first_fwd
            )

    def compute_mlp(self, hidden_states, is_first_fwd=False):
        timer_name = "moe-mlp" if isinstance(self.mlp, MoELayer) else "mlp"
        with profile(timer_name):
            return self._forward_mlp(hidden_states, is_first_fwd=is_first_fwd)

    def pre_process_compute(self, hidden_states):
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        residuals = hidden_states
        (
            capacity,
            topk_weights,
            topk_indices,
            gates_masked,
            mask,
            priorities,
            aux_loss,
            z_loss,
        ) = self.mlp.compute_gate(hidden_states)
        return (
            residual,
            hidden_states,
            residuals,
            topk_weights,
            topk_indices,
            aux_loss,
            z_loss,
        )

    def dispatch_preprocess_compute(self, args):
        hidden_states, topk_weights, topk_indices = args

        hidden_states, token_indices, token_weights = (
            self.mlp.dispatch_preprocess(
                (hidden_states, topk_weights, topk_indices)
            )
        )
        return hidden_states, token_indices, token_weights

    def post_process_compute(self, args, is_first_fwd=False):
        mlp_output, residual = args
        with paddle.enable_grad():
            output = self.mlp_bda(
                self.training, self.config.bias_dropout_fusion
            )((mlp_output, None), residual, self.hidden_dropout_prob)
        if is_first_fwd:
            output.stop_gradient = False
        return output


class TransformerLayerNode(ScheduleNode):
    def __init__(self, node, config, name="", layer_number=1):
        super().__init__(fwd_func=None, name=name)
        self.config = config
        self.layer_number = layer_number
        self.attn_node = ScheduleNode(
            node.compute_attention, name="attn_compute"
        )
        self.full_recompute = node.full_recompute
        self._is_sparse = True if isinstance(node.mlp, MoELayer) else False
        if self._is_sparse:
            self.pre_process_node = ScheduleNode(
                node.pre_process_compute, name="pre_process_compute"
            )
            self.dispatch_preprocess_node = ScheduleNode(
                node.dispatch_preprocess_compute,
                name="dispatch_preprocess_compute",
            )
            self.gate_node = ScheduleNode(
                node.mlp.compute_gate, name="gate_compute"
            )
            self.dispatch_node = ScheduleNode(
                node.mlp.compute_dispatch, name="dispatch_compute"
            )
            self.mlp_node = ScheduleNode(
                node.mlp.compute_experts, name="mlp_compute"
            )
            self.combine_node = ScheduleNode(
                node.mlp.compute_combine, name="combine_compute"
            )
            self.aux_loss_node = ScheduleNode(
                node.mlp.aux_loss_compute, name="aux_loss_compute"
            )
            self.post_process_node = ScheduleNode(
                node.post_process_compute, name="post_process_compute"
            )
            self.group_id = node.mlp.token_dispatcher._comm_manager.group.id
        else:
            self.mlp_node = ScheduleNode(node.compute_mlp, name="mlp_compute")

    def forward(self, inputs):
        inputs.pop("dynamic_inference_decode_only", None)
        mtp_tmp_dict = None
        assert (
            self.config.num_nextn_predict_layers is None
            or self.config.num_nextn_predict_layers == 0
        ), (
            f"current support num_nextn_predict_layers == 0, but get {self.config.num_nextn_predict_layers}"
        )
        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
        ):
            mtp_tmp_dict = {}
            for i in range(self.config.num_nextn_predict_layers):
                key = f"decoder_input_{i}"
                assert key in inputs
                mtp_tmp_dict[key] = inputs.pop(key)
        if self._is_sparse:
            if self.full_recompute:
                attn_state = tensors_clone(inputs)
                self.attn_recompute_args = attn_state
            hidden_states, context = self.attn_node.forward(
                inputs, is_first_fwd=self.full_recompute
            )
            (
                residual,
                hidden_states,
                residuals,
                topk_weights,
                topk_indices,
                aux_loss,
                z_loss,
            ) = self.pre_process_node.forward(hidden_states)

            hidden_states, token_indices, token_weights = (
                self.dispatch_preprocess_node.forward(
                    (hidden_states, topk_weights, topk_indices)
                )
            )

            hidden_states = self.dispatch_node.forward(
                (hidden_states, token_indices, token_weights),
                async_finish=True,
            )
            dispatch_fw_event = deep_ep.get_event_from_comm_stream(
                self.group_id
            )
            dispatch_fw_event.calc_stream_wait(self.group_id)

            if self.full_recompute:
                mlp_state = tensors_clone(hidden_states)
                self.mlp_recompute_args = mlp_state
            hidden_states = self.mlp_node.forward(
                hidden_states, is_first_fwd=self.full_recompute
            )

            hidden_states = self.combine_node.forward(
                hidden_states, async_finish=True
            )
            combine_fw_event = deep_ep.get_event_from_comm_stream(self.group_id)
            combine_fw_event.calc_stream_wait(self.group_id)

            hidden_states = self.aux_loss_node.forward(
                (hidden_states, aux_loss, z_loss, residuals)
            )

            self.post_process_recompute_args = (hidden_states, residual)
            output = self.post_process_node.forward(
                (hidden_states, residual), is_first_fwd=self.full_recompute
            )
        else:
            if self.full_recompute:
                attn_state = tensors_clone(inputs)
                self.attn_recompute_args = attn_state
            hidden_states, context = self.attn_node.forward(
                inputs, is_first_fwd=self.full_recompute
            )

            if self.full_recompute:
                mlp_state = tensors_clone(hidden_states)
                self.mlp_recompute_args = mlp_state
            output = self.mlp_node.forward(
                hidden_states, is_first_fwd=self.full_recompute
            )
        rst = {"hidden_states": output}
        if context is not None:
            rst["context"] = context
        rst = {**inputs, **rst}
        if mtp_tmp_dict is not None:
            rst = {**rst, **mtp_tmp_dict}
        return rst

    def backward(self, output_grad):
        if self.full_recompute:
            self.recompute_forward()
        mtp_tmp_grad = None
        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
        ):
            # maybe error, fix this by concat and split
            assert len(output_grad) == self.config.num_nextn_predict_layers + 1
            mtp_tmp_grad = output_grad[1:]
            output_grad = [output_grad[0]]
        if self._is_sparse:
            output_grad, residual_grad = self.post_process_node.backward(
                output_grad
            )

            output_grad, aux_loss_grad, z_loss_grad, residuals_grad = (
                self.aux_loss_node.backward(output_grad)
            )

            output_grad = self.combine_node.backward(output_grad)
            combine_bw_event = deep_ep.get_event_from_comm_stream(self.group_id)
            combine_bw_event.calc_stream_wait(self.group_id)
            output_grad = self.mlp_node.backward(output_grad)

            (output_grad, token_indices_grad, token_weights_grad) = (
                self.dispatch_node.backward(output_grad)
            )
            dispatch_bw_event = deep_ep.get_event_from_comm_stream(
                self.group_id
            )
            dispatch_bw_event.calc_stream_wait(self.group_id)

            (
                output_grad,
                topk_weights_grad,
                topk_indices_grad,
            ) = self.dispatch_preprocess_node.backward(
                (output_grad, token_indices_grad, token_weights_grad)
            )

            output_grad = self.pre_process_node.backward(
                (
                    residual_grad,
                    output_grad,
                    residuals_grad,
                    topk_weights_grad,
                    topk_indices_grad,
                    aux_loss_grad,
                    z_loss_grad,
                )
            )

            output_grad = self.attn_node.backward(output_grad)
        else:
            output_grad = self.mlp_node.backward(output_grad)
            output_grad = self.attn_node.backward(output_grad)

        if mtp_tmp_grad is not None:
            output_grad = output_grad + tuple(mtp_tmp_grad)
        return output_grad

    def recompute_forward(self):
        """Recompute the forwarding of mlp, attn and post_process"""
        if self._is_sparse:
            self.attn_node.forward(self.attn_recompute_args)
            del self.attn_recompute_args

            self.mlp_node.forward(self.mlp_recompute_args)
            del self.mlp_recompute_args

            self.post_process_node.forward(self.post_process_recompute_args)
            del self.post_process_recompute_args
        else:
            self.attn_node.forward(self.attn_recompute_args)
            del self.attn_recompute_args

            self.mlp_node.forward(self.mlp_recompute_args)
            del self.mlp_recompute_args


class TransformerLayerOverlappedScheduleNode(ScheduleNode):
    """Overlap schedule for TransformerLayer"""

    def __init__(self, forward_node, backward_node, name=""):
        assert isinstance(forward_node, TransformerLayerNode)
        assert isinstance(backward_node, TransformerLayerNode)
        super().__init__(fwd_func=None, name=name)
        self.forward_node = forward_node
        self.backward_node = backward_node
        self.config = forward_node.config

    def forward_backward(self, inputs, output_grad, split_bw=False):
        assert not split_bw
        mtp_tmp_dict = None
        mtp_tmp_grad = None
        if (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
        ):
            # maybe error, fix this by concat and split
            assert len(output_grad) == self.config.num_nextn_predict_layers + 1
            mtp_tmp_dict = {}
            mtp_tmp_grad = output_grad[1:]
            output_grad = [output_grad[0]]
            for i in range(self.config.num_nextn_predict_layers):
                key = f"decoder_input_{i}"
                assert key in inputs
                mtp_tmp_dict[key] = inputs.pop(key)
        if self.forward_node._is_sparse and self.backward_node._is_sparse:
            if self.backward_node.full_recompute:
                self.backward_node.recompute_forward()
            # 1. POST(B)
            output_grad, residual_grad = (
                self.backward_node.post_process_node.backward(output_grad)
            )
            output_grad, aux_loss_grad, z_loss_grad, residuals_grad = (
                self.backward_node.aux_loss_node.backward(output_grad)
            )

            # 2. COMBINE(B)
            output_grad = self.backward_node.combine_node.backward(output_grad)
            combine_bw_event = deep_ep.get_event_from_comm_stream(
                self.backward_node.group_id
            )

            # 3. ATTN(F)
            if self.forward_node.full_recompute:
                attn_state = tensors_clone(inputs)
                self.forward_node.attn_recompute_args = attn_state
            hidden_states, context = self.forward_node.attn_node.forward(
                inputs, is_first_fwd=self.forward_node.full_recompute
            )
            (
                residual,
                hidden_states,
                residuals,
                topk_weights,
                topk_indices,
                aux_loss,
                z_loss,
            ) = self.forward_node.pre_process_node.forward(hidden_states)

            hidden_states, token_indices, token_weights = (
                self.forward_node.dispatch_preprocess_node.forward(
                    (hidden_states, topk_weights, topk_indices)
                )
            )

            # 4. DISPATCH(F)
            hidden_states = self.forward_node.dispatch_node.forward(
                (hidden_states, token_indices, token_weights),
                async_finish=True,
            )
            dispatch_fw_event = deep_ep.get_event_from_comm_stream(
                self.forward_node.group_id
            )

            # 5. MLP(B)
            combine_bw_event.calc_stream_wait(self.backward_node.group_id)
            output_grad = self.backward_node.mlp_node.backward(output_grad)

            # 6. DISPATCH(B)
            output_grad, token_indices_grad, token_weights_grad = (
                self.backward_node.dispatch_node.backward(output_grad)
            )
            dispatch_bw_event = deep_ep.get_event_from_comm_stream(
                self.backward_node.group_id
            )

            # 7. MLP(F)
            dispatch_fw_event.calc_stream_wait(self.forward_node.group_id)
            if self.forward_node.full_recompute:
                mlp_state = tensors_clone(hidden_states)
                self.forward_node.mlp_recompute_args = mlp_state
            hidden_states = self.forward_node.mlp_node.forward(
                hidden_states, is_first_fwd=self.forward_node.full_recompute
            )

            # 8. COMBINE(F)
            hidden_states = self.forward_node.combine_node.forward(
                hidden_states, async_finish=True
            )
            combine_fw_event = deep_ep.get_event_from_comm_stream(
                self.forward_node.group_id
            )

            # 9. ATTN(B)
            dispatch_bw_event.calc_stream_wait(self.backward_node.group_id)
            (
                output_grad,
                topk_weights_grad,
                topk_indices_grad,
            ) = self.backward_node.dispatch_preprocess_node.backward(
                (output_grad, token_indices_grad, token_weights_grad)
            )

            output_grad = self.backward_node.pre_process_node.backward(
                (
                    residual_grad,
                    output_grad,
                    residuals_grad,
                    topk_weights_grad,
                    topk_indices_grad,
                    aux_loss_grad,
                    z_loss_grad,
                )
            )
            output_grad = self.backward_node.attn_node.backward(output_grad)

            # 10. POST(F)
            combine_fw_event.calc_stream_wait(self.forward_node.group_id)
            hidden_states = self.forward_node.aux_loss_node.forward(
                (hidden_states, aux_loss, z_loss, residuals)
            )
            if self.forward_node.full_recompute:
                self.forward_node.post_process_recompute_args = (
                    hidden_states,
                    residual,
                )
            output = self.forward_node.post_process_node.forward(
                (hidden_states, residual),
                is_first_fwd=self.forward_node.full_recompute,
            )
            rst = {"hidden_states": output}
            if context is not None:
                rst["context"] = context
            rst = {**inputs, **rst}
        else:
            # 1f
            rst = self.forward_node.forward(inputs)

            # 1b
            output_grad = self.backward_node.backward(output_grad)

        if mtp_tmp_dict is not None:
            rst = {**rst, **mtp_tmp_dict}
            output_grad = output_grad + tuple(mtp_tmp_grad)
        return rst, output_grad


@dataclass
class Gemma4TransformerLayerSublayersSpec(TransformerLayerSublayersSpec):
    """Extended spec for Gemma4 norm structure.

    Adds: post_self_attn_layernorm, pre_mlp_layernorm, post_mlp_layernorm.
    MoELayer internally handles post_moe_layernorm, post_shared_expert_layernorm,
    and pre_feedforward_layernorm_2.
    """

    post_self_attn_layernorm: LayerSpec | type = IdentityOp
    pre_mlp_layernorm: LayerSpec | type = IdentityOp
    post_mlp_layernorm: LayerSpec | type = IdentityOp


class Gemma4TransformerLayer(TransformerLayer):
    """Gemma4 transformer layer aligned with HF Gemma4TextDecoderLayer.

    Note: This layer has a fundamentally different forward topology (5-norm +
    layer_scalar) that cannot be parameterized into the base TransformerLayer.
    It is kept as a standalone subclass and wired via attention_layer_type="gemma4"
    through the standard get_gpt_layer_local_spec path.

    Forward flow:
        residual = x
        x = input_layernorm(x)
        x = self_attn(x)
        x = post_self_attn_layernorm(x)
        x = residual + x

        residual = x
        x = pre_mlp_layernorm(x)
        x = moe(x, residual)
        x = post_mlp_layernorm(x)
        x = (residual + x) * layer_scalar
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: Gemma4TransformerLayerSublayersSpec,
        layer_number: int = 1,
        hidden_dropout_prob: float | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config,
            sublayers_spec,
            layer_number,
            hidden_dropout_prob,
            pg_collection,
            is_mtp_layer,
        )

        norm_input_parallel = (
            self.config.sequence_parallel
            and self.config.tensor_model_parallel_size > 1
        )

        self.post_self_attn_layernorm = build_spec_layer(
            sublayers_spec.post_self_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            input_is_parallel=norm_input_parallel,
        )
        self.pre_mlp_layernorm = build_spec_layer(
            sublayers_spec.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            input_is_parallel=norm_input_parallel,
        )
        self.post_mlp_layernorm = build_spec_layer(
            sublayers_spec.post_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            input_is_parallel=norm_input_parallel,
        )

        # Per-layer output scalar (Google checkpoint key: "skip_scale").
        # Registered as a non-trainable buffer aligned with HF/Megatron: initialized
        # to 1.0 (no-op) and overwritten when loading pretrained weights.
        self.register_buffer("layer_scalar", paddle.ones([1], dtype="float32"))

    def _forward_impl(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        attn_mask_startend_row_indices: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        swa_rotary_pos_emb: Tensor | None = None,
        swa_rotary_pos_cos: Tensor | None = None,
        swa_rotary_pos_sin: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params=None,
        input_ids: Tensor | None = None,
        origin_input_ids: Tensor | None = None,
        **kwargs,
    ):
        # === Attention block ===
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            swa_rotary_pos_emb=swa_rotary_pos_emb,
            swa_rotary_pos_cos=swa_rotary_pos_cos,
            swa_rotary_pos_sin=swa_rotary_pos_sin,
            position_ids=position_ids,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            in_recompute=getattr(self, "full_recompute", False),
            past_key_values=kwargs.get("past_key_values"),
            layer_idx=getattr(self, "layer_number", None),
            use_cache=kwargs.get("use_cache", False),
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # === MLP/MoE block ===
        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)

        if isinstance(self.mlp, MoELayer):
            hidden_states, _ = self.mlp(
                hidden_states,
                input_ids=input_ids,
                residual=residual,
                origin_input_ids=origin_input_ids,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = (residual + hidden_states) * self.layer_scalar

        return hidden_states
